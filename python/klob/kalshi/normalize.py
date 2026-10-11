"""Turns recorded Kalshi messages into event tapes, one per market.

Message shapes follow the Kalshi AsyncAPI spec (orderbook_snapshot,
orderbook_delta, trade). Prices and sizes arrive as fixed-point strings and are
converted exactly.

Book deltas carry a per-subscription sequence number. If one is skipped, the
book for every market on that subscription is no longer known, so a GAP event
is written; the backtester refuses to quote until the next snapshot.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .. import tape as T
from ..fixedpoint import price_to_ticks, qty_to_units

SIDE = {"yes": T.YES, "no": T.NO}


def read_raw(path: str | Path) -> Iterator[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class Normalizer:
    def __init__(self) -> None:
        self.rows: dict[str, list[tuple]] = defaultdict(list)
        self.stats: dict[str, int] = defaultdict(int)
        self._seq: dict[int, int] = {}  # subscription id -> last sequence number
        self._sid_tickers: dict[int, set[str]] = defaultdict(set)
        self._book_sids: set[int] = set()
        self._last_ts: dict[str, int] = defaultdict(int)
        self._rest_book: dict[str, tuple[dict, dict]] = {}

    # -- output ------------------------------------------------------------

    def _emit(self, ticker: str, ts: int, exch_ts: int, qty: int, price: int, typ: int, side: int) -> None:
        ts = max(ts, self._last_ts[ticker])  # receive times must not run backwards
        self._last_ts[ticker] = ts
        self.rows[ticker].append((ts, exch_ts, qty, price, typ, side))

    def _gap(self, ticker: str, ts: int) -> None:
        self._emit(ticker, ts, 0, 0, 0, T.GAP, 0)
        self.stats["gaps"] += 1

    def tapes(self) -> dict[str, np.ndarray]:
        return {ticker: _to_events(rows) for ticker, rows in self.rows.items()}

    # -- input -------------------------------------------------------------

    def feed(self, record: dict) -> None:
        ts = int(record["recv_ns"])
        if "msg" in record:
            self._ws_message(ts, record["msg"])
        elif "rest" in record:
            self._rest_poll(ts, record["rest"])
        elif record.get("event") == "connected":
            # A fresh connection: sequence numbers restart and every book we
            # were tracking is stale until its new snapshot arrives.
            for ticker in list(self.rows):
                self._gap(ticker, ts)
            self._seq.clear()
            self._sid_tickers.clear()
            self._book_sids.clear()

    def feed_all(self, records: Iterable[dict]) -> "Normalizer":
        for r in records:
            self.feed(r)
        return self

    def _check_seq(self, ts: int, sid: int, seq: int) -> None:
        last = self._seq.get(sid)
        self._seq[sid] = seq
        if last is not None and seq != last + 1:
            self.stats["sequence_breaks"] += 1
            if sid in self._book_sids:
                for ticker in self._sid_tickers[sid]:
                    self._gap(ticker, ts)

    def _ws_message(self, ts: int, m: dict) -> None:
        kind = m.get("type")
        body = m.get("msg") or {}
        ticker = body.get("market_ticker")
        if kind in ("orderbook_snapshot", "orderbook_delta", "trade") and ticker:
            sid = int(m["sid"])
            if kind != "trade":
                self._book_sids.add(sid)
                self._sid_tickers[sid].add(ticker)
            self._check_seq(ts, sid, int(m["seq"]))
        if kind == "orderbook_snapshot":
            sent = int(m.get("sending_ts_ms") or 0) * 1_000_000
            self._snapshot(ticker, ts, sent, body.get("yes_dollars_fp") or [], body.get("no_dollars_fp") or [])
        elif kind == "orderbook_delta":
            exch = int(body.get("ts_ms") or 0) * 1_000_000
            self._emit(
                ticker, ts, exch, qty_to_units(body["delta_fp"]), price_to_ticks(body["price_dollars"]), T.DELTA, SIDE[body["side"]]
            )
            self.stats["deltas"] += 1
        elif kind == "trade":
            self._trade(ticker, ts, int(body.get("ts_ms") or 0) * 1_000_000, body)
        else:
            self.stats["other_messages"] += 1

    def _snapshot(self, ticker: str, ts: int, exch: int, yes: list, no: list) -> None:
        self._emit(ticker, ts, exch, 0, 0, T.SNAPSHOT_BEGIN, 0)
        for side, levels in ((T.YES, yes), (T.NO, no)):
            for price, size in levels:
                self._emit(ticker, ts, exch, qty_to_units(size), price_to_ticks(price), T.SNAPSHOT_LEVEL, side)
        self._emit(ticker, ts, exch, 0, 0, T.SNAPSHOT_END, 0)
        self.stats["snapshots"] += 1

    def _trade(self, ticker: str, ts: int, exch: int, body: dict) -> None:
        if body.get("is_block_trade"):
            self.stats["block_trades_skipped"] += 1  # matched off-book; never touched the queue
            return
        taker = body.get("taker_outcome_side") or body.get("taker_side")
        self._emit(
            ticker, ts, exch, qty_to_units(body["count_fp"]), price_to_ticks(body["yes_price_dollars"]), T.TRADE, SIDE[taker]
        )
        self.stats["trades"] += 1

    def _rest_poll(self, ts: int, poll: dict) -> None:
        """Converts consecutive REST snapshots into deltas, with the trades seen
        since the last poll placed just before them so that the simulator can
        tell trading from cancelling."""
        ticker = poll["ticker"]
        ob = poll["orderbook_fp"]
        book = tuple({price_to_ticks(p): qty_to_units(q) for p, q in ob.get(k) or []} for k in ("yes_dollars", "no_dollars"))
        prev = self._rest_book.get(ticker)
        self._rest_book[ticker] = book
        if prev is None:
            self._snapshot(ticker, ts, 0, ob.get("yes_dollars") or [], ob.get("no_dollars") or [])
            return
        for tr in sorted(poll.get("trades") or [], key=lambda t: t["created_time"]):
            self._trade(ticker, ts, _rfc3339_ns(tr["created_time"]), tr)
        for side in (T.YES, T.NO):
            for price in sorted(set(prev[side]) | set(book[side])):
                change = book[side].get(price, 0) - prev[side].get(price, 0)
                if change:
                    self._emit(ticker, ts, 0, change, price, T.DELTA, side)
                    self.stats["deltas"] += 1


def normalize_file(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    n = Normalizer().feed_all(read_raw(path))
    return n.tapes(), dict(n.stats)


def _rfc3339_ns(text: str) -> int:
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1e9)


def _to_events(rows: list[tuple]) -> np.ndarray:
    out = np.zeros(len(rows), dtype=T.EVENT_DTYPE)
    if rows:
        for name, col in zip(("ts", "exch_ts", "qty", "price", "type", "side"), zip(*rows)):
            out[name] = col
    return out
