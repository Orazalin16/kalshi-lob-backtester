"""Records raw Kalshi market data to gzip-compressed JSON lines.

The recorder stores exactly what the exchange sent plus the local receive time,
and does nothing else. Turning messages into book events is a separate, offline
step (normalize.py), so a bug in parsing never costs recorded data.

Two sources are supported:

* ``record_ws``: the WebSocket feed (order book snapshot + deltas, public
  trades). This is the real L2 stream and needs an API key, because Kalshi
  authenticates the WebSocket handshake even for public channels.
* ``record_rest``: polling of the public REST order book and trade endpoints.
  No key is needed. The result is coarser: changes inside a polling interval
  are seen only as their net effect.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import time
from pathlib import Path

WS_PROD = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
WS_DEMO = "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2"
WS_PATH = "/trade-api/ws/v2"


class SequenceGap(Exception):
    pass


def _sequence_broken(last_seq: dict[int, int], msg: dict) -> bool:
    if msg.get("type") not in ("orderbook_snapshot", "orderbook_delta"):
        return False
    sid, seq = msg.get("sid"), msg.get("seq")
    prev = last_seq.get(sid)
    last_seq[sid] = seq
    return prev is not None and seq != prev + 1


class RawWriter:
    def __init__(self, path: str | Path) -> None:
        self._f = gzip.open(Path(path), "at", encoding="utf-8")
        self.lines = 0

    def write(self, record: dict) -> None:
        self._f.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.lines += 1

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        self._f.close()


async def record_ws(tickers: list[str], out: str | Path, signer, url: str = WS_PROD, seconds: float | None = None) -> int:
    """Streams order book and trade messages for `tickers` until cancelled.

    Reconnects with exponential backoff. Each (re)connection writes a marker
    line; the normalizer turns it into a gap, because anything may have changed
    while we were away and only the next snapshot can be trusted.
    """
    import websockets

    writer = RawWriter(out)
    deadline = None if seconds is None else time.monotonic() + seconds
    backoff = 1.0
    try:
        while deadline is None or time.monotonic() < deadline:
            try:
                async with websockets.connect(url, additional_headers=signer.headers("GET", WS_PATH)) as ws:
                    writer.write({"recv_ns": time.time_ns(), "event": "connected", "tickers": tickers})
                    for cmd_id, channel in enumerate(("orderbook_delta", "trade"), start=1):
                        await ws.send(
                            json.dumps(
                                {"id": cmd_id, "cmd": "subscribe", "params": {"channels": [channel], "market_tickers": tickers}}
                            )
                        )
                    backoff = 1.0
                    last_seq: dict[int, int] = {}
                    while deadline is None or time.monotonic() < deadline:
                        timeout = None if deadline is None else max(deadline - time.monotonic(), 0.01)
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout)
                        except asyncio.TimeoutError:
                            break
                        msg = json.loads(raw)
                        writer.write({"recv_ns": time.time_ns(), "msg": msg})
                        if writer.lines % 2000 == 0:
                            writer.flush()
                        if _sequence_broken(last_seq, msg):
                            # A missed book message cannot be repaired in place.
                            # Reconnecting resubscribes and brings a fresh snapshot.
                            raise SequenceGap(f"sid {msg.get('sid')} jumped to seq {msg.get('seq')}")
            except (OSError, SequenceGap, websockets.WebSocketException) as exc:
                writer.write({"recv_ns": time.time_ns(), "event": "disconnected", "error": repr(exc)})
                writer.flush()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
    finally:
        writer.close()
    return writer.lines


def record_rest(ticker: str, out: str | Path, interval_s: float = 1.0, seconds: float | None = None, client=None) -> int:
    """Polls the public order book and trade endpoints for one market."""
    from .rest import KalshiREST

    client = client or KalshiREST()
    writer = RawWriter(out)
    seen: set[str] = set()
    first = True
    deadline = None if seconds is None else time.monotonic() + seconds
    try:
        while deadline is None or time.monotonic() < deadline:
            started = time.monotonic()
            book = client.orderbook(ticker)
            trades = client.trades(ticker)
            recv_ns = time.time_ns()
            fresh = [t for t in trades if t["trade_id"] not in seen]
            seen.update(t["trade_id"] for t in trades)
            # Trades that happened before the first poll are already reflected in the book.
            writer.write(
                {"recv_ns": recv_ns, "rest": {"ticker": ticker, "orderbook_fp": book, "trades": [] if first else fresh}}
            )
            first = False
            if writer.lines % 30 == 0:
                writer.flush()
            time.sleep(max(0.0, interval_s - (time.monotonic() - started)))
    finally:
        writer.close()
    return writer.lines
