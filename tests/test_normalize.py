"""Normalization of Kalshi messages.

The snapshot, delta and trade payloads below are the examples published in the
Kalshi AsyncAPI reference, so the field names and formats are the documented
ones. This code has not been run against a live connection.
"""

import pytest

import klob
from klob import reference as R
from klob import tape as T
from klob.kalshi.normalize import Normalizer

SNAPSHOT = {
    "type": "orderbook_snapshot",
    "sending_ts_ms": 1669149841234,
    "sid": 2,
    "seq": 2,
    "msg": {
        "market_ticker": "FED-23DEC-T3.00",
        "market_id": "9b0f6b43-5b68-4f9f-9f02-9a2d1b8ac1a1",
        "yes_dollars_fp": [["0.0800", "300.00"], ["0.2200", "333.00"]],
        "no_dollars_fp": [["0.5400", "20.00"], ["0.5600", "146.00"]],
    },
}
DELTA = {
    "type": "orderbook_delta",
    "sending_ts_ms": 1669149841123,
    "sid": 2,
    "seq": 3,
    "msg": {
        "market_ticker": "FED-23DEC-T3.00",
        "market_id": "9b0f6b43-5b68-4f9f-9f02-9a2d1b8ac1a1",
        "price_dollars": "0.9600",
        "delta_fp": "-54.00",
        "side": "yes",
        "ts": "2022-11-22T20:44:01Z",
        "ts_ms": 1669149841000,
    },
}
TRADE = {
    "type": "trade",
    "sending_ts_ms": 1669149841123,
    "sid": 11,
    "seq": 2,
    "msg": {
        "trade_id": "d91bc706-ee49-470d-82d8-11418bda6fed",
        "market_ticker": "HIGHNY-22DEC23-B53.5",
        "yes_price_dollars": "0.3600",
        "no_price_dollars": "0.6400",
        "count_fp": "136.00",
        "taker_side": "no",
        "taker_outcome_side": "no",
        "taker_book_side": "ask",
        "is_block_trade": False,
        "ts": 1669149841,
        "ts_ms": 1669149841000,
    },
}
FED = "FED-23DEC-T3.00"


def rows(events):
    return [(int(e["ts"]), int(e["qty"]), int(e["price"]), int(e["type"]), int(e["side"])) for e in events]


def test_documented_snapshot_delta_and_trade():
    n = Normalizer()
    n.feed({"recv_ns": 100, "msg": SNAPSHOT})
    n.feed({"recv_ns": 200, "msg": DELTA})
    n.feed({"recv_ns": 300, "msg": TRADE})
    tapes = n.tapes()
    assert rows(tapes[FED]) == [
        (100, 0, 0, T.SNAPSHOT_BEGIN, 0),
        (100, 30000, 800, T.SNAPSHOT_LEVEL, T.YES),
        (100, 33300, 2200, T.SNAPSHOT_LEVEL, T.YES),
        (100, 2000, 5400, T.SNAPSHOT_LEVEL, T.NO),
        (100, 14600, 5600, T.SNAPSHOT_LEVEL, T.NO),
        (100, 0, 0, T.SNAPSHOT_END, 0),
        (200, -5400, 9600, T.DELTA, T.YES),
    ]
    assert tapes[FED]["exch_ts"][-1] == 1669149841000 * 1_000_000
    assert rows(tapes["HIGHNY-22DEC23-B53.5"]) == [(300, 13600, 3600, T.TRADE, T.NO)]
    assert tapes[FED].dtype == T.EVENT_DTYPE


def test_normalized_tape_replays_in_both_books():
    n = Normalizer()
    n.feed({"recv_ns": 100, "msg": SNAPSHOT})
    events = n.tapes()[FED]
    books = [R.Book()] + ([klob.Book()] if klob.HAVE_CORE else [])
    for book in books:
        book.apply(events)
        assert (book.yes_bid, book.yes_ask) == (2200, 4400)  # best NO bid 56c offers YES at 44c
        assert book.qty(T.NO, 5600) == 14600 and book.valid


def delta(seq, price="0.2200", size="1.00", sid=2):
    m = {"type": "orderbook_delta", "sid": sid, "seq": seq, "msg": dict(DELTA["msg"], price_dollars=price, delta_fp=size)}
    return m


def test_sequence_gap_invalidates_the_book_until_the_next_snapshot():
    n = Normalizer()
    n.feed({"recv_ns": 100, "msg": SNAPSHOT})  # seq 2
    n.feed({"recv_ns": 200, "msg": delta(3)})
    n.feed({"recv_ns": 300, "msg": delta(5)})  # seq 4 never arrived
    events = n.tapes()[FED]
    assert [int(t) for t in events["type"][-3:]] == [T.DELTA, T.GAP, T.DELTA]
    assert n.stats["gaps"] == 1 and n.stats["sequence_breaks"] == 1
    book = R.Book()
    book.apply(events)
    assert not book.valid


def test_reconnect_marks_every_tracked_market_stale():
    n = Normalizer()
    n.feed({"recv_ns": 100, "msg": SNAPSHOT})
    n.feed({"recv_ns": 500, "event": "connected"})
    n.feed({"recv_ns": 600, "msg": dict(SNAPSHOT, seq=1)})  # sequence restarts without a spurious break
    events = n.tapes()[FED]
    assert int(events["type"][6]) == T.GAP
    assert n.stats["sequence_breaks"] == 0
    book = R.Book()
    book.apply(events)
    assert book.valid


def test_block_trades_are_dropped_and_time_never_runs_backwards():
    n = Normalizer()
    n.feed({"recv_ns": 300, "msg": TRADE})
    n.feed({"recv_ns": 250, "msg": dict(TRADE, seq=3, msg=dict(TRADE["msg"], is_block_trade=True))})
    n.feed({"recv_ns": 200, "msg": dict(TRADE, seq=4)})
    events = n.tapes()["HIGHNY-22DEC23-B53.5"]
    assert events["ts"].tolist() == [300, 300]
    assert n.stats["block_trades_skipped"] == 1


def test_rest_polls_become_a_snapshot_then_trades_then_deltas():
    def poll(ts, yes, no, trades=()):
        return {"recv_ns": ts, "rest": {"ticker": "MKT", "orderbook_fp": {"yes_dollars": yes, "no_dollars": no}, "trades": list(trades)}}

    trade = {
        "trade_id": "t1",
        "ticker": "MKT",
        "count_fp": "3.00",
        "yes_price_dollars": "0.4500",
        "no_price_dollars": "0.5500",
        "taker_outcome_side": "no",
        "taker_book_side": "ask",
        "created_time": "2026-10-01T12:00:00Z",
        "is_block_trade": False,
    }
    n = Normalizer()
    n.feed(poll(100, [["0.4500", "10.00"]], [["0.5300", "20.00"]]))
    n.feed(poll(200, [["0.4500", "5.00"], ["0.4400", "8.00"]], [], [trade]))
    assert rows(n.tapes()["MKT"])[4:] == [
        (200, 300, 4500, T.TRADE, T.NO),
        (200, 800, 4400, T.DELTA, T.YES),
        (200, -500, 4500, T.DELTA, T.YES),
        (200, -2000, 5300, T.DELTA, T.NO),
    ]

    # In the simulator, 3 of the 5 that left the 45c level were traded and 2 were cancelled.
    bt = R.Backtester(n.tapes()["MKT"], R.SimConfig(queue_model=R.OPTIMISTIC))
    bt.run_until(150)
    bt.submit(T.YES, 4500, 100)
    bt.run_until(10**9)
    assert bt.sim_stats["matched_trade_qty"] == 300 and bt.sim_stats["cancel_qty"] == 200 + 2000
    (order,) = bt.orders()
    assert order.ahead == 500  # 1000 ahead, 300 traded, 200 cancelled (optimistic: from the front)
