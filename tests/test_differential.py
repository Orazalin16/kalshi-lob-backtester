"""The C++ core and the Python reference must agree exactly.

Tapes here are adversarial rather than realistic: tied timestamps, deltas that
would drive a level negative, out-of-range prices, sequence gaps, snapshots in
the middle of the session, trades with no matching delta and deltas with no
matching trade. Orders are submitted and cancelled at random with random
latencies. After every step the two implementations are compared on the full
book, every live order, every fill and the account.
"""

import random

import numpy as np
import pytest

from klob import reference as R
from klob import tape as T

# Resting prices are drawn so the two sides usually leave a spread (YES bids up
# to 49c, NO bids up to 50c, so the best YES offer is 50c), with a few wild
# prices mixed in to cross the book on purpose.
YES_PRICES = [3800 + 100 * i for i in range(12)]  # 38c .. 49c
NO_PRICES = [3900 + 100 * i for i in range(12)]  # 39c .. 50c
WILD = [3000, 5500, 6200, 7000]
ORDER_PRICES = {T.YES: [4500, 4700, 4800, 4900, 4950, 5000, 5100], T.NO: [4600, 4800, 4900, 5000, 5050, 5100, 5300]}


def level_price(rng: random.Random, side: int) -> int:
    if rng.random() < 0.04:
        return rng.choice(WILD)
    return rng.choice(YES_PRICES if side == T.YES else NO_PRICES)


def random_tape(rng: random.Random, n: int) -> np.ndarray:
    rows = [(0, 0, 0, 0, T.SNAPSHOT_BEGIN, 0)]
    for p in YES_PRICES:
        rows.append((0, 0, rng.randrange(100, 5000), p, T.SNAPSHOT_LEVEL, T.YES))
    for p in NO_PRICES:
        rows.append((0, 0, rng.randrange(100, 5000), p, T.SNAPSHOT_LEVEL, T.NO))
    rows.append((0, 0, 0, 0, T.SNAPSHOT_END, 0))
    ts = 1
    for _ in range(n):
        ts += rng.choice([0, 0, 1, 1, 7, 1000, 60_000, 5_000_000, 80_000_000])
        u = rng.random()
        side = rng.randrange(2)
        price = level_price(rng, side)
        if u < 0.50:
            qty = rng.choice([-1, -1, 1]) * rng.randrange(1, 2500)
            rows.append((ts, ts, qty, price, T.DELTA, side))
        elif u < 0.88:
            # A trade whose taker is `side` hits the other side's bids.
            rest = level_price(rng, 1 - side)
            if rng.random() < 0.7:  # trade near the touch, where our orders are
                rest = rng.choice((YES_PRICES if side == T.NO else NO_PRICES)[-4:])
            yes_price = 10000 - rest if side == T.YES else rest
            qty = rng.randrange(1, 2500)
            pair = [(ts, ts, qty, yes_price, T.TRADE, side)]
            v = rng.random()
            if v < 0.8:  # usually the matching book delta is published too, before or after
                pair.append((ts, ts, -rng.choice([qty, qty, qty // 2 + 1]), rest, T.DELTA, 1 - side))
                if rng.random() < 0.5:
                    pair.reverse()
            for i, row in enumerate(pair):
                rows.append((ts + i * rng.choice([0, 1, 2_000_000]),) + row[1:])
            ts = rows[-1][0]
        elif u < 0.92:
            rows.append((ts, ts, rng.randrange(1, 100), rng.choice([-1, 10001, 5000]), T.DELTA, rng.choice([0, 1, 2])))
        elif u < 0.94:
            rows.append((ts, ts, 0, 0, T.GAP, 0))
        elif u < 0.98:
            rows.append((ts, ts, 0, 0, T.SNAPSHOT_BEGIN, 0))
            for sd in (T.YES, T.NO):
                for p in rng.sample(YES_PRICES if sd == T.YES else NO_PRICES, 7):
                    rows.append((ts, ts, rng.randrange(1, 4000), p, T.SNAPSHOT_LEVEL, sd))
            rows.append((ts, ts, 0, 0, T.SNAPSHOT_END, 0))
        else:
            rows.append((ts, ts, 5, price, 9, side))  # unknown event type
    return T.make_events(rows)


def random_actions(rng: random.Random, bt_ref):
    """Order actions for one step, chosen from the reference's view of live orders."""
    out = []
    for _ in range(rng.randrange(3)):
        live = bt_ref.orders()
        if live and rng.random() < 0.35:
            out.append(("cancel", rng.choice(live).id))
        elif rng.random() < 0.05:
            out.append(("cancel", rng.randrange(1, 500)))  # usually unknown or already closed
        else:
            side = rng.randrange(2)
            out.append(("submit", side, rng.choice(ORDER_PRICES[side]), rng.randrange(1, 3000)))
    return out


def state(bt, is_core: bool):
    orders = [(o.id, o.side, o.price, o.qty, o.remaining, o.ahead, int(o.state), o.cancel_requested) for o in bt.orders()]
    closed = [(o.id, o.remaining, int(o.state)) for o in bt.drain_closed()]
    depth = tuple((tuple(int(x) for x in px), tuple(int(x) for x in sz)) for px, sz in (bt.depth(0, 50), bt.depth(1, 50)))
    return dict(
        now=bt.now,
        cursor=bt.cursor,
        valid=bt.valid,
        top=tuple(bt.top()),
        depth=depth,
        orders=orders,
        closed=closed,
        account=(bt.position, bt.cash, bt.fees, bt.volume, bt.n_fills),
        last_trade=bt.last_trade,
        sim=bt.sim_stats,
        book=bt.book_stats,
    )


@pytest.mark.parametrize("seed", range(150))
def test_core_matches_reference(core, seed):
    rng = random.Random(seed)
    events = random_tape(rng, 400)
    kw = dict(
        order_latency_ns=rng.choice([0, 1, 1000, 3_000_000, 200_000_000]),
        cancel_latency_ns=rng.choice([0, 1, 1000, 3_000_000, 200_000_000]),
        attribution_window_ns=rng.choice([0, 1, 3_000_000, 50_000_000, 10**9]),
        maker_fee_ppm=rng.choice([0, 17_500, 70_000]),
        balance_precision_micro=rng.choice([100, 10_000]),
    )
    model = rng.randrange(3)
    a = core.Backtester(events, core.SimConfig(queue_model=core.QueueModel(model), **kw))
    b = R.Backtester(events, R.SimConfig(queue_model=model, **kw))

    stamps = events["ts"].tolist()
    end = stamps[-1]
    t = 0
    step = 0
    while t < end + 10**9:
        if rng.random() < 0.4 and b.cursor < len(stamps):
            # Stop exactly on an event's timestamp, so that zero-latency orders
            # arrive in the same nanosecond as book changes.
            t = max(t, stamps[min(b.cursor + rng.randrange(4), len(stamps) - 1)])
        else:
            t += rng.choice([1, 500, 100_000, 20_000_000, 400_000_000])
        more_a, more_b = a.run_until(t), b.run_until(t)
        assert more_a == more_b
        sa, sb = state(a, True), state(b, False)
        assert sa == sb, f"diverged at step {step}, t={t}"
        for action in random_actions(rng, b):
            if action[0] == "submit":
                assert a.submit(*action[1:]) == b.submit(*action[1:])
            else:
                assert a.cancel(action[1]) == b.cancel(action[1])
        step += 1

    fa = [tuple(int(x) for x in f) for f in a.fills()]
    fb = [tuple(int(x) for x in f) for f in b.fills()]
    assert fa == fb


@pytest.mark.parametrize("seed", range(20))
def test_book_only_replay_matches(core, seed):
    events = random_tape(random.Random(1000 + seed), 3000)
    a, b = core.Book(), R.Book()
    assert a.apply(events) == b.apply(events)
    assert a.stats == b.stats
    for side in (0, 1):
        pa, qa = a.depth(side, 100)
        pb, qb = b.depth(side, 100)
        assert pa.tolist() == pb and qa.tolist() == qb
        assert a.best(side) == b.best(side)


def test_submit_validation(core):
    events = random_tape(random.Random(0), 5)
    for bt in (core.Backtester(events, core.SimConfig()), R.Backtester(events, R.SimConfig())):
        for bad in [(0, 0, 100), (0, 10000, 100), (0, 5000, 0), (1, -5, 100)]:
            with pytest.raises(ValueError):
                bt.submit(*bad)


def test_random_tapes_exercise_the_simulator(core):
    """Guards the differential test itself: agreement on tapes where nothing
    happens would prove nothing."""
    totals = dict(fills=0, rejected=0, canceled=0, cancel_qty=0, matched=0, unmatched=0, fees=0, partial=0)
    for seed in range(40):
        rng = random.Random(seed)
        events = random_tape(rng, 400)
        b = R.Backtester(events, R.SimConfig(order_latency_ns=1000, cancel_latency_ns=1000, maker_fee_ppm=17_500))
        t, end = 0, int(events["ts"][-1])
        while t < end:
            t += rng.choice([500, 100_000, 20_000_000, 400_000_000])
            b.run_until(t)
            for action in random_actions(rng, b):
                getattr(b, action[0])(*action[1:])
        st = b.sim_stats
        ids = [f[1] for f in b.fills()]
        totals["fills"] += len(ids)
        totals["partial"] += len(ids) - len(set(ids))
        totals["rejected"] += st["rejected"]
        totals["canceled"] += st["canceled"]
        totals["cancel_qty"] += st["cancel_qty"]
        totals["matched"] += st["matched_trade_qty"]
        totals["unmatched"] += st["unmatched_trade_qty"]
        totals["fees"] += b.fees
    assert totals["fills"] > 400 and totals["partial"] > 100
    assert totals["rejected"] > 100 and totals["canceled"] > 100
    assert min(totals["cancel_qty"], totals["matched"], totals["unmatched"], totals["fees"]) > 0
