"""Hand-worked cases with exact expected numbers, run on both implementations.

Sizes are in units of 0.01 contract and prices in units of $0.0001, so
"1000 @ 4500" is ten contracts bid at 45 cents.
"""

import pytest

import klob
from klob import reference as R
from klob import tape as T
from klob.tape import DELTA, NO, TRADE, YES

CONS, PROP, OPT = 0, 1, 2
MS = 1_000_000


def snapshot(yes: dict, no: dict, ts: int = 0):
    rows = [(ts, 0, 0, 0, T.SNAPSHOT_BEGIN, 0)]
    rows += [(ts, 0, q, p, T.SNAPSHOT_LEVEL, YES) for p, q in yes.items()]
    rows += [(ts, 0, q, p, T.SNAPSHOT_LEVEL, NO) for p, q in no.items()]
    rows.append((ts, 0, 0, 0, T.SNAPSHOT_END, 0))
    return rows


# YES bid 45c x 10, NO bid 53c x 20: the market is 45 bid / 47 offered.
BASE = snapshot({4500: 1000}, {5300: 2000})


@pytest.fixture(params=["core", "reference"])
def make(request):
    backend = klob if request.param == "core" else R
    if backend is klob and not klob.HAVE_CORE:
        pytest.skip("C++ extension not built")

    def _make(rows, model=PROP, **kw):
        qm = klob.QueueModel(model) if backend is klob else model
        kw.setdefault("attribution_window_ns", 50 * MS)
        return backend.Backtester(T.make_events(rows), backend.SimConfig(queue_model=qm, **kw))

    return _make


def only_order(bt):
    (o,) = bt.orders()
    return o


def fills(bt):
    return [(int(f[0]), int(f[2]), int(f[4]), int(f[5])) for f in bt.fills()]  # ts, qty, price, side


def test_joins_the_back_and_fills_after_the_queue_trades(make):
    rows = BASE + [
        (10, 0, 500, 4500, DELTA, YES),  # arrives behind us
        (20, 0, 700, 4500, TRADE, NO),  # a seller hits the 45c bids
        (21, 0, -700, 4500, DELTA, YES),
        (30, 0, 1000, 4500, TRADE, NO),
        (31, 0, -800, 4500, DELTA, YES),
    ]
    bt = make(rows)
    bt.run_until(5)
    bt.submit(YES, 4500, 300)
    bt.run_until(25)
    assert only_order(bt).ahead == 300  # 1000 ahead, 700 traded
    bt.run_until(100)
    assert fills(bt) == [(30, 300, 4500, YES)]
    assert bt.position == 300 and bt.cash == -300 * 4500
    assert bt.orders() == []


@pytest.mark.parametrize("model, expected", [(CONS, 1000), (PROP, 750), (OPT, 500)])
def test_cancel_attribution_by_model(make, model, expected):
    rows = BASE + [
        (10, 0, 1000, 4500, DELTA, YES),  # 1000 behind us; the level is now 2000
        (20, 0, -500, 4500, DELTA, YES),  # 500 cancelled from somewhere
        (200 * MS, 0, 1, 3000, DELTA, YES),
    ]
    bt = make(rows, model)
    bt.run_until(5)
    bt.submit(YES, 4500, 100)
    bt.run_until(30)
    assert only_order(bt).ahead == 1000  # still inside the attribution window
    bt.run_until(100 * MS)
    assert only_order(bt).ahead == expected


def test_displayed_size_bounds_the_queue_even_when_conservative(make):
    rows = BASE + [(20, 0, -600, 4500, DELTA, YES)]
    bt = make(rows, CONS)
    bt.run_until(5)
    bt.submit(YES, 4500, 100)
    bt.run_until(100 * MS)
    assert only_order(bt).ahead == 400


@pytest.mark.parametrize("delta_first", [False, True])
@pytest.mark.parametrize("model", [CONS, PROP, OPT])
def test_trade_and_delta_order_does_not_matter(make, model, delta_first):
    pair = [(20, 0, 1200, 4500, TRADE, NO), (21, 0, -1000, 4500, DELTA, YES)]
    if delta_first:
        pair = [(20, 0, -1000, 4500, DELTA, YES), (21, 0, 1200, 4500, TRADE, NO)]
    bt = make(BASE + pair + [(200 * MS, 0, 1, 3000, DELTA, YES)], model)
    bt.run_until(5)
    bt.submit(YES, 4500, 500)
    bt.run_until(300 * MS)
    # 1000 ahead, 1200 sold: 200 reaches us regardless of message order or model.
    assert [(q, p) for _, q, p, _ in fills(bt)] == [(200, 4500)]
    assert only_order(bt).ahead == 0 and only_order(bt).remaining == 300


def test_order_joining_after_a_delta_is_not_advanced_by_its_late_trade(make):
    rows = BASE + [
        (10, 0, -300, 4500, DELTA, YES),  # the book update for a trade...
        (12, 0, 300, 4500, TRADE, NO),  # ...whose print arrives after we joined
        (200 * MS, 0, 1, 3000, DELTA, YES),
    ]
    bt = make(rows, OPT)
    bt.run_until(10)
    bt.submit(YES, 4500, 100)  # zero latency: joins at t=10 behind the 700 still displayed
    bt.run_until(11)
    assert only_order(bt).ahead == 700
    bt.run_until(300 * MS)
    assert only_order(bt).ahead == 700 and fills(bt) == []


def test_order_joining_after_a_cancel_is_not_advanced_by_it(make):
    rows = BASE + [(10, 0, -300, 4500, DELTA, YES), (200 * MS, 0, 1, 3000, DELTA, YES)]
    bt = make(rows, OPT)
    bt.run_until(10)
    bt.submit(YES, 4500, 100)  # joins at t=10, when the level already shows 700
    bt.run_until(11)
    assert only_order(bt).ahead == 700
    bt.run_until(300 * MS)  # the decrease times out as a cancel, but it happened before we joined
    assert only_order(bt).ahead == 700


def test_trade_at_a_worse_price_fills_an_improving_order(make):
    rows = BASE + [(20, 0, 250, 4500, TRADE, NO), (21, 0, -250, 4500, DELTA, YES)]
    bt = make(rows)
    bt.run_until(5)
    bt.submit(YES, 4600, 400)  # alone at 46c, ahead of the 45c bids
    bt.run_until(100)
    assert fills(bt) == [(20, 250, 4600, YES)]


def test_opposite_order_that_would_cross_us_fills_us(make):
    rows = BASE + [(20, 0, 300, 5400, DELTA, NO)]  # a NO bid at 54c offers YES at 46c
    bt = make(rows)
    bt.run_until(5)
    bt.submit(YES, 4600, 400)
    bt.run_until(100)
    assert fills(bt) == [(20, 300, 4600, YES)]
    assert only_order(bt).remaining == 100


def test_post_only_orders_that_would_cross_are_rejected(make):
    bt = make(BASE + [(100, 0, 1, 3000, DELTA, YES)])
    bt.run_until(5)
    crossing = bt.submit(YES, 4700, 100)  # the best offer is 47c
    resting = bt.submit(YES, 4699, 100)
    crossing_no = bt.submit(NO, 5500, 100)  # the best YES bid is 45c; 55c NO would lift it
    bt.run_until(50)
    assert [o.id for o in bt.orders()] == [resting]
    closed = {o.id: int(o.state) for o in bt.drain_closed()}
    assert closed == {crossing: R.REJECTED, crossing_no: R.REJECTED}


def test_order_can_fill_while_its_cancel_is_in_flight(make):
    rows = BASE + [(30, 0, 1500, 4500, TRADE, NO), (31, 0, -1000, 4500, DELTA, YES), (200, 0, 1, 3000, DELTA, YES)]
    bt = make(rows, cancel_latency_ns=50)
    bt.run_until(5)
    oid = bt.submit(YES, 4500, 800)
    bt.run_until(10)
    assert bt.cancel(oid)
    assert only_order(bt).cancel_requested
    bt.run_until(300)
    assert fills(bt) == [(30, 500, 4500, YES)]  # cancel landed at t=60, after the trade
    assert bt.orders() == []
    (closed,) = bt.drain_closed()
    assert int(closed.state) == R.CANCELED and closed.remaining == 300
    assert not bt.cancel(oid)


def test_order_latency_decides_queue_position(make):
    rows = BASE + [(50, 0, 4000, 4500, DELTA, YES), (300, 0, 1, 3000, DELTA, YES)]
    fast, slow = make(rows), make(rows, order_latency_ns=100)
    for bt in (fast, slow):
        bt.run_until(5)
        bt.submit(YES, 4500, 100)
        bt.run_until(400)
    assert only_order(fast).ahead == 1000
    assert only_order(slow).ahead == 5000  # arrived after the 4000 that beat it to the level


def test_sequence_gap_sends_resting_orders_to_the_back(make):
    resnap = snapshot({4500: 3000}, {5300: 2000}, ts=60)
    without_gap = BASE + [(20, 0, 700, 4500, TRADE, NO), (21, 0, -700, 4500, DELTA, YES)] + resnap
    with_gap = without_gap[:-len(resnap)] + [(50, 0, 0, 0, T.GAP, 0)] + resnap
    a, b = make(without_gap), make(with_gap)
    for bt in (a, b):
        bt.run_until(5)
        bt.submit(YES, 4500, 100)
        bt.run_until(100)
    assert only_order(a).ahead == 300  # the snapshot is consistent with what we knew
    assert only_order(b).ahead == 3000  # after a gap we cannot know, so assume the worst


def test_no_orders_are_accepted_while_the_book_is_invalid(make):
    rows = BASE + [(50, 0, 0, 0, T.GAP, 0), (100, 0, 1, 3000, DELTA, YES)]
    bt = make(rows)
    bt.run_until(60)
    assert not bt.valid
    bt.submit(YES, 4400, 100)
    bt.run_until(200)
    assert bt.orders() == [] and bt.sim_stats["rejected"] == 1


def test_yes_and_no_fills_net_to_a_locked_in_dollar(make):
    rows = BASE + [(20, 0, 1100, 4500, TRADE, NO), (30, 0, 2100, 4700, TRADE, YES)]
    bt = make(rows)
    bt.run_until(5)
    bt.submit(YES, 4500, 100)  # buy one YES at 45c
    bt.submit(NO, 5300, 100)  # buy one NO at 53c
    bt.run_until(100)
    assert fills(bt) == [(20, 100, 4500, YES), (30, 100, 5300, NO)]
    assert bt.position == 0
    assert bt.cash == 100 * (10000 - 4500 - 5300)  # $0.02: paid 98c for a pair worth $1


def test_fee_rounding_matches_the_documented_example(make):
    # Kalshi docs, "Fee Rounding": a buy with -$0.055 revenue and a model fee
    # of $0.00363825 pays a $0.003639 trade fee plus $0.001361 of rounding.
    cfg = R.SimConfig(maker_fee_ppm=70_000)
    trade, rounding, rebate, carried = R.compute_fee(cfg, 0, 550, 100, -55_000)
    assert (trade, rounding, rebate, carried) == (3_639, 1_361, 0, 1_361)

    rows = snapshot({500: 1000}, {5300: 2000}) + [(20, 0, 1100, 500, TRADE, NO)]
    bt = make(rows, maker_fee_ppm=70_000)
    bt.run_until(5)
    bt.submit(YES, 550, 100)  # one contract at 5.5c, inside the spread
    bt.run_until(100)
    assert bt.fees == 5_000 and bt.cash == -60_000


def test_rounding_overpayment_is_rebated_once_it_reaches_a_cent():
    cfg = R.SimConfig(maker_fee_ppm=0)
    carried = 0
    for expected in (4_000, 8_000, 12_000):  # one contract at $0.006, three times
        trade, rounding, rebate, carried = R.compute_fee(cfg, carried, 60, 100, -6_000)
        assert (trade, rounding, rebate, carried) == (0, 4_000, 0, expected)
    # A later fill whose own fee covers a cent gets the accumulated cent back.
    cfg = R.SimConfig(maker_fee_ppm=70_000)
    trade, rounding, rebate, carried = R.compute_fee(cfg, carried, 5000, 1000, -5_000_000)
    assert (trade, rounding, rebate) == (175_000, 5_000, 10_000) and carried == 7_000
