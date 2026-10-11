import math

import numpy as np
import pytest

import klob
from klob import backtest, synth
from klob import reference as R
from klob.strategy import MarketMaker, MMParams
from klob.tape import NO, YES


def test_quotes_straddle_the_market_and_never_cross():
    mm = MarketMaker()
    bid, ask = mm.quotes(0, 4500, 1000, 4700, 1000, position=0)
    assert bid is not None and ask is not None
    assert bid <= 4600 <= ask and bid < 4700 and ask > 4500
    assert bid % 100 == 0 and ask % 100 == 0


def test_inventory_skews_quotes_and_stops_at_the_limit():
    p = MMParams(max_position=100)
    flat = MarketMaker(p).quotes(0, 4500, 1000, 4700, 1000, position=0)
    mm_long, mm_short = MarketMaker(p), MarketMaker(p)
    for mm in (mm_long, mm_short):
        mm._var = 0.02**2  # a volatile market, so the skew is larger than a tick
    long_ = mm_long.quotes(0, 4500, 1000, 4700, 1000, position=8000)
    short = mm_short.quotes(0, 4500, 1000, 4700, 1000, position=-8000)
    assert long_[0] < short[0] and long_[1] < short[1]  # long inventory quotes lower on both sides
    assert flat[0] is not None
    assert MarketMaker(p).quotes(0, 4500, 1000, 4700, 1000, position=10000)[0] is None  # full: stop bidding
    assert MarketMaker(p).quotes(0, 4500, 1000, 4700, 1000, position=-10000)[1] is None


def test_price_volatility_shrinks_toward_the_boundaries():
    """The same log-odds volatility means a smaller price volatility at 5c than at 50c."""
    mid, edge = MarketMaker(), MarketMaker()
    mid._var = edge._var = 0.01**2
    mid.quotes(0, 4950, 100, 5050, 100, 0)
    edge.quotes(0, 450, 100, 550, 100, 0)
    assert mid.sigma_ticks == pytest.approx(0.01 * 0.25 * 10000)
    assert edge.sigma_ticks == pytest.approx(0.01 * 0.05 * 0.95 * 10000)


def test_microprice_leans_toward_the_thin_side():
    mm = MarketMaker(MMParams(min_half_spread=0.0, spread_mult=0.0))
    mm.quotes(0, 4500, 9000, 4700, 1000, 0)  # heavy bid, thin offer
    assert mm.reservation == pytest.approx((4700 * 9000 + 4500 * 1000) / 10000)


@pytest.fixture(scope="module")
def tape():
    return synth.generate(synth.SynthConfig(seed=9, duration_s=900, shadow_every_s=0)).events


def test_backtest_accounting_is_self_consistent(core, tape):
    params = MMParams(max_position=30, quote_size=10)
    res = backtest.run(tape, MarketMaker(params), core.SimConfig(order_latency_ns=20_000_000, cancel_latency_ns=20_000_000))
    f = res.fills
    assert len(f) > 5
    # Rebuild position and cash from the fill log alone.
    signed = np.where(f["side"] == YES, f["qty"], -f["qty"])
    yes_price = np.where(f["side"] == YES, f["price"], 10000 - f["price"])
    assert signed.sum() == res.position[-1]
    cash = -(signed * yes_price).sum() - f["fee"].sum()
    final_equity = (cash + signed.sum() * res.mid[-1]) * 1e-6
    assert math.isclose(final_equity, res.summary["pnl"], abs_tol=1e-9)
    # Position can exceed the limit by at most the two quotes that were resting when it was reached.
    assert res.summary["max_abs_position"] <= params.max_position + 2 * params.quote_size
    assert res.sim_stats["rejected"] <= res.sim_stats["submitted"] * 0.05


def test_strategy_runs_identically_on_core_and_reference(core, tape):
    short = tape[:40000]
    a = backtest.run(short, MarketMaker(), core.SimConfig(order_latency_ns=5_000_000))
    b = backtest.run(short, MarketMaker(), R.SimConfig(order_latency_ns=5_000_000), backend=R)
    assert np.array_equal(a.position, b.position) and np.array_equal(a.equity, b.equity)
    assert a.fills.tolist() == [tuple(x) for x in b.fills.tolist()]
    assert a.sim_stats == b.sim_stats
