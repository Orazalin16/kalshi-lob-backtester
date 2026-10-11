"""The synthetic market and the ground-truth check on the queue models."""

import numpy as np
import pytest

import klob
from klob import reference as R
from klob import synth, validate
from klob import tape as T

SMALL = dict(duration_s=600, shadow_every_s=1.0)


@pytest.fixture(scope="module")
def market():
    return synth.generate(synth.SynthConfig(seed=11, **SMALL))


def test_generation_is_deterministic(market):
    again = synth.generate(synth.SynthConfig(seed=11, **SMALL))
    assert np.array_equal(market.events, again.events)
    assert np.array_equal(market.shadows, again.shadows)
    other = synth.generate(synth.SynthConfig(seed=12, **SMALL))
    assert not np.array_equal(market.events["price"][:5000], other.events["price"][:5000])


def test_feed_is_a_consistent_book(market):
    ev = market.events
    assert np.all(np.diff(ev["ts"]) > 0)
    book = R.Book()
    crossed = 0
    for typ, side, price, qty in zip(ev["type"].tolist(), ev["side"].tolist(), ev["price"].tolist(), ev["qty"].tolist()):
        book.apply_event(typ, side, price, qty)
        yb, nb = book.best(T.YES), book.best(T.NO)
        crossed += yb >= 0 and nb >= 0 and yb + nb >= 10000
    # Deltas never remove more than is displayed, and price-time matching never leaves a crossed book.
    assert book.stats["negative_clamps"] == 0 and crossed == 0
    assert book.stats["trades"] > 1000


def test_every_trade_has_a_matching_book_decrease(core, market):
    bt = core.Backtester(market.events, core.SimConfig())
    bt.run_all()
    traded = int(market.events["qty"][market.events["type"] == T.TRADE].sum())
    assert bt.sim_stats["matched_trade_qty"] == traded
    assert bt.sim_stats["unmatched_trade_qty"] == 0


def test_queue_models_bracket_the_truth(core):
    """With all taker size visible, the conservative model can never fill more
    than really happened and the optimistic model can never fill less. This
    holds order by order, not just on average."""
    res = synth.generate(synth.SynthConfig(seed=5, cap_takers_to_displayed=True, **SMALL))
    truth = res.shadows["filled"]
    cons, prop, opt = (validate.simulate_shadows(res, m)[0] for m in range(3))
    assert len(truth) > 1000 and 0.2 < (truth > 0).mean() < 0.95
    assert np.all(cons <= truth) and np.all(truth <= opt)
    assert np.all(cons <= prop) and np.all(prop <= opt)
    assert (cons < truth).any() and (truth < opt).any()  # the bounds are not vacuous


def test_bracket_is_nearly_exact_with_hidden_taker_size(core, market):
    truth = market.shadows["filled"]
    cons, _, opt = (validate.simulate_shadows(market, m)[0] for m in range(3))
    assert np.all(cons <= truth)
    assert (truth > opt).mean() < 0.005


@pytest.mark.parametrize("delta_first", [0.0, 1.0])
def test_fills_do_not_depend_on_trade_and_delta_arrival_order(core, delta_first):
    """The same market published with book deltas always after, or always
    before, their trades must give every order the same fill."""
    base = synth.generate(synth.SynthConfig(seed=21, delta_first_prob=0.5, **SMALL))
    other = synth.generate(synth.SynthConfig(seed=21, delta_first_prob=delta_first, **SMALL))
    for model in range(3):
        a, _ = validate.simulate_shadows(base, model)
        b, _ = validate.simulate_shadows(other, model)
        assert np.array_equal(a, b)


def test_shadow_experiment_agrees_between_core_and_reference(core):
    res = synth.generate(synth.SynthConfig(seed=4, duration_s=240, shadow_every_s=1.0))
    for model in range(3):
        a = validate.simulate_shadows(res, model)
        b = validate.simulate_shadows(res, model, backend=R)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_scores_are_sane(core, market):
    scores = validate.validate(market)
    assert scores["conservative"]["false_fills"] == 0.0
    assert scores["optimistic"]["missed_fills"] <= 0.005
    assert scores["proportional"]["mae_contracts"] < scores["conservative"]["mae_contracts"]
    assert scores["conservative"]["filled_qty_ratio"] <= 1.0 <= scores["optimistic"]["filled_qty_ratio"] + 0.01
