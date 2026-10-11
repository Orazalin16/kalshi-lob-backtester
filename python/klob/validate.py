"""Scores the L2 queue models against order-level ground truth.

`synth.generate` records, for a sample of hypothetical passive orders, how much
each would really have been filled under price-time priority. Here the same
orders are placed in the backtester, which sees only the L2 feed, once per
queue model. The gap between the two is the model error.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from . import reference
from .synth import SynthResult

MODEL_NAMES = ("conservative", "proportional", "optimistic")


def simulate_shadows(result: SynthResult, queue_model: int, backend=None, attribution_window_ns: int = 50_000_000):
    """Replays the tape with one order per shadow.

    Returns (filled, t_first): size filled within the horizon and the time of
    the first fill (-1 if none), aligned with ``result.shadows``.
    """
    if backend is None:
        import klob as backend  # the C++ core
    horizon = int(result.config.shadow_horizon_s * 1e9)
    cfg = backend.SimConfig(queue_model=_model(backend, queue_model), attribution_window_ns=attribution_window_ns)
    bt = backend.Backtester(result.events, cfg)
    shadows = result.shadows
    index_of: dict[int, int] = {}
    expiring: deque[tuple[int, int]] = deque()
    for i in range(len(shadows)):
        t0 = int(shadows["t0"][i])
        bt.run_until(t0)
        while expiring and expiring[0][0] <= t0:
            bt.cancel(expiring.popleft()[1])
        oid = bt.submit(int(shadows["side"][i]), int(shadows["price"][i]), int(shadows["qty"][i]))
        index_of[oid] = i
        expiring.append((t0 + horizon, oid))
    bt.run_all()

    filled = np.zeros(len(shadows), dtype=np.int64)
    t_first = np.full(len(shadows), -1, dtype=np.int64)
    for f in bt.fills():
        ts, oid, qty = int(f[0]), int(f[1]), int(f[2])
        i = index_of[oid]
        if ts <= int(shadows["t0"][i]) + horizon:
            filled[i] += qty
            if t_first[i] < 0:
                t_first[i] = ts
    return filled, t_first


def _model(backend, queue_model: int):
    if backend is reference:
        return queue_model
    return backend.QueueModel(queue_model)


def score(result: SynthResult, filled: np.ndarray, t_first: np.ndarray) -> dict:
    sh = result.shadows
    truth = sh["filled"].astype(np.int64)
    truth_any = truth > 0
    model_any = filled > 0
    both = truth_any & model_any
    dt = (t_first[both] - sh["t_first"][both]) / 1e9
    return {
        "orders": int(len(sh)),
        "true_fill_rate": float(truth_any.mean()),
        "model_fill_rate": float(model_any.mean()),
        "filled_qty_ratio": float(filled.sum() / max(truth.sum(), 1)),
        "agreement": float((truth_any == model_any).mean()),
        "false_fills": float((model_any & ~truth_any).mean()),
        "missed_fills": float((~model_any & truth_any).mean()),
        "mae_contracts": float(np.abs(filled - truth).mean() / 100),
        "first_fill_delay_s_median": float(np.median(dt)) if dt.size else float("nan"),
        "first_fill_delay_s_mean": float(dt.mean()) if dt.size else float("nan"),
    }


def validate(result: SynthResult, backend=None) -> dict[str, dict]:
    return {
        name: score(result, *simulate_shadows(result, model, backend)) for model, name in enumerate(MODEL_NAMES)
    }
