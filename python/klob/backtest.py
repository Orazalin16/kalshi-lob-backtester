"""Runs a strategy over a tape and measures what happened."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fixedpoint import PRICE_MAX, QTY_SCALE
from .tape import YES

MICRO = 1e-6


@dataclass
class Result:
    ts: np.ndarray  # ns, one sample per strategy step with a two-sided book
    mid: np.ndarray  # ticks
    position: np.ndarray  # units, YES-equivalent
    equity: np.ndarray  # dollars, cash + position marked at mid
    fills: np.ndarray  # structured: ts, order_id, qty, fee, price, side
    fees: float
    sim_stats: dict
    book_stats: dict
    summary: dict = field(default_factory=dict)


def run(events: np.ndarray, strategy, config=None, backend=None, interval_ns: int = 250_000_000) -> Result:
    if backend is None:
        import klob as backend
    bt = backend.Backtester(events, config if config is not None else backend.SimConfig())
    ts, mid, pos, cash = [], [], [], []
    more = True
    while more:
        more = bt.elapse(interval_ns)
        yb, _, nb, _ = bt.top()
        if bt.valid and yb >= 0 and nb >= 0:
            ts.append(bt.now)
            mid.append(0.5 * (yb + PRICE_MAX - nb))
            pos.append(bt.position)
            cash.append(bt.cash)
        if more:
            strategy.on_step(bt)

    ts_a = np.asarray(ts, dtype=np.int64)
    mid_a = np.asarray(mid, dtype=np.float64)
    pos_a = np.asarray(pos, dtype=np.int64)
    equity = (np.asarray(cash, dtype=np.float64) + pos_a * mid_a) * MICRO
    fills = _fills_array(bt.fills())
    res = Result(ts_a, mid_a, pos_a, equity, fills, bt.fees * MICRO, dict(bt.sim_stats), dict(bt.book_stats))
    res.summary = summarize(res)
    return res


FILL_FIELDS = [("ts", "<i8"), ("order_id", "<u8"), ("qty", "<i8"), ("fee", "<i8"), ("price", "<i4"), ("side", "u1")]


def _fills_array(fills) -> np.ndarray:
    if isinstance(fills, np.ndarray):
        return fills
    out = np.zeros(len(fills), dtype=FILL_FIELDS)
    for i, f in enumerate(fills):
        out[i] = tuple(f)
    return out


def markouts(res: Result, horizons_s=(1.0, 10.0, 60.0)) -> dict:
    """Per-contract edge at the moment of each fill and after it, in cents.

    edge_at_fill is how far inside the mid we traded. markout_h is the same
    measured against the mid h seconds later; the difference between them is
    what adverse selection took back.
    """
    f = res.fills
    if len(f) == 0 or len(res.ts) == 0:
        return {"edge_at_fill": float("nan"), **{f"markout_{int(h)}s": float("nan") for h in horizons_s}}
    direction = np.where(f["side"] == YES, 1.0, -1.0)
    yes_price = np.where(f["side"] == YES, f["price"], PRICE_MAX - f["price"]).astype(np.float64)
    w = f["qty"].astype(np.float64)

    def mid_at(t: np.ndarray) -> np.ndarray:
        idx = np.clip(np.searchsorted(res.ts, t, side="right") - 1, 0, len(res.ts) - 1)
        return res.mid[idx]

    out = {"edge_at_fill": float(np.average(direction * (mid_at(f["ts"]) - yes_price), weights=w) / 100)}
    for h in horizons_s:
        later = mid_at(f["ts"] + int(h * 1e9))
        out[f"markout_{int(h)}s"] = float(np.average(direction * (later - yes_price), weights=w) / 100)
    return out


def summarize(res: Result, bar_s: float = 60.0) -> dict:
    if len(res.ts) == 0:
        return {"pnl": 0.0, "fills": 0}
    hours = (res.ts[-1] - res.ts[0]) / 3.6e12
    eq = res.equity
    # PnL per fixed bar, for a dispersion estimate that does not depend on step size.
    edges = np.arange(res.ts[0], res.ts[-1], int(bar_s * 1e9))
    idx = np.clip(np.searchsorted(res.ts, edges, side="right") - 1, 0, len(eq) - 1)
    bar_pnl = np.diff(eq[idx])
    peak = np.maximum.accumulate(eq)
    pos = res.position / QTY_SCALE
    f = res.fills
    volume = float(f["qty"].sum() / QTY_SCALE) if len(f) else 0.0
    out = {
        "hours": float(hours),
        "pnl": float(eq[-1]),
        "fees": float(res.fees),
        "fills": int(len(f)),
        "volume_contracts": volume,
        "pnl_per_contract_cents": float(100 * eq[-1] / volume) if volume else float("nan"),
        "max_drawdown": float((peak - eq).max()),
        "bar_pnl_mean": float(bar_pnl.mean()) if bar_pnl.size else float("nan"),
        "bar_pnl_std": float(bar_pnl.std(ddof=1)) if bar_pnl.size > 1 else float("nan"),
        "mean_abs_position": float(np.abs(pos).mean()),
        "max_abs_position": float(np.abs(pos).max()),
        "end_position": float(pos[-1]),
    }
    if bar_pnl.size > 1 and out["bar_pnl_std"] > 0:
        out["t_stat"] = float(bar_pnl.mean() / (bar_pnl.std(ddof=1) / np.sqrt(bar_pnl.size)))
    out.update(markouts(res))
    return out
