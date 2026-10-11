"""The synthetic study behind the numbers in the README.

Three questions, all answered on simulated markets where the truth is known:

1. How wrong is each queue model about fills, order by order?
2. How much does the simulated PnL of one fixed strategy move when only the
   queue model or the latency assumption changes?
3. How fast is the replay?

Run with ``klob study``. Everything is seeded; the same command reproduces the
same tables.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from . import backtest, reference, synth, validate
from .strategy import MarketMaker, MMParams

MODELS = validate.MODEL_NAMES


def queue_model_study(seeds=(1, 2, 3, 4, 5), hours: float = 2.0, biases=(0.0, 0.5, 1.0)) -> list[dict]:
    rows = []
    for bias in biases:
        pooled = {m: [] for m in MODELS}
        truth_all, n = [], 0
        for seed in seeds:
            res = synth.generate(
                synth.SynthConfig(seed=seed, duration_s=hours * 3600, cancel_back_bias=bias, shadow_every_s=2.0)
            )
            truth_all.append(res.shadows["filled"])
            n += len(res.shadows)
            for i, m in enumerate(MODELS):
                pooled[m].append(validate.simulate_shadows(res, i)[0])
        truth = np.concatenate(truth_all)
        for m in MODELS:
            filled = np.concatenate(pooled[m])
            rows.append(
                {
                    "cancel_back_bias": bias,
                    "model": m,
                    "orders": n,
                    "true_fill_rate": float((truth > 0).mean()),
                    "model_fill_rate": float((filled > 0).mean()),
                    "filled_qty_ratio": float(filled.sum() / truth.sum()),
                    "false_fills": float(((filled > 0) & (truth == 0)).mean()),
                    "missed_fills": float(((filled == 0) & (truth > 0)).mean()),
                    "mae_contracts": float(np.abs(filled - truth).mean() / 100),
                    "overfilled_orders": int((filled > truth).sum()),
                    "underfilled_orders": int((filled < truth).sum()),
                }
            )
    return rows


def strategy_study(seeds=(1, 2, 3, 4, 5), hours: float = 8.0, latencies_ms=(0, 50, 250, 1000)) -> list[dict]:
    import klob

    tapes = [synth.generate(synth.SynthConfig(seed=s, duration_s=hours * 3600, shadow_every_s=0)).events for s in seeds]
    rows = []
    for i, m in enumerate(MODELS):
        for lat in latencies_ms:
            runs = []
            for tape in tapes:
                cfg = klob.SimConfig(
                    queue_model=klob.QueueModel(i), order_latency_ns=lat * 10**6, cancel_latency_ns=lat * 10**6
                )
                runs.append(backtest.run(tape, MarketMaker(MMParams()), cfg).summary)
            pnl = np.array([r["pnl"] for r in runs])
            rows.append(
                {
                    "model": m,
                    "latency_ms": lat,
                    "sessions": len(runs),
                    "session_hours": hours,
                    "pnl_mean": float(pnl.mean()),
                    "pnl_std": float(pnl.std(ddof=1)) if len(pnl) > 1 else float("nan"),
                    "pnl_min": float(pnl.min()),
                    "pnl_max": float(pnl.max()),
                    "fills_mean": float(np.mean([r["fills"] for r in runs])),
                    "volume_mean": float(np.mean([r["volume_contracts"] for r in runs])),
                    "edge_at_fill": float(np.mean([r["edge_at_fill"] for r in runs])),
                    "markout_1s": float(np.mean([r["markout_1s"] for r in runs])),
                    "markout_10s": float(np.mean([r["markout_10s"] for r in runs])),
                    "markout_60s": float(np.mean([r["markout_60s"] for r in runs])),
                    "max_abs_position": float(np.max([r["max_abs_position"] for r in runs])),
                }
            )
    return rows


def benchmark(hours: float = 24.0, seed: int = 1, repeats: int = 15, reference_events: int = 300_000) -> dict:
    import klob

    tape = synth.generate(synth.SynthConfig(seed=seed, duration_s=hours * 3600, shadow_every_s=0)).events
    best_book = min(_timed(lambda: klob.Book().apply(tape)) for _ in range(repeats))
    best_sim = min(_timed(lambda: klob.Backtester(tape, klob.SimConfig()).run_all()) for _ in range(repeats))
    head = tape[:reference_events]
    ref_book = min(_timed(lambda: reference.Book().apply(head)) for _ in range(3))
    core_head = min(_timed(lambda: klob.Book().apply(head)) for _ in range(repeats))
    # A full strategy backtest: Python decides quotes four times a second, C++ replays in between.
    t0 = time.perf_counter()
    res = backtest.run(tape, MarketMaker(), klob.SimConfig(order_latency_ns=50_000_000, cancel_latency_ns=50_000_000))
    strat_s = time.perf_counter() - t0
    return {
        "events": int(len(tape)),
        "tape_hours": hours,
        "core_book_events_per_s": len(tape) / best_book,
        "core_book_ns_per_event": best_book / len(tape) * 1e9,
        "core_sim_events_per_s": len(tape) / best_sim,
        "reference_book_events_per_s": len(head) / ref_book,
        "speedup_vs_reference": (len(head) / core_head) / (len(head) / ref_book),
        "strategy_backtest_seconds": strat_s,
        "strategy_steps": int(len(res.ts)),
        "machine": f"{platform.processor() or platform.machine()}, {_cpu_model()}",
        "python": platform.python_version(),
    }


def _timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown CPU"


# -- report ----------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render(queue_rows: list[dict], strat_rows: list[dict], bench: dict) -> str:
    out = ["# Results on synthetic markets", ""]
    out.append(
        "Generated by `klob study`. Every number below comes from simulated markets "
        "(see `klob/synth.py`), not from Kalshi data, and says nothing about whether the "
        "strategy would make money on a real exchange."
    )
    out += ["", "## 1. Queue models against order-level truth", ""]
    out.append(
        "Each row pools hypothetical 25-contract passive orders placed at or one tick behind the best "
        "price and held for 60 seconds. `cancel bias` is how strongly cancellations in the simulated "
        "market favour newer orders (0 = uniform)."
    )
    out += [
        "",
        "| cancel bias | model | orders | true fill rate | model fill rate | filled size vs truth | false fills | missed fills | mean abs error (contracts) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in queue_rows:
        out.append(
            f"| {r['cancel_back_bias']:.1f} | {r['model']} | {r['orders']:,} | {_pct(r['true_fill_rate'])} | "
            f"{_pct(r['model_fill_rate'])} | {_pct(r['filled_qty_ratio'])} | {_pct(r['false_fills'])} | "
            f"{_pct(r['missed_fills'])} | {r['mae_contracts']:.2f} |"
        )
    out += ["", "## 2. One strategy, twelve sets of assumptions", ""]
    n, h = strat_rows[0]["sessions"], strat_rows[0]["session_hours"]
    out.append(
        f"The default market maker (10 contracts a side, 100-contract limit) on {n} independent {h:g}-hour "
        "sessions. PnL is in dollars per session, marked at the final mid. Edge and markouts are cents per contract."
    )
    out += [
        "",
        "| queue model | latency | PnL mean | PnL std | PnL range | fills | edge at fill | markout 1s | markout 10s | markout 60s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in strat_rows:
        out.append(
            f"| {r['model']} | {r['latency_ms']} ms | {r['pnl_mean']:.2f} | {r['pnl_std']:.2f} | "
            f"{r['pnl_min']:.2f} to {r['pnl_max']:.2f} | {r['fills_mean']:.0f} | {r['edge_at_fill']:.2f} | "
            f"{r['markout_1s']:.2f} | {r['markout_10s']:.2f} | {r['markout_60s']:.2f} |"
        )
    out += ["", "## 3. Replay speed", ""]
    out.append(f"Machine: {bench['machine']}; Python {bench['python']}. Tape: {bench['events']:,} events ({bench['tape_hours']:.0f} simulated hours).")
    out += [
        "",
        "| measurement | result |",
        "|---|---|",
        f"| C++ book replay | {bench['core_book_events_per_s'] / 1e6:.1f} M events/s ({bench['core_book_ns_per_event']:.1f} ns/event) |",
        f"| C++ replay with the fill simulator attached, no orders resting | {bench['core_sim_events_per_s'] / 1e6:.1f} M events/s |",
        f"| Python reference book replay | {bench['reference_book_events_per_s'] / 1e6:.2f} M events/s |",
        f"| Speedup over the reference | {bench['speedup_vs_reference']:.0f}x |",
        f"| Full strategy backtest of the tape ({bench['strategy_steps']:,} Python decision steps) | {bench['strategy_backtest_seconds']:.1f} s |",
        "",
    ]
    return "\n".join(out)


def run(out_dir: str | Path = "results", quick: bool = False) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if quick:
        q = queue_model_study(seeds=(1,), hours=0.5, biases=(0.5,))
        s = strategy_study(seeds=(1, 2), hours=1.0, latencies_ms=(0, 250))
        b = benchmark(hours=1.0, repeats=3, reference_events=50_000)
    else:
        q, s, b = queue_model_study(), strategy_study(), benchmark()
    (out / "results.json").write_text(json.dumps({"queue_models": q, "strategy": s, "benchmark": b}, indent=2))
    report = render(q, s, b)
    (out / "RESULTS.md").write_text(report)
    return report
