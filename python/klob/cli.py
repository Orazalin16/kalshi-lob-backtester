"""Command-line entry point: ``klob <command>``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

from . import tape as T


def _core():
    import klob

    if not klob.HAVE_CORE:
        sys.exit("The C++ extension is not built. Run `pip install .` from the repository root.")
    return klob


def cmd_synth(a) -> None:
    from . import synth

    res = synth.generate(synth.SynthConfig(seed=a.seed, duration_s=a.hours * 3600, shadow_every_s=0))
    T.save(a.out, res.events)
    print(json.dumps(T.describe(res.events), indent=2))


def cmd_info(a) -> None:
    print(json.dumps(T.describe(T.load(a.tape)), indent=2))


def cmd_tobin(a) -> None:
    np.ascontiguousarray(T.load(a.tape)).tofile(a.out)


def cmd_backtest(a) -> None:
    klob = _core()
    from . import backtest
    from .strategy import MarketMaker, MMParams

    cfg = klob.SimConfig(
        queue_model=getattr(klob.QueueModel, a.queue_model.upper()),
        order_latency_ns=int(a.latency_ms * 1e6),
        cancel_latency_ns=int(a.latency_ms * 1e6),
        maker_fee_ppm=a.maker_fee_ppm,
    )
    params = MMParams(quote_size=a.size, max_position=a.max_position, tick=a.tick)
    res = backtest.run(T.load(a.tape), MarketMaker(params), cfg)
    print(json.dumps({"summary": res.summary, "sim": res.sim_stats, "book": res.book_stats}, indent=2))


def cmd_validate(a) -> None:
    _core()
    from . import synth, validate

    res = synth.generate(synth.SynthConfig(seed=a.seed, duration_s=a.hours * 3600, shadow_every_s=2.0))
    print(json.dumps(validate.validate(res), indent=2))


def cmd_bench(a) -> None:
    _core()
    from . import study

    print(json.dumps(study.benchmark(hours=a.hours), indent=2))


def cmd_study(a) -> None:
    _core()
    from . import study

    print(study.run(a.out_dir, quick=a.quick))


def cmd_record(a) -> None:
    from .kalshi import recorder

    if a.source == "rest":
        from .kalshi.rest import DEMO, PROD, KalshiREST

        n = recorder.record_rest(a.tickers[0], a.out, a.interval, a.seconds, KalshiREST(DEMO if a.demo else PROD))
    else:
        from .kalshi.auth import Signer

        if not (a.key_id and a.key_file):
            sys.exit("The WebSocket feed needs --key-id and --key-file (see README, 'Recording live data').")
        url = recorder.WS_DEMO if a.demo else recorder.WS_PROD
        try:
            n = asyncio.run(recorder.record_ws(a.tickers, a.out, Signer.from_file(a.key_id, a.key_file), url, a.seconds))
        except KeyboardInterrupt:
            return
    print(f"wrote {n} lines to {a.out}")


def cmd_normalize(a) -> None:
    from .kalshi.normalize import normalize_file

    tapes, stats = normalize_file(a.raw)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ticker, events in tapes.items():
        T.save(out / f"{ticker}.npy", events)
        print(ticker, json.dumps(T.describe(events)))
    print(json.dumps(stats, indent=2))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="klob", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate a synthetic tape")
    s.add_argument("out")
    s.add_argument("--hours", type=float, default=8.0)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("info", help="summarize a tape")
    s.add_argument("tape")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("tobin", help="write a tape as raw records for the C++ benchmark")
    s.add_argument("tape")
    s.add_argument("out")
    s.set_defaults(fn=cmd_tobin)

    s = sub.add_parser("backtest", help="run the market maker over a tape")
    s.add_argument("tape")
    s.add_argument("--queue-model", choices=["conservative", "proportional", "optimistic"], default="proportional")
    s.add_argument("--latency-ms", type=float, default=50.0)
    s.add_argument("--maker-fee-ppm", type=int, default=0)
    s.add_argument("--size", type=int, default=10)
    s.add_argument("--max-position", type=int, default=100)
    s.add_argument("--tick", type=int, default=100, help="price ticks per level (100 = one cent)")
    s.set_defaults(fn=cmd_backtest)

    s = sub.add_parser("validate", help="score the queue models against synthetic ground truth")
    s.add_argument("--hours", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("bench", help="measure replay throughput")
    s.add_argument("--hours", type=float, default=24.0)
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("study", help="run the full synthetic study and write results/RESULTS.md")
    s.add_argument("--out-dir", default="results")
    s.add_argument("--quick", action="store_true")
    s.set_defaults(fn=cmd_study)

    s = sub.add_parser("record", help="record raw Kalshi market data")
    s.add_argument("source", choices=["ws", "rest"])
    s.add_argument("tickers", nargs="+")
    s.add_argument("--out", required=True)
    s.add_argument("--key-id")
    s.add_argument("--key-file")
    s.add_argument("--demo", action="store_true")
    s.add_argument("--seconds", type=float)
    s.add_argument("--interval", type=float, default=1.0, help="REST polling interval in seconds")
    s.set_defaults(fn=cmd_record)

    s = sub.add_parser("normalize", help="convert a raw recording into per-market tapes")
    s.add_argument("raw")
    s.add_argument("--out-dir", default="tapes")
    s.set_defaults(fn=cmd_normalize)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
