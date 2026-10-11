#!/usr/bin/env python3
"""Checks that the test suite notices bugs.

Each mutant below is a small, plausible mistake in the C++ core: an off-by-one
comparison, a forgotten clamp, a wake-up that is never scheduled. The script
applies one at a time to a scratch copy of the repository, rebuilds the
extension, and runs the tests. A mutant "survives" if the tests still pass,
which means the suite cannot tell the right behaviour from the wrong one there.

    python scripts/mutation_check.py          # all mutants
    python scripts/mutation_check.py 2 7      # only mutants 2 and 7

Exits non-zero if any mutant survives. Takes a few minutes: every mutant is a
full rebuild.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM = "cpp/include/klob/exchange_sim.hpp"

MUTANTS = [
    # (file, original, replacement, what it breaks)
    (SIM, "if (p.ts <= o.active_ts) seen -= p.qty;", "if (p.ts < o.active_ts) seen -= p.qty;",
     "late trade advances an order that joined in the same nanosecond as its delta"),
    (SIM, "if (o.active_ts >= d.ts) continue;", "if (o.active_ts > d.ts) continue;",
     "cancel advances an order that joined in the same nanosecond"),
    (SIM, "o.ahead = after_gap_ ? level : std::min(o.ahead, level);", "o.ahead = std::min(o.ahead, level);",
     "queue position trusted across a sequence gap"),
    (SIM, "o.side == ours && o.price + price >= kPriceMax", "o.side == ours && o.price + price > kPriceMax",
     "opposite order that exactly meets our price does not fill us"),
    (SIM, "debits_[n].ts + w <= t) apply_cancel", "debits_[n].ts + w < t) apply_cancel",
     "attribution window is one nanosecond too long"),
    (SIM, "next_due_ = std::min(next_due_, std::max(o.cancel_ts, o.active_ts));", "",
     "cancel is requested but never scheduled"),
    (SIM, "      o.ahead = std::min(o.ahead, level);\n    }\n  }\n\n  void expire_attribution", "    }\n  }\n\n  void expire_attribution",
     "size ahead may exceed the displayed size"),
    (SIM, "const Qty consumed = std::min(o.ahead, seen);", "const Qty consumed = std::min(o.ahead, qty);",
     "trade size already seen as a delta is counted twice"),
    (SIM, "o.state == OrderState::Active && o.cancel_ts <= t) o.state", "o.state == OrderState::Active && o.cancel_ts < t) o.state",
     "cancel takes effect one nanosecond late"),
    (SIM, "if (!book.valid() || (opp >= 0 && o.price + opp >= kPriceMax))", "if (!book.valid() || (opp >= 0 && o.price + opp > kPriceMax))",
     "post-only order that locks the market is accepted"),
    (SIM, "if (o.price > price) {\n        // The taker", "if (o.price >= price) {\n        // The taker",
     "order at the traded price jumps the queue"),
    (SIM, "case QueueModel::Optimistic: o.ahead = std::max<Qty>(0, o.ahead - d.qty); break;", "case QueueModel::Optimistic: o.ahead -= d.qty; break;",
     "size ahead goes negative"),
    ("cpp/include/klob/backtester.hpp", "sim_.on_time(e.ts - 1, book_);", "sim_.on_time(e.ts, book_);",
     "order arriving at t is placed before the event at t"),
    ("cpp/include/klob/backtester.hpp", "sim_.on_decrease(e.ts, side, e.price, before - after);", "sim_.on_decrease(e.ts, side, e.price, -e.qty);",
     "clamped delta reported at its nominal size"),
    ("cpp/include/klob/fees.hpp", "(scaled + kDenom - 1) / kDenom", "scaled / kDenom",
     "trade fee rounded down instead of up"),
    ("cpp/include/klob/fees.hpp", "std::min(acc.carried / step, gross / step) * step", "(acc.carried / step) * step",
     "rebate can exceed the fill's own fee"),
    ("cpp/include/klob/level_bitmap.hpp", "if (words_[w] == 0) summary_[w >> 6] &= ~bit(w);", "",
     "summary bit left set for an empty block"),
    ("cpp/include/klob/book.hpp", "if (q < 0) {\n      ++stats_.negative_clamps;\n      q = 0;\n    }", "if (q < 0) ++stats_.negative_clamps;",
     "level allowed to go negative"),
]

TESTS = ["tests/test_differential.py", "tests/test_scenarios.py"]


def build_and_test(tree: Path) -> bool:
    """Returns True if the tests pass on this tree."""
    build = tree / "build"
    cmake_dir = subprocess.run([sys.executable, "-m", "pybind11", "--cmakedir"], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(["cmake", "-S", str(tree), "-B", str(build), f"-Dpybind11_DIR={cmake_dir}", "-Wno-dev"], check=True, capture_output=True)
    made = subprocess.run(["cmake", "--build", str(build)], capture_output=True, text=True)
    if made.returncode != 0:
        raise RuntimeError(made.stdout[-2000:] + made.stderr[-2000:])
    for so in build.glob("_core*"):
        shutil.copy(so, tree / "python" / "klob")
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *TESTS],
        cwd=tree, env={"PYTHONPATH": str(tree / "python"), "PATH": "/usr/local/bin:/usr/bin:/bin"}, capture_output=True, text=True,
    )
    return run.returncode == 0


def main() -> int:
    survivors = []
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "repo"
        for name in ("cpp", "tests", "CMakeLists.txt"):
            src = ROOT / name
            shutil.copytree(src, tree / name) if src.is_dir() else shutil.copy(src, tree / name)
        shutil.copytree(ROOT / "python", tree / "python", ignore=shutil.ignore_patterns("_core*", "__pycache__"))

        if not build_and_test(tree):
            print("the unmodified tree does not pass; fix that first")
            return 2
        only = {int(a) for a in sys.argv[1:]}
        for i, (rel, old, new, why) in enumerate(MUTANTS, 1):
            if only and i not in only:
                continue
            path = tree / rel
            original = path.read_text()
            if original.count(old) != 1:
                print(f"{i:2d}. STALE     {why}  (pattern not found exactly once in {rel})")
                survivors.append(why)
                continue
            path.write_text(original.replace(old, new))
            try:
                caught = not build_and_test(tree)
            finally:
                path.write_text(original)
            print(f"{i:2d}. {'caught  ' if caught else 'SURVIVED'}  {why}")
            if not caught:
                survivors.append(why)
    tried = len(only) if only else len(MUTANTS)
    print(f"\n{tried - len(survivors)} of {tried} mutants caught")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
