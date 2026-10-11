"""The normalized event tape: one 32-byte record per book or trade event.

The dtype mirrors ``klob::Event`` in cpp/include/klob/types.hpp field for
field, so a tape loaded with ``np.load(..., mmap_mode="r")`` is handed to the
C++ core without a copy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

EVENT_DTYPE = np.dtype(
    [
        ("ts", "<i8"),  # local receive time, ns
        ("exch_ts", "<i8"),  # exchange time, ns (0 if the feed gave none)
        ("qty", "<i8"),  # 1e-2 contracts; signed for deltas
        ("price", "<i4"),  # 1e-4 dollars
        ("type", "u1"),
        ("side", "u1"),
        ("flags", "<u2"),
    ]
)
assert EVENT_DTYPE.itemsize == 32

# Event types
SNAPSHOT_BEGIN = 0
SNAPSHOT_LEVEL = 1
SNAPSHOT_END = 2
DELTA = 3
TRADE = 4  # side = taker side, price = YES price of the trade
GAP = 5

# Sides. The book holds YES bids and NO bids; a NO bid at q offers YES at 1 - q.
YES = 0
NO = 1

TYPE_NAMES = {
    SNAPSHOT_BEGIN: "snapshot_begin",
    SNAPSHOT_LEVEL: "snapshot_level",
    SNAPSHOT_END: "snapshot_end",
    DELTA: "delta",
    TRADE: "trade",
    GAP: "gap",
}


def make_events(rows: list[tuple]) -> np.ndarray:
    """Builds a tape from (ts, exch_ts, qty, price, type, side) tuples."""
    out = np.zeros(len(rows), dtype=EVENT_DTYPE)
    for i, (ts, exch_ts, qty, price, typ, side) in enumerate(rows):
        out[i] = (ts, exch_ts, qty, price, typ, side, 0)
    return out


def save(path: str | Path, events: np.ndarray) -> None:
    if events.dtype != EVENT_DTYPE:
        raise TypeError("events must use tape.EVENT_DTYPE")
    np.save(Path(path), np.ascontiguousarray(events))


def load(path: str | Path, mmap: bool = True) -> np.ndarray:
    events = np.load(Path(path), mmap_mode="r" if mmap else None)
    if events.dtype != EVENT_DTYPE:
        raise TypeError(f"{path} is not an event tape")
    return events


def describe(events: np.ndarray) -> dict:
    """Counts by event type plus the time span, for a quick sanity check."""
    counts = np.bincount(events["type"], minlength=len(TYPE_NAMES))
    out = {TYPE_NAMES[i]: int(counts[i]) for i in TYPE_NAMES}
    out["events"] = int(len(events))
    if len(events):
        out["seconds"] = float(events["ts"][-1] - events["ts"][0]) / 1e9
        out["monotonic"] = bool(np.all(np.diff(events["ts"]) >= 0))
    return out
