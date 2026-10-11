"""Exact conversion between Kalshi's fixed-point strings and integers.

Kalshi sends prices as dollar strings with up to four decimals ("0.4200") and
sizes as contract strings with up to two ("10.00"). Parsing them through float
would make book state depend on rounding, so they are parsed digit by digit.
"""

from __future__ import annotations

PRICE_SCALE = 10_000  # price ticks per dollar
QTY_SCALE = 100  # size units per contract
PRICE_MAX = PRICE_SCALE  # $1.0000


def _parse(text: str, decimals: int) -> int:
    s = text.strip()
    neg = s.startswith("-")
    if neg or s.startswith("+"):
        s = s[1:]
    whole, _, frac = s.partition(".")
    if not (whole or frac) or not (whole + frac).isdigit():
        raise ValueError(f"not a fixed-point number: {text!r}")
    if len(frac) > decimals:
        if frac[decimals:].strip("0"):
            raise ValueError(f"{text!r} has more than {decimals} decimal places")
        frac = frac[:decimals]
    value = int(whole or "0") * 10**decimals + int(frac.ljust(decimals, "0") or "0")
    return -value if neg else value


def price_to_ticks(text: str) -> int:
    """'0.4200' -> 4200."""
    return _parse(text, 4)


def qty_to_units(text: str) -> int:
    """'10.00' -> 1000. Negative values are allowed (book deltas are signed)."""
    return _parse(text, 2)


def ticks_to_dollars(ticks: int) -> float:
    return ticks / PRICE_SCALE


def units_to_contracts(units: int) -> float:
    return units / QTY_SCALE
