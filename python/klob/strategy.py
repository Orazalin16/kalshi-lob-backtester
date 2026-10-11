"""Inventory-aware market making for a binary contract.

The classic Avellaneda-Stoikov quoting rule assumes the mid price is an
arithmetic Brownian motion, which a probability is not: it lives in (0, 1) and
moves less the closer it gets to either end. Here volatility is estimated in
log-odds, where a random walk is a reasonable local model, and mapped back to
price space through the derivative of the logistic function:

    sigma_price = sigma_logit * p * (1 - p)

Everything else follows the usual shape. The quote centre is a size-weighted
mid (microprice) shifted against inventory, and the half-spread widens with
volatility. Kalshi has no asks, so "selling YES at a" is sent as a NO bid at
1 - a.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .fixedpoint import PRICE_MAX, QTY_SCALE
from .tape import NO, YES


@dataclass
class MMParams:
    quote_size: int = 10  # contracts per side
    max_position: int = 100  # contracts, YES-equivalent, either direction
    tick: int = 100  # price ticks per level; 100 = one cent
    min_half_spread: float = 50.0  # ticks
    spread_mult: float = 1.0  # half-spread in units of expected move over quote_horizon_s
    quote_horizon_s: float = 5.0
    risk_aversion: float = 1.0  # skew at full inventory, in units of expected move over inventory_horizon_s
    inventory_horizon_s: float = 60.0
    microprice_weight: float = 1.0  # 0 = quote around the mid, 1 = around the microprice
    vol_halflife_s: float = 120.0
    vol_floor: float = 1e-4  # log-odds per sqrt(second)


class MarketMaker:
    def __init__(self, params: MMParams | None = None) -> None:
        self.p = params or MMParams()
        self._last_x: float | None = None
        self._last_ts = 0
        self._var = self.p.vol_floor**2  # EWMA variance of log-odds per second
        self.sigma_ticks = 0.0  # latest price volatility, ticks per sqrt(second)
        self.reservation = float("nan")

    # -- signal ------------------------------------------------------------

    def _update_vol(self, ts: int, mid: float) -> float:
        p = min(max(mid / PRICE_MAX, 1e-4), 1 - 1e-4)
        x = math.log(p / (1 - p))
        if self._last_x is not None:
            dt = (ts - self._last_ts) / 1e9
            if dt > 0:
                dx = x - self._last_x
                alpha = 1.0 - 0.5 ** (dt / self.p.vol_halflife_s)
                self._var += alpha * (dx * dx / dt - self._var)
        self._last_x, self._last_ts = x, ts
        sigma_x = math.sqrt(max(self._var, self.p.vol_floor**2))
        self.sigma_ticks = sigma_x * p * (1 - p) * PRICE_MAX
        return self.sigma_ticks

    def quotes(self, ts: int, yes_bid: int, bid_size: int, yes_ask: int, ask_size: int, position: int):
        """Returns (YES bid price or None, YES ask price or None) in ticks."""
        prm = self.p
        mid = 0.5 * (yes_bid + yes_ask)
        sigma = self._update_vol(ts, mid)
        micro = (yes_ask * bid_size + yes_bid * ask_size) / (bid_size + ask_size) if bid_size + ask_size else mid
        fair = mid + prm.microprice_weight * (micro - mid)

        inv = position / (prm.max_position * QTY_SCALE)  # -1 .. 1
        skew = prm.risk_aversion * inv * sigma * math.sqrt(prm.inventory_horizon_s)
        self.reservation = fair - skew
        half = max(prm.min_half_spread, prm.spread_mult * sigma * math.sqrt(prm.quote_horizon_s))

        tick = prm.tick
        bid = math.floor((self.reservation - half) / tick) * tick
        ask = math.ceil((self.reservation + half) / tick) * tick
        # Post-only: never quote through the other side of the book.
        bid = min(bid, yes_ask - tick)
        ask = max(ask, yes_bid + tick)
        bid_ok = tick <= bid <= PRICE_MAX - tick and inv < 1.0
        ask_ok = tick <= ask <= PRICE_MAX - tick and inv > -1.0
        return (bid if bid_ok else None), (ask if ask_ok else None)

    # -- order management --------------------------------------------------

    def on_step(self, bt) -> None:
        yb, bid_size, nb, ask_size = bt.top()
        live = bt.orders()
        if not bt.valid or yb < 0 or nb < 0:
            for o in live:
                if not o.cancel_requested:
                    bt.cancel(o.id)
            return
        bid, ask = self.quotes(bt.now, yb, bid_size, PRICE_MAX - nb, ask_size, bt.position)
        want = {YES: bid, NO: None if ask is None else PRICE_MAX - ask}
        busy = {YES: False, NO: False}
        for o in live:
            busy[o.side] = True
            # An order already at the wanted price keeps its place in the queue.
            if o.price != want[o.side] and not o.cancel_requested:
                bt.cancel(o.id)
        size = self.p.quote_size * QTY_SCALE
        for side in (YES, NO):
            if want[side] is not None and not busy[side]:
                bt.submit(side, want[side], size)
