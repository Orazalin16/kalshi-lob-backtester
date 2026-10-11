"""Synthetic Kalshi-style market with order-level ground truth.

Recorded Kalshi data is L2: it never reveals where an order stood in its queue,
so a queue model cannot be checked against it. This module simulates a binary
market at the order level (L3) with strict price-time priority, emits exactly
the L2 feed Kalshi would publish (snapshot, per-level deltas, public trades),
and separately tracks what would truly have happened to "shadow" orders placed
in that market. The backtester only sees the L2 feed; its fills can then be
scored against the truth.

The market itself is a zero-intelligence model in the style of Smith, Farmer et
al.: limit orders arrive around a latent fair probability, every resting order
is cancelled at a constant hazard, and takers are a mix of noise and informed
flow that picks off stale quotes after the fair value moves. The fair value is
a random walk with jumps in log-odds, which keeps it inside (0, 1) and makes
price volatility scale with p(1 - p) the way a binary contract's does.

Nothing here is calibrated to real Kalshi flow. It exists to test machinery.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import tape as T

CENT = 100  # price ticks per cent
CONTRACT = 100  # size units per contract

SHADOW_DTYPE = np.dtype(
    [
        ("t0", "<i8"),  # time the shadow joins, ns
        ("side", "u1"),
        ("price", "<i4"),  # ticks
        ("qty", "<i8"),  # units
        ("ahead0", "<i8"),  # true size ahead at t0, units
        ("filled", "<i8"),  # true filled size within the horizon, units
        ("t_first", "<i8"),  # time of first fill, or -1
        ("t_done", "<i8"),  # time fully filled, or -1
    ]
)


@dataclass
class SynthConfig:
    seed: int = 7
    duration_s: float = 3600.0
    warmup_s: float = 120.0
    start_prob: float = 0.50
    logit_vol: float = 0.004  # per sqrt(second)
    jump_rate: float = 1 / 400  # per second
    jump_scale: float = 0.12  # std of a jump in log-odds
    limit_rate: float = 16.0  # limit orders per second
    market_rate: float = 3.0  # taker orders per second
    cancel_hazard: float = 0.05  # per resting order per second
    informed_frac: float = 0.35  # share of takers who know the fair value
    informed_edge: float = 0.6  # cents of edge an informed taker requires
    quote_noise: float = 0.5  # std (cents) of a liquidity provider's fair-value error
    depth_decay: float = 0.6  # geometric decay of limit prices away from fair
    size_mu: float = 3.0  # log-mean of order size in contracts
    size_sigma: float = 0.9
    cancel_back_bias: float = 0.5  # 0 = cancels uniform over orders, 1 = strongly prefer newer orders
    delta_first_prob: float = 0.5  # chance a trade's book delta is published before the trade
    feed_latency_ns: int = 0
    # A taker's unfilled remainder never appears in a real feed. Capping taker
    # size to what is displayed removes that blind spot, which makes the
    # conservative/optimistic bracket on fills exact (see tests/test_synth.py).
    cap_takers_to_displayed: bool = False
    shadow_every_s: float = 5.0  # 0 disables ground-truth shadows
    shadow_qty: int = 25  # contracts
    shadow_horizon_s: float = 60.0
    shadow_behind_prob: float = 0.3  # chance a shadow rests one tick behind the best price


class _Order:
    __slots__ = ("oid", "side", "price", "qty", "pos", "watchers")

    def __init__(self, oid: int, side: int, price: int, qty: int) -> None:
        self.oid, self.side, self.price, self.qty = oid, side, price, qty
        self.pos = -1  # index in the per-side list used for cancel sampling
        self.watchers: list | None = None  # shadows that have this order ahead of them


class _Shadow:
    __slots__ = ("t0", "side", "price", "qty", "rem", "ahead", "ahead0", "t_first", "t_done", "t_end")


@dataclass
class SynthResult:
    events: np.ndarray
    shadows: np.ndarray
    fair: np.ndarray  # (ts ns, fair probability) sampled once per second
    config: SynthConfig


class _Market:
    """Order-level book for one binary market. Prices are in cents, sizes in contracts."""

    def __init__(self, cfg: SynthConfig) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.queues: tuple[dict[int, deque], dict[int, deque]] = ({}, {})
        self.level_qty: tuple[dict[int, int], dict[int, int]] = ({}, {})
        self.resting: tuple[list[_Order], list[_Order]] = ([], [])
        self.next_oid = 0
        self.x = math.log(cfg.start_prob / (1 - cfg.start_prob))
        self.rows: list[tuple] = []
        self.recording = False
        self.last_ts = 0
        self.now_ns = 0
        self.live_shadows: list[_Shadow] = []
        self.done_shadows: list[_Shadow] = []

    # -- helpers -----------------------------------------------------------

    @property
    def fair_cents(self) -> float:
        return 100.0 / (1.0 + math.exp(-self.x))

    def best(self, side: int) -> int:
        lv = self.level_qty[side]
        return max(lv) if lv else -1

    def _emit(self, typ: int, side: int, price_c: int, qty_c: int) -> None:
        if not self.recording:
            return
        ts = max(self.now_ns + self.cfg.feed_latency_ns, self.last_ts + 1)
        self.last_ts = ts
        self.rows.append((ts, self.now_ns, qty_c * CONTRACT, price_c * CENT, typ, side))

    def _reduce(self, o: _Order, r: int) -> None:
        """Removes r contracts from a resting order and keeps every index in step."""
        o.qty -= r
        lv = self.level_qty[o.side]
        left = lv[o.price] - r
        if left > 0:
            lv[o.price] = left
        else:
            del lv[o.price]
            del self.queues[o.side][o.price]
        if o.watchers:
            for sh in o.watchers:
                sh.ahead -= r
        if o.qty == 0:
            lst = self.resting[o.side]
            last = lst.pop()
            if last is not o:
                lst[o.pos] = last
                last.pos = o.pos
            o.watchers = None

    def _rest(self, side: int, price: int, qty: int) -> None:
        self.next_oid += 1
        o = _Order(self.next_oid, side, price, qty)
        self.queues[side].setdefault(price, deque()).append(o)
        self.level_qty[side][price] = self.level_qty[side].get(price, 0) + qty
        o.pos = len(self.resting[side])
        self.resting[side].append(o)
        self._emit(T.DELTA, side, price, qty)

    # -- order flow --------------------------------------------------------

    def incoming(self, side_in: int, limit: int, size: int, rests: bool) -> None:
        """An order to buy `side_in` at up to `limit` cents.

        It trades against resting bids on the other side priced at 100 - limit
        or better, in price-time priority. Any remainder rests if `rests`.
        """
        s = 1 - side_in
        floor_price = 100 - limit
        if not rests and self.cfg.cap_takers_to_displayed:
            size = min(size, sum(q for p, q in self.level_qty[s].items() if p >= floor_price))
            if size == 0:
                return
        if self.live_shadows:
            self._fill_shadows(s, floor_price, size)
        z = size
        lv = self.level_qty[s]
        while z > 0 and lv:
            p = max(lv)
            if p < floor_price:
                break
            q = self.queues[s][p]
            while z > 0 and q:
                o = q[0]
                if o.qty == 0:  # cancelled earlier, lazily removed
                    q.popleft()
                    continue
                m = o.qty if o.qty <= z else z
                z -= m
                if m == o.qty:
                    q.popleft()
                yes_price = 100 - p if side_in == T.YES else p
                if self.rng.random() < self.cfg.delta_first_prob:
                    self._emit(T.DELTA, s, p, -m)
                    self._emit(T.TRADE, side_in, yes_price, m)
                else:
                    self._emit(T.TRADE, side_in, yes_price, m)
                    self._emit(T.DELTA, s, p, -m)
                self._reduce(o, m)
                if p not in lv:
                    break
        if rests and z > 0 and 1 <= limit <= 99:
            self._rest(side_in, limit, z)

    def cancel_random(self) -> None:
        n0, n1 = len(self.resting[0]), len(self.resting[1])
        if n0 + n1 == 0:
            return
        side = 0 if self.rng.random() * (n0 + n1) < n0 else 1
        lst = self.resting[side]
        o = lst[self.rng.randrange(len(lst))]
        if self.rng.random() < self.cfg.cancel_back_bias:
            other = lst[self.rng.randrange(len(lst))]
            if other.oid > o.oid:  # newer orders sit further back
                o = other
        r = o.qty
        q = self.queues[side][o.price]
        self._emit(T.DELTA, side, o.price, -r)
        if q[0] is o:
            q.popleft()
        elif len(q) > 64 and sum(1 for x in q if x.qty) * 2 < len(q):
            self._reduce(o, r)
            if o.price in self.queues[side]:
                self.queues[side][o.price] = deque(x for x in q if x.qty)
            return
        self._reduce(o, r)

    def limit_order(self) -> None:
        cfg, rng = self.cfg, self.rng
        side = T.YES if rng.random() < 0.5 else T.NO
        ref = self.fair_cents + rng.gauss(0.0, cfg.quote_noise)
        if side == T.NO:
            ref = 100.0 - ref
        k = 0
        while rng.random() < cfg.depth_decay:
            k += 1
        price = math.ceil(ref) - 1 - k
        if 1 <= price <= 99:
            self.incoming(side, price, self._size(), rests=True)

    def market_order(self) -> None:
        cfg, rng = self.cfg, self.rng
        if rng.random() < cfg.informed_frac:
            fair = self.fair_cents
            nb, yb = self.best(T.NO), self.best(T.YES)
            if nb >= 0 and fair - (100 - nb) >= cfg.informed_edge:
                self.incoming(T.YES, math.floor(fair - cfg.informed_edge), 4 * self._size(), rests=False)
            elif yb >= 0 and yb - fair >= cfg.informed_edge:
                self.incoming(T.NO, math.floor(100 - fair - cfg.informed_edge), 4 * self._size(), rests=False)
            return
        side = T.YES if rng.random() < 0.5 else T.NO
        opp = self.best(1 - side)
        if opp >= 0:
            self.incoming(side, 100 - opp, self._size(), rests=False)

    def _size(self) -> int:
        return max(1, min(2000, int(self.rng.lognormvariate(self.cfg.size_mu, self.cfg.size_sigma))))

    # -- ground truth ------------------------------------------------------

    def add_shadow(self, side: int) -> None:
        best = self.best(side)
        if best < 0:
            return
        price = best - 1 if self.rng.random() < self.cfg.shadow_behind_prob else best
        if price < 1:
            return
        sh = _Shadow()
        sh.t0 = self.last_ts
        sh.t_end = sh.t0 + int(self.cfg.shadow_horizon_s * 1e9)
        sh.side, sh.price = side, price
        sh.qty = sh.rem = self.cfg.shadow_qty
        sh.t_first = sh.t_done = -1
        ahead = 0
        for o in self.queues[side].get(price, ()):
            if o.qty:
                ahead += o.qty
                if o.watchers is None:
                    o.watchers = []
                o.watchers.append(sh)
        sh.ahead = sh.ahead0 = ahead
        self.live_shadows.append(sh)

    def _fill_shadows(self, s: int, floor_price: int, size: int) -> None:
        """What an incoming order of `size` would have given each shadow on side s.

        The taker first clears every real order at a better price, then the real
        orders that were ahead of the shadow at its own price, then the shadow.
        """
        lv = self.level_qty[s]
        ts = max(self.now_ns + self.cfg.feed_latency_ns, self.last_ts + 1)
        for sh in self.live_shadows:
            if sh.side != s or sh.price < floor_price or sh.rem == 0 or ts > sh.t_end:
                continue
            better = 0
            for p, q in lv.items():
                if p > sh.price:
                    better += q
            got = size - better - sh.ahead
            if got <= 0:
                continue
            got = min(got, sh.rem)
            sh.rem -= got
            if sh.t_first < 0:
                sh.t_first = ts
            if sh.rem == 0:
                sh.t_done = ts

    def retire_shadows(self) -> None:
        keep = []
        for sh in self.live_shadows:
            if sh.rem == 0 or self.now_ns > sh.t_end:
                self.done_shadows.append(sh)
            else:
                keep.append(sh)
        self.live_shadows = keep

    def snapshot(self) -> None:
        self._emit(T.SNAPSHOT_BEGIN, 0, 0, 0)
        for side in (T.YES, T.NO):
            for p in sorted(self.level_qty[side]):
                self._emit(T.SNAPSHOT_LEVEL, side, p, self.level_qty[side][p])
        self._emit(T.SNAPSHOT_END, 0, 0, 0)


def generate(cfg: SynthConfig | None = None) -> SynthResult:
    cfg = cfg or SynthConfig()
    mkt = _Market(cfg)
    rng = mkt.rng
    base_ns = 1_760_000_000 * 10**9
    t = -cfg.warmup_s
    next_shadow = 0.0
    next_fair = 0.0
    fair_rows: list[tuple[int, float]] = []
    sqrt = math.sqrt

    while True:
        n_rest = len(mkt.resting[0]) + len(mkt.resting[1])
        cancel_rate = cfg.cancel_hazard * n_rest
        total = cfg.limit_rate + cfg.market_rate + cancel_rate
        dt = rng.expovariate(total)
        t += dt
        if t >= cfg.duration_s:
            break
        mkt.x += cfg.logit_vol * sqrt(dt) * rng.gauss(0.0, 1.0)
        if rng.random() < cfg.jump_rate * dt:
            mkt.x += rng.gauss(0.0, cfg.jump_scale)
        mkt.x = max(-3.5, min(3.5, mkt.x))  # keep fair within roughly 3c..97c
        mkt.now_ns = base_ns + int(t * 1e9)

        if not mkt.recording and t >= 0:
            mkt.recording = True
            mkt.snapshot()
        if mkt.recording:
            while next_fair <= t:
                fair_rows.append((base_ns + int(next_fair * 1e9), mkt.fair_cents / 100.0))
                next_fair += 1.0
            if cfg.shadow_every_s > 0 and t >= next_shadow:
                next_shadow += cfg.shadow_every_s
                mkt.retire_shadows()
                if t + cfg.shadow_horizon_s < cfg.duration_s:
                    mkt.add_shadow(T.YES)
                    mkt.add_shadow(T.NO)

        u = rng.random() * total
        if u < cfg.limit_rate:
            mkt.limit_order()
        elif u < cfg.limit_rate + cfg.market_rate:
            mkt.market_order()
        else:
            mkt.cancel_random()

    mkt.now_ns = base_ns + int(cfg.duration_s * 1e9)
    mkt.done_shadows.extend(mkt.live_shadows)
    events = T.make_events(mkt.rows) if len(mkt.rows) < 1000 else _rows_to_events(mkt.rows)
    shadows = np.zeros(len(mkt.done_shadows), dtype=SHADOW_DTYPE)
    for i, sh in enumerate(sorted(mkt.done_shadows, key=lambda s: (s.t0, s.side))):
        shadows[i] = (
            sh.t0,
            sh.side,
            sh.price * CENT,
            sh.qty * CONTRACT,
            sh.ahead0 * CONTRACT,
            (sh.qty - sh.rem) * CONTRACT,
            sh.t_first,
            sh.t_done,
        )
    fair = np.array(fair_rows, dtype=[("ts", "<i8"), ("fair", "<f8")])
    return SynthResult(events=events, shadows=shadows, fair=fair, config=cfg)


def _rows_to_events(rows: list[tuple]) -> np.ndarray:
    cols = list(zip(*rows))
    out = np.zeros(len(rows), dtype=T.EVENT_DTYPE)
    for name, col in zip(("ts", "exch_ts", "qty", "price", "type", "side"), cols):
        out[name] = col
    return out
