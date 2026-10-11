"""Pure-Python reference implementation of the book and the fill simulator.

This is the specification. It is written for clarity, uses dictionaries where
the C++ core uses dense arrays and bitmaps, and is checked against the core on
randomized tapes (tests/test_differential.py): after every step both must agree
on every level, every order, every fill and the cash balance, to the unit.

It also serves as the baseline for the replay benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import tape as T
from .fixedpoint import PRICE_MAX

NEVER = 2**63 - 1

CONSERVATIVE, PROPORTIONAL, OPTIMISTIC = 0, 1, 2
PENDING_NEW, ACTIVE, FILLED, CANCELED, REJECTED = 0, 1, 2, 3, 4


class Book:
    def __init__(self) -> None:
        self.levels: tuple[dict[int, int], dict[int, int]] = ({}, {})
        self.valid = False
        self.stats = dict(deltas=0, trades=0, snapshots=0, gaps=0, negative_clamps=0, bad_events=0)

    def qty(self, side: int, price: int) -> int:
        return self.levels[side].get(price, 0)

    def best(self, side: int) -> int:
        lv = self.levels[side]
        return max(lv) if lv else -1

    @property
    def yes_bid(self) -> int:
        return self.best(T.YES)

    @property
    def yes_ask(self) -> int:
        nb = self.best(T.NO)
        return -1 if nb < 0 else PRICE_MAX - nb

    def depth(self, side: int, k: int = 10) -> tuple[list[int], list[int]]:
        prices = sorted(self.levels[side], reverse=True)[:k]
        return prices, [self.levels[side][p] for p in prices]

    def _set(self, side: int, price: int, q: int) -> None:
        if q > 0:
            self.levels[side][price] = q
        else:
            self.levels[side].pop(price, None)

    def apply_delta(self, side: int, price: int, dq: int) -> int:
        q = self.qty(side, price) + dq
        if q < 0:
            self.stats["negative_clamps"] += 1
            q = 0
        self._set(side, price, q)
        self.stats["deltas"] += 1
        return q

    def begin_snapshot(self) -> None:
        self.levels[0].clear()
        self.levels[1].clear()
        self.valid = False

    def set_level(self, side: int, price: int, q: int) -> None:
        self._set(side, price, max(q, 0))

    def end_snapshot(self) -> None:
        self.valid = True
        self.stats["snapshots"] += 1

    def mark_gap(self) -> None:
        self.valid = False
        self.stats["gaps"] += 1

    def apply_event(self, typ: int, side: int, price: int, qty: int) -> None:
        if side > 1 or not 0 <= price <= PRICE_MAX:
            self.stats["bad_events"] += 1
        elif typ == T.DELTA:
            self.apply_delta(side, price, qty)
        elif typ == T.TRADE:
            self.stats["trades"] += 1
        elif typ == T.SNAPSHOT_LEVEL:
            self.set_level(side, price, qty)
        elif typ == T.SNAPSHOT_BEGIN:
            self.begin_snapshot()
        elif typ == T.SNAPSHOT_END:
            self.end_snapshot()
        elif typ == T.GAP:
            self.mark_gap()
        else:
            self.stats["bad_events"] += 1

    def apply(self, events: np.ndarray) -> int:
        for typ, side, price, qty in zip(
            events["type"].tolist(), events["side"].tolist(), events["price"].tolist(), events["qty"].tolist()
        ):
            self.apply_event(typ, side, price, qty)
        return len(events)


@dataclass
class SimConfig:
    queue_model: int = PROPORTIONAL
    order_latency_ns: int = 0
    cancel_latency_ns: int = 0
    attribution_window_ns: int = 50_000_000
    maker_fee_ppm: int = 0
    balance_precision_micro: int = 10_000


@dataclass
class Order:
    id: int
    side: int
    price: int
    qty: int
    submit_ts: int
    active_ts: int
    remaining: int = 0
    ahead: int = 0
    cancel_ts: int = NEVER
    state: int = PENDING_NEW
    fee_carried: int = 0

    @property
    def cancel_requested(self) -> bool:
        return self.cancel_ts != NEVER


@dataclass
class _Pending:
    ts: int
    qty: int
    price: int
    side: int


def compute_fee(cfg: SimConfig, carried: int, price: int, qty: int, revenue: int) -> tuple[int, int, int, int]:
    """Returns (trade_fee, rounding_fee, rebate, new_carried) in micro-dollars."""
    scaled = cfg.maker_fee_ppm * qty * price * (PRICE_MAX - price)
    trade = -(-scaled // 10_000_000_000)  # ceiling
    step = cfg.balance_precision_micro
    raw = revenue - trade
    rounding = raw - (raw // step) * step  # // floors toward -inf
    carried += rounding
    rebate = min(carried // step, (trade + rounding) // step) * step
    return trade, rounding, rebate, carried - rebate


@dataclass
class ExchangeSim:
    cfg: SimConfig
    position: int = 0
    cash: int = 0
    fees: int = 0
    volume: int = 0
    live: list[Order] = field(default_factory=list)
    closed: list[Order] = field(default_factory=list)
    fills: list[tuple] = field(default_factory=list)  # (ts, order_id, qty, fee, price, side)
    stats: dict = field(
        default_factory=lambda: dict(
            submitted=0, rejected=0, canceled=0, filled=0, cancel_qty=0, matched_trade_qty=0, unmatched_trade_qty=0
        )
    )
    _debits: list[_Pending] = field(default_factory=list)
    _credits: list[_Pending] = field(default_factory=list)
    _next_id: int = 0
    _next_due: int = NEVER
    _after_gap: bool = False

    def submit(self, now: int, side: int, price: int, qty: int) -> int:
        if not 0 < price < PRICE_MAX:
            raise ValueError("price must be inside (0, 1)")
        if qty <= 0:
            raise ValueError("qty must be positive")
        self._next_id += 1
        o = Order(self._next_id, side, price, qty, now, now + self.cfg.order_latency_ns, remaining=qty)
        self.live.append(o)
        self._next_due = min(self._next_due, o.active_ts)
        self.stats["submitted"] += 1
        return o.id

    def cancel(self, now: int, order_id: int) -> bool:
        for o in self.live:
            if o.id == order_id:
                if o.cancel_ts == NEVER:
                    o.cancel_ts = now + self.cfg.cancel_latency_ns
                    self._next_due = min(self._next_due, max(o.cancel_ts, o.active_ts))
                return True
        return False

    def on_time(self, t: int, book: Book) -> None:
        if t < self._next_due:
            return
        self._expire(t, book)
        keep = []
        for o in self.live:
            if o.state == PENDING_NEW and o.active_ts <= t:
                self._activate(o, book)
            if o.state == ACTIVE and o.cancel_ts <= t:
                o.state = CANCELED
            if o.state in (CANCELED, REJECTED):
                self._close(o)
            else:
                keep.append(o)
        self.live = keep
        self._recompute_due()

    def on_decrease(self, t: int, side: int, price: int, removed: int) -> None:
        left = removed
        for c in self._credits:
            if left == 0:
                break
            if c.side == side and c.price == price:
                m = min(c.qty, left)
                c.qty -= m
                left -= m
                self.stats["matched_trade_qty"] += m
        self._credits = [c for c in self._credits if c.qty]
        if left > 0:
            self._debits.append(_Pending(t, left, price, side))
            self._next_due = min(self._next_due, t + self.cfg.attribution_window_ns)

    def on_increase(self, t: int, side: int, price: int, added: int) -> None:
        ours = 1 - side
        for o in list(self.live):
            if o.state == ACTIVE and o.side == ours and o.price + price >= PRICE_MAX:
                o.ahead = 0
                self._fill(o, t, min(o.remaining, added))

    def on_trade(self, t: int, taker: int, yes_price: int, qty: int) -> None:
        side = 1 - taker
        price = PRICE_MAX - yes_price if taker == T.YES else yes_price
        paired: list[tuple[int, int]] = []
        left = qty
        for d in self._debits:
            if left == 0:
                break
            if d.side == side and d.price == price:
                m = min(d.qty, left)
                d.qty -= m
                left -= m
                paired.append((d.ts, m))
                self.stats["matched_trade_qty"] += m
        self._debits = [d for d in self._debits if d.qty]
        if left > 0:
            self._credits.append(_Pending(t, left, price, side))
            self._next_due = min(self._next_due, t + self.cfg.attribution_window_ns)

        for o in list(self.live):
            if o.state != ACTIVE or o.side != side or o.price < price:
                continue
            seen = qty - sum(m for ts, m in paired if ts <= o.active_ts)
            if o.price > price:
                o.ahead = 0
                take = min(o.remaining, seen)
            else:
                consumed = min(o.ahead, seen)
                o.ahead -= consumed
                take = min(o.remaining, seen - consumed)
            self._fill(o, t, take)

    def on_snapshot_end(self, book: Book) -> None:
        self._debits.clear()
        self._credits.clear()
        for o in self.live:
            if o.state == ACTIVE:
                level = book.qty(o.side, o.price)
                o.ahead = level if self._after_gap else min(o.ahead, level)
        self._after_gap = False
        self._recompute_due()

    def on_gap(self) -> None:
        self._after_gap = True

    # -- internals ---------------------------------------------------------

    def _activate(self, o: Order, book: Book) -> None:
        opp = book.best(1 - o.side)
        if not book.valid or (opp >= 0 and o.price + opp >= PRICE_MAX):
            o.state = REJECTED
            self.stats["rejected"] += 1
            return
        o.ahead = book.qty(o.side, o.price)
        o.state = ACTIVE

    def _close(self, o: Order) -> None:
        if o.state == CANCELED:
            self.stats["canceled"] += 1
        if o.state == FILLED:
            self.stats["filled"] += 1
        self.closed.append(o)

    def _fill(self, o: Order, t: int, q: int) -> None:
        if q <= 0:
            return
        o.remaining -= q
        notional = o.price * q
        trade, rounding, rebate, o.fee_carried = compute_fee(self.cfg, o.fee_carried, o.price, q, -notional)
        fee = trade + rounding - rebate
        if o.side == T.YES:
            self.position += q
            self.cash -= notional
        else:
            self.position -= q
            self.cash += (PRICE_MAX - o.price) * q
        self.cash -= fee
        self.fees += fee
        self.volume += q
        self.fills.append((t, o.id, q, fee, o.price, o.side))
        if o.remaining == 0:
            o.state = FILLED
            self._close(o)
            self.live.remove(o)

    def _apply_cancel(self, d: _Pending, book: Book) -> None:
        self.stats["cancel_qty"] += d.qty
        level = book.qty(d.side, d.price)
        model = self.cfg.queue_model
        for o in self.live:
            if o.state != ACTIVE or o.side != d.side or o.price != d.price or o.active_ts >= d.ts:
                continue
            if model == PROPORTIONAL:
                before = level + d.qty
                if o.ahead > 0 and before > 0:
                    o.ahead -= d.qty * o.ahead // before
            elif model == OPTIMISTIC:
                o.ahead = max(0, o.ahead - d.qty)
            o.ahead = min(o.ahead, level)

    def _expire(self, t: int, book: Book) -> None:
        w = self.cfg.attribution_window_ns
        n = 0
        while n < len(self._debits) and self._debits[n].ts + w <= t:
            self._apply_cancel(self._debits[n], book)
            n += 1
        del self._debits[:n]
        n = 0
        while n < len(self._credits) and self._credits[n].ts + w <= t:
            self.stats["unmatched_trade_qty"] += self._credits[n].qty
            n += 1
        del self._credits[:n]

    def _recompute_due(self) -> None:
        due = NEVER
        for o in self.live:
            if o.state == PENDING_NEW:
                due = min(due, o.active_ts)
            elif o.cancel_ts != NEVER:
                due = min(due, o.cancel_ts)
        w = self.cfg.attribution_window_ns
        if self._debits:
            due = min(due, self._debits[0].ts + w)
        if self._credits:
            due = min(due, self._credits[0].ts + w)
        self._next_due = due


class Backtester:
    """Same interface as the C++ ``klob.Backtester``."""

    def __init__(self, events: np.ndarray, config: SimConfig | None = None) -> None:
        self._ts = events["ts"].tolist()
        self._typ = events["type"].tolist()
        self._side = events["side"].tolist()
        self._price = events["price"].tolist()
        self._qty = events["qty"].tolist()
        self._n = len(events)
        self.cursor = 0
        self.now = self._ts[0] if self._n else 0
        self.book = Book()
        self.sim = ExchangeSim(config or SimConfig())
        self.last_trade = -1

    @property
    def done(self) -> bool:
        return self.cursor >= self._n

    @property
    def valid(self) -> bool:
        return self.book.valid

    @property
    def position(self) -> int:
        return self.sim.position

    @property
    def cash(self) -> int:
        return self.sim.cash

    @property
    def fees(self) -> int:
        return self.sim.fees

    @property
    def volume(self) -> int:
        return self.sim.volume

    @property
    def n_fills(self) -> int:
        return len(self.sim.fills)

    @property
    def book_stats(self) -> dict:
        return dict(self.book.stats)

    @property
    def sim_stats(self) -> dict:
        return dict(self.sim.stats)

    def submit(self, side: int, price: int, qty: int) -> int:
        return self.sim.submit(self.now, side, price, qty)

    def cancel(self, order_id: int) -> bool:
        return self.sim.cancel(self.now, order_id)

    def qty(self, side: int, price: int) -> int:
        return self.book.qty(side, price)

    def best(self, side: int) -> int:
        return self.book.best(side)

    def depth(self, side: int, k: int = 10):
        return self.book.depth(side, k)

    def top(self) -> tuple[int, int, int, int]:
        yb, nb = self.book.best(T.YES), self.book.best(T.NO)
        return yb, self.book.qty(T.YES, yb) if yb >= 0 else 0, nb, self.book.qty(T.NO, nb) if nb >= 0 else 0

    def orders(self) -> list[Order]:
        return list(self.sim.live)

    def drain_closed(self) -> list[Order]:
        out, self.sim.closed = self.sim.closed, []
        return out

    def fills(self) -> list[tuple]:
        return list(self.sim.fills)

    def run_until(self, t: int) -> bool:
        while self.cursor < self._n and self._ts[self.cursor] <= t:
            self._step(self.cursor)
            self.cursor += 1
        if t > self.now:
            self.now = t
        self.sim.on_time(self.now, self.book)
        return self.cursor < self._n

    def elapse(self, dt: int) -> bool:
        return self.run_until(self.now + dt)

    def run_all(self) -> None:
        self.run_until(NEVER - 1)

    def _step(self, i: int) -> None:
        ts, typ, side, price, qty = self._ts[i], self._typ[i], self._side[i], self._price[i], self._qty[i]
        book, sim = self.book, self.sim
        sim.on_time(ts - 1, book)  # an order arriving at exactly ts is placed after this event
        if side > 1 or not 0 <= price <= PRICE_MAX:
            book.stats["bad_events"] += 1
        elif typ == T.DELTA:
            before = book.qty(side, price)
            after = book.apply_delta(side, price, qty)
            if after < before:
                sim.on_decrease(ts, side, price, before - after)
            elif after > before:
                sim.on_increase(ts, side, price, after - before)
        elif typ == T.TRADE:
            book.stats["trades"] += 1
            self.last_trade = price
            if qty > 0:
                sim.on_trade(ts, side, price, qty)
        elif typ == T.SNAPSHOT_LEVEL:
            book.set_level(side, price, qty)
        elif typ == T.SNAPSHOT_BEGIN:
            book.begin_snapshot()
        elif typ == T.SNAPSHOT_END:
            book.end_snapshot()
            sim.on_snapshot_end(book)
        elif typ == T.GAP:
            book.mark_gap()
            sim.on_gap()
        else:
            book.stats["bad_events"] += 1
