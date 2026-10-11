# Design notes

These notes explain the decisions in klob that are not obvious from the code, in the order a tape flows through the system.

## 1. Kalshi's book

A Kalshi market is a binary contract that pays $1 if an event happens. The exchange does not keep asks. It keeps bids to buy YES and bids to buy NO, and because one YES plus one NO is always worth exactly $1, a NO bid at 53 cents is the same thing as an offer to sell YES at 47 cents. A taker who "buys YES at 47" is matched against that resting NO bid.

klob keeps this representation instead of converting to bids and asks. Every order, level and fill is stored on its own side at its own price, and the conversion (`yes_ask = 1 - best_no_bid`) happens only where a strategy needs it. The alternative, mirroring NO prices into an ask book on ingest, makes it easy to get a sign wrong in exactly one place and hard to compare the book against the raw feed.

Position is kept as one number, net YES-equivalent contracts. Buying a NO contract at price q is booked as selling a YES at 1 - q: position falls by one and cash rises by 1 - q. This is exact once the $1 pair is netted, and it means PnL is always `cash + position * mid`.

## 2. Integers everywhere

Kalshi sends prices as strings with up to four decimals and sizes with up to two. klob stores prices in units of $0.0001 and sizes in units of 0.01 contract, both as integers, and parses the strings digit by digit rather than through `float`.

The reason is not speed. A price times a size lands exactly on a $0.000001 grid, so cash is an exact integer number of micro-dollars, and two implementations of the simulator can be required to agree to the last unit. With floating point, the C++ core and the Python reference would differ in the last bits and the differential test (section 8) would have to compare with a tolerance, which is where real bugs hide.

Two intermediate products do not fit in 64 bits: the fee `multiplier * size * P * (1 - P)` and the proportional queue update `cancelled * ahead / level`. Both use a 128-bit intermediate.

## 3. The tape

A recording is normalized into an array of 32-byte records:

```
ts       int64   local receive time, ns (the clock the simulation runs on)
exch_ts  int64   exchange time, ns, when the feed provides one
qty      int64   size; signed for deltas
price    int32   price level
type     uint8   snapshot begin / level / end, delta, trade, gap
side     uint8   YES or NO (for trades: the taker's side)
flags    uint16
```

The same layout is the C++ struct, the NumPy dtype and the file format, so a tape is memory-mapped from disk and read by the C++ core in place. Two records fit in a cache line and replay is a linear scan, which is the access pattern hardware prefetchers handle best.

The simulation clock is the local receive time, not the exchange time. A strategy can only react to a message when it arrives, and using exchange timestamps would let it act on information it did not have yet.

## 4. The book data structure

Prices live on a bounded grid: 0 to 10,000 ticks. Each side is a flat array of 10,001 sizes indexed by price, so applying a delta is one load, one add and one store with no allocation and no pointer chasing.

A flat array makes "what is the best price?" a scan. klob keeps a two-level bitmap beside it: 157 words with one bit per price level, and 3 summary words with one bit per word. The best price is the highest set bit of the highest non-zero summary word, found with two count-leading-zeros instructions. Updating the bitmap on a delta is two ORs, or two ANDs and a compare.

The array is 80 KB per side, more than an L1 cache, but activity is concentrated within a few ticks of the touch, so the lines that are actually touched stay hot. A snapshot clears only the occupied levels by walking the bitmap.

On the 24-hour benchmark tape this is 2.6 to 3.3 times faster than the same book on `std::map`. The gap is modest because the synthetic book has only about 10 levels a side, so the tree is small and stays in cache. I would expect it to widen on a deeper book but have not measured that.

## 5. Fill simulation

This is the core of the project. An order of ours rests at price p. The simulator keeps `ahead`, an estimate of the size in front of it at p, and updates it from the feed.

**Joining.** An order sent at time t reaches the exchange at `t + order_latency` and joins behind everything displayed at its price at that moment. Orders are post-only: if the order would cross the other side when it arrives, it is rejected. An order arriving at exactly the timestamp of a book event is placed after that event.

**A trade at our price** consumes the queue from the front. `ahead` falls by the trade size, and if the trade is larger than `ahead`, the remainder fills us.

**A trade at a worse price on our side** means the taker went past us. Had we really been there we would have been hit first, so we are filled up to the trade size.

**A new order on the other side that crosses our price** could only rest in the real book because we were not there. With us in the book it would have traded against us, so we are filled up to its size.

**A decrease at our price that no trade explains** is a cancel, and this is where L2 data runs out. The cancelled order may have been ahead of us or behind us. Three models:

- *Conservative*: assume it was behind us whenever that is possible. `ahead` changes only if the displayed size falls below it.
- *Proportional*: assume the cancelled size was spread evenly over the queue. If the level held L before the cancel of r, `ahead` falls by `r * ahead / L`.
- *Optimistic*: assume it was ahead of us. `ahead` falls by r.

The first and last are bounds: a correct simulation can never fill less than the conservative model or more than the optimistic one, as long as every taker's full size is visible. The synthetic study confirms this order by order. Running a strategy under both gives an interval for its PnL, and if a strategy is only profitable under the optimistic model, the backtest has not shown that it is profitable.

### Telling trades from cancels

Kalshi publishes trades on one channel and book deltas on another, and does not guarantee which arrives first. A trade of 100 at our level shows up as a trade message and as a delta of -100, in either order. Treating both naively would advance the queue twice.

The simulator pairs them:

- A trade arrives first. Its effect on the queue is applied at once, and its size is remembered as a *credit* at that level. When the delta arrives it consumes the credit and has no further effect.
- A delta arrives first. Nothing is applied yet; the decrease is held as a *debit*. If a trade at that level arrives within the attribution window (50 ms by default), it claims the debit and is applied as a trade. If the window passes with no trade, the debit is applied as a cancel under the chosen queue model.

Two details matter. First, the queue estimate is not clamped to the displayed size while a debit is pending, because the delta has already shrunk the display and clamping would lose the size that the late trade is about to consume. Second, an order that joins a level between a delta and its late trade already saw the smaller size when it joined, so that trade must not advance it. Each debit carries the time it was observed, and an order ignores pieces observed before it arrived.

The property this buys is tested directly: the same market published with every delta before its trade, or every delta after it, gives every order the same fill.

### Gaps

Book deltas carry a sequence number. If one is skipped, or the connection drops, the book is unknown until the next full snapshot. The normalizer writes a `GAP` event, the book is marked invalid, new orders are rejected, and when the snapshot arrives any resting order is moved to the back of its level. Anything could have happened in the gap, and the back of the queue is the only position that cannot overstate fills.

### Fees

Kalshi's fee is quadratic in price, `multiplier * contracts * P * (1 - P)`, and is rounded in two stages: the model fee is rounded up to a micro-dollar, then a rounding fee is added so the balance lands on the account's precision grid (one cent for most accounts). The rounding overpayment accumulates per order and is rebated in whole grid steps, capped so no fill has a negative fee. `fees.hpp` implements this in integer micro-dollars and is checked against the worked example in Kalshi's documentation. The multiplier is a parameter; maker fees apply only on some series.

### What is not modelled

- **Market impact.** Our orders do not change anyone else's behaviour, and the trades that hit us still happen to the real orders behind us.
- **Hidden taker size.** A taker's unfilled remainder never appears in a feed, so some fills that would have happened are invisible.
- **Taking liquidity.** There are no marketable orders.
- **Our own orders interacting.** Two of our orders at one price each track the public queue independently.

## 6. The strategy

`strategy.py` is deliberately simple. It exists to exercise the simulator on something with realistic behaviour, not to be an edge.

Avellaneda and Stoikov's rule quotes around a reservation price shifted against inventory, with a spread that widens with volatility. It assumes an arithmetic Brownian mid, and a probability is not one: it is confined to (0, 1) and its volatility shrinks near the ends. klob estimates volatility on the log-odds `x = log(p / (1 - p))`, where a random walk is a reasonable local model, and converts it to price terms with the derivative of the logistic function, `sigma_price = sigma_x * p * (1 - p)`. A contract at 5 cents gets a much tighter quote than one at 50 cents for the same log-odds volatility.

The centre of the quote is the mid adjusted toward the thin side of the book (a size-weighted mid). The inventory skew and half-spread are both expressed in units of the expected price move over a horizon, which makes the parameters easier to reason about than a raw risk-aversion coefficient.

An order whose target price has not changed is left alone, because cancelling and replacing it would give up its queue position.

## 7. The synthetic market and ground truth

Real L2 data cannot validate a queue model, because the thing being modelled is not in the data. `synth.py` simulates a market at the order level, where queue positions are known, and emits the L2 feed Kalshi would publish for it.

The order flow is a zero-intelligence model: limit orders arrive at a constant rate around a latent fair value, every resting order is cancelled at a constant hazard, and takers arrive at a constant rate. Two additions make it less benign. The fair value is a random walk with jumps in log-odds, and a share of takers know it, so quotes left behind by a jump are picked off. And cancellations can be biased toward newer orders, which sit at the back of their queues, so the proportional model's uniformity assumption is deliberately violated.

For ground truth, the generator tracks *shadow orders*: hypothetical orders that join a level at a known time. For each one it records exactly which real orders were ahead of it and updates that set as they trade or cancel. When a taker arrives, the shadow would have received whatever was left after the better-priced orders and the orders ahead of it were filled. The backtester then replays the L2 feed with a real simulated order in place of each shadow, and the two fills are compared.

This says how well each queue model recovers the truth *in this simulated market*. It does not say the simulated market resembles Kalshi.

## 8. Testing

The Python reference in `reference.py` implements the same rules with dictionaries and lists. The differential test replays randomized tapes through both implementations with random order activity and compares the complete state after every step: all levels, all live and closed orders including `ahead`, all fills, position, cash, fees and the diagnostic counters.

The tapes are adversarial. They contain tied timestamps, deltas larger than the level, out-of-range prices, unknown event types, gaps, snapshots mid-session, trades with no delta and deltas with no trade. A separate test asserts that those tapes actually produce fills, partial fills, cancels, rejections and fees, since agreement on tapes where nothing happens would prove nothing.

`scripts/mutation_check.py` measures whether the tests would notice a mistake. It applies 18 small bugs to the C++ one at a time and requires the tests to fail for each. Its first run found a real hole. The scenario for an order that joins a level in the same nanosecond as a cancel never let the order go live before the attribution window closed, so one mutant survived until the scenario was fixed.

## 9. Performance notes

Numbers are in the README and `results/RESULTS.md`. A few observations behind them:

- Book-only replay costs about 6 to 7 ns per event, which means reading roughly 5 GB of tape per second on one core. I have not profiled where the remaining time goes; at that rate a smaller record may matter as much as a faster book.
- The fill simulator scans a flat vector of live orders on each trade or delta. A market maker has a handful of resting orders, so this beats any map. It would not suit thousands of simultaneous orders.
- `on_time` is called before every event but returns after one comparison unless something is due: the simulator keeps the earliest time at which an order arrives, a cancel lands or an attribution window closes.
- Strategies step the engine from Python at a fixed interval and C++ replays everything in between, so the cost of Python is per decision, not per event.
