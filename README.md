# klob

An order-book replay engine and backtester for market making on [Kalshi](https://kalshi.com) event contracts. The core is C++20 with Python bindings.

The question it is built to answer: **if I had been resting a passive order in this market, would it have been filled?** Kalshi publishes a level-2 feed, which gives the total size at each price but not where any one order sits in the queue. A backtest has to assume something about that, and the assumption can matter more than the strategy. klob makes the assumption explicit, runs the same data under a pessimistic, a neutral and an optimistic queue model, and measures how far each one is from the truth on simulated markets where the truth is known.

## What is in here

| Part | Where | What it does |
|---|---|---|
| Order book | `cpp/include/klob/book.hpp` | L2 book for Kalshi's two-sided representation (YES bids and NO bids, no asks). Dense price array plus a two-level bitmap for the best price. |
| Fill simulator | `cpp/include/klob/exchange_sim.hpp` | Tracks the size ahead of each of our resting orders through trades and cancels, with order and cancel latency, post-only rejection and Kalshi's fee rounding. |
| Replay | `cpp/include/klob/backtester.hpp` | Drives the book and simulator over a tape of 32-byte events. Python steps it; C++ does the work between steps. |
| Reference | `python/klob/reference.py` | The same book and simulator in plain Python. It is the specification the C++ is tested against. |
| Kalshi I/O | `python/klob/kalshi/` | Request signing, REST client, WebSocket and REST recorders, and a normalizer from raw messages to tapes. |
| Synthetic market | `python/klob/synth.py` | An order-level (L3) simulated market that emits a Kalshi-style L2 feed and records what would truly have happened to test orders. |
| Strategy | `python/klob/strategy.py` | An inventory-aware market maker with volatility estimated in log-odds. |
| Study | `python/klob/study.py` | Reproduces every number below with `klob study`. |

## Results

All of the following comes from **simulated markets**, not from Kalshi data. The simulated order flow is not calibrated to any real market, and nothing here is evidence that the strategy is profitable. What the results do show is how the machinery behaves. Full tables are in [`results/RESULTS.md`](results/RESULTS.md).

**1. The queue models bracket the truth, order by order.** 107,100 test orders (25 contracts, placed at or one tick behind the best price, held 60 seconds) across three cancellation regimes:

| queue model | filled size vs truth | orders filled more than truth | orders filled less than truth | mean abs error |
|---|---|---|---|---|
| conservative | 89% to 92% | 0 | 8,564 | 1.6 to 2.2 contracts |
| proportional | 99.6% to 102.5% | 1,912 | 747 | 0.3 to 0.6 contracts |
| optimistic | 105% to 108% | 6,329 | 27 | 1.0 to 1.5 contracts |

The conservative model never over-fills. The optimistic model under-fills 27 orders out of 107,100. The cause is takers who wanted more than the book displayed, which no market data feed reveals: when taker size is capped to displayed size the bracket is exact, and a test asserts it (`tests/test_synth.py`).

**2. The queue assumption moves simulated PnL more than anything else.** The same market maker on the same five 8-hour sessions:

| queue model | PnL per session, 50 ms latency | fills | markout after 60 s (cents/contract) |
|---|---|---|---|
| conservative | $4.95 | 303 | 0.25 |
| proportional | $24.56 | 758 | 0.57 |
| optimistic | $65.67 | 1,276 | 0.75 |

A 13x spread in PnL from one modelling choice, with data and strategy held fixed. Moving latency from 0 to 250 ms changes PnL by under 5% in the proportional and optimistic rows. The conservative fills are also worse fills: an order that only executes once the whole queue ahead has traded is being hit by the largest takers, and its markout decays from 0.58 cents at the fill to 0.25 cents a minute later.

**3. Replay speed.** On a 2.1 GHz Xeon cloud VM, single thread, 3.3 million events (24 simulated hours):

| | events per second |
|---|---|
| C++ book replay | 148 M (6.8 ns per event) |
| the same book on `std::map` | about 60 M |
| C++ replay with the fill simulator and two resting orders | 30 M |
| Python reference | 2.7 M |

A 24-hour strategy backtest with 345,600 Python decision steps takes 1.8 seconds.

## Quick start

```bash
git clone https://github.com/Orazalin16/kalshi-lob-backtester
cd kalshi-lob-backtester
pip install ".[dev]"        # builds the C++ extension; needs a C++20 compiler and CMake >= 3.20

pytest                      # 260 tests
klob validate               # queue models against ground truth, one seed
klob study                  # the full study; writes results/RESULTS.md (about two minutes)
```

C++ unit tests and the native benchmark:

```bash
cmake -S . -B build -DKLOB_BUILD_PYTHON=OFF -DKLOB_BUILD_TESTS=ON -DKLOB_BUILD_BENCH=ON
cmake --build build -j && ctest --test-dir build

klob synth day.npy --hours 24 && klob tobin day.npy day.bin
./build/klob_bench day.bin
```

Add `-DKLOB_SANITIZE=ON` to run the same tests under AddressSanitizer and UBSan.

## Using it

```python
import klob
from klob import backtest, tape
from klob.strategy import MarketMaker, MMParams

events = tape.load("tapes/KXHIGHNY-26OCT05-T75.npy")      # memory-mapped, passed to C++ without a copy
config = klob.SimConfig(
    queue_model=klob.QueueModel.CONSERVATIVE,
    order_latency_ns=50_000_000,
    cancel_latency_ns=50_000_000,
)
result = backtest.run(events, MarketMaker(MMParams(quote_size=10, max_position=100)), config)
print(result.summary)
```

Writing a strategy means writing `on_step(bt)`. The backtester exposes `bt.top()`, `bt.depth(side, k)`, `bt.orders()`, `bt.position`, `bt.submit(side, price, qty)` and `bt.cancel(order_id)`. Prices are integers in units of $0.0001 and sizes in units of 0.01 contract, the finest grid Kalshi uses.

## Recording live data

Market data over REST is public. The WebSocket feed, which carries the real per-level deltas, authenticates its handshake even for public channels, so it needs an API key (Account settings -> API keys on kalshi.com; the demo environment issues keys too).

```bash
# No key: poll the public order book and trades once a second
klob record rest KXHIGHNY-26OCT05-T75 --out raw.jsonl.gz

# With a key: the full L2 stream for one or more markets
klob record ws KXHIGHNY-26OCT05-T75 --key-id <id> --key-file kalshi.key --out raw.jsonl.gz

klob normalize raw.jsonl.gz --out-dir tapes/
klob backtest tapes/KXHIGHNY-26OCT05-T75.npy --queue-model conservative --latency-ms 50
```

The recorder stores raw messages with a local receive timestamp and nothing else; parsing happens offline, so a parsing bug cannot lose data. A skipped sequence number or a reconnect becomes a `GAP` event, and the simulator neither quotes against an invalid book nor trusts a queue position across one.

## Status and limitations

- **No results on recorded Kalshi data yet.** The recorder and normalizer are tested against the message examples in Kalshi's API reference, not against a live session.
- **No market impact.** Our orders are assumed not to change what anyone else does. This is reasonable for small size and wrong for large size.
- **Passive orders only.** Every order is post-only. Taking liquidity is not simulated.
- **One market at a time**, and one tick size per run. Kalshi markets with tapered tick grids need the `tick` parameter chosen for the price region being quoted.
- **Fees** follow the documented rounding rules, with the maker multiplier supplied by the user. Check the series' `fee_type` and the current fee schedule before trusting a fee-sensitive result.
- **Hidden taker size** is unobservable in any feed, so even the optimistic model can under-fill slightly (27 in 107,100 above).

## How it is tested

- **Differential tests.** The C++ core and the Python reference replay 150 randomized adversarial tapes (tied timestamps, negative levels, sequence gaps, mid-session snapshots, unmatched trades) with random orders, cancels, latencies and fee settings, and must agree on every level, order, fill and the cash balance after every step.
- **Mutation checks.** `scripts/mutation_check.py` plants 18 small bugs in the C++ core one at a time (off-by-one comparisons, a missing clamp, a wake-up that is never scheduled, a fee rounded the wrong way), rebuilds, and confirms the tests fail for each. All 18 are caught.
- **Hand-worked scenarios** with exact expected numbers for each rule in the simulator, run on both implementations.
- **Ground truth.** Conservative <= truth <= optimistic, per order, on the synthetic market; fills are identical whether a trade is published before or after its book delta.
- **Kalshi's own examples**: the documented snapshot, delta and trade payloads, and the worked example from the fee rounding page.
- 21 GoogleTest cases for the bitmap, book, fees and simulator, clean under ASan and UBSan and with `-Wall -Wextra -Wpedantic -Wconversion -Wshadow`.

## Design

[`docs/DESIGN.md`](docs/DESIGN.md) explains the decisions: why prices are integers, how the book finds the best price, what exactly happens when a trade and its book update arrive in either order, how each queue model treats a cancel, and what the synthetic market assumes.

## Prior work

- [hftbacktest](https://github.com/nkaz001/hftbacktest) by nkaz001 is the reference point for L2 backtesting with queue position and latency. klob is a much smaller, Kalshi-specific take on the same problem with a different fill model for the two-channel feed; no code is shared.
- Erik Rigtorp, *Estimating order queue position* (2013), for the probabilistic treatment of cancels.
- Avellaneda and Stoikov, *High-frequency trading in a limit order book* (2008), for the shape of the quoting rule.
- Smith, Farmer, Gillemot and Krishnamurthy, *Statistical theory of the continuous double auction* (2003), for the zero-intelligence order flow behind the synthetic market.
- The [Kalshi API documentation](https://docs.kalshi.com) for message formats, fixed-point conventions and fee rounding.

## License

MIT. Not affiliated with Kalshi. This is research software; trading real money with it is at your own risk.
