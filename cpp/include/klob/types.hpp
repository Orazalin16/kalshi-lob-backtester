#pragma once
// Fixed-point units shared by every component.
//
// Kalshi quotes prices as dollar strings with up to four decimals and sizes as
// contract strings with up to two decimals. Both are kept as integers here so
// that book state, fills and cash are exact and reproducible across the C++
// core and the Python reference implementation.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace klob {

using Price = std::int32_t;   // 1e-4 USD  ("0.4200" -> 4200)
using Qty = std::int64_t;     // 1e-2 contracts ("10.00" -> 1000)
using Micro = std::int64_t;   // 1e-6 USD; Price * Qty lands exactly on this grid
using TimeNs = std::int64_t;  // nanoseconds since the Unix epoch
// Intermediate products (fee multiplier * size * P * (1 - P), size * size) can
// exceed 64 bits. GCC and Clang provide a 128-bit integer; MSVC would need a
// different widening multiply here.
__extension__ using Int128 = __int128;

inline constexpr Price kPriceMax = 10'000;      // $1.0000
inline constexpr std::size_t kLevels = 10'001;  // price indices 0..10000
inline constexpr TimeNs kNever = std::numeric_limits<TimeNs>::max();

// A Kalshi book has no asks. It holds bids to buy YES and bids to buy NO; a NO
// bid at q is the same thing as an offer to sell YES at 1 - q.
enum class Side : std::uint8_t { Yes = 0, No = 1 };

constexpr Side opposite(Side s) noexcept {
  return s == Side::Yes ? Side::No : Side::Yes;
}

enum class EventType : std::uint8_t {
  SnapshotBegin = 0,  // book is about to be restated in full
  SnapshotLevel = 1,  // absolute size at one price level
  SnapshotEnd = 2,    // book is consistent again
  Delta = 3,          // signed size change at one price level
  Trade = 4,          // side = taker side, price = YES price of the trade
  Gap = 5,            // sequence gap: state is unknown until the next snapshot
};

// One record of the normalized tape. The layout is the on-disk format and the
// NumPy dtype, so the fields are plain integers and the size is pinned.
struct Event {
  TimeNs ts;       // local receive time; the clock the simulation runs on
  TimeNs exch_ts;  // exchange timestamp when the feed provides one, else 0
  Qty qty;
  Price price;
  std::uint8_t type;
  std::uint8_t side;
  std::uint16_t flags;

  EventType event_type() const noexcept { return static_cast<EventType>(type); }
  Side event_side() const noexcept { return static_cast<Side>(side); }
};

static_assert(sizeof(Event) == 32, "Event is the on-disk record; keep it at 32 bytes");
static_assert(std::is_trivially_copyable_v<Event>);

}  // namespace klob
