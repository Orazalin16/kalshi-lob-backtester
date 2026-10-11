#pragma once
// Level-2 order book for one Kalshi market.

#include <array>
#include <cstddef>
#include <cstdint>

#include "klob/level_bitmap.hpp"
#include "klob/types.hpp"

namespace klob {

class BookSide {
 public:
  Qty at(Price p) const noexcept { return qty_[static_cast<std::size_t>(p)]; }
  Price best() const noexcept { return bits_.highest(); }
  Price next_below(Price p) const noexcept {
    return bits_.next_below(static_cast<std::size_t>(p));
  }
  std::size_t levels() const noexcept { return bits_.count(); }

  void set(Price p, Qty q) noexcept {
    const auto i = static_cast<std::size_t>(p);
    qty_[i] = q;
    if (q > 0) {
      bits_.set(i);
    } else {
      bits_.reset(i);
    }
  }

  // Zero only the occupied levels; a snapshot should not cost 80 KB of writes.
  void clear() noexcept {
    for (Price p = best(); p >= 0; p = next_below(p)) qty_[static_cast<std::size_t>(p)] = 0;
    bits_.clear();
  }

 private:
  std::array<Qty, kLevels> qty_{};
  LevelBitmap<kLevels> bits_;
};

struct BookStats {
  std::uint64_t deltas = 0;
  std::uint64_t trades = 0;
  std::uint64_t snapshots = 0;
  std::uint64_t gaps = 0;
  std::uint64_t negative_clamps = 0;  // a delta would have driven a level below zero
  std::uint64_t bad_events = 0;       // price outside [0, 1] or unknown type/side
};

class Book {
 public:
  static constexpr bool in_range(Price p) noexcept { return p >= 0 && p <= kPriceMax; }

  Qty qty(Side s, Price p) const noexcept { return side(s).at(p); }
  Price best(Side s) const noexcept { return side(s).best(); }
  Price next_below(Side s, Price p) const noexcept { return side(s).next_below(p); }
  std::size_t levels(Side s) const noexcept { return side(s).levels(); }

  Price yes_bid() const noexcept { return best(Side::Yes); }
  // Best offer for YES implied by the best NO bid, or -1 if there is none.
  Price yes_ask() const noexcept {
    const Price nb = best(Side::No);
    return nb < 0 ? -1 : kPriceMax - nb;
  }
  bool crossed() const noexcept {
    const Price yb = best(Side::Yes);
    const Price nb = best(Side::No);
    return yb >= 0 && nb >= 0 && yb + nb >= kPriceMax;
  }

  // False between a sequence gap (or the start of the tape) and the next
  // completed snapshot. Nothing should be quoted against an invalid book.
  bool valid() const noexcept { return valid_; }
  const BookStats& stats() const noexcept { return stats_; }

  // Applies a signed size change and returns the new size at that level.
  Qty apply_delta(Side s, Price p, Qty dq) noexcept {
    BookSide& bs = side(s);
    Qty q = bs.at(p) + dq;
    if (q < 0) {
      ++stats_.negative_clamps;
      q = 0;
    }
    bs.set(p, q);
    ++stats_.deltas;
    return q;
  }

  void begin_snapshot() noexcept {
    sides_[0].clear();
    sides_[1].clear();
    valid_ = false;
  }
  void set_level(Side s, Price p, Qty q) noexcept { side(s).set(p, q < 0 ? 0 : q); }
  void end_snapshot() noexcept {
    valid_ = true;
    ++stats_.snapshots;
  }
  void mark_gap() noexcept {
    valid_ = false;
    ++stats_.gaps;
  }
  void note_trade() noexcept { ++stats_.trades; }
  void note_bad_event() noexcept { ++stats_.bad_events; }

  // Book-only replay of one event (no order simulation).
  void apply(const Event& e) noexcept {
    if (e.side > 1 || !in_range(e.price)) {
      ++stats_.bad_events;
      return;
    }
    switch (e.event_type()) {
      case EventType::Delta: apply_delta(e.event_side(), e.price, e.qty); break;
      case EventType::Trade: ++stats_.trades; break;
      case EventType::SnapshotLevel: set_level(e.event_side(), e.price, e.qty); break;
      case EventType::SnapshotBegin: begin_snapshot(); break;
      case EventType::SnapshotEnd: end_snapshot(); break;
      case EventType::Gap: mark_gap(); break;
      default: ++stats_.bad_events; break;
    }
  }

  // Writes up to k best levels (best first) and returns how many were written.
  std::size_t depth(Side s, std::size_t k, Price* prices, Qty* sizes) const noexcept {
    const BookSide& bs = side(s);
    std::size_t n = 0;
    for (Price p = bs.best(); p >= 0 && n < k; p = bs.next_below(p)) {
      prices[n] = p;
      sizes[n] = bs.at(p);
      ++n;
    }
    return n;
  }

 private:
  BookSide& side(Side s) noexcept { return sides_[static_cast<std::size_t>(s)]; }
  const BookSide& side(Side s) const noexcept { return sides_[static_cast<std::size_t>(s)]; }

  BookSide sides_[2];
  BookStats stats_;
  bool valid_ = false;
};

}  // namespace klob
