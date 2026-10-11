#pragma once
// Event-driven replay: advances a Book and an ExchangeSim over a recorded tape.

#include <cstddef>
#include <span>

#include "klob/book.hpp"
#include "klob/exchange_sim.hpp"
#include "klob/types.hpp"

namespace klob {

// Book-only replay. Returns the number of events applied.
inline std::size_t replay(Book& book, std::span<const Event> events) noexcept {
  for (const Event& e : events) book.apply(e);
  return events.size();
}

class Backtester {
 public:
  Backtester(std::span<const Event> tape, const SimConfig& cfg)
      : tape_(tape), sim_(cfg), now_(tape.empty() ? 0 : tape.front().ts) {}

  TimeNs now() const noexcept { return now_; }
  std::size_t cursor() const noexcept { return cursor_; }
  std::size_t size() const noexcept { return tape_.size(); }
  bool done() const noexcept { return cursor_ >= tape_.size(); }
  const Book& book() const noexcept { return book_; }
  ExchangeSim& sim() noexcept { return sim_; }
  const ExchangeSim& sim() const noexcept { return sim_; }
  Price last_trade_yes_price() const noexcept { return last_trade_; }

  std::uint64_t submit(Side side, Price price, Qty qty) { return sim_.submit(now_, side, price, qty); }
  bool cancel(std::uint64_t id) noexcept { return sim_.cancel(now_, id); }

  // Applies every event with ts <= t, then brings orders up to t.
  // Returns false once the tape is exhausted.
  bool run_until(TimeNs t) {
    while (cursor_ < tape_.size() && tape_[cursor_].ts <= t) step(tape_[cursor_++]);
    if (t > now_) now_ = t;
    sim_.on_time(now_, book_);
    return cursor_ < tape_.size();
  }

  bool elapse(TimeNs dt) { return run_until(now_ + dt); }

 private:
  void step(const Event& e) {
    // Settle everything strictly before this event. An order that arrives at
    // exactly e.ts is placed after it, so "arrives at t" always means "sees
    // every event stamped t or earlier", with or without latency.
    sim_.on_time(e.ts - 1, book_);
    if (e.side > 1 || !Book::in_range(e.price)) {
      book_.note_bad_event();
      return;
    }
    const Side side = e.event_side();
    switch (e.event_type()) {
      case EventType::Delta: {
        // Use the change actually applied: a delta that would take a level
        // below zero is clamped, and the simulator should see only what left.
        const Qty before = book_.qty(side, e.price);
        const Qty after = book_.apply_delta(side, e.price, e.qty);
        if (after < before) {
          sim_.on_decrease(e.ts, side, e.price, before - after);
        } else if (after > before) {
          sim_.on_increase(e.ts, side, e.price, after - before);
        }
        break;
      }
      case EventType::Trade:
        book_.note_trade();
        last_trade_ = e.price;
        if (e.qty > 0) sim_.on_trade(e.ts, side, e.price, e.qty);
        break;
      case EventType::SnapshotLevel: book_.set_level(side, e.price, e.qty); break;
      case EventType::SnapshotBegin: book_.begin_snapshot(); break;
      case EventType::SnapshotEnd:
        book_.end_snapshot();
        sim_.on_snapshot_end(book_);
        break;
      case EventType::Gap:
        book_.mark_gap();
        sim_.on_gap();
        break;
      default: book_.note_bad_event(); break;
    }
  }

  std::span<const Event> tape_;
  Book book_;
  ExchangeSim sim_;
  std::size_t cursor_ = 0;
  TimeNs now_;
  Price last_trade_ = -1;
};

}  // namespace klob
