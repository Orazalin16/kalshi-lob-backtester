// Replay throughput on a recorded tape (raw array of 32-byte Event records).
//
//   klob_bench tape.bin [repeats]
//
// Reports events per second and per-event latency percentiles for book-only
// replay, the same replay on a std::map book for comparison, and events per
// second with the fill simulator attached and a pair of orders resting at the
// touch.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "klob/backtester.hpp"

using namespace klob;
using Clock = std::chrono::steady_clock;

namespace {

std::vector<Event> load(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path);
    std::exit(1);
  }
  const auto bytes = static_cast<std::size_t>(f.tellg());
  std::vector<Event> out(bytes / sizeof(Event));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(Event)));
  return out;
}

double seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// The textbook book: one ordered map per side, best price at begin().
struct MapBook {
  std::map<Price, Qty, std::greater<>> sides[2];

  void apply(const Event& e) {
    if (e.side > 1 || !Book::in_range(e.price)) return;
    auto& side = sides[e.side];
    switch (e.event_type()) {
      case EventType::Delta: {
        const auto it = side.try_emplace(e.price, 0).first;
        it->second += e.qty;
        if (it->second <= 0) side.erase(it);
        break;
      }
      case EventType::SnapshotLevel:
        if (e.qty > 0) side[e.price] = e.qty;
        break;
      case EventType::SnapshotBegin:
        sides[0].clear();
        sides[1].clear();
        break;
      default: break;
    }
  }
  Price best(int s) const { return sides[s].empty() ? -1 : sides[s].begin()->first; }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s tape.bin [repeats]\n", argv[0]);
    return 1;
  }
  const std::vector<Event> tape = load(argv[1]);
  const int repeats = argc > 2 ? std::atoi(argv[2]) : 20;
  std::printf("events: %zu (%.1f MB)\n", tape.size(), static_cast<double>(tape.size() * sizeof(Event)) / 1e6);

  // Book-only replay, best of `repeats`.
  double best = 1e300;
  std::int64_t checksum = 0;
  for (int r = 0; r < repeats; ++r) {
    auto book = std::make_unique<Book>();
    const auto t0 = Clock::now();
    replay(*book, tape);
    const auto t1 = Clock::now();
    best = std::min(best, seconds(t0, t1));
    checksum = book->yes_bid() + book->yes_ask();
  }
  std::printf("book replay: %.1f M events/s (%.1f ns/event), checksum %lld\n",
              static_cast<double>(tape.size()) / best / 1e6, best / static_cast<double>(tape.size()) * 1e9,
              static_cast<long long>(checksum));

  {
    double best_map = 1e300;
    std::int64_t map_checksum = 0;
    for (int r = 0; r < std::max(1, repeats / 2); ++r) {
      MapBook book;
      const auto t0 = Clock::now();
      for (const Event& e : tape) book.apply(e);
      best_map = std::min(best_map, seconds(t0, Clock::now()));
      map_checksum = book.best(0) + (kPriceMax - book.best(1));
    }
    std::printf("std::map book: %.1f M events/s (%.1f ns/event), checksum %lld -> dense book is %.1fx faster\n",
                static_cast<double>(tape.size()) / best_map / 1e6, best_map / static_cast<double>(tape.size()) * 1e9,
                static_cast<long long>(map_checksum), best_map / best);
  }

  // Per-event latency distribution. Timing every event adds the cost of two
  // clock reads, so these numbers are an upper bound on the true cost.
  {
    auto book = std::make_unique<Book>();
    std::vector<std::uint32_t> ns(tape.size());
    for (std::size_t i = 0; i < tape.size(); ++i) {
      const auto t0 = Clock::now();
      book->apply(tape[i]);
      ns[i] = static_cast<std::uint32_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
    }
    std::sort(ns.begin(), ns.end());
    const auto pct = [&](double p) { return ns[static_cast<std::size_t>(p * static_cast<double>(ns.size() - 1))]; };
    std::printf("per-event (incl. timer): p50 %u ns, p99 %u ns, p99.9 %u ns\n", pct(0.50), pct(0.99), pct(0.999));
  }

  // With the fill simulator: keep one order resting at each best price and
  // replace it once a second of tape time.
  {
    best = 1e300;
    std::size_t fills = 0;
    for (int r = 0; r < std::max(1, repeats / 4); ++r) {
      SimConfig cfg;
      cfg.order_latency = cfg.cancel_latency = 20'000'000;
      auto bt = std::make_unique<Backtester>(tape, cfg);
      const auto t0 = Clock::now();
      while (bt->elapse(1'000'000'000)) {
        for (const Order& o : bt->sim().live_orders()) bt->cancel(o.id);
        const Price yb = bt->book().yes_bid();
        const Price nb = bt->book().best(Side::No);
        if (bt->book().valid() && yb > 0 && nb > 0) {
          bt->submit(Side::Yes, yb, 1000);
          bt->submit(Side::No, nb, 1000);
        }
      }
      best = std::min(best, seconds(t0, Clock::now()));
      fills = bt->sim().fills().size();
    }
    std::printf("replay + fill simulation: %.1f M events/s (%zu fills)\n",
                static_cast<double>(tape.size()) / best / 1e6, fills);
  }
  return 0;
}
