#include <gtest/gtest.h>

#include <map>
#include <random>
#include <vector>

#include "klob/backtester.hpp"
#include "klob/book.hpp"

using namespace klob;

namespace {
Event ev(EventType t, Side s, Price p, Qty q, TimeNs ts = 0) {
  return Event{ts, 0, q, p, static_cast<std::uint8_t>(t), static_cast<std::uint8_t>(s), 0};
}
}  // namespace

TEST(Book, SnapshotThenDeltas) {
  Book b;
  EXPECT_FALSE(b.valid());
  b.apply(ev(EventType::SnapshotBegin, Side::Yes, 0, 0));
  b.apply(ev(EventType::SnapshotLevel, Side::Yes, 4500, 1000));
  b.apply(ev(EventType::SnapshotLevel, Side::No, 5300, 2000));
  b.apply(ev(EventType::SnapshotEnd, Side::Yes, 0, 0));
  EXPECT_TRUE(b.valid());
  EXPECT_EQ(b.yes_bid(), 4500);
  EXPECT_EQ(b.yes_ask(), 4700);  // a NO bid at 53c offers YES at 47c
  EXPECT_FALSE(b.crossed());

  b.apply(ev(EventType::Delta, Side::Yes, 4600, 300));
  EXPECT_EQ(b.yes_bid(), 4600);
  b.apply(ev(EventType::Delta, Side::Yes, 4600, -300));
  EXPECT_EQ(b.yes_bid(), 4500);
  EXPECT_EQ(b.levels(Side::Yes), 1u);
}

TEST(Book, NegativeLevelIsClampedAndCounted) {
  Book b;
  b.apply(ev(EventType::Delta, Side::No, 5000, 100));
  b.apply(ev(EventType::Delta, Side::No, 5000, -250));
  EXPECT_EQ(b.qty(Side::No, 5000), 0);
  EXPECT_EQ(b.best(Side::No), -1);
  EXPECT_EQ(b.stats().negative_clamps, 1u);
}

TEST(Book, RejectsOutOfRangeEvents) {
  Book b;
  b.apply(ev(EventType::Delta, Side::Yes, -1, 100));
  b.apply(ev(EventType::Delta, Side::Yes, kPriceMax + 1, 100));
  Event bad_side = ev(EventType::Delta, Side::Yes, 5000, 100);
  bad_side.side = 2;
  b.apply(bad_side);
  EXPECT_EQ(b.stats().bad_events, 3u);
  EXPECT_EQ(b.best(Side::Yes), -1);
}

TEST(Book, GapInvalidatesUntilNextSnapshot) {
  Book b;
  b.apply(ev(EventType::SnapshotBegin, Side::Yes, 0, 0));
  b.apply(ev(EventType::SnapshotEnd, Side::Yes, 0, 0));
  b.apply(ev(EventType::Gap, Side::Yes, 0, 0));
  EXPECT_FALSE(b.valid());
  b.apply(ev(EventType::SnapshotBegin, Side::Yes, 0, 0));
  b.apply(ev(EventType::SnapshotEnd, Side::Yes, 0, 0));
  EXPECT_TRUE(b.valid());
}

// The dense array + bitmap must behave like the obvious ordered map.
TEST(Book, MatchesMapOnRandomDeltas) {
  std::mt19937 rng(3);
  Book b;
  std::map<Price, Qty> ref[2];
  for (int i = 0; i < 300'000; ++i) {
    const auto s = static_cast<Side>(rng() % 2);
    const Price p = static_cast<Price>(rng() % kLevels);
    const Qty dq = static_cast<Qty>(rng() % 2001) - 900;
    b.apply(ev(EventType::Delta, s, p, dq));
    auto& m = ref[static_cast<int>(s)];
    Qty q = m[p] + dq;
    if (q <= 0) {
      m.erase(p);
    } else {
      m[p] = q;
    }
    ASSERT_EQ(b.best(s), m.empty() ? -1 : m.rbegin()->first);
  }
  for (int s = 0; s < 2; ++s) {
    std::vector<Price> px(kLevels);
    std::vector<Qty> sz(kLevels);
    const std::size_t n = b.depth(static_cast<Side>(s), kLevels, px.data(), sz.data());
    ASSERT_EQ(n, ref[s].size());
    std::size_t i = 0;
    for (auto it = ref[s].rbegin(); it != ref[s].rend(); ++it, ++i) {
      EXPECT_EQ(px[i], it->first);
      EXPECT_EQ(sz[i], it->second);
    }
  }
}

TEST(Book, ClearAfterSnapshotLeavesNoStaleLevels) {
  Book b;
  for (Price p = 100; p < 9900; p += 37) b.apply(ev(EventType::Delta, Side::Yes, p, 500));
  b.apply(ev(EventType::SnapshotBegin, Side::Yes, 0, 0));
  EXPECT_EQ(b.best(Side::Yes), -1);
  for (Price p = 0; p <= kPriceMax; ++p) ASSERT_EQ(b.qty(Side::Yes, p), 0);
}
