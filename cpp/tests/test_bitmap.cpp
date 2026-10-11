#include <gtest/gtest.h>

#include <random>
#include <set>

#include "klob/level_bitmap.hpp"
#include "klob/types.hpp"

using klob::LevelBitmap;

TEST(LevelBitmap, EmptyHasNoHighest) {
  LevelBitmap<klob::kLevels> b;
  EXPECT_EQ(b.highest(), -1);
  EXPECT_EQ(b.next_below(10'000), -1);
  EXPECT_EQ(b.count(), 0u);
}

TEST(LevelBitmap, WordAndSummaryBoundaries) {
  LevelBitmap<klob::kLevels> b;
  for (std::size_t i : {0u, 63u, 64u, 4095u, 4096u, 10'000u}) {
    b.set(i);
    EXPECT_TRUE(b.test(i));
    EXPECT_EQ(b.highest(), static_cast<int>(i));
  }
  EXPECT_EQ(b.next_below(10'000), 4096);
  EXPECT_EQ(b.next_below(4096), 4095);
  EXPECT_EQ(b.next_below(4095), 64);
  EXPECT_EQ(b.next_below(64), 63);
  EXPECT_EQ(b.next_below(63), 0);
  EXPECT_EQ(b.next_below(0), -1);
  b.reset(10'000);
  b.reset(4096);
  EXPECT_EQ(b.highest(), 4095);
}

// Against std::set on random operations: same maximum, same predecessor.
TEST(LevelBitmap, MatchesOrderedSet) {
  std::mt19937 rng(7);
  std::uniform_int_distribution<std::size_t> pick(0, klob::kLevels - 1);
  LevelBitmap<klob::kLevels> b;
  std::set<std::size_t> ref;
  for (int i = 0; i < 200'000; ++i) {
    const std::size_t p = pick(rng);
    if (rng() % 3 == 0) {
      b.reset(p);
      ref.erase(p);
    } else {
      b.set(p);
      ref.insert(p);
    }
    ASSERT_EQ(b.highest(), ref.empty() ? -1 : static_cast<int>(*ref.rbegin()));
    const std::size_t q = pick(rng);
    const auto it = ref.lower_bound(q);
    ASSERT_EQ(b.next_below(q), it == ref.begin() ? -1 : static_cast<int>(*std::prev(it)));
  }
  EXPECT_EQ(b.count(), ref.size());
}
