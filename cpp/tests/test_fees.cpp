#include <gtest/gtest.h>

#include "klob/fees.hpp"

using namespace klob;

// Worked example from the Kalshi "Fee Rounding" documentation.
TEST(Fees, DocumentedExample) {
  FeeModel model{70'000, 10'000};
  FeeAccumulator acc;
  const FeeBreakdown f = compute_fee(model, acc, 550, 100, -55'000);
  EXPECT_EQ(f.trade, 3'639);     // ceil($0.00363825) to the micro-dollar
  EXPECT_EQ(f.rounding, 1'361);  // brings the balance change to -$0.06
  EXPECT_EQ(f.rebate, 0);
  EXPECT_EQ(f.net(), 5'000);
  EXPECT_EQ(acc.carried, 1'361);
}

TEST(Fees, AccumulatorRebatesWholeCentsOnly) {
  FeeModel model{0, 10'000};
  FeeAccumulator acc;
  for (Micro expected : {4'000, 8'000, 12'000}) {
    const FeeBreakdown f = compute_fee(model, acc, 60, 100, -6'000);
    EXPECT_EQ(f.rounding, 4'000);
    EXPECT_EQ(f.rebate, 0);  // this fill's own fee cannot cover a cent
    EXPECT_EQ(acc.carried, expected);
  }
  model.maker_ppm = 70'000;
  const FeeBreakdown f = compute_fee(model, acc, 5000, 1000, -5'000'000);
  EXPECT_EQ(f.trade, 175'000);
  EXPECT_EQ(f.rebate, 10'000);
  EXPECT_EQ(acc.carried, 7'000);
}

TEST(Fees, FeeIsSymmetricInPriceAndZeroAtTheBoundaries) {
  FeeModel model{70'000, 100};
  for (Price p : {100, 2500, 5000}) {
    FeeAccumulator a, b;
    EXPECT_EQ(compute_fee(model, a, p, 1000, 0).trade, compute_fee(model, b, kPriceMax - p, 1000, 0).trade);
  }
  FeeAccumulator acc;
  EXPECT_EQ(compute_fee(model, acc, 0, 1000, 0).trade, 0);
}

TEST(Fees, LargeSizesDoNotOverflow) {
  FeeModel model{70'000, 10'000};
  FeeAccumulator acc;
  const Qty hundred_million_contracts = 10'000'000'000;
  const FeeBreakdown f = compute_fee(model, acc, 5000, hundred_million_contracts, 0);
  EXPECT_EQ(f.trade, 1'750'000'000'000);  // 0.07 * 1e8 * 0.25 = $1.75M
}
