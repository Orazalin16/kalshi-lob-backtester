#include <gtest/gtest.h>

#include <vector>

#include "klob/backtester.hpp"

using namespace klob;

namespace {

Event ev(TimeNs ts, EventType t, Side s, Price p, Qty q) {
  return Event{ts, 0, q, p, static_cast<std::uint8_t>(t), static_cast<std::uint8_t>(s), 0};
}

// 45c bid for 10 contracts, YES offered at 47c (NO bid at 53c).
std::vector<Event> base() {
  return {
      ev(0, EventType::SnapshotBegin, Side::Yes, 0, 0),
      ev(0, EventType::SnapshotLevel, Side::Yes, 4500, 1000),
      ev(0, EventType::SnapshotLevel, Side::No, 5300, 2000),
      ev(0, EventType::SnapshotEnd, Side::Yes, 0, 0),
  };
}

SimConfig config(QueueModel m) {
  SimConfig c;
  c.queue_model = m;
  return c;
}

}  // namespace

TEST(ExchangeSim, TradeConsumesQueueThenFillsUs) {
  auto tape = base();
  tape.push_back(ev(20, EventType::Trade, Side::No, 4500, 700));
  tape.push_back(ev(21, EventType::Delta, Side::Yes, 4500, -700));
  tape.push_back(ev(30, EventType::Trade, Side::No, 4500, 500));
  Backtester bt(tape, config(QueueModel::Proportional));
  bt.run_until(5);
  bt.submit(Side::Yes, 4500, 300);
  bt.run_until(25);
  ASSERT_EQ(bt.sim().live_orders().size(), 1u);
  EXPECT_EQ(bt.sim().live_orders()[0].ahead, 300);
  bt.run_until(100);
  ASSERT_EQ(bt.sim().fills().size(), 1u);
  EXPECT_EQ(bt.sim().fills()[0].qty, 200);
  EXPECT_EQ(bt.sim().account().position, 200);
  EXPECT_EQ(bt.sim().account().cash, -200 * 4500);
}

class CancelModel : public ::testing::TestWithParam<std::pair<QueueModel, Qty>> {};

TEST_P(CancelModel, UnexplainedDecrease) {
  auto tape = base();
  tape.push_back(ev(10, EventType::Delta, Side::Yes, 4500, 1000));  // behind us
  tape.push_back(ev(20, EventType::Delta, Side::Yes, 4500, -500));  // cancelled from somewhere
  Backtester bt(tape, config(GetParam().first));
  bt.run_until(5);
  bt.submit(Side::Yes, 4500, 100);
  bt.run_until(1'000'000'000);
  EXPECT_EQ(bt.sim().live_orders().at(0).ahead, GetParam().second);
  EXPECT_EQ(bt.sim().stats().cancel_qty, 500);
}

INSTANTIATE_TEST_SUITE_P(Models, CancelModel,
                         ::testing::Values(std::pair{QueueModel::Conservative, Qty{1000}},
                                           std::pair{QueueModel::Proportional, Qty{750}},
                                           std::pair{QueueModel::Optimistic, Qty{500}}));

TEST(ExchangeSim, MessageOrderWithinWindowDoesNotChangeFills) {
  for (bool delta_first : {false, true}) {
    auto tape = base();
    if (delta_first) {
      tape.push_back(ev(20, EventType::Delta, Side::Yes, 4500, -1000));
      tape.push_back(ev(21, EventType::Trade, Side::No, 4500, 1200));
    } else {
      tape.push_back(ev(20, EventType::Trade, Side::No, 4500, 1200));
      tape.push_back(ev(21, EventType::Delta, Side::Yes, 4500, -1000));
    }
    Backtester bt(tape, config(QueueModel::Optimistic));
    bt.run_until(5);
    bt.submit(Side::Yes, 4500, 500);
    bt.run_until(1'000'000'000);
    ASSERT_EQ(bt.sim().fills().size(), 1u) << "delta_first=" << delta_first;
    EXPECT_EQ(bt.sim().fills()[0].qty, 200);
    EXPECT_EQ(bt.sim().stats().cancel_qty, 0);
  }
}

TEST(ExchangeSim, PostOnlyRejectsCrossingOrders) {
  auto tape = base();
  tape.push_back(ev(100, EventType::Delta, Side::Yes, 3000, 1));
  Backtester bt(tape, config(QueueModel::Proportional));
  bt.run_until(5);
  bt.submit(Side::Yes, 4700, 100);  // would lift the 47c offer
  bt.submit(Side::Yes, 4699, 100);
  bt.run_until(50);
  EXPECT_EQ(bt.sim().stats().rejected, 1u);
  ASSERT_EQ(bt.sim().live_orders().size(), 1u);
  EXPECT_EQ(bt.sim().live_orders()[0].price, 4699);
}

TEST(ExchangeSim, RejectsInvalidOrders) {
  auto tape = base();
  Backtester bt(tape, config(QueueModel::Proportional));
  EXPECT_THROW(bt.submit(Side::Yes, 0, 100), std::invalid_argument);
  EXPECT_THROW(bt.submit(Side::Yes, kPriceMax, 100), std::invalid_argument);
  EXPECT_THROW(bt.submit(Side::Yes, 5000, 0), std::invalid_argument);
}

TEST(ExchangeSim, NoBidAccountingNetsAgainstYes) {
  auto tape = base();
  tape.push_back(ev(20, EventType::Trade, Side::No, 4500, 1100));
  tape.push_back(ev(30, EventType::Trade, Side::Yes, 4700, 2100));
  Backtester bt(tape, config(QueueModel::Proportional));
  bt.run_until(5);
  bt.submit(Side::Yes, 4500, 100);
  bt.submit(Side::No, 5300, 100);
  bt.run_until(100);
  EXPECT_EQ(bt.sim().account().position, 0);
  EXPECT_EQ(bt.sim().account().cash, 100 * (kPriceMax - 4500 - 5300));
}
