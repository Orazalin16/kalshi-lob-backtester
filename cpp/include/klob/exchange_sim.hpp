#pragma once
// Fill simulation for our own passive orders against a recorded L2 feed.
//
// An L2 feed tells us the total size at each price, not where our order would
// have stood in the queue. The simulator tracks, for every resting order, an
// estimate of the size still ahead of it and updates that estimate from three
// kinds of evidence:
//
//   * a trade at our level consumes the queue from the front, so it always
//     reduces the size ahead and fills us with whatever is left over;
//   * a size decrease that no trade explains is a cancel, and whether it came
//     from in front of us or behind us is unknowable from L2 data. The
//     QueueModel chooses the assumption;
//   * a trade at a worse price, or a new opposite order that would have crossed
//     us, means we would have been filled first.
//
// Kalshi publishes trades and book deltas on separate channels with no ordering
// guarantee between them, so a decrease is held for a short attribution window
// before being classified as a cancel. A trade that arrives inside the window
// claims it instead. The result is the same whichever message arrives first.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "klob/book.hpp"
#include "klob/fees.hpp"
#include "klob/types.hpp"

namespace klob {

enum class QueueModel : std::uint8_t {
  Conservative = 0,  // cancels come from behind us whenever that is possible
  Proportional = 1,  // cancels are spread evenly over the queue
  Optimistic = 2,    // cancels come from in front of us
};

enum class OrderState : std::uint8_t {
  PendingNew = 0,  // sent, not yet at the exchange
  Active = 1,
  Filled = 2,
  Canceled = 3,
  Rejected = 4,  // post-only order that would have crossed, or book was invalid
};

struct Order {
  std::uint64_t id = 0;
  TimeNs submit_ts = 0;
  TimeNs active_ts = 0;       // arrival time at the exchange
  TimeNs cancel_ts = kNever;  // arrival time of a requested cancel
  Qty qty = 0;
  Qty remaining = 0;
  Qty ahead = 0;  // estimated size in front of us at our price
  Price price = 0;
  Side side = Side::Yes;
  OrderState state = OrderState::PendingNew;
  FeeAccumulator fee_acc;
};

struct Fill {
  TimeNs ts;
  std::uint64_t order_id;
  Qty qty;
  Micro fee;
  Price price;  // on the order's own side: the YES price for a YES bid, the NO price for a NO bid
  std::uint8_t side;
  std::uint8_t pad_[3];
};
static_assert(sizeof(Fill) == 40);

struct SimConfig {
  QueueModel queue_model = QueueModel::Proportional;
  TimeNs order_latency = 0;
  TimeNs cancel_latency = 0;
  TimeNs attribution_window = 50'000'000;  // 50 ms
  FeeModel fees;
};

struct Account {
  Qty position = 0;  // net YES-equivalent contracts (a NO contract counts as -1 YES + $1)
  Micro cash = 0;
  Micro fees = 0;
  Qty volume = 0;
  std::uint64_t fills = 0;
};

struct SimStats {
  std::uint64_t submitted = 0;
  std::uint64_t rejected = 0;
  std::uint64_t canceled = 0;
  std::uint64_t filled = 0;
  Qty cancel_qty = 0;           // decreases classified as cancels
  Qty matched_trade_qty = 0;    // trade size paired with a book decrease
  Qty unmatched_trade_qty = 0;  // trade size never seen in the book (data-quality signal)
};

class ExchangeSim {
 public:
  explicit ExchangeSim(const SimConfig& cfg) : cfg_(cfg) {}

  const SimConfig& config() const noexcept { return cfg_; }
  const Account& account() const noexcept { return account_; }
  const SimStats& stats() const noexcept { return stats_; }
  const std::vector<Order>& live_orders() const noexcept { return live_; }
  const std::vector<Order>& closed_orders() const noexcept { return closed_; }
  const std::vector<Fill>& fills() const noexcept { return fills_; }
  void clear_closed() noexcept { closed_.clear(); }

  // Post-only limit order. The order reaches the book at now + order_latency.
  std::uint64_t submit(TimeNs now, Side side, Price price, Qty qty) {
    if (price <= 0 || price >= kPriceMax) throw std::invalid_argument("price must be inside (0, 1)");
    if (qty <= 0) throw std::invalid_argument("qty must be positive");
    Order o;
    o.id = ++next_id_;
    o.submit_ts = now;
    o.active_ts = now + cfg_.order_latency;
    o.qty = o.remaining = qty;
    o.price = price;
    o.side = side;
    live_.push_back(o);
    next_due_ = std::min(next_due_, o.active_ts);
    ++stats_.submitted;
    return o.id;
  }

  // Requests a cancel. Returns false if the order is unknown or already closed.
  // The order can still be filled until the cancel arrives.
  bool cancel(TimeNs now, std::uint64_t id) noexcept {
    for (Order& o : live_) {
      if (o.id != id) continue;
      if (o.cancel_ts == kNever) {
        o.cancel_ts = now + cfg_.cancel_latency;
        next_due_ = std::min(next_due_, std::max(o.cancel_ts, o.active_ts));
      }
      return true;
    }
    return false;
  }

  // Brings order state up to and including time t: orders whose arrival time
  // has passed join the book as it stands now, due cancels take effect, and
  // decreases that have waited out the attribution window become cancels.
  void on_time(TimeNs t, const Book& book) {
    if (t < next_due_) return;
    expire_attribution(t, book);
    for (std::size_t i = 0; i < live_.size();) {
      Order& o = live_[i];
      if (o.state == OrderState::PendingNew && o.active_ts <= t) activate(o, book);
      if (o.state == OrderState::Active && o.cancel_ts <= t) o.state = OrderState::Canceled;
      if (o.state == OrderState::Canceled || o.state == OrderState::Rejected) {
        close(i);
      } else {
        ++i;
      }
    }
    recompute_next_due();
  }

  // Size at (side, price) dropped by `removed`. Either a trade already seen
  // explains it, or it waits in the attribution window.
  void on_decrease(TimeNs t, Side side, Price price, Qty removed) {
    Qty left = removed;
    for (Pending& c : credits_) {
      if (left == 0) break;
      if (c.side != side || c.price != price) continue;
      const Qty m = std::min(c.qty, left);
      c.qty -= m;
      left -= m;
      stats_.matched_trade_qty += m;
    }
    drop_empty(credits_);
    if (left > 0) {
      debits_.push_back({t, left, price, side});
      next_due_ = std::min(next_due_, t + cfg_.attribution_window);
    }
  }

  // Size was added on `side`. An order that rests at a price crossing one of
  // ours would have traded with us instead of resting.
  void on_increase(TimeNs t, Side side, Price price, Qty added) {
    const Side ours = opposite(side);
    for (std::size_t i = 0; i < live_.size();) {
      Order& o = live_[i];
      if (o.state == OrderState::Active && o.side == ours && o.price + price >= kPriceMax) {
        o.ahead = 0;
        if (fill(i, t, std::min(o.remaining, added))) continue;
      }
      ++i;
    }
  }

  // Public trade. `yes_price` is the YES price of the trade; the resting side
  // is the one opposite the taker.
  void on_trade(TimeNs t, Side taker, Price yes_price, Qty qty) {
    const Side side = opposite(taker);
    const Price price = taker == Side::Yes ? kPriceMax - yes_price : yes_price;

    // Pair the trade with decreases that arrived before it. Each paired piece
    // remembers when its decrease was observed: an order that joined the level
    // after that moment already saw the smaller size and must not be advanced
    // by it a second time.
    paired_.clear();
    Qty left = qty;
    for (Pending& d : debits_) {
      if (left == 0) break;
      if (d.side != side || d.price != price) continue;
      const Qty m = std::min(d.qty, left);
      d.qty -= m;
      left -= m;
      paired_.push_back({d.ts, m, price, side});
      stats_.matched_trade_qty += m;
    }
    drop_empty(debits_);
    if (left > 0) {
      credits_.push_back({t, left, price, side});
      next_due_ = std::min(next_due_, t + cfg_.attribution_window);
    }

    for (std::size_t i = 0; i < live_.size();) {
      Order& o = live_[i];
      if (o.state != OrderState::Active || o.side != side || o.price < price) {
        ++i;
        continue;
      }
      Qty seen = qty;
      for (const Pending& p : paired_) {
        if (p.ts <= o.active_ts) seen -= p.qty;
      }
      Qty take = 0;
      if (o.price > price) {
        // The taker reached a worse price than ours, so nothing real was left
        // in front of us.
        o.ahead = 0;
        take = std::min(o.remaining, seen);
      } else {
        const Qty consumed = std::min(o.ahead, seen);
        o.ahead -= consumed;
        take = std::min(o.remaining, seen - consumed);
      }
      if (take > 0 && fill(i, t, take)) continue;
      ++i;
    }
  }

  // A full snapshot replaces whatever we believed about queue positions. After
  // a sequence gap we cannot know what happened in between, so resting orders
  // are sent to the back of their level.
  void on_snapshot_end(const Book& book) noexcept {
    debits_.clear();
    credits_.clear();
    for (Order& o : live_) {
      if (o.state != OrderState::Active) continue;
      const Qty level = book.qty(o.side, o.price);
      o.ahead = after_gap_ ? level : std::min(o.ahead, level);
    }
    after_gap_ = false;
    recompute_next_due();
  }

  void on_gap() noexcept { after_gap_ = true; }

 private:
  struct Pending {
    TimeNs ts;
    Qty qty;
    Price price;
    Side side;
  };

  static void drop_empty(std::vector<Pending>& v) {
    v.erase(std::remove_if(v.begin(), v.end(), [](const Pending& p) { return p.qty == 0; }), v.end());
  }

  void activate(Order& o, const Book& book) noexcept {
    const Price opp = book.best(opposite(o.side));
    if (!book.valid() || (opp >= 0 && o.price + opp >= kPriceMax)) {
      o.state = OrderState::Rejected;
      ++stats_.rejected;
      return;
    }
    o.ahead = book.qty(o.side, o.price);
    o.state = OrderState::Active;
  }

  void close(std::size_t i) {
    if (live_[i].state == OrderState::Canceled) ++stats_.canceled;
    if (live_[i].state == OrderState::Filled) ++stats_.filled;
    closed_.push_back(live_[i]);
    live_.erase(live_.begin() + static_cast<std::ptrdiff_t>(i));
  }

  // Books a fill on live_[i]. Returns true if the order was completed and
  // removed, in which case index i now refers to the next order.
  bool fill(std::size_t i, TimeNs t, Qty q) {
    if (q <= 0) return false;
    Order& o = live_[i];
    o.remaining -= q;
    const Micro notional = static_cast<Micro>(o.price) * q;
    const FeeBreakdown fee = compute_fee(cfg_.fees, o.fee_acc, o.price, q, -notional);
    if (o.side == Side::Yes) {
      account_.position += q;
      account_.cash -= notional;
    } else {
      // Buying NO at q is selling YES at 1 - q once the $1 pair is netted.
      account_.position -= q;
      account_.cash += static_cast<Micro>(kPriceMax - o.price) * q;
    }
    account_.cash -= fee.net();
    account_.fees += fee.net();
    account_.volume += q;
    ++account_.fills;
    fills_.push_back({t, o.id, q, fee.net(), o.price, static_cast<std::uint8_t>(o.side), {0, 0, 0}});
    if (o.remaining == 0) {
      o.state = OrderState::Filled;
      close(i);
      return true;
    }
    return false;
  }

  // Applies a decrease that no trade claimed within the attribution window.
  void apply_cancel(const Pending& d, const Book& book) noexcept {
    stats_.cancel_qty += d.qty;
    const Qty level = book.qty(d.side, d.price);
    for (Order& o : live_) {
      if (o.state != OrderState::Active || o.side != d.side || o.price != d.price) continue;
      if (o.active_ts >= d.ts) continue;  // joined after the decrease was already visible
      switch (cfg_.queue_model) {
        case QueueModel::Conservative: break;
        case QueueModel::Proportional: {
          const Qty before = level + d.qty;
          if (o.ahead > 0 && before > 0) {
            o.ahead -= static_cast<Qty>(Int128{d.qty} * o.ahead / before);
          }
          break;
        }
        case QueueModel::Optimistic: o.ahead = std::max<Qty>(0, o.ahead - d.qty); break;
      }
      // Whatever the model, no more can be ahead of us than is displayed.
      o.ahead = std::min(o.ahead, level);
    }
  }

  void expire_attribution(TimeNs t, const Book& book) {
    const TimeNs w = cfg_.attribution_window;
    std::size_t n = 0;
    while (n < debits_.size() && debits_[n].ts + w <= t) apply_cancel(debits_[n++], book);
    debits_.erase(debits_.begin(), debits_.begin() + static_cast<std::ptrdiff_t>(n));
    n = 0;
    while (n < credits_.size() && credits_[n].ts + w <= t) stats_.unmatched_trade_qty += credits_[n++].qty;
    credits_.erase(credits_.begin(), credits_.begin() + static_cast<std::ptrdiff_t>(n));
  }

  void recompute_next_due() noexcept {
    TimeNs due = kNever;
    for (const Order& o : live_) {
      if (o.state == OrderState::PendingNew) {
        due = std::min(due, o.active_ts);
      } else if (o.cancel_ts != kNever) {
        due = std::min(due, o.cancel_ts);
      }
    }
    const TimeNs w = cfg_.attribution_window;
    if (!debits_.empty()) due = std::min(due, debits_.front().ts + w);
    if (!credits_.empty()) due = std::min(due, credits_.front().ts + w);
    next_due_ = due;
  }

  SimConfig cfg_;
  Account account_;
  SimStats stats_;
  std::vector<Order> live_;    // a market maker rests a handful of orders; a flat scan beats a map
  std::vector<Order> closed_;  // finished orders, drained by the caller
  std::vector<Fill> fills_;
  std::vector<Pending> debits_;   // decreases waiting for a trade to claim them
  std::vector<Pending> credits_;  // trades waiting for their decrease to arrive
  std::vector<Pending> paired_;   // scratch
  std::uint64_t next_id_ = 0;
  TimeNs next_due_ = kNever;
  bool after_gap_ = false;
};

}  // namespace klob
