#pragma once
// Kalshi fee arithmetic in integer micro-dollars.
//
// The fee model is quadratic in price: multiplier * contracts * P * (1 - P).
// Kalshi then rounds in two stages (docs: "Fee Rounding"):
//   1. the model fee is rounded up to $0.000001;
//   2. a rounding fee is added so the balance change lands on the account's
//      precision grid ($0.01 for most accounts), and the overpayment is carried
//      per order and rebated once it accumulates to a whole grid step.

#include <algorithm>
#include <cstdint>

#include "klob/types.hpp"

namespace klob {

struct FeeModel {
  std::int64_t maker_ppm = 0;         // fee multiplier in parts per million (0.0175 -> 17500)
  Micro balance_precision = 10'000;   // $0.01; direct members use 100 ($0.0001)
};

struct FeeAccumulator {
  Micro carried = 0;  // rounding overpayment not yet rebated on this order
};

struct FeeBreakdown {
  Micro trade = 0;
  Micro rounding = 0;
  Micro rebate = 0;
  Micro net() const noexcept { return trade + rounding - rebate; }
};

namespace detail {
constexpr Micro floor_to(Micro x, Micro step) noexcept {
  Micro q = x / step;
  if ((x % step != 0) && (x < 0)) --q;
  return q * step;
}
}  // namespace detail

// revenue is the signed cash flow of the fill before fees (negative for a buy).
inline FeeBreakdown compute_fee(const FeeModel& model, FeeAccumulator& acc, Price price, Qty qty,
                                Micro revenue) noexcept {
  FeeBreakdown out;
  // ppm (1e-6) * qty (1e-2) * P (1e-4) * (1-P) (1e-4) = 1e-16 USD; micro = 1e-6.
  const Int128 scaled = Int128{model.maker_ppm} * qty * price * (kPriceMax - price);
  constexpr Int128 kDenom = 10'000'000'000;
  out.trade = static_cast<Micro>((scaled + kDenom - 1) / kDenom);

  const Micro step = model.balance_precision;
  const Micro raw_change = revenue - out.trade;
  out.rounding = raw_change - detail::floor_to(raw_change, step);
  acc.carried += out.rounding;

  const Micro gross = out.trade + out.rounding;
  out.rebate = std::min(acc.carried / step, gross / step) * step;
  acc.carried -= out.rebate;
  return out;
}

}  // namespace klob
