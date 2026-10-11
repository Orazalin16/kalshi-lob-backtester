#pragma once
// Two-level bitmap over price levels.
//
// The book stores sizes in a dense array indexed by price, which makes updates
// a single store but leaves "where is the best bid?" as a scan. This bitmap
// answers it with two count-leading-zeros instructions: a summary word says
// which 64-level block is non-empty, and that block says which level.

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

namespace klob {

template <std::size_t N>
class LevelBitmap {
  static constexpr std::size_t kWords = (N + 63) / 64;
  static constexpr std::size_t kSummary = (kWords + 63) / 64;

 public:
  void set(std::size_t i) noexcept {
    const std::size_t w = i >> 6;
    words_[w] |= bit(i);
    summary_[w >> 6] |= bit(w);
  }

  void reset(std::size_t i) noexcept {
    const std::size_t w = i >> 6;
    words_[w] &= ~bit(i);
    if (words_[w] == 0) summary_[w >> 6] &= ~bit(w);
  }

  bool test(std::size_t i) const noexcept { return (words_[i >> 6] & bit(i)) != 0; }

  void clear() noexcept {
    words_.fill(0);
    summary_.fill(0);
  }

  // Highest set index, or -1 when empty.
  int highest() const noexcept {
    for (std::size_t s = kSummary; s-- > 0;) {
      if (summary_[s] != 0) return top_of_word((s << 6) + msb(summary_[s]));
    }
    return -1;
  }

  // Highest set index strictly below i, or -1. Used to walk depth downward.
  int next_below(std::size_t i) const noexcept {
    if (i == 0) return -1;
    const std::size_t j = i - 1;
    const std::size_t w = j >> 6;
    if (const std::uint64_t m = words_[w] & low_mask(j & 63); m != 0) {
      return static_cast<int>((w << 6) + msb(m));
    }
    if (w == 0) return -1;
    const std::size_t k = w - 1;
    std::size_t s = k >> 6;
    std::uint64_t sm = summary_[s] & low_mask(k & 63);
    for (;;) {
      if (sm != 0) return top_of_word((s << 6) + msb(sm));
      if (s == 0) return -1;
      sm = summary_[--s];
    }
  }

  std::size_t count() const noexcept {
    std::size_t n = 0;
    for (const std::uint64_t w : words_) n += static_cast<std::size_t>(std::popcount(w));
    return n;
  }

 private:
  static constexpr std::uint64_t bit(std::size_t i) noexcept { return std::uint64_t{1} << (i & 63); }
  // Bits 0..hi inclusive.
  static constexpr std::uint64_t low_mask(std::size_t hi) noexcept {
    return ~std::uint64_t{0} >> (63 - hi);
  }
  static std::size_t msb(std::uint64_t x) noexcept {
    return 63 - static_cast<std::size_t>(std::countl_zero(x));
  }
  int top_of_word(std::size_t w) const noexcept {
    return static_cast<int>((w << 6) + msb(words_[w]));
  }

  std::array<std::uint64_t, kWords> words_{};
  std::array<std::uint64_t, kSummary> summary_{};
};

}  // namespace klob
