/**
 * @file Filter.cpp
 * @brief Both versions of the filter, and the input generators.
 */

#include "src/bench/demo/examples/filter/inc/Filter.hpp"

#include <algorithm>
#include <random>

namespace vernier {
namespace bench {
namespace demo {

/* ----------------------------- API ----------------------------- */

std::size_t filterBranchy(std::span<const double> values, double threshold, std::span<double> out) {
  std::size_t kept = 0;
  for (const double value : values) {
    if (value > threshold) {
      // A conditional store: the branch around it survives optimization
      out[kept++] = value;
    }
  }
  return kept;
}

std::size_t filterBranchless(std::span<const double> values, double threshold,
                             std::span<double> out) {
  std::size_t kept = 0;
  for (const double value : values) {
    out[kept] = value; // Always stored; a rejected value is overwritten by the next
    kept += static_cast<std::size_t>(value > threshold);
  }
  return kept;
}

std::vector<double> makeValues(std::size_t count, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  std::vector<double> values(count);
  for (double& value : values) {
    value = unit(rng);
  }
  return values;
}

std::vector<double> makeSortedValues(std::size_t count, unsigned seed) {
  std::vector<double> values = makeValues(count, seed);
  std::sort(values.begin(), values.end());
  return values;
}

} // namespace demo
} // namespace bench
} // namespace vernier
