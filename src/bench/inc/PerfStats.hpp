#ifndef VERNIER_PERFSTATS_HPP
#define VERNIER_PERFSTATS_HPP
/**
 * @file PerfStats.hpp
 * @brief Statistical summaries for microbenchmarks (median, p10/p90, min/max, mean, stddev, CV).
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace vernier {
namespace bench {

/* ------------------------------- Stats --------------------------------- */

/** @brief Summary of wall-time samples (microseconds per call). */
struct Stats {
  double median{}; ///< 50th percentile
  double p10{};    ///< 10th percentile
  double p90{};    ///< 90th percentile
  double min{};    ///< minimum
  double max{};    ///< maximum
  double mean{};   ///< arithmetic mean
  double stddev{}; ///< standard deviation (jitter measure)
  double cv{};     ///< coefficient of variation (relative jitter: stddev/mean)
};

/* --------------------------------- API --------------------------------- */

/**
 * @brief Compute summary statistics with linear interpolation for quantiles.
 * @param values Vector of samples (modified: sorted in-place for performance).
 * @return Stats summary; zero-initialized if empty.
 * @note NOT RT-safe (heap allocation via vector sort).
 */
inline Stats summarize(std::vector<double>& values) {
  if (values.empty()) {
    return {};
  }
  std::sort(values.begin(), values.end());

  const auto QUANTILE = [&](double f) {
    const double IDX = f * static_cast<double>(values.size() - 1);
    const std::size_t LO = static_cast<std::size_t>(IDX);
    const std::size_t HI = (LO + 1 < values.size()) ? (LO + 1) : LO;
    const double FRAC = IDX - static_cast<double>(LO);
    return values[LO] * (1.0 - FRAC) + values[HI] * FRAC;
  };

  // Mean
  double sum = 0.0;
  for (const double VAL : values) {
    sum += VAL;
  }
  const double MEAN = sum / static_cast<double>(values.size());

  // Standard deviation (population formula)
  double sumSquaredDiff = 0.0;
  for (const double VAL : values) {
    const double DIFF = VAL - MEAN;
    sumSquaredDiff += DIFF * DIFF;
  }
  const double VARIANCE = sumSquaredDiff / static_cast<double>(values.size());
  const double STDDEV = std::sqrt(VARIANCE);

  // Coefficient of variation (relative jitter)
  const double CV = (MEAN != 0.0) ? (STDDEV / MEAN) : 0.0;

  return {QUANTILE(0.50), // median
          QUANTILE(0.10), // p10
          QUANTILE(0.90), // p90
          values.front(), // min
          values.back(),  // max
          MEAN,           // mean
          STDDEV,         // stddev
          CV};            // cv
}

} // namespace bench
} // namespace vernier

#endif // VERNIER_PERFSTATS_HPP