#ifndef VERNIER_MONITORSUMMARY_HPP
#define VERNIER_MONITORSUMMARY_HPP
/**
 * @file MonitorSummary.hpp
 * @brief Accumulates per-scope statistics for end-of-run reporting.
 */

#include "src/monitor/inc/MonitorTag.hpp"

#include <cmath>
#include <cstdio>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace vernier {
namespace monitor {

/* ----------------------------- ScopeStats ----------------------------- */

/** @brief Accumulated stats for a single scope+tag combination. */
struct ScopeStats {
  std::string label; ///< "tag_name/id"
  std::string scope; ///< scope name
  SampleKind kind{SampleKind::SCOPE};
  std::uint64_t count{0};
  double minVal{1e18};
  double maxVal{0.0};
  double sum{0.0};
  std::uint64_t breaches{0};

  std::vector<double> values; ///< all durations (for percentile calc)

  /**
   * @brief Record a value into the accumulator.
   * @param v The value to record.
   * @param breach True if this sample is a threshold breach.
   * @note NOT RT-safe: Heap allocation via vector push_back.
   */
  void record(double v, bool breach) {
    count++;
    sum += v;
    if (v < minVal)
      minVal = v;
    if (v > maxVal)
      maxVal = v;
    if (breach)
      breaches++;
    values.push_back(v);
  }

  /**
   * @brief Compute median of recorded values.
   * @return Median value, or 0.0 if empty.
   * @note NOT RT-safe: Heap allocation via vector copy and sort.
   */
  [[nodiscard]] double median() const {
    if (values.empty())
      return 0.0;
    auto sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const auto N = sorted.size();
    if (N % 2 == 1)
      return sorted[N / 2];
    return (sorted[N / 2 - 1] + sorted[N / 2]) / 2.0;
  }

  /**
   * @brief Compute 99th percentile of recorded values.
   * @return P99 value, or 0.0 if empty.
   * @note NOT RT-safe: Heap allocation via vector copy and sort.
   */
  [[nodiscard]] double p99() const {
    if (values.empty())
      return 0.0;
    auto sorted = values;
    std::sort(sorted.begin(), sorted.end());
    auto idx = static_cast<std::size_t>(std::ceil(0.99 * static_cast<double>(sorted.size())) - 1);
    if (idx >= sorted.size())
      idx = sorted.size() - 1;
    return sorted[idx];
  }
};

/* ----------------------------- MonitorSummary ----------------------------- */

/** @brief Accumulator that collects samples and produces a summary table. */
class MonitorSummary {
public:
  /**
   * @brief Record a sample into the accumulator.
   * @param sample The sample to record.
   * @note NOT RT-safe: Heap allocation via map insertion and string operations.
   */
  void record(const Sample& sample) {
    const std::string KEY = std::string(sample.tag.name) + "/" + std::to_string(sample.tag.id) +
                            "::" + std::string(sample.scope);

    auto& stats = entries_[KEY];
    if (stats.count == 0) {
      stats.label = std::string(sample.tag.name) + "/" + std::to_string(sample.tag.id);
      stats.scope = sample.scope;
      stats.kind = sample.kind;
    }

    const double VAL =
        (sample.kind == SampleKind::SCOPE || sample.kind == SampleKind::THRESHOLD_BREACH)
            ? static_cast<double>(sample.durationNs) / 1e6 // ms
            : sample.value;

    const bool BREACH = (sample.kind == SampleKind::THRESHOLD_BREACH);
    stats.record(VAL, BREACH);
  }

  /**
   * @brief Print the summary table to stderr.
   * @param wallTimeNs Wall-clock duration in nanoseconds.
   * @param totalSamples Total number of samples recorded.
   * @param droppedSamples Number of samples dropped due to overflow.
   * @note NOT RT-safe: Console I/O.
   */
  void print(std::uint64_t wallTimeNs, std::uint64_t totalSamples,
             std::uint64_t droppedSamples) const {
    std::fprintf(stderr, "\nvernier::monitor summary\n");
    std::fprintf(stderr, "%-20s %-24s %8s %10s %10s %10s %8s\n", "Tag", "Scope", "Calls", "Median",
                 "P99", "Max", "Breaches");
    std::fprintf(stderr, "%-20s %-24s %8s %10s %10s %10s %8s\n", "--------------------",
                 "------------------------", "--------", "----------", "----------", "----------",
                 "--------");

    for (const auto& [KEY, S] : entries_) {
      if (S.kind == SampleKind::GAUGE) {
        std::fprintf(stderr, "%-20s %-24s %8lu %10.3f %10.3f %10.3f %8s\n", S.label.c_str(),
                     (S.scope + " (g)").c_str(), static_cast<unsigned long>(S.count), S.median(),
                     S.p99(), S.maxVal, "-");
      } else if (S.kind == SampleKind::COUNTER) {
        std::fprintf(stderr, "%-20s %-24s %8lu %10s %10s %10.0f %8s\n", S.label.c_str(),
                     (S.scope + " (c)").c_str(), static_cast<unsigned long>(S.count), "-", "-",
                     S.sum, "-");
      } else {
        char breachStr[16];
        std::snprintf(breachStr, sizeof(breachStr), "%lu", static_cast<unsigned long>(S.breaches));
        std::fprintf(stderr, "%-20s %-24s %8lu %8.3f ms %8.3f ms %8.3f ms %8s\n", S.label.c_str(),
                     S.scope.c_str(), static_cast<unsigned long>(S.count), S.median(), S.p99(),
                     S.maxVal, breachStr);
      }
    }

    const double WALL_SEC = static_cast<double>(wallTimeNs) / 1e9;
    std::fprintf(stderr, "\nTotal samples: %lu | Dropped: %lu | Wall time: %.1f s\n",
                 static_cast<unsigned long>(totalSamples),
                 static_cast<unsigned long>(droppedSamples), WALL_SEC);
  }

  /**
   * @brief Number of distinct scopes recorded.
   * @return Entry count.
   * @note RT-safe: No allocation.
   */
  [[nodiscard]] std::size_t size() const { return entries_.size(); }

  /**
   * @brief Access entries (for testing).
   * @return Const reference to the entries map.
   * @note RT-safe: No allocation.
   */
  [[nodiscard]] const auto& entries() const { return entries_; }

private:
  std::map<std::string, ScopeStats> entries_;
};

} // namespace monitor
} // namespace vernier

#endif // VERNIER_MONITORSUMMARY_HPP
