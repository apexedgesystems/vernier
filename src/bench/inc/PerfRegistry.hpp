#ifndef VERNIER_PERFREGISTRY_HPP
#define VERNIER_PERFREGISTRY_HPP
/**
 * @file PerfRegistry.hpp
 * @brief Minimal per-test result handoff between PerfCase and a gtest listener.
 *
 * Extended with multi-GPU and Unified Memory fields.
 */

#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "src/bench/inc/PerfStats.hpp"

namespace vernier {
namespace bench {

/* --------------------------------- API --------------------------------- */

struct PerfConfig;

inline std::atomic<const PerfConfig*> gPerfCfg{nullptr};

/**
 * @brief Set global perf config pointer.
 * @note RT-safe (atomic store).
 */
inline void setGlobalPerfConfig(const PerfConfig* cfg) noexcept {
  gPerfCfg.store(cfg, std::memory_order_release);
}

/**
 * @brief Get global perf config pointer.
 * @note RT-safe (atomic load).
 */
inline const PerfConfig* globalPerfConfig() noexcept {
  return gPerfCfg.load(std::memory_order_acquire);
}

/* -------------------------------- PerfRow -------------------------------- */

/**
 * @brief Flat row of fields to emit into CSV/JSONL.
 *
 * Includes multi-GPU fields (deviceId, deviceCount, multiGpuEfficiency, p2pBandwidthGBs)
 * and Unified Memory fields.
 */
struct PerfRow {
  std::string testName;
  int cycles{};
  int repeats{};
  int warmup{};
  int threads{};
  int msgBytes{};
  bool console{};
  bool nonBlocking{};
  std::string minLevel;
  Stats stats{};
  double callsPerSecond{};

  // Profiling metadata
  std::optional<std::string> profileTool{};
  std::optional<std::string> profileDir{};

  // Run metadata
  std::string timestamp;
  std::string gitHash;
  std::string hostname;
  std::string platform;

  // GPU-specific metadata (original)
  std::optional<std::string> gpuModel;
  std::optional<std::string> computeCapability;
  std::optional<double> kernelTimeUs;
  std::optional<double> transferTimeUs;
  std::optional<size_t> h2dBytes;
  std::optional<size_t> d2hBytes;
  std::optional<double> speedupVsCpu;
  std::optional<double> memBandwidthGBs;
  std::optional<double> occupancy;
  std::optional<int> smClockMHz;
  std::optional<bool> throttling;

  // Multi-GPU fields
  std::optional<int> deviceId;              ///< Which GPU (0-N), -1 = single-GPU test
  std::optional<int> deviceCount;           ///< Total GPUs used in this test
  std::optional<double> multiGpuEfficiency; ///< Speedup / N_gpus (ideal=1.0)
  std::optional<double> p2pBandwidthGBs;    ///< Peer-to-peer bandwidth

  // Unified Memory fields
  std::optional<size_t> umPageFaults;      ///< Total UM page faults
  std::optional<size_t> umH2DMigrations;   ///< Host->Device migrations
  std::optional<size_t> umD2HMigrations;   ///< Device->Host migrations
  std::optional<double> umMigrationTimeUs; ///< Time spent migrating
  std::optional<bool> umThrashing;         ///< Thrashing detected?

  // Stability assessment
  bool stable{true};        ///< CV% below adaptive threshold
  double cvThreshold{0.05}; ///< Threshold used (from recommendedCVThreshold)
};

/* ----------------------------- PerfSummaryEntry ----------------------------- */

/** @brief Lightweight summary entry for end-of-run table (avoids copying full PerfRow). */
struct PerfSummaryEntry {
  std::string testName;
  double medianUs{};
  double cv{};
  double callsPerSecond{};
  bool stable{true};
  double cvThreshold{0.05};
};

/* ------------------------------ PerfRegistry ------------------------------ */

/**
 * @brief Thread-safe single-slot registry for the last PerfRow produced by PerfCase.
 *
 * Also accumulates lightweight summary entries for the end-of-run table.
 *
 * @note NOT RT-safe (mutex locking, heap allocation).
 */
class PerfRegistry {
public:
  static PerfRegistry& instance() {
    static PerfRegistry r;
    return r;
  }

  void set(PerfRow row) {
    std::lock_guard<std::mutex> lock(mu_);
    // Accumulate summary for end-of-run table
    summary_.push_back(PerfSummaryEntry{row.testName, row.stats.median, row.stats.cv,
                                        row.callsPerSecond, row.stable, row.cvThreshold});
    last_ = std::move(row);
  }

  void updateProfileMeta(const std::string& tool, const std::string& dir) {
    std::lock_guard<std::mutex> lock(mu_);
    if (last_) {
      last_->profileTool = tool;
      last_->profileDir = dir;
    }
  }

  std::optional<PerfRow> take() {
    std::lock_guard<std::mutex> lock(mu_);
    auto out = std::move(last_);
    last_.reset();
    return out;
  }

  /** @brief Get accumulated summary entries (for end-of-run table). */
  [[nodiscard]] const std::vector<PerfSummaryEntry>& summary() const { return summary_; }

private:
  PerfRegistry() = default;
  std::mutex mu_;
  std::optional<PerfRow> last_;
  std::vector<PerfSummaryEntry> summary_;
};

} // namespace bench
} // namespace vernier

#endif // VERNIER_PERFREGISTRY_HPP