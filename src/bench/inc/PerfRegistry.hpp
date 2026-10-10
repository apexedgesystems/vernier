#ifndef VERNIER_PERFREGISTRY_HPP
#define VERNIER_PERFREGISTRY_HPP
/**
 * @file PerfRegistry.hpp
 * @brief Handoff of each test's measurement rows from the harness to the
 * GoogleTest listeners that publish them.
 *
 * Extended with multi-GPU and Unified Memory fields.
 */

#include <atomic>
#include <cstddef>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
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
 * @brief Flat row of fields to emit into CSV/JSONL: one completed measurement.
 *
 * Includes multi-GPU fields (deviceId, deviceCount, multiGpuEfficiency, p2pBandwidthGBs)
 * and Unified Memory fields.
 */
struct PerfRow {
  /// The measuring case's name when published; PerfRegistry::takeAll() replaces
  /// it with the row's name in the test (see detail::rowIdentities()).
  std::string testName;
  int cycles{};
  int repeats{};
  int warmup{};
  /// Threads that made the measured CPU calls: contentionRun()'s workers, 1 for
  /// measured() and throughputLoop(). A GPU row records 1, the host thread that
  /// drove it; a multi-GPU row's devices are its deviceCount.
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
  // The declared transfers' bytes over their time (empty when a test declares
  // none) and the harness's occupancy estimate (empty without a launch
  // configuration; not a measured occupancy).
  std::optional<double> memBandwidthGBs;
  std::optional<double> occupancy;

  // NVML samples at a kernel measurement's start and end (multi-GPU rows carry
  // none). Each cell is empty unless NVML reported every reading it needs, and
  // the run names the readings it did not report.
  std::optional<int> smClockMHz;
  std::optional<bool> throttling;
  std::optional<double> powerDrawW;  ///< Mean of the power draw sampled at both ends (W)
  std::optional<double> powerLimitW; ///< Configured power limit (W)
  std::optional<int> temperatureC;   ///< End-of-measure GPU core temperature (C)
  std::optional<int>
      temperatureDeltaC; ///< Delta over the measured window (C, positive = warmed up)

  // CUPTI in-process kernel records (empty when CUPTI recorded no launch for
  // the row: a build without CUPTI, a collector that stood down, or a failure).
  std::optional<std::size_t> cuptiKernelLaunches;
  std::optional<int> cuptiRegistersMedian;
  std::optional<int> cuptiRegistersMax;
  std::optional<std::uint32_t> cuptiStaticSmemBytes;
  std::optional<std::uint32_t> cuptiDynamicSmemBytes;

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

  /// The measurement's label argument. Not a CSV column: it names the row when
  /// its case measures more than once in a test.
  std::string label;
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

/* ------------------------------ Row Identity ------------------------------ */

namespace detail {

/**
 * @brief Name one test's rows, given in the order their measurements completed.
 *
 * A case name (PerfRow::testName as published) that occurs once among the rows
 * names its row unchanged. Each row of a case that occurs more than once is
 * named "<case>/<label>"; a label that is empty, or that repeats among that
 * case's rows, is followed by "#n", n counting that label's rows from 1. The
 * unchanged names are given first, then the others in completion order; one
 * that equals a name already given (possible only through a '/' in a case name
 * or a '#' in a label) takes the smallest "#k", k >= 2, that no row of the
 * test uses. The names are therefore unique within the test.
 *
 * @return One name per row, in the order of @p rows.
 * @note NOT RT-safe (heap allocation).
 */
inline std::vector<std::string> rowIdentities(const std::vector<PerfRow>& rows) {
  std::map<std::string, std::size_t> caseRows;
  std::map<std::pair<std::string, std::string>, std::size_t> labelRows;
  for (const PerfRow& row : rows) {
    ++caseRows[row.testName];
    ++labelRows[{row.testName, row.label}];
  }

  std::vector<std::string> names(rows.size());
  std::vector<bool> qualified(rows.size(), false);
  std::map<std::pair<std::string, std::string>, std::size_t> labelSeen;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const PerfRow& row = rows[i];
    if (caseRows[row.testName] == 1) {
      names[i] = row.testName;
      continue;
    }
    qualified[i] = true;
    const std::pair<std::string, std::string> KEY{row.testName, row.label};
    const std::size_t NTH = ++labelSeen[KEY];
    names[i] = row.testName + "/" + row.label;
    if (row.label.empty() || labelRows[KEY] > 1) {
      names[i] += "#" + std::to_string(NTH);
    }
  }

  std::set<std::string> given;
  std::multiset<std::string> pending;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (qualified[i]) {
      pending.insert(names[i]);
    } else {
      given.insert(names[i]);
    }
  }
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (!qualified[i]) {
      continue;
    }
    pending.erase(pending.find(names[i]));
    if (given.count(names[i]) != 0) {
      const std::string BASE = names[i];
      std::size_t k = 2;
      while (given.count(BASE + "#" + std::to_string(k)) != 0 ||
             pending.count(BASE + "#" + std::to_string(k)) != 0) {
        ++k;
      }
      names[i] = BASE + "#" + std::to_string(k);
    }
    given.insert(names[i]);
  }
  return names;
}

} // namespace detail

/* ------------------------------ PerfRegistry ------------------------------ */

/**
 * @brief Thread-safe handoff of each test's measurement rows.
 *
 * The harness publishes one row per completed measurement with set(). The rows
 * wait, in publication order, until a GoogleTest listener takes them at the
 * test's end with takeAll(), which names them (detail::rowIdentities()) and
 * gives their end-of-run summary entries the same names. The listeners
 * installPerfEventListener() installs do that after every test; a program that
 * installs neither keeps every row until it exits.
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

  /**
   * @brief Publish one completed measurement's row, after every row already
   * waiting. Nothing is replaced: each row leaves with the next take.
   */
  void set(PerfRow row) {
    std::lock_guard<std::mutex> lock(mu_);
    // Accumulate summary for end-of-run table
    summary_.push_back(PerfSummaryEntry{row.testName, row.stats.median, row.stats.cv,
                                        row.callsPerSecond, row.stable, row.cvThreshold});
    waiting_.push_back(WaitingRow{std::move(row), summary_.size() - 1, std::this_thread::get_id()});
  }

  /**
   * @brief Stamp profiler identity on the waiting row the calling thread
   * published last.
   *
   * A profiler's after hook runs on the thread that published the measurement
   * it brackets, right after publishing it, so the stamp reaches that
   * measurement's row even when another thread has published since. Nothing
   * is stamped when that row has already been taken.
   *
   * A case has one profiler, so every row of a case that measures more than
   * once names the same artifact folder; a backend that writes each capture
   * under a fixed file name keeps only the latest one there.
   */
  void updateProfileMeta(const std::string& tool, const std::string& dir) {
    std::lock_guard<std::mutex> lock(mu_);
    const std::thread::id SELF = std::this_thread::get_id();
    for (auto it = waiting_.rbegin(); it != waiting_.rend(); ++it) {
      if (it->publisher == SELF) {
        it->row.profileTool = tool;
        it->row.profileDir = dir;
        return;
      }
    }
  }

  /**
   * @brief Take every waiting row, in publication order, each named by
   * detail::rowIdentities(); their summary entries take the same names.
   * Nothing waits afterwards. Listeners call this at a test's end.
   */
  std::vector<PerfRow> takeAll() {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<PerfRow> rows;
    rows.reserve(waiting_.size());
    for (WaitingRow& waiting : waiting_) {
      rows.push_back(std::move(waiting.row));
    }
    const std::vector<std::string> NAMES = detail::rowIdentities(rows);
    for (std::size_t i = 0; i < rows.size(); ++i) {
      rows[i].testName = NAMES[i];
      summary_[waiting_[i].summaryIndex].testName = NAMES[i];
    }
    waiting_.clear();
    return rows;
  }

  /**
   * @brief Take every waiting row, as takeAll() does, and return the most
   * recent one, or std::nullopt when none waited. For a caller that made one
   * measurement and reads its row.
   */
  std::optional<PerfRow> take() {
    std::vector<PerfRow> rows = takeAll();
    if (rows.empty()) {
      return std::nullopt;
    }
    return std::move(rows.back());
  }

  /** @brief Get accumulated summary entries (for end-of-run table). */
  [[nodiscard]] const std::vector<PerfSummaryEntry>& summary() const { return summary_; }

private:
  /// A published row waiting for its test's end.
  struct WaitingRow {
    PerfRow row;
    std::size_t summaryIndex{}; ///< Its entry in summary_.
    std::thread::id publisher;  ///< The thread that published it.
  };

  PerfRegistry() = default;
  std::mutex mu_;
  std::vector<WaitingRow> waiting_;
  std::vector<PerfSummaryEntry> summary_;
};

} // namespace bench
} // namespace vernier

#endif // VERNIER_PERFREGISTRY_HPP