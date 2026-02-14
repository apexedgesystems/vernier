#ifndef VERNIER_PERFHARNESS_HPP
#define VERNIER_PERFHARNESS_HPP
/**
 * @file PerfHarness.hpp
 * @brief High-level perf test constructs layered over GoogleTest.
 *
 * Includes console printing, metadata capture, memory bandwidth analysis,
 * and the PerfCase class that drives warmup/measure/teardown phases.
 */

#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <thread>

#include <unistd.h> // gethostname

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/PerfUtils.hpp"
#include "src/bench/inc/PerfRegistry.hpp"
#include "src/bench/inc/PerfValidation.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- Console Printing ----------------------------- */

/**
 * @brief Format a large number with K/M/G suffix for readability.
 * @note RT-safe (pure computation).
 */
inline void formatCompact(char* buf, std::size_t bufLen, double val) {
  if (val >= 1e9) {
    std::snprintf(buf, bufLen, "%.1fG", val / 1e9);
  } else if (val >= 1e6) {
    std::snprintf(buf, bufLen, "%.1fM", val / 1e6);
  } else if (val >= 1e3) {
    std::snprintf(buf, bufLen, "%.1fK", val / 1e3);
  } else {
    std::snprintf(buf, bufLen, "%.0f", val);
  }
}

/**
 * @brief Print a compact, scannable result line for a test.
 *
 * Format: [TestName] 3.456 us/call  CV=7.2%  ~289K calls/s  (p10=3.10 p90=3.80 sd=0.25)
 *
 * Key metrics (median, CV, calls/s) are first for quick scanning.
 * Secondary stats (percentiles, stddev) are in parentheses.
 *
 * @note NOT RT-safe (console I/O).
 */
inline void printStats(const char* label, const Stats& s, double callsPerSecond, bool stable) {
  char cpsBuf[16];
  formatCompact(cpsBuf, sizeof(cpsBuf), callsPerSecond);

  const char* STABILITY = stable ? "" : " [UNSTABLE]";

  std::printf("%s  %.3f us/call  CV=%.1f%%  ~%s calls/s  (p10=%.3f p90=%.3f sd=%.3f)%s\n", label,
              s.median, s.cv * 100.0, cpsBuf, s.p10, s.p90, s.stddev, STABILITY);
}

/**
 * @brief Print actionable hints based on benchmark results.
 * @note NOT RT-safe (console I/O).
 */
inline void printHints(const Stats& s, const PerfConfig& cfg) {
  if (cfg.quickMode && s.cv > 0.15) {
    std::fprintf(stderr,
                 "  [INFO] High CV%% (%.1f%%) in quick mode is expected.\n"
                 "         For stable baselines, run without --quick\n",
                 s.cv * 100.0);
    return;
  }

  if (s.cv > 0.20 && cfg.profileTool.empty()) {
    std::fprintf(stderr,
                 "  [!] High variance (CV=%.1f%%). Consider:\n"
                 "      --profile perf    # CPU counters + flamegraph\n"
                 "      --profile nsight  # GPU kernel analysis\n",
                 s.cv * 100.0);
  } else if (!cfg.quickMode && s.cv > 0.10 && cfg.profileTool.empty()) {
    std::fprintf(stderr, "  [WARN] Moderate variance (CV=%.1f%%). Try --repeats 20 or --profile\n",
                 s.cv * 100.0);
  }
}

/**
 * @brief Print stats with automatic hints (convenience wrapper).
 * @note NOT RT-safe (console I/O).
 */
inline void printStatsWithHints(const char* label, const Stats& s, double callsPerSecond,
                                const PerfConfig& cfg, bool stable) {
  printStats(label, s, callsPerSecond, stable);
  printHints(s, cfg);
}

/**
 * @brief Print progress during long measurement runs.
 *
 * Emits a carriage-return overwrite on stderr every ~2 seconds.
 * Only prints if the measurement has been running for >2 seconds.
 * The line is cleared before the final result is printed.
 *
 * @note NOT RT-safe (console I/O).
 */
inline void printProgress(const char* testName, int repeat, int totalRepeats,
                          std::chrono::steady_clock::time_point startTime) {
  auto now = std::chrono::steady_clock::now();
  double elapsedS =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count() / 1000.0;
  double estTotalS = elapsedS * totalRepeats / (repeat + 1);
  double remainS = estTotalS - elapsedS;
  std::fprintf(stderr, "\r  [%s] repeat %d/%d (%.0f%%)  elapsed=%.1fs  remaining~%.1fs   ",
               testName, repeat + 1, totalRepeats, 100.0 * (repeat + 1) / totalRepeats, elapsedS,
               remainS);
}

/** @brief Clear progress line from stderr. */
inline void clearProgress() { std::fprintf(stderr, "\r%*s\r", 80, ""); }

/* ----------------------------- Metadata Capture ----------------------------- */

/**
 * @brief Capture current timestamp in ISO 8601 format.
 * @note NOT RT-safe (system calls, heap allocation).
 */
inline std::string captureTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};

#if defined(__unix__) || defined(__APPLE__)
  gmtime_r(&t, &tm);
#else
  std::tm* tmp = std::gmtime(&t);
  if (tmp)
    tm = *tmp;
#endif

  std::array<char, 32> buf{};
  std::strftime(buf.data(), buf.size(), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return std::string(buf.data());
}

/**
 * @brief Capture git commit hash (short form).
 * @note NOT RT-safe (spawns subprocess via popen).
 */
inline std::string captureGitHash() {
  std::array<char, 128> buf{};
  FILE* pipe = popen("git describe --always --dirty --tags 2>/dev/null", "r");
  if (!pipe) {
    return "unknown";
  }

  if (fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
    pclose(pipe);
    std::string result(buf.data());
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
      result.pop_back();
    }
    return result.empty() ? "unknown" : result;
  }

  pclose(pipe);
  return "unknown";
}

/**
 * @brief Capture system hostname.
 * @note NOT RT-safe (system call, heap allocation).
 */
inline std::string captureHostname() {
  std::array<char, 256> buf{};

#if defined(__unix__) || defined(__APPLE__)
  if (gethostname(buf.data(), buf.size()) == 0) {
    buf.back() = '\0';
    return std::string(buf.data());
  }
#endif

  return "unknown";
}

/**
 * @brief Detect platform architecture.
 * @note RT-safe (compile-time constant string).
 */
inline std::string capturePlatform() {
#if defined(__x86_64__) || defined(_M_X64)
  return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
  return "aarch64";
#elif defined(__arm__) || defined(_M_ARM)
  return "arm";
#elif defined(__i386__) || defined(_M_IX86)
  return "x86";
#elif defined(__riscv)
  return "riscv";
#else
  return "unknown";
#endif
}

/**
 * @brief Capture all metadata at once (cached for performance).
 * @param cacheMetadata If true, capture once and reuse. If false, always fresh.
 * @return Tuple of (timestamp, gitHash, hostname, platform).
 * @note NOT RT-safe (system calls, subprocess, heap allocation).
 */
inline std::tuple<std::string, std::string, std::string, std::string>
captureMetadata(bool cacheMetadata = true) {
  static bool cached = false;
  static std::string cachedGitHash;
  static std::string cachedHostname;
  static std::string cachedPlatform;

  if (cacheMetadata && !cached) {
    cachedGitHash = captureGitHash();
    cachedHostname = captureHostname();
    cachedPlatform = capturePlatform();
    cached = true;
  }

  return std::make_tuple(captureTimestamp(), cacheMetadata ? cachedGitHash : captureGitHash(),
                         cacheMetadata ? cachedHostname : captureHostname(),
                         cacheMetadata ? cachedPlatform : capturePlatform());
}

/* ----------------------------- MemoryProfile ----------------------------- */

/**
 * @brief Memory profile for bandwidth analysis.
 *
 * Describes memory access patterns for an operation to enable
 * bandwidth calculations and CPU-bound vs memory-bound analysis.
 */
struct MemoryProfile {
  size_t bytesRead = 0;      ///< Input data size per operation (bytes)
  size_t bytesWritten = 0;   ///< Output data size per operation (bytes)
  size_t bytesAllocated = 0; ///< Heap allocations per operation (optional)

  /**
   * @brief Calculate total bandwidth in MB/s.
   * @param durationUs Duration in microseconds
   * @return Bandwidth in MB/s (read + write)
   * @note RT-safe (pure calculation, no allocations).
   */
  [[nodiscard]] double bandwidthMBs(double durationUs) const {
    if (durationUs <= 0.0)
      return 0.0;
    const double TOTAL_BYTES = static_cast<double>(bytesRead + bytesWritten);
    const double DURATION_S = durationUs / 1e6;
    return (TOTAL_BYTES / DURATION_S) / 1e6; // MB/s
  }

  /**
   * @brief Calculate efficiency vs theoretical peak.
   * @param durationUs Duration in microseconds
   * @param peakMBs Theoretical peak bandwidth (MB/s)
   * @return Efficiency percentage [0-100]
   * @note RT-safe (pure calculation, no allocations).
   */
  [[nodiscard]] double efficiency(double durationUs, double peakMBs) const {
    if (peakMBs <= 0.0)
      return 0.0;
    const double ACHIEVED = bandwidthMBs(durationUs);
    return (ACHIEVED / peakMBs) * 100.0;
  }
};

/* ------------------------------ PerfResult ------------------------------ */

/** @brief Result of a measured section. */
struct PerfResult {
  Stats stats{};           ///< Summarized per-call wall us
  double callsPerSecond{}; ///< Derived from median
  std::string label;       ///< Section label (e.g., "measured")

  // Memory bandwidth analysis (optional)
  std::optional<MemoryProfile> memoryProfile{};

  // Stability assessment
  bool stable{true};        ///< CV% below adaptive threshold
  double cvThreshold{0.05}; ///< Threshold used (from recommendedCVThreshold)
};

/* ----------------------------- Row Builder ----------------------------- */

/**
 * @brief Build a PerfRow from test state and measurement results.
 *
 * Centralizes PerfRow construction so callers don't need 38 positional fields.
 * GPU/multi-GPU/Unified Memory fields remain at default (nullopt) for CPU tests.
 *
 * @note NOT RT-safe (captures metadata via subprocess).
 */
inline PerfRow buildPerfRow(const std::string& testName, const PerfConfig& cfg, int actualWarmup,
                            int threadCount, const Stats& stats, double callsPerSecond) {
  auto [timestamp, gitHash, hostname, platform] = captureMetadata(/*cache=*/true);

  const double CV_THRESHOLD = recommendedCVThreshold(cfg);

  PerfRow row;
  row.testName = testName;
  row.cycles = cfg.cycles;
  row.repeats = cfg.repeats;
  row.warmup = actualWarmup;
  row.threads = threadCount;
  row.msgBytes = cfg.msgBytes;
  row.console = cfg.console;
  row.nonBlocking = cfg.nonBlocking;
  row.minLevel = cfg.minLevel;
  row.stats = stats;
  row.callsPerSecond = callsPerSecond;
  row.timestamp = timestamp;
  row.gitHash = gitHash;
  row.hostname = hostname;
  row.platform = platform;
  row.stable = stats.cv < CV_THRESHOLD;
  row.cvThreshold = CV_THRESHOLD;
  return row;
}

/* ------------------------------- PerfCase ------------------------------- */

/**
 * @brief A convenience facade to structure a perf test into phases.
 *
 * Typical usage:
 *   PerfCase perf{"Suite.Name", cfg};
 *   perf.setup(...);
 *   perf.warmup([&]{ body(cycles); });
 *   const PerfResult r = perf.measured([&]{ body(cycles); });
 *   perf.teardown(...);
 *
 * @note NOT RT-safe (heap allocation, threads, console/file I/O).
 */
class PerfCase {
public:
  using Fn = std::function<void()>;
  using BeforeHook = std::function<void(const PerfCase&)>;
  using AfterHook = std::function<void(const PerfCase&, const Stats&)>;

  PerfCase(std::string testName, PerfConfig cfg)
      : testName_(std::move(testName)), cfg_(std::move(cfg)) {}

  /** @brief Optional one-time setup before warmup/measure. */
  void setup(Fn fn) {
    if (fn) {
      fn();
    }
  }

  /**
   * @brief Run warmup repeats (user provides the inner loop).
   *
   * Auto-scaling warmup: If cfg.warmup == 0, automatically determine warmup count
   * based on the number of cycles:
   *   - cycles < 1000:  warmup = 5 (small workloads need more warmup)
   *   - cycles < 10000: warmup = 3 (medium workloads)
   *   - cycles >= 10000: warmup = 1 (large workloads already warm caches)
   *
   * Otherwise, use the explicitly specified warmup count.
   */
  void warmup(Fn fn) {
    if (!fn) {
      return;
    }

    // Determine warmup count
    actualWarmup_ = cfg_.warmup;

    // Auto-scaling: if warmup is 0, calculate based on cycles
    if (actualWarmup_ == 0) {
      if (cfg_.cycles < 1000) {
        actualWarmup_ = 5; // Small workloads need more warmup
      } else if (cfg_.cycles < 10000) {
        actualWarmup_ = 3; // Medium workloads
      } else {
        actualWarmup_ = 1; // Large workloads (already warmed by sheer volume)
      }
    }

    for (int w = 0; w < actualWarmup_; ++w) {
      fn();
    }
  }

  /** @brief Install hooks bracketing the measured phase (optional). */
  void setBeforeMeasureHook(BeforeHook h) { beforeHook_ = std::move(h); }
  void setAfterMeasureHook(AfterHook h) { afterHook_ = std::move(h); }

  /**
   * @brief Measure repeats. User-provided body should perform exactly `cycles()` ops.
   * Records wall us per call per repeat, then summarizes.
   *
   * Progress is printed to stderr every ~2 seconds for long-running measurements.
   */
  PerfResult measured(Fn fn, std::string label = "measured") {
    if (beforeHook_) {
      beforeHook_(*this);
    }

    const auto MEASURE_START = std::chrono::steady_clock::now();
    auto lastProgressTime = MEASURE_START;
    bool showedProgress = false;

    std::vector<double> perCall;
    perCall.reserve(static_cast<std::size_t>(cfg_.repeats));
    for (int r = 0; r < cfg_.repeats; ++r) {
      const double T0 = nowUs();
      fn();
      const double T1 = nowUs();
      perCall.push_back((T1 - T0) / static_cast<double>(cfg_.cycles));

      // Progress reporting: print every ~2 seconds for long runs
      auto now = std::chrono::steady_clock::now();
      auto sinceLast =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - lastProgressTime);
      if (sinceLast.count() >= 2000) {
        printProgress(testName_.c_str(), r, cfg_.repeats, MEASURE_START);
        lastProgressTime = now;
        showedProgress = true;
      }
    }

    if (showedProgress) {
      clearProgress();
    }

    auto vals = perCall; // copy so summarize() can sort
    const Stats S = summarize(vals);
    const double CPS = (S.median > 0.0) ? 1e6 / S.median : 0.0;
    const double CV_THRESHOLD = recommendedCVThreshold(cfg_);
    const bool IS_STABLE = S.cv < CV_THRESHOLD;

    // Print stats with automatic hints
    const std::string LABEL_STR = "[" + testName_ + "]";
    printStatsWithHints(LABEL_STR.c_str(), S, CPS, cfg_, IS_STABLE);

    PerfRegistry::instance().set(buildPerfRow(testName_, cfg_, actualWarmup_, threads(), S, CPS));

    if (afterHook_) {
      afterHook_(*this, S);
    }

    return PerfResult{S, CPS, std::move(label), std::nullopt, IS_STABLE, CV_THRESHOLD};
  }

  /** @brief Optional teardown after measurement. */
  void teardown(Fn fn) {
    if (fn) {
      fn();
    }
  }

  // ---------------------------------------------------------------------------
  // Pre-baked constructs (lightweight wrappers around measured())
  // ---------------------------------------------------------------------------

  /**
   * @brief Throughput loop: runs a lambda `op()` exactly `cycles()` times.
   * The lambda should perform one operation and be allocation-free in hot paths.
   *
   * Optional memory profile for bandwidth analysis.
   */
  PerfResult throughputLoop(const std::function<void()>& op, std::string label = "throughput",
                            std::optional<MemoryProfile> memProfile = std::nullopt) {
    auto loop = [&]() {
      for (int i = 0; i < cfg_.cycles; ++i) {
        op();
      }
    };

    auto result = measured(loop, std::move(label));
    result.memoryProfile = memProfile;

    // Print bandwidth analysis if provided
    if (memProfile.has_value()) {
      const double BW = memProfile->bandwidthMBs(result.stats.median);
      std::printf("  Memory bandwidth: %.1f MB/s (%.1f MB read, %.1f MB written per call)\n", BW,
                  memProfile->bytesRead / 1e6, memProfile->bytesWritten / 1e6);

      // Estimate theoretical peak (rough heuristic: DDR4-2400 ~19 GB/s per channel)
      // Users should provide actual peak for their system
      const double ESTIMATED_PEAK = 12000.0; // MB/s (conservative DDR4 estimate)
      const double EFF = memProfile->efficiency(result.stats.median, ESTIMATED_PEAK);
      std::printf("  Estimated efficiency: %.1f%% of theoretical peak (~%.0f MB/s)\n", EFF,
                  ESTIMATED_PEAK);

      if (EFF < 10.0) {
        std::printf(
            "  Hint: Low bandwidth utilization -> CPU-bound (algorithm optimization needed)\n");
      } else if (EFF > 50.0) {
        std::printf(
            "  Hint: High bandwidth utilization -> Memory-bound (consider memory layout)\n");
      }
    }

    return result;
  }

  /**
   * @brief Throughput loop with simple memory tracking (read-only).
   *
   * Simplified overload for common case of read-only memory operations.
   *
   * @param op Operation to benchmark (called cycles() times)
   * @param label Label for output and CSV
   * @param bytesRead Bytes read per operation
   * @return PerfResult with memory bandwidth analysis
   *
   * Example:
   * @code
   * auto result = perf.throughputLoop([&]{ sum(data); }, "sum", data.size());
   * @endcode
   */
  PerfResult throughputLoop(const std::function<void()>& op, std::string label,
                            std::size_t bytesRead) {
    return throughputLoop(op, std::move(label), MemoryProfile{bytesRead, 0, 0});
  }

  /**
   * @brief Throughput loop with read/write memory tracking.
   *
   * Simplified overload for read/write memory operations.
   *
   * @param op Operation to benchmark (called cycles() times)
   * @param label Label for output and CSV
   * @param bytesRead Bytes read per operation
   * @param bytesWritten Bytes written per operation
   * @return PerfResult with memory bandwidth analysis
   *
   * Example:
   * @code
   * auto result = perf.throughputLoop([&]{ copy(src, dst); }, "copy", srcSize, dstSize);
   * @endcode
   */
  PerfResult throughputLoop(const std::function<void()>& op, std::string label,
                            std::size_t bytesRead, std::size_t bytesWritten) {
    return throughputLoop(op, std::move(label), MemoryProfile{bytesRead, bytesWritten, 0});
  }

  /**
   * @brief Contention run: start `threads()` workers simultaneously, each doing `cycles()` ops.
   * The `worker` lambda is invoked in each thread. Caller is free to compute extras (e.g., drop%).
   * This returns only timing stats (per-call across all operations). For drop%, derive externally.
   */
  PerfResult contentionRun(const std::function<void()>& worker, std::string label = "contention") {
    const int THREAD_COUNT = threads();
    std::vector<double> perCall;
    perCall.reserve(static_cast<std::size_t>(cfg_.repeats));

    if (beforeHook_) {
      beforeHook_(*this);
    }

    const auto MEASURE_START = std::chrono::steady_clock::now();
    auto lastProgressTime = MEASURE_START;
    bool showedProgress = false;

    for (int r = 0; r < cfg_.repeats; ++r) {
      StartGate gate(THREAD_COUNT);
      const double T0 = nowUs();

      std::vector<std::thread> ts;
      ts.reserve(static_cast<std::size_t>(THREAD_COUNT));
      for (int t = 0; t < THREAD_COUNT; ++t) {
        ts.emplace_back([&]() {
          gate.start();
          for (int i = 0; i < cfg_.cycles; ++i) {
            worker();
          }
        });
      }
      gate.releaseWhenAllReady();
      for (auto& th : ts) {
        th.join();
      }

      const double T1 = nowUs();
      const double TOTAL_CALLS = static_cast<double>(THREAD_COUNT) * cfg_.cycles;
      perCall.push_back((T1 - T0) / TOTAL_CALLS);

      // Progress reporting
      auto now = std::chrono::steady_clock::now();
      auto sinceLast =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - lastProgressTime);
      if (sinceLast.count() >= 2000) {
        printProgress(testName_.c_str(), r, cfg_.repeats, MEASURE_START);
        lastProgressTime = now;
        showedProgress = true;
      }
    }

    if (showedProgress) {
      clearProgress();
    }

    auto vals = perCall;
    const Stats S = summarize(vals);
    const double CPS = (S.median > 0.0) ? 1e6 / S.median : 0.0;
    const double CV_THRESHOLD = recommendedCVThreshold(cfg_);
    const bool IS_STABLE = S.cv < CV_THRESHOLD;

    // Print stats with automatic hints
    const std::string LABEL_STR = "[" + testName_ + "]";
    printStatsWithHints(LABEL_STR.c_str(), S, CPS, cfg_, IS_STABLE);

    PerfRegistry::instance().set(
        buildPerfRow(testName_, cfg_, actualWarmup_, THREAD_COUNT, S, CPS));

    if (afterHook_) {
      afterHook_(*this, S);
    }

    return PerfResult{S, CPS, std::move(label), std::nullopt, IS_STABLE, CV_THRESHOLD};
  }

  // Accessors
  [[nodiscard]] int cycles() const noexcept { return cfg_.cycles; }
  [[nodiscard]] int repeats() const noexcept { return cfg_.repeats; }
  [[nodiscard]] int threads() const noexcept { return (cfg_.threads > 0) ? cfg_.threads : 1; }
  [[nodiscard]] int warmup() const noexcept { return actualWarmup_; }
  [[nodiscard]] const PerfConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] const std::string& testName() const noexcept { return testName_; }

private:
  std::string testName_;
  PerfConfig cfg_{};
  int actualWarmup_ = 1;

  // Optional hooks (no-ops unless set)
  BeforeHook beforeHook_{};
  AfterHook afterHook_{};
};

} // namespace bench
} // namespace vernier

#endif // VERNIER_PERFHARNESS_HPP