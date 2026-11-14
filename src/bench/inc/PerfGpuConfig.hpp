#ifndef VERNIER_PERFGPUCONFIG_HPP
#define VERNIER_PERFGPUCONFIG_HPP
/**
 * @file PerfGpuConfig.hpp
 * @brief GPU-specific configuration extensions.
 */

#include <string>
#include "src/bench/inc/PerfConfig.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- PerfGpuConfig ----------------------------- */

/**
 * @brief GPU-specific configuration (extends PerfConfig).
 */
struct PerfGpuConfig {
  // ---- GPU warmup settings ----
  int gpuWarmup = 10;

  // ---- Memory transfer strategy ----
  enum class MemoryStrategy {
    Explicit, ///< Explicit H2D/D2H transfers (default)
    Unified,  ///< CUDA Unified Memory
    Pinned,   ///< Pinned host memory
    Mapped    ///< Zero-copy mapped memory
  };
  /// The memory strategy for the tests that allocate by one: --gpu-memory sets
  /// it (Explicit without the flag). Such a test reads it
  /// (PerfGpuCase::gpuConfig()) and makes its own allocation, as
  /// GpuMemoryStrategies.SelectedStrategy does. The harness reads it nowhere:
  /// it allocates no test's memory, and the choice asks for no unified-memory
  /// counters (captureUnifiedMemory).
  MemoryStrategy memStrategy = MemoryStrategy::Explicit;

  // ---- Profiling options ----
  bool captureOccupancy = true;
  bool captureClockSpeeds = true;
  bool captureMemoryBandwidth = true;

  /// An explicit request for unified-memory counters: --capture-um sets it,
  /// and a caller that sets it makes the same request. Choosing a memory
  /// strategy (memStrategy, --gpu-memory) does not set it. Each measurement
  /// made with this set records CUPTI's unified-memory counters over its
  /// window (GpuStats::unifiedMemory). Where it cannot (CUPTI refuses them on
  /// a platform without them, or stood down, or this build has none), or a
  /// window's records are incomplete, the measurement reports the request as
  /// failed, its counters stay empty, and a run whose tests pass exits with
  /// status 4 (ProfilerRegistry::finishRun()).
  bool captureUnifiedMemory = false;

  // ---- Device selection ----
  int deviceId = 0;

  // ---- Stream configuration ----
  bool useHighPriorityStream = true;

  // ---- Validation thresholds ----
  double minSpeedupVsCpu = 1.0;
  double maxTransferOverhead = 50.0;

  /**
   * @brief GPU settings derived from @p base: the warmup count, and the
   *        Unified memory strategy when the profile args contain "unified".
   *
   * The strategy is recorded in memStrategy only. It does not ask for
   * unified-memory counters (captureUnifiedMemory stays false), and the
   * harness does not allocate a test's memory: the test makes its own
   * allocation.
   */
  static PerfGpuConfig fromPerfConfig(const PerfConfig& base) {
    PerfGpuConfig gpu;

    if (base.warmup <= 3) {
      gpu.gpuWarmup = 10;
    } else {
      gpu.gpuWarmup = 15;
    }

    if (base.profileArgs.find("unified") != std::string::npos) {
      gpu.memStrategy = MemoryStrategy::Unified;
    }

    return gpu;
  }
};

/* --------------------------------- API --------------------------------- */

/**
 * @brief Parse GPU-specific flags (extends parsePerfFlags).
 *
 * Call it after parsePerfFlags(), as PERF_GPU_MAIN() does; the macro then
 * makes the result the process's GPU configuration, which each GPU case
 * copies when it is built (PerfGpuCase::gpuConfig()). A missing value, or a
 * --gpu-memory value other than explicit, unified, pinned or mapped, is a
 * usage error: the parser names it on stderr and exits with status 2.
 */
inline void parseGpuFlags(PerfGpuConfig& cfg, int* argc, char** argv) {
  const auto NEED_ARG = [&](const char* name, int i) -> const char* {
    if (i + 1 >= *argc) {
      std::fprintf(stderr, "Missing value for %s\n", name);
      std::exit(2);
    }
    return argv[i + 1];
  };

  int w = 1;
  for (int i = 1; i < *argc; ++i) {
    std::string_view a = argv[i];

    if (a == "--gpu-warmup") {
      cfg.gpuWarmup = std::max(1, std::atoi(NEED_ARG("--gpu-warmup", i)));
      ++i;
    } else if (a == "--gpu-device") {
      cfg.deviceId = std::max(0, std::atoi(NEED_ARG("--gpu-device", i)));
      ++i;
    } else if (a == "--gpu-memory") {
      // A memory strategy only: no request for unified-memory counters. Any
      // other value is refused, so a mistyped strategy cannot run another.
      std::string_view strategy = NEED_ARG("--gpu-memory", i);
      if (strategy == "unified") {
        cfg.memStrategy = PerfGpuConfig::MemoryStrategy::Unified;
      } else if (strategy == "pinned") {
        cfg.memStrategy = PerfGpuConfig::MemoryStrategy::Pinned;
      } else if (strategy == "explicit") {
        cfg.memStrategy = PerfGpuConfig::MemoryStrategy::Explicit;
      } else if (strategy == "mapped") {
        cfg.memStrategy = PerfGpuConfig::MemoryStrategy::Mapped;
      } else {
        std::fprintf(stderr,
                     "--gpu-memory: '%.*s' is not a memory strategy (explicit, unified, pinned "
                     "or mapped)\n",
                     static_cast<int>(strategy.size()), strategy.data());
        std::exit(2);
      }
      ++i;
    } else if (a == "--min-speedup") {
      cfg.minSpeedupVsCpu = std::atof(NEED_ARG("--min-speedup", i));
      ++i;
    } else if (a == "--capture-um") {
      // Asks for unified-memory counters (see captureUnifiedMemory)
      cfg.captureUnifiedMemory = true;
    } else {
      argv[w++] = argv[i];
    }
  }
  *argc = w;
}

/* ------------------------------ GPU Mode Flag ------------------------------ */

/**
 * @brief Lightweight cross-cutting flag: was PERF_GPU_MAIN called in this
 *        process? Read by CPU-side components (PerfListener) that need to
 *        decide whether to emit the GPU section of the schema but can't
 *        pull PerfGpuHarness.hpp (CUDA-dependent).
 *
 * Lives here rather than in a dedicated header because PerfGpuConfig is the
 * existing CUDA-free GPU-configuration boundary already included by both
 * PERF_GPU_MAIN and PerfListener; co-locating the flag avoids a one-file
 * header just for a bool.
 */
namespace detail {

inline bool& gpuModeFlag() {
  static bool active = false;
  return active;
}

inline void markGpuMode() { gpuModeFlag() = true; }
inline bool isGpuModeActive() { return gpuModeFlag(); }

} // namespace detail

} // namespace bench
} // namespace vernier

#endif // VERNIER_PERFGPUCONFIG_HPP