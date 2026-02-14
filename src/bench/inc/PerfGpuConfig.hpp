#ifndef VERNIER_PERFGPUCONFIG_HPP
#define VERNIER_PERFGPUCONFIG_HPP
/**
 * @file PerfGpuConfig.hpp
 * @brief GPU-specific configuration extensions with Unified Memory profiling support.
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
  MemoryStrategy memStrategy = MemoryStrategy::Explicit;

  // ---- Profiling options ----
  bool captureOccupancy = true;
  bool captureClockSpeeds = true;
  bool captureMemoryBandwidth = true;

  // Unified Memory profiling (opt-in)
  bool captureUnifiedMemory = false; ///< Track UM page faults and migrations

  // ---- Device selection ----
  int deviceId = 0;

  // ---- Stream configuration ----
  bool useHighPriorityStream = true;

  // ---- Validation thresholds ----
  double minSpeedupVsCpu = 1.0;
  double maxTransferOverhead = 50.0;

  // Parse GPU-specific flags from PerfConfig
  static PerfGpuConfig fromPerfConfig(const PerfConfig& base) {
    PerfGpuConfig gpu;

    if (base.warmup <= 3) {
      gpu.gpuWarmup = 10;
    } else {
      gpu.gpuWarmup = 15;
    }

    // Detect unified memory request from profile args
    if (base.profileArgs.find("unified") != std::string::npos) {
      gpu.memStrategy = MemoryStrategy::Unified;
      gpu.captureUnifiedMemory = true;
    }

    return gpu;
  }
};

/* --------------------------------- API --------------------------------- */

/**
 * @brief Parse GPU-specific flags (extends parsePerfFlags).
 *
 * NOTE: This function should be called after parsePerfFlags() in main().
 * GPU tests typically use fromPerfConfig() to derive GPU settings from the
 * base PerfConfig, so this function is only needed for standalone GPU binaries.
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
      std::string_view strategy = NEED_ARG("--gpu-memory", i);
      if (strategy == "unified") {
        cfg.memStrategy = PerfGpuConfig::MemoryStrategy::Unified;
        cfg.captureUnifiedMemory = true; // Auto-enable UM tracking
      } else if (strategy == "pinned") {
        cfg.memStrategy = PerfGpuConfig::MemoryStrategy::Pinned;
      } else if (strategy == "explicit") {
        cfg.memStrategy = PerfGpuConfig::MemoryStrategy::Explicit;
      } else if (strategy == "mapped") {
        cfg.memStrategy = PerfGpuConfig::MemoryStrategy::Mapped;
      }
      ++i;
    } else if (a == "--min-speedup") {
      cfg.minSpeedupVsCpu = std::atof(NEED_ARG("--min-speedup", i));
      ++i;
    } else if (a == "--capture-um") {
      // Enable Unified Memory profiling (page faults, migrations)
      cfg.captureUnifiedMemory = true;
    } else {
      argv[w++] = argv[i];
    }
  }
  *argc = w;
}

} // namespace bench
} // namespace vernier

#endif // VERNIER_PERFGPUCONFIG_HPP