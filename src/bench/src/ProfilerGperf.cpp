/**
 * @file ProfilerGperf.cpp
 * @brief Implementation of gperftools profiler backend.
 */

#include "src/bench/inc/ProfilerGperf.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

// Only include gperftools headers if available
#if UB_HAS_GPERF_CPU
#include <gperftools/profiler.h>
#endif

#if UB_HAS_GPERF_HEAP
#include <gperftools/heap-profiler.h>
#endif

namespace vernier {
namespace bench {

/* ----------------------------- Constants ----------------------------- */

namespace {

/**
 * @brief Detect if the current binary was compiled with DWARF v5.
 *
 * gperftools uses libunwind for stack unwinding, which has known issues with
 * DWARF v5 debug info (clang-14+ defaults to v5). Symptoms: 20-50% of samples
 * misattributed to wrong functions.
 *
 * Workaround: compile with -gdwarf-4 -fno-omit-frame-pointer
 */
bool detectDwarfV5Warning() {
  // Clang 14+ defaults to DWARF v5; GCC 11+ uses DWARF v5 with -gdwarf-5
  // We can only detect at compile time whether we're likely to have the issue
#if defined(__clang_major__) && __clang_major__ >= 14
#if !defined(__DWARF_VERSION__) || __DWARF_VERSION__ >= 5
  return true;
#endif
#endif
  return false;
}

} // namespace

/* ----------------------------- GperfProfiler Methods ----------------------------- */

GperfProfiler::GperfProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  // Resolve artifact directory: <artifactRoot>/<Suite.Case>.gperf/
  if (!cfg_.artifactRoot.empty()) {
    artifactDir_ = cfg_.artifactRoot + "/" + testName_ + ".gperf";
  } else {
    artifactDir_ = "./" + testName_ + ".gperf";
  }
  std::error_code ec;
  std::filesystem::create_directories(artifactDir_, ec);

  // Parse mode from profileArgs (simple substring contains)
  std::string args = cfg_.profileArgs;
  auto containsKey = [&](const char* k) {
    return args.find(k) != std::string::npos ||
           args.find(std::string(k) + "=1") != std::string::npos;
  };

  wantCpu_ = (args.empty() || containsKey("cpu") || containsKey("both"));
  wantHeap_ = (containsKey("heap") || containsKey("both"));

#if !UB_HAS_GPERF_CPU
  wantCpu_ = false;
#endif
#if !UB_HAS_GPERF_HEAP
  wantHeap_ = false;
#endif

  // DWARF v5 warning (once per process)
  static bool warned = false;
  if (!warned && wantCpu_ && detectDwarfV5Warning()) {
    std::fprintf(stderr,
                 "\n[WARN] gperftools: Clang DWARF v5 detected. Sample attribution may be "
                 "inaccurate (20-50%% misattribution).\n"
                 "   Fix: compile with -gdwarf-4 -fno-omit-frame-pointer\n"
                 "   Also recommended: setarch -R (disable ASLR) for consistent addresses\n\n");
    warned = true;
  }
}

void GperfProfiler::beforeMeasure() {
#if UB_HAS_GPERF_CPU || UB_HAS_GPERF_HEAP
  // Set sampling frequency before starting the profiler.
  // gperftools reads CPUPROFILE_FREQUENCY at ProfilerStart() time.
  // Default (100 Hz) is too coarse for sub-millisecond operations.
  if (wantCpu_ && cfg_.profileFrequency > 0) {
    std::string freq = std::to_string(cfg_.profileFrequency);
    ::setenv("CPUPROFILE_FREQUENCY", freq.c_str(), /*overwrite=*/1);
  }

  if (wantHeap_) {
#if UB_HAS_GPERF_HEAP
    // HeapProfilerStart uses a prefix (it creates <prefix>.<N>.heap files)
    heapPrefix_ = artifactDir_ + "/heap";
    HeapProfilerStart(heapPrefix_.c_str());
#endif
  }
  if (wantCpu_) {
#if UB_HAS_GPERF_CPU
    cpuPath_ = artifactDir_ + "/cpu.prof";
    ProfilerStart(cpuPath_.c_str());
#endif
  }
#endif
}

void GperfProfiler::afterMeasure(const Stats& /*s*/) {
#if UB_HAS_GPERF_CPU || UB_HAS_GPERF_HEAP
  if (wantCpu_) {
#if UB_HAS_GPERF_CPU
    ProfilerFlush();
    ProfilerStop();

    // Auto-analyze: run pprof and print top functions
    if (cfg_.profileAnalyze && !cpuPath_.empty()) {
      runPprofAnalysis();
    }
#endif
  }
  if (wantHeap_) {
#if UB_HAS_GPERF_HEAP
    // Emit a final snapshot (optional), then stop.
    HeapProfilerDump("final");
    HeapProfilerStop();
#endif
  }
#endif
}

void GperfProfiler::runPprofAnalysis() const {
#ifdef __linux__
  // Check if pprof/google-pprof is available
  bool hasPprof = (std::system("command -v google-pprof >/dev/null 2>&1") == 0);
  if (!hasPprof) {
    hasPprof = (std::system("command -v pprof >/dev/null 2>&1") == 0);
  }

  if (!hasPprof) {
    std::fprintf(stderr,
                 "\n[INFO] --profile-analyze: google-pprof not found. Install gperftools.\n"
                 "   Profile saved to: %s\n"
                 "   Manual analysis: google-pprof --text <binary> %s\n\n",
                 cpuPath_.c_str(), cpuPath_.c_str());
    return;
  }

  // Get the path to our own binary from /proc/self/exe
  std::array<char, 4096> exePath{};
  ssize_t len = ::readlink("/proc/self/exe", exePath.data(), exePath.size() - 1);
  if (len <= 0) {
    std::fprintf(stderr, "[WARN] --profile-analyze: Could not determine binary path\n");
    return;
  }
  exePath[static_cast<std::size_t>(len)] = '\0';

  std::printf("\n=== gperftools Auto-Analysis (top 15 by cumulative) ===\n");
  std::printf("Profile: %s\n\n", cpuPath_.c_str());

  // Run pprof --text --cum (cumulative view, most useful for finding hotspots)
  std::string cmd = "google-pprof --text --cum --lines '" + std::string(exePath.data()) + "' '" +
                    cpuPath_ + "' 2>/dev/null | head -20";
  [[maybe_unused]] int rc = std::system(cmd.c_str());

  std::printf("\n--- Self time (top 10) ---\n\n");

  cmd = "google-pprof --text --lines '" + std::string(exePath.data()) + "' '" + cpuPath_ +
        "' 2>/dev/null | head -15";
  rc = std::system(cmd.c_str());

  std::printf("\n");
#endif
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeGperfProfiler(const PerfConfig& cfg, const std::string& testName) {
#if UB_HAS_GPERF_CPU || UB_HAS_GPERF_HEAP
  // Only create if at least one mode is compiled in
  return std::make_unique<GperfProfiler>(cfg, testName);
#else
  (void)cfg;
  (void)testName;
  return std::unique_ptr<Profiler>{}; // unsupported at build time -> let factory fall back to no-op
#endif
}

} // namespace bench
} // namespace vernier
