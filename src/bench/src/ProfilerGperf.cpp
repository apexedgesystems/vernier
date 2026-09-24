/**
 * @file ProfilerGperf.cpp
 * @brief Implementation of gperftools profiler backend.
 */

#include "src/bench/inc/ProfilerGperf.hpp"

#include "src/bench/inc/ProfilerRegistry.hpp"

#include <unistd.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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

/** @brief The analyzers --profile-analyze may run, in the order they are tried. */
constexpr const char* ANALYZERS[] = {"google-pprof", "pprof"};

// Used only by the analysis, which exists only where CPU profiling does.
#if UB_HAS_GPERF_CPU && defined(__linux__)
/** @brief Bound on one analyzer run; symbolizing a large binary takes a while. */
constexpr int ANALYZER_TIMEOUT_MS = 120000;

/** @brief The first @p lines lines of @p text. */
std::string firstLines(const std::string& text, std::size_t lines) {
  std::istringstream in(text);
  std::string line;
  std::string out;
  for (std::size_t i = 0; i < lines && std::getline(in, line); ++i) {
    out += line + "\n";
  }
  return out;
}
#endif

} // namespace

/* ----------------------------- Readiness ----------------------------- */

GperfModes parseGperfModes(const std::string& profileArgs) {
  // Simple substring match, as the flag has always been read.
  const auto HAS = [&](const char* key) { return profileArgs.find(key) != std::string::npos; };
  GperfModes modes;
  modes.cpu = profileArgs.empty() || HAS("cpu") || HAS("both");
  modes.heap = HAS("heap") || HAS("both");
  return modes;
}

ReadinessResult checkGperfRequest(const ReadinessRequest& request, const ReadinessContext& ctx) {
  auto plan = std::make_shared<GperfPlan>();
  plan->modes = parseGperfModes(request.profileArgs);
  plan->analyze = request.analyze;

  constexpr bool CPU_BUILT = UB_HAS_GPERF_CPU != 0;
  constexpr bool HEAP_BUILT = UB_HAS_GPERF_HEAP != 0;
  if (!CPU_BUILT && !HEAP_BUILT) {
    return readinessResult(ReadinessCause::MISSING,
                           "gperftools headers were not present when libbench was built",
                           "apt install libgperftools-dev (or equivalent), then rebuild.");
  }
  if (plan->modes.heap && !HEAP_BUILT) {
    return readinessResult(ReadinessCause::UNSUPPORTED,
                           "heap profiling is not compiled in: it needs tcmalloc, which replaces "
                           "the process allocator and is therefore opt-in",
                           "Reconfigure with -DVERNIER_LINK_TCMALLOC=ON and rebuild; CPU "
                           "profiling (--profile-args cpu) needs no change.");
  }
  if (plan->modes.cpu && !CPU_BUILT) {
    return readinessResult(ReadinessCause::MISSING,
                           "CPU profiling is not compiled in (gperftools/profiler.h was absent)",
                           "apt install libgperftools-dev (or equivalent), then rebuild.");
  }

  std::string modes;
  if (plan->modes.cpu) {
    modes += "cpu";
  }
  if (plan->modes.heap) {
    modes += modes.empty() ? "heap" : " and heap";
  }

  // The analyzer: the first candidate found on PATH, and nothing else.
  std::string notExecutable;
  for (const char* NAME : ANALYZERS) {
    const auto FOUND = resolveExecutable(NAME, ctx);
    if (FOUND && FOUND->executable) {
      plan->analyzer = FOUND->path;
      break;
    }
    if (FOUND && notExecutable.empty()) {
      notExecutable = FOUND->path;
    }
  }

  ReadinessResult result;
  if (plan->analyze && plan->modes.cpu && plan->analyzer.empty()) {
    // A promised analysis cannot run; the capture still can, and is kept.
    result = readinessResult(
        ReadinessCause::MISSING,
        "--profile-analyze needs google-pprof or pprof, and neither is on PATH" +
            (notExecutable.empty() ? std::string{}
                                   : " as an executable (" + notExecutable + " is not one)") +
            "; the " + modes + " capture still runs and cpu.prof is kept",
        "Install an analyzer (google-pprof from gperftools, or Go's pprof), or drop "
        "--profile-analyze.",
        ReadinessStage::ANALYSIS);
  } else {
    std::string message =
        "gperftools profiles " + modes + (HEAP_BUILT ? " (built: cpu, heap)" : " (built: cpu)");
    if (plan->analyze && plan->modes.cpu) {
      message += "; --profile-analyze runs " + plan->analyzer;
    } else if (!plan->analyzer.empty()) {
      message += "; analyzer " + plan->analyzer;
    } else {
      message += "; no analyzer on PATH (only --profile-analyze needs one)";
    }
    result = readinessResult(ReadinessCause::READY, std::move(message), "");
  }
  result.plan = std::move(plan);
  return result;
}

/* ----------------------------- GperfProfiler Methods ----------------------------- */

namespace {

std::shared_ptr<const GperfPlan> readyPlan(const ReadinessResult& result) {
  if (!result.collectionReady()) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<const GperfPlan>(result.plan);
}

ReadinessResult decideNow(const PerfConfig& cfg) {
  const ReadinessContext CTX = ReadinessContext::capture();
  ReadinessRequest request = readinessRequestFor(cfg, ReadinessScope::RUNTIME, CTX);
  request.backend = "gperf";
  return checkGperfRequest(request, CTX);
}

} // namespace

GperfProfiler::GperfProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  // A profiler built for a test owns that test's folder, ready or not.
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "gperf");
  const ReadinessResult DECISION = decideNow(cfg_);
  plan_ = readyPlan(DECISION);
  if (!plan_) {
    std::fprintf(stderr, "[gperf] not started: %s\n", DECISION.report.message.c_str());
    if (!DECISION.report.hint.empty()) {
      std::fprintf(stderr, "[gperf] %s\n", DECISION.report.hint.c_str());
    }
    return;
  }
  applyPlan();
}

GperfProfiler::GperfProfiler(const PerfConfig& cfg, std::string testName,
                             std::shared_ptr<const GperfPlan> plan)
    : cfg_(cfg), testName_(std::move(testName)), plan_(std::move(plan)) {
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "gperf");
  if (plan_) {
    applyPlan();
  }
}

void GperfProfiler::applyPlan() {
  // The check refused modes that are not compiled in; the guards keep a
  // hand-made plan from naming one.
  wantCpu_ = plan_->modes.cpu && UB_HAS_GPERF_CPU != 0;
  wantHeap_ = plan_->modes.heap && UB_HAS_GPERF_HEAP != 0;

  // DWARF v5 warning (once per process)
  static std::atomic<bool> warned{false};
  if (wantCpu_ && detectDwarfV5Warning() && !warned.exchange(true)) {
    std::fprintf(stderr,
                 "\n[WARN] gperftools: Clang DWARF v5 detected. Sample attribution may be "
                 "inaccurate (20-50%% misattribution).\n"
                 "   Fix: compile with -gdwarf-4 -fno-omit-frame-pointer\n"
                 "   Also recommended: setarch -R (disable ASLR) for consistent addresses\n\n");
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

    // Auto-analyze: run the analyzer the check found and print top functions
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
// Only ever called from the UB_HAS_GPERF_CPU branch of afterMeasure();
// the body must compile out with it because cpuPath_ exists only there.
#if UB_HAS_GPERF_CPU && defined(__linux__)
  if (!plan_ || plan_->analyzer.empty()) {
    // Reported when the profiler was created: no analyzer to run.
    std::fprintf(stderr,
                 "[gperf] analysis skipped: no google-pprof or pprof; raw profile kept at %s\n",
                 cpuPath_.c_str());
    return;
  }

  // The binary to symbolize: this process's own executable.
  std::array<char, 4096> exePath{};
  ssize_t len = ::readlink("/proc/self/exe", exePath.data(), exePath.size() - 1);
  if (len <= 0) {
    std::fprintf(stderr, "[WARN] --profile-analyze: Could not determine binary path\n");
    return;
  }
  exePath[static_cast<std::size_t>(len)] = '\0';
  const std::string EXE{exePath.data()};
  const ReadinessContext CTX = ReadinessContext::capture();

  // Runs one view with the resolved analyzer, directly (no shell, no head),
  // prints its first lines, and reports a failure without touching the raw
  // profile. Returns false on failure.
  const auto VIEW = [&](const std::vector<std::string>& args, std::size_t lines) {
    std::vector<std::string> argv{plan_->analyzer};
    argv.insert(argv.end(), args.begin(), args.end());
    argv.push_back(EXE);
    argv.push_back(cpuPath_);
    const ProbeResult RUN = runBoundedProbe(argv, ANALYZER_TIMEOUT_MS, CTX, ProbeStreams::SEPARATE);
    if (!RUN.succeeded()) {
      const std::string TAIL = outputTail(RUN.errorOutput);
      std::fprintf(stderr, "[gperf] %s failed: %s%s%s; raw profile kept at %s\n",
                   plan_->analyzer.c_str(), RUN.describe().c_str(), TAIL.empty() ? "" : ": ",
                   TAIL.c_str(), cpuPath_.c_str());
      return false;
    }
    const std::string REPORT = firstLines(RUN.output, lines);
    // google-pprof prints nothing at all for a profile without samples.
    std::fputs(REPORT.empty() ? "(the analyzer printed no report; a very short run may hold no "
                                "samples)\n"
                              : REPORT.c_str(),
               stdout);
    return true;
  };

  std::printf("\n=== gperftools Auto-Analysis (top 15 by cumulative) ===\n");
  std::printf("Profile: %s\nAnalyzer: %s\n\n", cpuPath_.c_str(), plan_->analyzer.c_str());
  // Cumulative view: the most useful for finding hotspots.
  if (!VIEW({"--text", "--cum", "--lines"}, 20)) {
    return;
  }
  std::printf("\n--- Self time (top 10) ---\n\n");
  (void)VIEW({"--text", "--lines"}, 15);
  std::printf("\n");
  std::fflush(stdout);
#endif
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeGperfProfiler(const PerfConfig& cfg, const std::string& testName) {
  auto plan = readyPlan(decideNow(cfg));
  if (!plan) {
    return nullptr; // not compiled in, or an unsupported mode -> the factory falls back to no-op
  }
  return std::make_unique<GperfProfiler>(cfg, testName, std::move(plan));
}

namespace {

std::unique_ptr<Profiler> makePlannedGperfProfiler(const PerfConfig& cfg,
                                                   const std::string& testName,
                                                   const ReadinessResult& result) {
  auto plan = readyPlan(result);
  if (!plan) {
    return nullptr;
  }
  return std::make_unique<GperfProfiler>(cfg, testName, std::move(plan));
}

} // namespace

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_READINESS_BACKEND("gperf", ::vernier::bench::checkGperfRequest,
                                   ::vernier::bench::makePlannedGperfProfiler,
                                   "Install libgperftools-dev and rebuild.")
