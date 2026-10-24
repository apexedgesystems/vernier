/**
 * @file ProfilerGperf.cpp
 * @brief Implementation of gperftools profiler backend.
 */

#include "src/bench/inc/ProfilerGperf.hpp"

#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/inc/ValgrindTool.hpp"

#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
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

/** @brief Bound on the analyzer's --help probe. */
constexpr int ANALYZER_PROBE_TIMEOUT_MS = 10000;

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

/** @brief What the capture holds, for messages: "cpu", "heap" or "cpu and heap". */
std::string capturedModes(const GperfModes& modes) {
  if (modes.cpu && modes.heap) {
    return "cpu and heap";
  }
  return modes.cpu ? "cpu" : "heap";
}

/** @brief The rate gperftools samples at when CPUPROFILE_FREQUENCY is not set, in Hz. */
constexpr int GPERF_DEFAULT_SAMPLING_HZ = 100;

constexpr int MICROSECONDS_PER_SECOND = 1000000;

/** @brief gperftools' sampling period for a rate, in microseconds, rounded down as it rounds. */
constexpr int periodUsFor(int hz) { return MICROSECONDS_PER_SECOND / hz; }

/// The sampling periods gperftools writes, those of its highest rate and of 1 Hz.
constexpr long long SHORTEST_PERIOD_US = periodUsFor(GPERF_MAX_SAMPLING_HZ);
constexpr long long LONGEST_PERIOD_US = periodUsFor(1);

/** @brief This program as a command line names it: its argv[0] as one shell word. */
std::string programWord() {
#if defined(__GLIBC__) && defined(_GNU_SOURCE)
  if (program_invocation_name != nullptr && *program_invocation_name != '\0') {
    return detail::shellQuote(program_invocation_name);
  }
#endif
  return "<this-binary>";
}

// Used only by the check of a finished profile, which exists only where CPU
// profiling does.
#if UB_HAS_GPERF_CPU
/** @brief Where a capture's rate came from, for messages: "1000 Hz requested". */
std::string rateSourceText(GperfRateSource source, int hz) {
  switch (source) {
  case GperfRateSource::REQUESTED:
    return std::to_string(hz) + " Hz requested";
  case GperfRateSource::INHERITED:
    return std::string{SAMPLING_RATE_VARIABLE} + "=" + std::to_string(hz) + " from the environment";
  case GperfRateSource::DEFAULT:
    break;
  }
  return "gperftools' default " + std::to_string(GPERF_DEFAULT_SAMPLING_HZ) + " Hz";
}
#endif

} // namespace

/* ----------------------------- Readiness ----------------------------- */

std::optional<ReadinessResult> parseGperfModes(const std::string& profileArgs, GperfModes& modes) {
  modes = GperfModes{};
  const std::vector<std::string> WORDS = valgrind_tool::modeWords(profileArgs);
  for (const std::string& WORD : WORDS) {
    const bool CPU = WORD == "cpu" || WORD == "both";
    const bool HEAP = WORD == "heap" || WORD == "both";
    if (!CPU && !HEAP) {
      modes = GperfModes{};
      return valgrind_tool::refusedWord("gperf", WORD, {"cpu", "heap", "both"});
    }
    modes.cpu = modes.cpu || CPU;
    modes.heap = modes.heap || HEAP;
  }
  if (WORDS.empty()) {
    modes.cpu = true;
  }
  return std::nullopt;
}

ReadinessResult checkGperfRequest(const ReadinessRequest& request, const ReadinessContext& ctx) {
  auto plan = std::make_shared<GperfPlan>();
  if (auto refused = parseGperfModes(request.profileArgs, plan->modes)) {
    return *refused;
  }
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

  // The rate the CPU capture samples at; a rate it cannot take collects nothing.
  const GperfSampling SAMPLING = decideGperfSampling(plan->modes, request.profileFrequency, ctx);
  if (SAMPLING.error) {
    return *SAMPLING.error;
  }
  plan->expectedPeriodUs = SAMPLING.expectedPeriodUs;
  plan->samplingHz = SAMPLING.hz;
  plan->samplingSource = SAMPLING.source;

  const std::string MODES = capturedModes(plan->modes);
  const GperfAnalysis ANALYSIS = decideGperfAnalysis(plan->modes, plan->analyze, ctx);
  plan->analyzer = ANALYSIS.analyzer;
  plan->analysisReady = plan->analyze && plan->modes.cpu && !ANALYSIS.error;

  ReadinessResult result;
  if (ANALYSIS.error) {
    // The promised analysis cannot run; the capture still can, and is kept.
    result = *ANALYSIS.error;
    plan->analysisSkipped = plan->analyzer.empty() ? std::string{"no google-pprof or pprof on PATH"}
                                                   : plan->analyzer + " does not run";
  } else {
    std::string message =
        "gperftools profiles " + MODES + (HEAP_BUILT ? " (built: cpu, heap)" : " (built: cpu)");
    if (plan->analysisReady) {
      message += "; --profile-analyze runs " + plan->analyzer;
    } else if (!plan->analyzer.empty()) {
      message += "; analyzer " + plan->analyzer;
    } else {
      message += "; no analyzer on PATH (only --profile-analyze needs one)";
    }
    if (!SAMPLING.clause.empty()) {
      message += "; " + SAMPLING.clause;
    }
    result = readinessResult(ReadinessCause::READY, std::move(message), "");
  }
  result.plan = std::move(plan);
  return result;
}

GperfAnalysis decideGperfAnalysis(const GperfModes& modes, bool analyze,
                                  const ReadinessContext& ctx) {
  GperfAnalysis analysis;
  // The analyzer: the first candidate found on PATH, and nothing else.
  std::string notExecutable;
  for (const char* NAME : ANALYZERS) {
    const auto FOUND = resolveExecutable(NAME, ctx);
    if (FOUND && FOUND->executable) {
      analysis.analyzer = FOUND->path;
      break;
    }
    if (FOUND && notExecutable.empty()) {
      notExecutable = FOUND->path;
    }
  }
  if (!analyze || !modes.cpu) {
    return analysis; // no analysis promised: the analyzer is only named
  }
  const std::string KEPT =
      "; the " + capturedModes(modes) + " capture still runs and cpu.prof is kept";
  if (analysis.analyzer.empty()) {
    analysis.error = readinessResult(
        ReadinessCause::MISSING,
        "--profile-analyze needs google-pprof or pprof, and neither is on PATH" +
            (notExecutable.empty() ? std::string{}
                                   : " as an executable (" + notExecutable + " is not one)") +
            KEPT,
        "Install an analyzer (google-pprof from gperftools, or Go's pprof), or drop "
        "--profile-analyze.",
        ReadinessStage::ANALYSIS);
    return analysis;
  }
  // The analyzer runs: both google-pprof and Go's pprof answer --help.
  const ProbeResult HELP =
      runBoundedProbe({analysis.analyzer, "--help"}, ANALYZER_PROBE_TIMEOUT_MS, ctx);
  if (HELP.endUnobserved()) {
    analysis.error = readinessResult(ReadinessCause::UNUSABLE,
                                     "--profile-analyze would run " + analysis.analyzer +
                                         ", whose --help " + HELP.describe() + KEPT,
                                     CHILD_STATUS_REMEDY, ReadinessStage::ANALYSIS);
  } else if (!HELP.succeeded()) {
    const std::string TAIL = outputTail(HELP.output);
    analysis.error = readinessResult(
        ReadinessCause::UNUSABLE,
        "--profile-analyze would run " + analysis.analyzer + ", which does not run: --help " +
            HELP.describe() + (TAIL.empty() ? std::string{} : ": " + TAIL) + KEPT,
        "Repair or reinstall that analyzer, put a working google-pprof or pprof first on PATH, "
        "or drop --profile-analyze.",
        ReadinessStage::ANALYSIS);
  }
  return analysis;
}

GperfSampling decideGperfSampling(const GperfModes& modes, int requestedHz,
                                  const ReadinessContext& ctx) {
  GperfSampling sampling;
  const std::string HZ = std::to_string(requestedHz);
  const std::string MAX = std::to_string(GPERF_MAX_SAMPLING_HZ);
  // Only a source value can be out of range: the command line refuses one.
  if (requestedHz != 0 && (requestedHz < 1 || requestedHz > GPERF_MAX_SAMPLING_HZ)) {
    sampling.error = readinessResult(
        ReadinessCause::CONFIGURATION,
        "a sampling rate of " + HZ +
            " Hz was asked for (--profile-frequency or PerfConfig::profileFrequency): "
            "gperftools samples at 1 to " +
            MAX + " Hz",
        "Ask for a whole number from 1 to " + MAX +
            ", or leave PerfConfig::profileFrequency at 0 for gperftools' default " +
            std::to_string(GPERF_DEFAULT_SAMPLING_HZ) + " Hz.");
    return sampling;
  }
  if (!modes.cpu) {
    // Heap profiling alone samples nothing, so the variable is not read.
    if (requestedHz != 0) {
      sampling.error = readinessResult(
          ReadinessCause::CONFIGURATION,
          "--profile-frequency " + HZ +
              " sets the CPU sampling rate, and --profile-args heap profiles the heap only",
          "Profile the CPU too (--profile-args both), or drop --profile-frequency.");
    }
    return sampling;
  }

  // gperftools read the variable once, as this process started; the snapshot
  // stands for that environment.
  const std::string VARIABLE = SAMPLING_RATE_VARIABLE;
  const std::optional<std::string> SET = ctx.get(VARIABLE);
  const std::optional<int> SET_HZ = SET ? parseProfileFrequency(*SET) : std::nullopt;
  if (requestedHz == 0) {
    if (!SET) {
      sampling.expectedPeriodUs = periodUsFor(GPERF_DEFAULT_SAMPLING_HZ);
      return sampling;
    }
    if (!SET_HZ) {
      sampling.error = readinessResult(
          ReadinessCause::CONFIGURATION,
          VARIABLE + "='" + *SET + "' in the environment is not a whole number of Hz from 1 to " +
              MAX + ": gperftools reads such a value without saying what rate it takes from it",
          "Unset it, or set it to a whole number from 1 to " + MAX +
              " before the benchmark starts.");
      return sampling;
    }
    sampling.hz = *SET_HZ;
    sampling.source = GperfRateSource::INHERITED;
    sampling.expectedPeriodUs = periodUsFor(*SET_HZ);
    sampling.clause = "sampling at " + VARIABLE + "=" + std::to_string(*SET_HZ) +
                      " from the environment: expected period " +
                      std::to_string(*sampling.expectedPeriodUs) +
                      " us, checked in each finished profile";
    return sampling;
  }
  if (SET_HZ && *SET_HZ == requestedHz) {
    sampling.hz = requestedHz;
    sampling.source = GperfRateSource::REQUESTED;
    sampling.expectedPeriodUs = periodUsFor(requestedHz);
    sampling.clause = HZ + " Hz requested; the current environment matches the request (" +
                      VARIABLE + "=" + HZ + "); expected period " +
                      std::to_string(*sampling.expectedPeriodUs) +
                      " us, checked in each finished profile";
    return sampling;
  }
  const std::string REQUEST = std::string{"--profile gperf"} +
                              (modes.heap ? " --profile-args both" : "") + " --profile-frequency " +
                              HZ;
  const std::string PROGRAM = programWord();
  sampling.error =
      readinessResult(ReadinessCause::CONFIGURATION,
                      HZ + " Hz requested, and this process's environment has " +
                          (SET ? VARIABLE + "='" + *SET + "'" : "no " + VARIABLE) +
                          ": gperftools reads the rate only when the process starts",
                      "Run it through bench run " + PROGRAM + " " + REQUEST + ", which sets " +
                          VARIABLE + " before the benchmark starts, or start it as " + VARIABLE +
                          "=" + HZ + " " + PROGRAM + " " + REQUEST + ".");
  return sampling;
}

GperfProfileSummary readGperfProfile(const std::string& path) {
  GperfProfileSummary summary;
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    summary.problem = "it cannot be opened";
    return summary;
  }
  const std::string BYTES{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  constexpr std::size_t SLOT_BYTES = sizeof(std::uintptr_t);
  const std::size_t SLOTS = BYTES.size() / SLOT_BYTES;
  const auto SLOT = [&BYTES](std::size_t index) {
    std::uintptr_t value = 0;
    std::memcpy(&value, BYTES.data() + index * SLOT_BYTES, SLOT_BYTES);
    return value;
  };
  const auto BYTE = [](std::size_t index) { return std::to_string(index * SLOT_BYTES); };

  // The header: 0, n (at least 3), the version 0, the period, padding to n.
  if (SLOTS < 5) {
    summary.problem = "its " + std::to_string(BYTES.size()) + " bytes hold no header (5 slots of " +
                      std::to_string(SLOT_BYTES) + " bytes)";
    return summary;
  }
  const std::uintptr_t HEADER_SLOTS = SLOT(1);
  if (SLOT(0) != 0 || HEADER_SLOTS < 3 || SLOT(2) != 0) {
    summary.problem = "its header starts " + std::to_string(SLOT(0)) + "," +
                      std::to_string(HEADER_SLOTS) + "," + std::to_string(SLOT(2)) +
                      " where gperftools writes 0, then 3 or more, then 0";
    return summary;
  }
  const std::uintptr_t PERIOD = SLOT(3);
  if (PERIOD < static_cast<std::uintptr_t>(SHORTEST_PERIOD_US) ||
      PERIOD > static_cast<std::uintptr_t>(LONGEST_PERIOD_US)) {
    summary.problem = "its header records a sampling period of " + std::to_string(PERIOD) +
                      " us, outside the " + std::to_string(SHORTEST_PERIOD_US) + " to " +
                      std::to_string(LONGEST_PERIOD_US) + " us gperftools writes";
    return summary;
  }
  summary.periodUs = static_cast<long long>(PERIOD);
  if (HEADER_SLOTS > SLOTS - 2) {
    summary.problem = "its header of " + std::to_string(HEADER_SLOTS + 2) +
                      " slots runs past the end of the file";
    return summary;
  }

  // The records, each a count, a depth and that many program counters, up to
  // the trailer 0, 1, 0. Every step is bounded by the slots the file holds.
  for (std::size_t at = 2 + HEADER_SLOTS;;) {
    if (SLOTS - at < 2) {
      summary.problem =
          "it ends at byte " + std::to_string(BYTES.size()) + " with no trailer (0, 1, 0)";
      return summary;
    }
    const std::uintptr_t COUNT = SLOT(at);
    const std::uintptr_t DEPTH = SLOT(at + 1);
    if (DEPTH > SLOTS - at - 2) {
      summary.problem = "the record at byte " + BYTE(at) + " runs past the end of the file";
      return summary;
    }
    if (COUNT == 0) {
      if (DEPTH == 1 && SLOT(at + 2) == 0) {
        summary.binaryBytes = static_cast<long long>((at + 3) * SLOT_BYTES);
        summary.valid = true;
        return summary;
      }
      summary.problem =
          "the record at byte " + BYTE(at) + " counts no samples and is not the trailer (0, 1, 0)";
      return summary;
    }
    if (DEPTH == 0) {
      summary.problem = "the record at byte " + BYTE(at) + " holds no program counters";
      return summary;
    }
    summary.samples += COUNT;
    at += 2 + DEPTH;
  }
}

/* ----------------------------- GperfProfiler Methods ----------------------------- */

namespace {

#if UB_HAS_GPERF_CPU || UB_HAS_GPERF_HEAP
/** @brief The size of @p path, or -1 when there is no such file. */
long long fileSize(const std::string& path) {
  std::error_code ec;
  const auto SIZE = std::filesystem::file_size(path, ec);
  return ec ? -1 : static_cast<long long>(SIZE);
}

/**
 * @brief Remove each of @p paths that exists; a missing one is fine.
 * @return "" when none is left, or "<path>: <why>" for the first that could
 * not be removed (the rest are left as they are).
 */
std::string removeFiles(const std::vector<std::string>& paths) {
  for (const std::string& PATH : paths) {
    std::error_code ec;
    std::filesystem::remove(PATH, ec);
    if (ec) {
      return PATH + ": " + ec.message();
    }
  }
  return {};
}
#endif

#if UB_HAS_GPERF_HEAP
/** @brief The heap dumps in @p dir named by HeapProfilerStart's @p prefix: <prefix>.<N>.heap. */
std::vector<std::string> heapDumps(const std::string& dir, const std::string& prefix) {
  std::vector<std::string> dumps;
  const std::string STEM = std::filesystem::path(prefix).filename().string() + ".";
  std::error_code ec;
  for (const auto& ENTRY : std::filesystem::directory_iterator(dir, ec)) {
    const std::string NAME = ENTRY.path().filename().string();
    if (NAME.size() > STEM.size() + 5 && NAME.compare(0, STEM.size(), STEM) == 0 &&
        NAME.compare(NAME.size() - 5, 5, ".heap") == 0) {
      dumps.push_back(ENTRY.path().string());
    }
  }
  return dumps;
}
#endif

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
  // Reported as the registry reports its own decisions; a rejected request
  // leaves no folder behind.
  const ReadinessResult DECISION = decideNow(cfg_);
  if (!ProfilerRegistry::instance().reportDecision("gperf", DECISION)) {
    return;
  }
  plan_ = readyPlan(DECISION);
  if (!plan_) {
    return;
  }
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "gperf");
  applyPlan();
}

GperfProfiler::GperfProfiler(const PerfConfig& cfg, std::string testName,
                             std::shared_ptr<const GperfPlan> plan)
    : cfg_(cfg), testName_(std::move(testName)), plan_(std::move(plan)) {
  if (plan_) {
    artifactDir_ =
        profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "gperf");
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

GperfProfiler::~GperfProfiler() { stopCapture(); }

void GperfProfiler::beforeMeasure() {
#if UB_HAS_GPERF_CPU || UB_HAS_GPERF_HEAP
  // gperftools runs one CPU profile and one heap profile per process. A
  // capture this profiler started for an earlier window is still on when
  // that window ended through neither afterMeasure() nor abortMeasure()
  // (hooks attached without the abort hook): it is this profiler's, so it is
  // stopped first, unanalyzed. A start gperftools then refuses is another
  // profiler's capture running, which is this case's failure and is left
  // alone: a capture is stopped only by the profiler that started it. A
  // previous run's files are removed first, so what is there afterwards is
  // this run's; a capture whose previous files cannot all be removed does
  // not start, since what it wrote could not be told from them.
  stopCapture();
  if (wantHeap_) {
#if UB_HAS_GPERF_HEAP
    // HeapProfilerStart uses a prefix (it creates <prefix>.<N>.heap files)
    heapPrefix_ = artifactDir_ + "/heap";
    if (IsHeapProfilerRunning() != 0) {
      fail(ReadinessCause::UNUSABLE,
           "a gperftools heap profile already runs in this process, so this case's could not "
           "start; its heap is not profiled",
           "Unset HEAPPROFILE (it starts a heap profile at launch), and profile one case at a "
           "time.",
           ReadinessStage::COLLECTION);
    } else if (const std::string LEFT = removeFiles(heapDumps(artifactDir_, heapPrefix_));
               !LEFT.empty()) {
      fail(ReadinessCause::UNUSABLE,
           "the previous run's heap dumps could not be cleared (" + LEFT +
               "), so this case's could not be told from them; its heap is not profiled",
           "Remove them, or give --profile-output-dir a folder this user can write.",
           ReadinessStage::COLLECTION);
    } else {
      HeapProfilerStart(heapPrefix_.c_str());
      heapActive_ = IsHeapProfilerRunning() != 0;
      if (!heapActive_) {
        fail(ReadinessCause::UNUSABLE,
             "gperftools did not start a heap profile into " + heapPrefix_ +
                 "; this case's heap is not profiled",
             "", ReadinessStage::COLLECTION);
      }
    }
#endif
  }
  if (wantCpu_) {
#if UB_HAS_GPERF_CPU
    cpuPath_ = artifactDir_ + "/cpu.prof";
    if (const std::string LEFT = removeFiles({cpuPath_}); !LEFT.empty()) {
      fail(ReadinessCause::UNUSABLE,
           "the previous run's CPU profile could not be cleared (" + LEFT +
               "), so this case's could not be told from it; this case is not profiled",
           "Remove it, or give --profile-output-dir a folder this user can write.",
           ReadinessStage::COLLECTION);
    } else {
      cpuActive_ = ProfilerStart(cpuPath_.c_str()) != 0;
      if (!cpuActive_) {
        fail(ReadinessCause::UNUSABLE,
             "gperftools did not start a CPU profile into " + cpuPath_ +
                 " (ProfilerStart returned 0): another one already runs in this process, or "
                 "the file cannot be written; this case is not profiled",
             "Unset CPUPROFILE (it starts a profile at launch), profile one case at a time, "
             "and give --profile-output-dir a folder this user can write.",
             ReadinessStage::COLLECTION);
      }
    }
#endif
  }
#endif
}

void GperfProfiler::afterMeasure(const Stats& /*s*/) {
#if UB_HAS_GPERF_CPU
  if (cpuActive_) {
    ProfilerFlush();
    ProfilerStop();
    cpuActive_ = false;

    // The capture is this run's file, whether or not an analysis follows;
    // only a file that holds a complete profile is analyzed.
    const long long SIZE = fileSize(cpuPath_);
    if (SIZE <= 0) {
      fail(SIZE < 0 ? ReadinessCause::MISSING : ReadinessCause::UNUSABLE,
           cpuPath_ + (SIZE < 0 ? " was not written" : " is empty") +
               ": gperftools stopped the CPU profile and left no capture there",
           "", ReadinessStage::COMPLETION);
    } else if (checkCapture() && cfg_.profileAnalyze) {
      // Auto-analyze: run the analyzer the check found and print top functions
      runPprofAnalysis();
    }
  }
#endif
#if UB_HAS_GPERF_HEAP
  if (heapActive_) {
    // Emit a final snapshot (optional), then stop.
    HeapProfilerDump("final");
    HeapProfilerStop();
    heapActive_ = false;

    bool written = false;
    for (const std::string& DUMP : heapDumps(artifactDir_, heapPrefix_)) {
      written = written || fileSize(DUMP) > 0;
    }
    if (!written) {
      fail(ReadinessCause::MISSING,
           "no heap dump (" + heapPrefix_ +
               ".NNNN.heap) was written: gperftools stopped the heap profile and left none",
           "", ReadinessStage::COMPLETION);
    }
  }
#endif
}

void GperfProfiler::abortMeasure() noexcept { stopCapture(); }

bool GperfProfiler::checkCapture() const {
// Only ever called from the UB_HAS_GPERF_CPU branch of afterMeasure().
#if UB_HAS_GPERF_CPU
  const GperfProfileSummary PROFILE = readGperfProfile(cpuPath_);
  if (!PROFILE.valid) {
    fail(ReadinessCause::UNUSABLE,
         cpuPath_ + " is not a complete gperftools CPU profile: " + PROFILE.problem +
             "; raw profile kept",
         "", ReadinessStage::COMPLETION);
    return false;
  }
  if (!plan_ || !plan_->expectedPeriodUs) {
    return true; // a plan made by hand expects no period: the format alone is checked
  }
  // The period gperftools configured for this capture, as its finished file
  // records it: the rate the process started with, not the rate taken, which
  // the kernel's timer tick may hold lower. A request for exactly 100 Hz
  // records the period of no request, 10000 us, so a request lost at that rate
  // passes this comparison.
  const std::string SOURCE = rateSourceText(plan_->samplingSource, plan_->samplingHz);
  if (PROFILE.periodUs != *plan_->expectedPeriodUs) {
    fail(ReadinessCause::UNUSABLE,
         cpuPath_ + " records a period of " + std::to_string(PROFILE.periodUs) + " us (" +
             std::to_string(MICROSECONDS_PER_SECOND / PROFILE.periodUs) + " Hz) where " +
             std::to_string(*plan_->expectedPeriodUs) + " us was expected for " + SOURCE +
             ": gperftools started with another rate; raw profile kept",
         "Set " + std::string{SAMPLING_RATE_VARIABLE} +
             " before the benchmark starts (bench run --profile-frequency does), not from inside "
             "it.",
         ReadinessStage::COMPLETION);
    return true; // a complete profile, at another rate: still analyzed
  }
  if (plan_->samplingSource != GperfRateSource::DEFAULT) {
    std::fprintf(stderr, "[gperf] %s: cpu.prof records a period of %lld us (%s), %llu sample%s\n",
                 testName_.c_str(), PROFILE.periodUs, SOURCE.c_str(), PROFILE.samples,
                 PROFILE.samples == 1 ? "" : "s");
  }
  return true;
#else
  return false;
#endif
}

void GperfProfiler::fail(ReadinessCause cause, const std::string& detail, const std::string& remedy,
                         ReadinessStage stage) const {
  ProfilerRegistry::instance().reportFailure("gperf", testName_,
                                             readinessResult(cause, detail, remedy, stage));
}

void GperfProfiler::stopCapture() noexcept {
#if UB_HAS_GPERF_CPU
  if (cpuActive_) {
    ProfilerStop();
  }
#endif
#if UB_HAS_GPERF_HEAP
  if (heapActive_) {
    HeapProfilerStop();
  }
#endif
  cpuActive_ = false;
  heapActive_ = false;
}

void GperfProfiler::runPprofAnalysis() const {
// Only ever called from the UB_HAS_GPERF_CPU branch of afterMeasure();
// the body must compile out with it because cpuPath_ exists only there.
#if UB_HAS_GPERF_CPU && defined(__linux__)
  if (!plan_ || !plan_->analysisReady) {
    // Reported when the profiler was created: the analysis cannot run.
    const std::string WHY = (plan_ && !plan_->analysisSkipped.empty())
                                ? plan_->analysisSkipped
                                : std::string{"no analyzer was selected"};
    std::fprintf(stderr, "[gperf] analysis skipped: %s; raw profile kept at %s\n", WHY.c_str(),
                 cpuPath_.c_str());
    return;
  }

  // The binary to symbolize: this process's own executable.
  std::array<char, 4096> exePath{};
  ssize_t len = ::readlink("/proc/self/exe", exePath.data(), exePath.size() - 1);
  if (len <= 0) {
    const std::string WHY = len < 0 ? std::strerror(errno) : "empty link";
    fail(ReadinessCause::MISSING,
         "the program to symbolize, /proc/self/exe, could not be read: " + WHY +
             "; raw profile kept at " + cpuPath_,
         "Analyze it by hand: " + plan_->analyzer + " --text <this program> " + cpuPath_,
         ReadinessStage::ANALYSIS);
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
    if (auto unseen = unobservedProbe(plan_->analyzer + " on the profile (raw profile kept at " +
                                          cpuPath_ + ")",
                                      RUN, ReadinessStage::ANALYSIS)) {
      ProfilerRegistry::instance().reportFailure("gperf", testName_, *unseen);
      return false;
    }
    if (!RUN.succeeded()) {
      const std::string TAIL = outputTail(RUN.errorOutput);
      std::string command;
      for (const std::string& part : argv) {
        command += (command.empty() ? "" : " ") + part;
      }
      fail(ReadinessCause::UNUSABLE,
           plan_->analyzer + " failed on the profile: " + RUN.describe() +
               (TAIL.empty() ? "" : ": " + TAIL) + "; raw profile kept at " + cpuPath_,
           "Run it by hand to see why: " + command, ReadinessStage::ANALYSIS);
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
  const ReadinessResult DECISION = decideNow(cfg);
  auto plan = ProfilerRegistry::instance().reportDecision("gperf", DECISION) ? readyPlan(DECISION)
                                                                             : nullptr;
  if (!plan) {
    return nullptr; // not compiled in, or an unsupported mode: reported, and the run fails
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

// gperf takes a sampling rate: its check reads the variable gperftools takes
// the rate from at start, so a change to it is a new decision.
VERNIER_REGISTER_READINESS_BACKEND("gperf", ::vernier::bench::checkGperfRequest,
                                   ::vernier::bench::makePlannedGperfProfiler,
                                   "Install libgperftools-dev and rebuild.",
                                   ::vernier::bench::SAMPLING_RATE_VARIABLE)
