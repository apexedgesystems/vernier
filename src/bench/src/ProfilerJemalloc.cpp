/**
 * @file ProfilerJemalloc.cpp
 * @brief jemalloc heap-profiler backend implementation.
 *
 * jemalloc profiles a process when it is that process's allocator, built with
 * profiling and told by MALLOC_CONF to profile; it writes its dump when the
 * process exits. The doctor answers for a run by preloading jemalloc into one
 * probe program with the options `bench run` sets; a run asks its own
 * allocator through mallctl and starts nothing. Neither announces a dump
 * before the process has exited: `bench run` checks it after the exit.
 */

#include "src/bench/inc/ProfilerJemalloc.hpp"

#include <dlfcn.h>
#if defined(__GLIBC__)
#include <link.h> // link_map, for dladdr1
#endif
#include <sys/types.h>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/inc/ValgrindTool.hpp"

namespace vernier {
namespace bench {

namespace {

/// jemalloc is preloaded through the Linux loader's LD_PRELOAD.
#if defined(__linux__)
constexpr bool LINUX_LOADER = true;
#else
constexpr bool LINUX_LOADER = false;
#endif

/// The probe program and what it is asked: it prints its own memory map,
/// which names the library the loader preloaded.
constexpr const char* ENV_PROGRAM = "/usr/bin/env";
constexpr const char* PROBE_PROGRAM = "/bin/cat";
constexpr const char* PROBE_ARGUMENT = "/proc/self/maps";

/// The options `bench run` sets, before the dumps' prefix.
constexpr const char* RUN_OPTIONS = "prof:true,prof_final:true,prof_accum:true";

/// What jemalloc prints for an option a build without profiling rejects.
constexpr const char* PROFILING_REJECTED = "Invalid conf pair: prof:true";

/// What ld.so prints before its reason when it cannot preload a library.
constexpr std::string_view PRELOAD_REFUSED = "cannot be preloaded (";

/// The remedy of a jemalloc that cannot profile, or of none.
constexpr const char* PROFILING_REMEDY =
    "Use a jemalloc built with profiling (configure --enable-prof; Debian's and Ubuntu's "
    "libjemalloc2 have it on amd64 and i386, not on arm64), with its library folder first on "
    "LD_LIBRARY_PATH when it is not the system's; or profile the heap with --profile heaptrack.";

/// The remedy of a run whose allocator is not jemalloc.
constexpr const char* WRAP_REMEDY =
    "Wrap it: LD_PRELOAD=libjemalloc.so.2 "
    "MALLOC_CONF=prof:true,prof_final:true,prof_accum:true,prof_prefix:./jeprof <this-binary> "
    "--profile jemalloc [...]; or run it with bench run --profile jemalloc, which wraps it.";

/// The remedy of options that do not profile, or not fully.
constexpr const char* OPTIONS_REMEDY =
    "Set MALLOC_CONF=prof:true,prof_final:true,prof_accum:true,prof_prefix:./jeprof, as bench "
    "run --profile jemalloc does.";

/// The probe as reports name it.
std::string probeText() {
  return std::string{ENV_PROGRAM} + " LD_PRELOAD=" + JEMALLOC_PRELOAD +
         " MALLOC_CONF=" + RUN_OPTIONS + ",prof_prefix:<private directory>/jeprof " +
         PROBE_PROGRAM + " " + PROBE_ARGUMENT;
}

/// The probe as a user runs it by hand.
std::string probeByHand() {
  return std::string{"LD_PRELOAD="} + JEMALLOC_PRELOAD + " MALLOC_CONF=" + RUN_OPTIONS +
         ",prof_prefix:/tmp/jeprof " + PROBE_PROGRAM + " " + PROBE_ARGUMENT;
}

/** @brief The memory map of process @p pid, or "" when it cannot be read. */
std::string mapsOf(pid_t pid) {
  std::ifstream maps("/proc/" + std::to_string(static_cast<long>(pid)) + "/maps");
  std::stringstream text;
  text << maps.rdbuf();
  return text.str();
}

/**
 * @brief The path of the first libjemalloc a memory map holds, "" when it
 * holds none. A map line's path starts at its first '/': the address, flags,
 * offset, device and inode fields before it hold none.
 */
std::string mappedJemalloc(const std::string& mapsText) {
  std::istringstream lines(mapsText);
  std::string line;
  while (std::getline(lines, line)) {
    const std::size_t SLASH = line.find('/');
    if (SLASH == std::string::npos) {
      continue;
    }
    const std::string PATH = line.substr(SLASH);
    const std::size_t NAME = PATH.rfind('/') + 1;
    if (PATH.compare(NAME, 11, "libjemalloc") == 0) {
      return PATH;
    }
  }
  return {};
}

/** @brief The first line of @p text that contains @p needle, trimmed; "" when none does. */
std::string lineWith(const std::string& text, const std::string& needle) {
  std::istringstream lines(text);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.find(needle) != std::string::npos) {
      while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
        line.pop_back();
      }
      return line;
    }
  }
  return {};
}

/** @brief True for a final dump's name: jeprof.<pid>.<n>.f.heap. */
bool finalDumpName(const std::string& name) {
  const std::string HEAD = "jeprof.";
  const std::string TAIL = ".f.heap";
  if (name.size() <= HEAD.size() + TAIL.size() || name.compare(0, HEAD.size(), HEAD) != 0 ||
      name.compare(name.size() - TAIL.size(), TAIL.size(), TAIL) != 0) {
    return false;
  }
  const std::string MIDDLE = name.substr(HEAD.size(), name.size() - HEAD.size() - TAIL.size());
  const std::size_t DOT = MIDDLE.find('.');
  if (DOT == std::string::npos || DOT == 0 || DOT + 1 == MIDDLE.size()) {
    return false;
  }
  for (std::size_t i = 0; i < MIDDLE.size(); ++i) {
    if (i != DOT && (MIDDLE[i] < '0' || MIDDLE[i] > '9')) {
      return false;
    }
  }
  return true;
}

/** @brief The files in @p dir, each with its first line. */
std::vector<JemallocDump> dumpsIn(const std::string& dir) {
  std::vector<JemallocDump> dumps;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    JemallocDump dump;
    dump.name = entry.path().filename().string();
    std::ifstream in(entry.path());
    std::getline(in, dump.firstLine);
    dumps.push_back(std::move(dump));
  }
  return dumps;
}

/** @brief "<library>" when one is named, else "the jemalloc the loader preloads". */
std::string libraryOrLoader(const std::string& library) {
  return library.empty() ? std::string{"the jemalloc the loader preloads"} : library;
}

/** @brief The analyzer, beside a verdict that collects: it reads dumps and decides nothing. */
std::string analyzerNote(const ReadinessContext& ctx) {
  const auto JEPROF = resolveExecutable("jeprof", ctx);
  if (JEPROF && JEPROF->executable) {
    return "; analyzer " + JEPROF->path;
  }
  return "; jeprof is not on PATH (only reading the dumps needs it)";
}

/* ----------------------------- Doctor ----------------------------- */

/** @brief The doctor's verdict from what its probe observed. */
ReadinessResult probeVerdict(const JemallocProbeFacts& probe, const ReadinessContext& ctx,
                             JemallocPlan& plan) {
  const ProbeResult& RUN = probe.run;
  const std::string TAIL = outputTail(RUN.errorOutput);
  const std::string ENDED = "jemalloc's probe, " + probeText() + ": " + RUN.describe() +
                            (TAIL.empty() ? "" : ": " + TAIL);
  if (!RUN.started) {
    return readinessResult(ReadinessCause::UNUSABLE, ENDED,
                           "Run the probe by hand to see why: " + probeByHand());
  }
  if (RUN.endUnobserved()) {
    // Its end was not read here: neither a success nor a failure.
    return readinessResult(ReadinessCause::UNUSABLE, ENDED, CHILD_STATUS_REMEDY);
  }
  // The loader's refusal: ld.so says why and runs the program without it.
  const std::string REFUSAL = lineWith(RUN.errorOutput, "cannot be preloaded");
  if (!REFUSAL.empty()) {
    const std::size_t OPEN = REFUSAL.find(PRELOAD_REFUSED);
    const std::size_t FROM = OPEN == std::string::npos ? OPEN : OPEN + PRELOAD_REFUSED.size();
    const std::size_t CLOSE = REFUSAL.rfind(')');
    const std::string WHY = FROM != std::string::npos && CLOSE != std::string::npos && CLOSE > FROM
                                ? REFUSAL.substr(FROM, CLOSE - FROM)
                                : REFUSAL;
    return readinessResult(ReadinessCause::MISSING,
                           std::string{"the loader cannot preload "} + JEMALLOC_PRELOAD + " (" +
                               WHY + ")",
                           PROFILING_REMEDY);
  }
  plan.library = mappedJemalloc(RUN.output);
  const std::string REJECTION = lineWith(RUN.errorOutput, PROFILING_REJECTED);
  if (!REJECTION.empty()) {
    return readinessResult(ReadinessCause::MISSING,
                           libraryOrLoader(plan.library) +
                               " is built without profiling: preloaded with bench run's options, "
                               "it rejects them (\"" +
                               REJECTION + "\")",
                           PROFILING_REMEDY);
  }
  if (!RUN.succeeded()) {
    return readinessResult(ReadinessCause::UNUSABLE, ENDED,
                           "Run the probe by hand to see why: " + probeByHand());
  }
  const std::string COMPLAINT = lineWith(RUN.errorOutput, "<jemalloc>:");
  if (!COMPLAINT.empty()) {
    return readinessResult(ReadinessCause::UNUSABLE,
                           libraryOrLoader(plan.library) +
                               " does not take bench run's options: \"" + COMPLAINT + "\"",
                           "Run the probe by hand to see why: " + probeByHand());
  }
  if (plan.library.empty()) {
    return readinessResult(
        ReadinessCause::UNUSABLE,
        std::string{"the loader preloaded "} + JEMALLOC_PRELOAD +
            " without a word, but no libjemalloc is mapped in its probe (" + probeText() + ")",
        "Run the probe by hand to see what the loader preloads: " + probeByHand());
  }
  for (const JemallocDump& DUMP : probe.dumps) {
    if (!finalDumpName(DUMP.name)) {
      continue;
    }
    if (DUMP.firstLine.compare(0, 8, "heap_v2/") != 0) {
      return readinessResult(ReadinessCause::UNUSABLE,
                             plan.library + " wrote " + DUMP.name +
                                 " at its probe's exit, which is not a heap profile (it starts '" +
                                 DUMP.firstLine.substr(0, 40) + "')",
                             "Run the probe by hand to see what it writes: " + probeByHand());
    }
    return readinessResult(ReadinessCause::READY,
                           plan.library +
                               " profiles: preloaded with bench run's options, it wrote its final "
                               "heap dump at exit (probe: " +
                               PROBE_PROGRAM + " " + PROBE_ARGUMENT + ")" + analyzerNote(ctx),
                           "");
  }
  return readinessResult(ReadinessCause::UNUSABLE,
                         plan.library +
                             " takes bench run's options but wrote no final heap dump when its "
                             "probe exited (" +
                             probeText() + ")",
                         "Run the probe by hand to see why: " + probeByHand());
}

/** @brief The doctor's probe: one preloaded run of the probe program, bounded. */
JemallocProbeFacts runProbe(const ReadinessContext& ctx, std::optional<ReadinessResult>& failed) {
  JemallocProbeFacts facts;
  for (const char* HELPER : {ENV_PROGRAM, PROBE_PROGRAM}) {
    const auto FOUND = resolveExecutable(HELPER, ctx);
    if (!FOUND || !FOUND->executable) {
      failed = readinessResult(ReadinessCause::MISSING_HELPER,
                               std::string{HELPER} + ", which jemalloc's probe runs, is missing",
                               std::string{"Restore "} + HELPER + " (coreutils).");
      return facts;
    }
  }
  const ProbeScratch SCRATCH(ctx);
  if (!SCRATCH.ok()) {
    failed = readinessResult(ReadinessCause::INTERNAL,
                             "no private directory could be created for jemalloc's probe",
                             "Check that TMPDIR (or /tmp) is writable.");
    return facts;
  }
  if (SCRATCH.path().find(',') != std::string::npos) {
    failed = readinessResult(ReadinessCause::INTERNAL,
                             "the private directory of jemalloc's probe, " + SCRATCH.path() +
                                 ", holds ',', which ends an option in MALLOC_CONF",
                             "Set TMPDIR to a writable directory whose path holds no ','.");
    return facts;
  }
  std::vector<std::string> argv{ENV_PROGRAM};
  for (const auto& [NAME, VALUE] : jemallocWrapEnvironment(SCRATCH.path())) {
    argv.push_back(NAME + "=" + VALUE);
  }
  argv.emplace_back(PROBE_PROGRAM);
  argv.emplace_back(PROBE_ARGUMENT);
  facts.run = runBoundedProbe(argv, JEMALLOC_PROBE_TIMEOUT_MS, ctx, ProbeStreams::SEPARATE);
  facts.dumps = dumpsIn(SCRATCH.path());
  return facts;
}

/* ----------------------------- Runtime ----------------------------- */

/**
 * @brief True when the undefined-behaviour sanitizer's runtime is part of the
 * program itself (clang links it so), not a shared library (GCC's libubsan):
 * its handler resolves into the main program's link map, whose name is empty.
 * Measured with clang 21: there a preloaded jemalloc 5.3.0 does not take
 * MALLOC_CONF, where GCC 13's shared runtime leaves it alone.
 */
bool ubsanInTheProgram() {
#if defined(__GLIBC__)
  void* const HANDLER = ::dlsym(RTLD_DEFAULT, "__ubsan_handle_add_overflow");
  if (HANDLER == nullptr) {
    return false;
  }
  Dl_info info{};
  link_map* map = nullptr;
  return ::dladdr1(HANDLER, &info, reinterpret_cast<void**>(&map), RTLD_DL_LINKMAP) != 0 &&
         map != nullptr && map->l_name != nullptr && map->l_name[0] == '\0';
#else
  return false;
#endif
}

/// jemalloc's mallctl, as its header declares it.
using Mallctl = int (*)(const char*, void*, std::size_t*, void*, std::size_t);

/** @brief Read @p name into @p out; false (and @p out untouched) when mallctl refuses. */
template <typename T> bool readMallctl(Mallctl mallctl, const char* name, T& out) {
  T value{};
  std::size_t size = sizeof(value);
  if (mallctl(name, &value, &size, nullptr, 0) != 0 || size != sizeof(value)) {
    return false;
  }
  out = value;
  return true;
}

/** @brief True when @p prefix lies in the folder @p wrapDir names. */
bool insideWrapFolder(const std::string& prefix, const std::optional<std::string>& wrapDir) {
  if (!wrapDir || wrapDir->empty()) {
    return false;
  }
  const std::string FOLDER = wrapDir->back() == '/' ? *wrapDir : *wrapDir + "/";
  return prefix.compare(0, FOLDER.size(), FOLDER) == 0;
}

/** @brief The library and version a runtime verdict names. */
std::string whichJemalloc(const JemallocRuntimeFacts& facts) {
  const std::string VERSION = facts.version.empty() ? std::string{} : " " + facts.version;
  if (facts.library.empty()) {
    return "jemalloc" + VERSION + ", linked into this program";
  }
  return facts.library + (facts.version.empty() ? std::string{} : " (" + facts.version + ")");
}

/** @brief A run's verdict from what its own allocator says. */
ReadinessResult runtimeVerdict(const ReadinessRequest& request, const ReadinessContext& ctx,
                               const JemallocRuntimeFacts& facts, JemallocPlan& plan) {
  plan.library = facts.library;
  if (!facts.jemalloc) {
    plan.launch = LaunchContext::NOT_WRAPPED;
    const auto PRELOAD = ctx.get("LD_PRELOAD");
    if (PRELOAD && PRELOAD->find("jemalloc") != std::string::npos) {
      return readinessResult(ReadinessCause::MISSING,
                             "LD_PRELOAD names " + *PRELOAD +
                                 ", but jemalloc is not this process's allocator: the loader did "
                                 "not preload it (it said why when this program started)",
                             PROFILING_REMEDY);
    }
    return readinessResult(ReadinessCause::MISSING,
                           "jemalloc collects only as the process's allocator, and this process "
                           "does not run on jemalloc",
                           WRAP_REMEDY);
  }
  const std::string WHICH = whichJemalloc(facts);
  if (!facts.configProf) {
    return readinessResult(ReadinessCause::MISSING,
                           "the jemalloc this process runs on, " + WHICH +
                               ", is built without profiling",
                           PROFILING_REMEDY);
  }
  if (!facts.optProf) {
    const auto CONF = ctx.get("MALLOC_CONF");
    if (CONF && CONF->find("prof:true") != std::string::npos) {
      // Asked for, and off: jemalloc read its options without MALLOC_CONF.
      return readinessResult(ReadinessCause::UNUSABLE,
                             "jemalloc is this process's allocator (" + WHICH +
                                 "), but it did not take MALLOC_CONF: MALLOC_CONF holds prof:true, "
                                 "and jemalloc's profiling is off",
                             "jemalloc reads MALLOC_CONF once, when it starts: set it before the "
                             "program starts, as bench run --profile jemalloc does, and profile a "
                             "build without -fsanitize, whose runtime can keep jemalloc from "
                             "reading it.");
    }
    return readinessResult(ReadinessCause::CONFIGURATION,
                           "jemalloc is this process's allocator (" + WHICH +
                               "), but its profiling is off: MALLOC_CONF has no prof:true",
                           OPTIONS_REMEDY);
  }
  const bool PERIODIC = facts.lgProfInterval >= 0 || facts.optProfGdump;
  if (!facts.optProfFinal && !PERIODIC) {
    return readinessResult(ReadinessCause::CONFIGURATION,
                           "jemalloc profiles this process (" + WHICH +
                               ") but writes no dump: MALLOC_CONF has no prof_final:true, and no "
                               "interval or growth dumps",
                           OPTIONS_REMEDY);
  }
  plan.prefix = facts.prefix;
  plan.finalDump = facts.optProfFinal;
  const bool BY_RUNNER = request.launch == LaunchContext::RUNNER_WRAPPED &&
                         insideWrapFolder(facts.prefix, ctx.get("VERNIER_EXTERNAL_WRAP_DIR"));
  plan.launch = BY_RUNNER ? LaunchContext::RUNNER_WRAPPED : LaunchContext::MANUALLY_WRAPPED;

  std::vector<std::string> limits;
  if (!facts.optProfFinal) {
    limits.emplace_back(
        "no dump at exit (prof_final is off): only its interval or growth dumps are written");
  }
  if (!facts.profActive) {
    limits.emplace_back(
        "nothing is sampled (prof.active is off) until the program turns sampling on");
  }
  if (!facts.optProfAccum) {
    limits.emplace_back("its dumps hold the bytes in use when they are written, not those "
                        "allocated over the run (prof_accum is off), which jeprof --alloc_space "
                        "reads");
  }
  if (!limits.empty()) {
    std::string message = "jemalloc profiles this process (" + WHICH + "), with limits: ";
    for (std::size_t i = 0; i < limits.size(); ++i) {
      message += (i == 0 ? "" : "; ") + limits[i];
    }
    return readinessResult(ReadinessCause::CAVEAT, std::move(message), OPTIONS_REMEDY);
  }
  return readinessResult(
      ReadinessCause::READY,
      "jemalloc profiles this process (" + WHICH + "), under " +
          (BY_RUNNER ? std::string{"bench run's wrap"} : std::string{"a wrap started by hand"}),
      "");
}

/* ----------------------------- Decision ----------------------------- */

ReadinessResult decideJemalloc(const ReadinessRequest& request, const ReadinessContext& ctx,
                               const JemallocFacts& facts) {
  for (const std::string& WORD : valgrind_tool::modeWords(request.profileArgs)) {
    return valgrind_tool::refusedWord("jemalloc", WORD, {});
  }
  auto plan = std::make_shared<JemallocPlan>();
  ReadinessResult result;
  JemallocRuntimeFacts live;
  const bool NEEDS_LIVE = LINUX_LOADER && facts.runtime == nullptr;
  if (NEEDS_LIVE) {
    live = jemallocRuntimeFacts();
  }
  const JemallocRuntimeFacts& RUNTIME = facts.runtime == nullptr ? live : *facts.runtime;
  if (!LINUX_LOADER) {
    result = readinessResult(ReadinessCause::UNSUPPORTED,
                             "jemalloc is preloaded through the Linux loader (LD_PRELOAD), which "
                             "this system does not have",
                             "Profile the heap on Linux.");
  } else if (RUNTIME.sanitizer) {
    result = readinessResult(ReadinessCause::UNSUPPORTED,
                             "jemalloc does not profile a program built with the address or "
                             "thread sanitizer: the sanitizer's allocator serves its allocations",
                             "Profile a build without -fsanitize=address or -fsanitize=thread.");
  } else if (RUNTIME.ubsanLinkedIn) {
    result = readinessResult(ReadinessCause::UNSUPPORTED,
                             "jemalloc does not profile a program built with clang's "
                             "undefined-behaviour sanitizer: with its runtime linked into the "
                             "program, a preloaded jemalloc does not take MALLOC_CONF, so its "
                             "profiling stays off",
                             "Profile a build without -fsanitize=undefined.");
  } else if (request.scope == ReadinessScope::RUNTIME) {
    result = runtimeVerdict(request, ctx, RUNTIME, *plan);
  } else if (facts.probe != nullptr) {
    result = probeVerdict(*facts.probe, ctx, *plan);
  } else {
    std::optional<ReadinessResult> failed;
    const JemallocProbeFacts PROBE = runProbe(ctx, failed);
    result = failed ? *failed : probeVerdict(PROBE, ctx, *plan);
  }
  result.plan = std::move(plan);
  if (!request.analyze) {
    return result;
  }
  return valgrind_tool::withAnalysis(
      std::move(result),
      readinessResult(ReadinessCause::UNSUPPORTED,
                      "--profile-analyze: jemalloc has no automatic analysis; the capture still "
                      "runs and its dump is kept",
                      "Read the dump with jeprof --text --alloc_space <this-binary> <dump> after "
                      "the process exits, and drop --profile-analyze.",
                      ReadinessStage::ANALYSIS));
}

std::shared_ptr<const JemallocPlan> readyPlan(const ReadinessResult& result) {
  if (!result.collectionReady()) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<const JemallocPlan>(result.plan);
}

ReadinessResult decideNow(const PerfConfig& cfg) {
  const ReadinessContext CTX = ReadinessContext::capture();
  ReadinessRequest request = readinessRequestFor(cfg, ReadinessScope::RUNTIME, CTX);
  request.backend = "jemalloc";
  return checkJemallocRequest(request, CTX);
}

} // namespace

/* ----------------------------- Facts ----------------------------- */

JemallocRuntimeFacts jemallocRuntimeFacts() {
  JemallocRuntimeFacts facts;
  // The sanitizers' runtimes export their initializers; their allocator is
  // in the program itself, ahead of any preload.
  facts.sanitizer = ::dlsym(RTLD_DEFAULT, "__asan_init") != nullptr ||
                    ::dlsym(RTLD_DEFAULT, "__tsan_init") != nullptr;
  facts.ubsanLinkedIn = ubsanInTheProgram();
  facts.library = mappedJemalloc(mapsOf(::getpid()));
  void* const SYMBOL = ::dlsym(RTLD_DEFAULT, "mallctl");
  if (SYMBOL == nullptr) {
    return facts;
  }
  facts.jemalloc = true;
  const auto MALLCTL = reinterpret_cast<Mallctl>(SYMBOL);
  const char* version = nullptr;
  if (readMallctl(MALLCTL, "version", version) && version != nullptr) {
    facts.version = version;
  }
  static_cast<void>(readMallctl(MALLCTL, "config.prof", facts.configProf));
  if (!facts.configProf) {
    return facts;
  }
  static_cast<void>(readMallctl(MALLCTL, "opt.prof", facts.optProf));
  static_cast<void>(readMallctl(MALLCTL, "prof.active", facts.profActive));
  static_cast<void>(readMallctl(MALLCTL, "opt.prof_final", facts.optProfFinal));
  static_cast<void>(readMallctl(MALLCTL, "opt.prof_accum", facts.optProfAccum));
  static_cast<void>(readMallctl(MALLCTL, "opt.prof_gdump", facts.optProfGdump));
  ssize_t interval = -1;
  if (readMallctl(MALLCTL, "opt.lg_prof_interval", interval)) {
    facts.lgProfInterval = static_cast<long>(interval);
  }
  const char* prefix = nullptr;
  if (readMallctl(MALLCTL, "opt.prof_prefix", prefix) && prefix != nullptr) {
    facts.prefix = prefix;
  }
  return facts;
}

/* ----------------------------- Check ----------------------------- */

std::vector<std::pair<std::string, std::string>> jemallocWrapEnvironment(const std::string& dir) {
  return {{"LD_PRELOAD", JEMALLOC_PRELOAD},
          {"MALLOC_CONF", std::string{RUN_OPTIONS} + ",prof_prefix:" + dir + "/jeprof"}};
}

ReadinessResult checkJemallocRequest(const ReadinessRequest& request, const ReadinessContext& ctx) {
  return decideJemalloc(request, ctx, JemallocFacts{});
}

ReadinessResult checkJemallocRequestWithFacts(const ReadinessRequest& request,
                                              const ReadinessContext& ctx,
                                              const JemallocFacts& facts) {
  return decideJemalloc(request, ctx, facts);
}

/* ----------------------------- JemallocProfiler ----------------------------- */

JemallocProfiler::JemallocProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  // Decided first and reported as the registry reports its own decisions; a
  // request that cannot collect creates nothing.
  const ReadinessResult DECISION = decideNow(cfg_);
  if (ProfilerRegistry::instance().reportDecision("jemalloc", DECISION)) {
    plan_ = readyPlan(DECISION);
    artifactDir_ = profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_,
                                                    "jemalloc");
  }
}

JemallocProfiler::JemallocProfiler(const PerfConfig& cfg, std::string testName,
                                   std::shared_ptr<const JemallocPlan> plan)
    : cfg_(cfg), testName_(std::move(testName)), plan_(std::move(plan)) {
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "jemalloc");
}

void JemallocProfiler::beforeMeasure() {
  if (!plan_) {
    return;
  }
  // What jemalloc will do; nothing is written before the process exits.
  const long PID = static_cast<long>(::getpid());
  if (!plan_->finalDump) {
    std::fprintf(stderr,
                 "[jemalloc] jemalloc profiles this process and writes interval or growth\n"
                 "[jemalloc] dumps as %s.%ld.<n>.<i|u>.heap; none when it exits.\n",
                 plan_->prefix.c_str(), PID);
    return;
  }
  std::fprintf(stderr,
               "[jemalloc] jemalloc profiles this process and writes its heap dump when the\n"
               "[jemalloc] process exits, as %s.%ld.<n>.f.heap.\n",
               plan_->prefix.c_str(), PID);
  if (plan_->launch == LaunchContext::RUNNER_WRAPPED) {
    std::fprintf(stderr, "[jemalloc] bench run checks that dump after the exit.\n");
  } else {
    std::fprintf(stderr, "[jemalloc] This run does not check that dump; bench run --profile "
                         "jemalloc does.\n");
  }
}

void JemallocProfiler::afterMeasure(const Stats& /*s*/) {
  // jemalloc writes its dump at process exit; nothing to do per measurement.
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeJemallocProfiler(const PerfConfig& cfg, const std::string& testName) {
  return std::make_unique<JemallocProfiler>(cfg, testName);
}

namespace {

std::unique_ptr<Profiler> makePlannedJemallocProfiler(const PerfConfig& cfg,
                                                      const std::string& testName,
                                                      const ReadinessResult& result) {
  auto plan = readyPlan(result);
  if (!plan) {
    return nullptr;
  }
  return std::make_unique<JemallocProfiler>(cfg, testName, std::move(plan));
}

} // namespace

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_READINESS_BACKEND(
    "jemalloc", ::vernier::bench::checkJemallocRequest,
    ::vernier::bench::makePlannedJemallocProfiler,
    "Use a jemalloc built with profiling (configure --enable-prof), preloaded: bench run "
    "--profile jemalloc wraps the benchmark.",
    "LD_LIBRARY_PATH", "LD_PRELOAD", "MALLOC_CONF")
