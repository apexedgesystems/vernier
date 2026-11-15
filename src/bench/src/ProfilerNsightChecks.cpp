/**
 * @file ProfilerNsightChecks.cpp
 * @brief The Nsight backend's mode parser and readiness check, and libbench's
 * default registration of `nsight` and `ncu` with a passive profiler.
 */

#include "src/bench/inc/ProfilerNsightChecks.hpp"

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/bench/inc/Profiler.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/inc/ValgrindTool.hpp"

namespace vernier {
namespace bench {

namespace {

/** @brief How long the doctor's `<tool> --version` may run. */
constexpr int VERSION_PROBE_TIMEOUT_MS = 10000;

/** @brief The driver parameter file whose RmProfilingAdminOnly the ncu rows read. */
constexpr const char* DRIVER_PARAMS = "/proc/driver/nvidia/params";

std::string join(const std::vector<std::string>& parts, const std::string& separator) {
  std::string out;
  for (const std::string& PART : parts) {
    out += (out.empty() ? "" : separator) + PART;
  }
  return out;
}

/** @brief "--profile <backend>[ --profile-args <value>]", the value whole, as one shell word. */
std::string requestText(const std::string& backend, const std::string& profileArgs) {
  const std::size_t FIRST = profileArgs.find_first_not_of(" \t\n,");
  if (FIRST == std::string::npos) {
    return "--profile " + backend;
  }
  const std::size_t LAST = profileArgs.find_last_not_of(" \t\n,");
  return "--profile " + backend + " --profile-args " +
         detail::shellQuote(profileArgs.substr(FIRST, LAST - FIRST + 1));
}

/** @brief ncu's options for @p selection, each as one shell word, each after a space. */
std::string ncuOptionsText(const NsightSelection& selection) {
  std::string out;
  for (const std::string& OPTION : selection.ncuOptions()) {
    out += " " + detail::shellQuote(OPTION);
  }
  return out;
}

/** @brief "; the request collects <what>" for a request that names what, else "". */
std::string collectsNote(const NsightSelection& selection) {
  const std::string WHAT = selection.describe();
  return WHAT.empty() ? std::string{} : "; the request collects " + WHAT;
}

/** @brief The line of a `--version` output that names the version, else its last line. */
std::string versionLine(const std::string& output) {
  std::istringstream in(output);
  std::string line;
  while (std::getline(in, line)) {
    if (line.find("ersion") != std::string::npos) {
      return outputTail(line, 1);
    }
  }
  return outputTail(output, 1);
}

/** @brief True when @p driverParams says `RmProfilingAdminOnly: 1`. */
bool countersRootOnly(const std::string& driverParams) {
  std::ifstream in(driverParams);
  std::string line;
  constexpr std::string_view KEY = "RmProfilingAdminOnly:";
  while (std::getline(in, line)) {
    if (line.rfind(KEY, 0) == 0) {
      const std::size_t VALUE = line.find_first_not_of(" \t", KEY.size());
      return VALUE != std::string::npos && line[VALUE] == '1';
    }
  }
  return false;
}

std::string installHint(const std::string& tool) {
  return tool == "nsys"
             ? "Install the CUDA toolkit's Nsight Systems (nsys), or add its bin folder (for "
               "example /usr/local/cuda/bin) to PATH."
             : "Install the CUDA toolkit's Nsight Compute (ncu), or add its bin folder (for "
               "example /usr/local/cuda/bin) to PATH.";
}

/**
 * @brief The remedy of a request no Nsight tool started: the command that
 * captures it, and `bench run`, which wraps every mode.
 */
std::string wrapRemedy(const std::string& backend, const NsightSelection& selection,
                       const std::string& profileArgs) {
  const std::string REQUEST = requestText(backend, profileArgs);
  if (selection.mode == NsightMode::Systems) {
    return "Wrap it: nsys profile -o ./profile -t cuda,nvtx --force-overwrite true <this-binary> " +
           REQUEST + " [...]; or run it with bench run " + REQUEST +
           ", which wraps it and writes the summary reports.";
  }
  return std::string{"Wrap it, with few launches (ncu replays each one): ncu -o ./"} +
         selection.ncuReport() + " -f --target-processes all" + ncuOptionsText(selection) +
         " <this-binary> " + REQUEST + " --cycles 3 --repeats 1 [...]; or run it with bench run " +
         REQUEST + " --cycles 3 --repeats 1, which wraps it.";
}

/** @brief "; the driver gives ... root only ..." for an ncu mode outside root, else "". */
std::string countersNote(NsightMode mode, const ReadinessContext& ctx,
                         const std::string& driverParams) {
  if (mode == NsightMode::Systems || ctx.euid() == 0 || !countersRootOnly(driverParams)) {
    return {};
  }
  return "; the driver gives GPU performance counters to root only (RmProfilingAdminOnly: 1 in " +
         driverParams + "), and this process is not root";
}

/** @brief The doctor's decision: whether the mode's tool runs here. */
ReadinessResult probeTool(const std::string& tool, const NsightSelection& selection,
                          const ReadinessContext& ctx, const std::string& driverParams) {
  const auto RESOLVED = resolveExecutable(tool, ctx);
  if (!RESOLVED) {
    return readinessResult(ReadinessCause::MISSING, tool + " not found on PATH", installHint(tool));
  }
  if (!RESOLVED->executable) {
    return readinessResult(ReadinessCause::UNUSABLE, RESOLVED->path + " is not an executable file",
                           "Reinstall it, or fix PATH so it finds a working " + tool + ".");
  }
  const ProbeResult VERSION =
      runBoundedProbe({RESOLVED->path, "--version"}, VERSION_PROBE_TIMEOUT_MS, ctx);
  if (auto unseen = unobservedProbe(RESOLVED->path + " --version", VERSION)) {
    return *unseen;
  }
  if (!VERSION.succeeded()) {
    const std::string TAIL = outputTail(VERSION.output, 3);
    return readinessResult(ReadinessCause::UNUSABLE,
                           RESOLVED->path + " --version: " + VERSION.describe() +
                               (TAIL.empty() ? std::string{} : ": " + TAIL),
                           "Reinstall it, or fix PATH so it finds a working " + tool + ".");
  }
  return readinessResult(ReadinessCause::UNVERIFIED,
                         RESOLVED->path + " runs here (" + versionLine(VERSION.output) + ")" +
                             collectsNote(selection) +
                             "; whether it captures the benchmark's GPU work is not checked "
                             "before the run" +
                             countersNote(selection.mode, ctx, driverParams),
                         "");
}

/** @brief A run's decision: whether the mode's tool started this process. */
ReadinessResult decideRuntime(const std::string& backend, const std::string& tool,
                              const NsightSelection& selection, const std::string& profileArgs,
                              const ReadinessContext& ctx, const std::string& driverParams) {
  const std::string SESSION = profiler_env::nsightSessionTool(ctx);
  if (SESSION.empty()) {
    if (!resolveExecutable(tool, ctx)) {
      return readinessResult(ReadinessCause::MISSING, tool + " not found on PATH",
                             installHint(tool));
    }
    return readinessResult(ReadinessCause::MISSING,
                           backend + " collects only when " + tool + " starts the process, and " +
                               tool + " did not start this one",
                           wrapRemedy(backend, selection, profileArgs));
  }
  if (SESSION != tool) {
    return readinessResult(ReadinessCause::UNSUPPORTED,
                           requestText(backend, profileArgs) + " needs " + tool +
                               ", and this process runs under " + SESSION,
                           wrapRemedy(backend, selection, profileArgs));
  }
  return readinessResult(ReadinessCause::UNVERIFIED,
                         SESSION +
                             " started this process and writes its report when the "
                             "process exits" +
                             collectsNote(selection) +
                             "; whether it captured the benchmark's GPU work "
                             "is not checked from inside the process" +
                             countersNote(selection.mode, ctx, driverParams),
                         "");
}

/** @brief The analysis a request with --profile-analyze promises, which Nsight has none of. */
ReadinessResult analysisDecision(const std::string& backend) {
  return readinessResult(
      ReadinessCause::UNSUPPORTED,
      "--profile-analyze: " + backend +
          " has no analysis of its own; the capture still runs and its report is kept",
      backend == "ncu" ? "Read the report with ncu --import after the process exits, and drop "
                         "--profile-analyze."
                       : "Read the report with nsys stats (an ncu report with ncu --import) after "
                         "the process exits, and drop --profile-analyze.",
      ReadinessStage::ANALYSIS);
}

} // namespace

namespace {

/** @brief An ncu option bench run passes on, by what its value names. */
enum class NcuOption { NONE, SET, SECTION, METRICS };

/** @brief The option @p name spells, "--set" and the others. */
NcuOption ncuOptionOf(const std::string& name) {
  if (name == "--set") {
    return NcuOption::SET;
  }
  if (name == "--section") {
    return NcuOption::SECTION;
  }
  return name == "--metrics" ? NcuOption::METRICS : NcuOption::NONE;
}

/** @brief The refusal of @p option given without its value. */
std::string takesText(NcuOption option) {
  switch (option) {
  case NcuOption::SET:
    return "--set takes one section set's name";
  case NcuOption::SECTION:
    return "--section takes one section's identifier";
  case NcuOption::METRICS:
  case NcuOption::NONE:
    break;
  }
  return "--metrics takes one or more metric names";
}

/** @brief True when @p value is letters, digits, '_' and @p extra, and not empty. */
bool nameForm(const std::string& value, char extra) {
  if (value.empty()) {
    return false;
  }
  for (const char C : value) {
    const bool ALNUM = (C >= 'a' && C <= 'z') || (C >= 'A' && C <= 'Z') || (C >= '0' && C <= '9');
    if (!ALNUM && C != '_' && C != extra) {
      return false;
    }
  }
  return true;
}

/** @brief Add @p value, given to @p option, to @p selection: "", or the refusal's text. */
std::string takeValue(NsightSelection& selection, NcuOption option, const std::string& value) {
  constexpr const char* NO_PATTERN = "bench run passes ncu no regular expression or metric group";
  if (value.empty()) {
    return takesText(option);
  }
  switch (option) {
  case NcuOption::SET:
    if (!nameForm(value, '_')) {
      return "'" + value + "' is not a section set's name (letters, digits and '_')";
    }
    if (!selection.set.empty()) {
      return "--set is given twice ('" + selection.set + "', '" + value + "'); ncu takes one set";
    }
    selection.set = value;
    return {};
  case NcuOption::SECTION:
    if (!nameForm(value, '_')) {
      return "'" + value + "' is not a section's identifier (letters, digits and '_'); " +
             NO_PATTERN;
    }
    selection.sections.push_back(value);
    return {};
  case NcuOption::METRICS:
  case NcuOption::NONE:
    break;
  }
  if (!nameForm(value, '.')) {
    return "'" + value + "' is not a metric name (letters, digits, '_' and '.'); " + NO_PATTERN;
  }
  selection.metrics.push_back(value);
  return {};
}

} // namespace

std::vector<std::string> NsightSelection::ncuOptions() const {
  if (mode == NsightMode::Systems) {
    return {};
  }
  if (mode == NsightMode::ComputeReplay) {
    return {"--metrics", replayMetricList()};
  }
  std::vector<std::string> out;
  if (!set.empty()) {
    out.insert(out.end(), {"--set", set});
  }
  for (const std::string& SECTION : sections) {
    out.insert(out.end(), {"--section", SECTION});
  }
  if (!metrics.empty()) {
    out.insert(out.end(), {"--metrics", join(metrics, ",")});
  }
  return out;
}

const char* NsightSelection::ncuReport() const {
  return mode == NsightMode::ComputeReplay ? "kernel_replay" : "kernel_profile";
}

std::string NsightSelection::describe() const {
  if (mode == NsightMode::ComputeReplay) {
    return "the replay metrics (" + std::to_string(REPLAY_METRICS.size()) + ")";
  }
  return join(ncuOptions(), " ");
}

std::optional<ReadinessResult> parseNsightRequest(const std::string& backend,
                                                  const std::string& profileArgs,
                                                  NsightSelection& selection) {
  const bool NSIGHT = backend != "ncu";
  NsightSelection taken;
  bool compute = !NSIGHT;
  bool replay = false;
  // The option whose value comes next, whether a metric list is open, and
  // the first problem: the rest of the words still select the mode.
  NcuOption awaiting = NcuOption::NONE;
  bool listing = false;
  std::string problem;
  const auto NOTE = [&problem](std::string text) {
    if (problem.empty()) {
      problem = std::move(text);
    }
  };
  for (const std::string& WORD : valgrind_tool::modeWords(profileArgs)) {
    const bool MODE = WORD == "replay" || (NSIGHT && (WORD == "compute" || WORD == "ncu"));
    const bool OPTION = WORD[0] == '-';
    if (MODE || OPTION) {
      if (awaiting != NcuOption::NONE) {
        NOTE(takesText(awaiting));
        awaiting = NcuOption::NONE;
      }
      listing = false;
    }
    if (MODE) {
      compute = true;
      replay = replay || WORD == "replay";
      continue;
    }
    if (OPTION) {
      const std::size_t EQUALS = WORD.find('=');
      const NcuOption NAMED = ncuOptionOf(WORD.substr(0, EQUALS));
      if (NAMED == NcuOption::NONE) {
        NOTE("'" + WORD +
             "' is not an option bench run passes to ncu; it passes --set, --section and "
             "--metrics (the report, its overwrite and --target-processes are bench run's own)");
        continue;
      }
      compute = true;
      if (EQUALS == std::string::npos) {
        awaiting = NAMED;
        continue;
      }
      const std::string WHY = takeValue(taken, NAMED, WORD.substr(EQUALS + 1));
      NOTE(WHY);
      listing = WHY.empty() && NAMED == NcuOption::METRICS;
      continue;
    }
    if (awaiting != NcuOption::NONE) {
      const NcuOption FOR = awaiting;
      awaiting = NcuOption::NONE;
      const std::string WHY = takeValue(taken, FOR, WORD);
      NOTE(WHY);
      listing = WHY.empty() && FOR == NcuOption::METRICS;
      continue;
    }
    if (listing) {
      NOTE(takeValue(taken, NcuOption::METRICS, WORD));
      continue;
    }
    NOTE("'" + WORD + "' is not a mode of " + (NSIGHT ? "nsight" : "ncu") + "; it takes " +
         (NSIGHT ? "compute, ncu, replay, --set NAME, --section NAME and --metrics NAMES"
                 : "replay, --set NAME, --section NAME and --metrics NAMES"));
  }
  if (awaiting != NcuOption::NONE) {
    NOTE(takesText(awaiting));
  }
  const bool SELECTS = !taken.set.empty() || !taken.sections.empty() || !taken.metrics.empty();
  if (replay && SELECTS) {
    NOTE("replay collects its own metric list; give replay alone, or name what to collect "
         "without it");
  }
  taken.mode =
      replay ? NsightMode::ComputeReplay : (compute ? NsightMode::Compute : NsightMode::Systems);
  if (replay) {
    taken.set.clear();
    taken.sections.clear();
    taken.metrics.clear();
  }
  selection = std::move(taken);
  if (problem.empty()) {
    return std::nullopt;
  }
  return readinessResult(ReadinessCause::CONFIGURATION, problem,
                         "Give --profile-args replay, --set NAME, --section NAME or --metrics "
                         "NAMES (under nsight also compute or ncu), Vernier's supported subset "
                         "of ncu's options, or drop it; for any other ncu option, start the "
                         "binary under ncu yourself.");
}

std::optional<ReadinessResult> parseNsightMode(const std::string& backend,
                                               const std::string& profileArgs, NsightMode& mode) {
  NsightSelection selection;
  auto refused = parseNsightRequest(backend, profileArgs, selection);
  mode = selection.mode;
  return refused;
}

const char* nsightModeTool(NsightMode mode) { return mode == NsightMode::Systems ? "nsys" : "ncu"; }

ReadinessResult checkNsightRequestWith(const ReadinessRequest& request, const ReadinessContext& ctx,
                                       const std::string& driverParams) {
  const std::string BACKEND = request.backend == "ncu" ? "ncu" : "nsight";
  NsightSelection selection;
  if (auto refused = parseNsightRequest(BACKEND, request.profileArgs, selection)) {
    return *refused;
  }
  const std::string TOOL = nsightModeTool(selection.mode);
  ReadinessResult result =
      request.scope == ReadinessScope::RUNTIME
          ? decideRuntime(BACKEND, TOOL, selection, request.profileArgs, ctx, driverParams)
          : probeTool(TOOL, selection, ctx, driverParams);
  if (!request.analyze) {
    return result;
  }
  return valgrind_tool::withAnalysis(std::move(result), analysisDecision(BACKEND));
}

ReadinessResult checkNsightRequest(const ReadinessRequest& request, const ReadinessContext& ctx) {
  return checkNsightRequestWith(request, ctx, DRIVER_PARAMS);
}

namespace {

/**
 * @brief libbench's profiler for a request Nsight may capture: it names the
 * backend and its folder and does nothing else (no NVTX without CUDA). A CUDA
 * build's registration replaces it with the Nsight backend.
 */
std::unique_ptr<Profiler> makePassiveNsightProfiler(const PerfConfig& cfg,
                                                    const std::string& testName,
                                                    const ReadinessResult& result) {
  if (!result.collectionReady()) {
    return nullptr;
  }
  const char* SUFFIX = ProfilerRegistry::canonicalName(cfg.profileTool) == "ncu" ? "ncu" : "nsight";
  return std::make_unique<detail::NoOpProfiler>(
      "nsight",
      profiler_env::resolveArtifactDir(cfg.profileTool, cfg.artifactRoot, testName, SUFFIX));
}

} // namespace

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_READINESS_FALLBACK("nsight", ::vernier::bench::checkNsightRequest,
                                    ::vernier::bench::makePassiveNsightProfiler,
                                    "Install NVIDIA Nsight Systems (nsys) or Nsight Compute "
                                    "(ncu), the tool the mode needs.",
                                    "NSYS_PROFILING_SESSION_ID", "NV_NSIGHT_INJECTION_PORT_BASE")

VERNIER_REGISTER_READINESS_FALLBACK("ncu", ::vernier::bench::checkNsightRequest,
                                    ::vernier::bench::makePassiveNsightProfiler,
                                    "Install NVIDIA Nsight Compute (ncu).",
                                    "NSYS_PROFILING_SESSION_ID", "NV_NSIGHT_INJECTION_PORT_BASE")
