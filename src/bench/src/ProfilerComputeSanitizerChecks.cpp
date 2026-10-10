/**
 * @file ProfilerComputeSanitizerChecks.cpp
 * @brief The Compute Sanitizer backend's tool parser and readiness check, and
 * libbench's default registration of `compute-sanitizer` with a passive
 * profiler.
 */

#include "src/bench/inc/ProfilerComputeSanitizerChecks.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "src/bench/inc/Profiler.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/inc/ValgrindTool.hpp"

namespace vernier {
namespace bench {

namespace {

/** @brief How long the doctor's `compute-sanitizer --version` may run. */
constexpr int VERSION_PROBE_TIMEOUT_MS = 10000;

/** @brief The status `bench run` asks the tool to end with when it finds errors. */
constexpr int FINDINGS_EXIT_CODE = 5;

const std::vector<std::string>& sanitizerTools() {
  static const std::vector<std::string> TOOLS = {"memcheck", "racecheck", "synccheck", "initcheck"};
  return TOOLS;
}

std::string requestText(const std::string& profileArgs) {
  std::string words;
  for (const std::string& WORD : valgrind_tool::modeWords(profileArgs)) {
    words += (words.empty() ? "" : ",") + WORD;
  }
  return "--profile compute-sanitizer" +
         (words.empty() ? std::string{} : " --profile-args " + words);
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

/** @brief The text of /proc/<pid>/maps, or "" when it cannot be read. */
std::string mapsOf(pid_t pid) {
  std::ifstream in("/proc/" + std::to_string(static_cast<long>(pid)) + "/maps");
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

ReadinessResult missingTool() {
  return readinessResult(ReadinessCause::MISSING, "compute-sanitizer not found on PATH",
                         "Install the CUDA toolkit, which ships compute-sanitizer, or add its bin "
                         "folder (for example /usr/local/cuda/bin) to PATH.");
}

/**
 * @brief The by-hand wrap's log: sanitizer.log in this process's working
 * directory, named whole. compute-sanitizer joins a relative log name to its
 * own working directory and reads a '%' in the result as a macro, so a
 * relative name fails wherever that directory's path holds one; the whole
 * name is written with each '%' doubled for the tool and quoted for the shell.
 */
std::string byHandLog() {
  std::error_code ec;
  const std::filesystem::path HERE = std::filesystem::current_path(ec);
  const std::string LOG = ec ? std::string("./sanitizer.log") : (HERE / "sanitizer.log").string();
  return detail::shellQuote(detail::escapePercent(LOG));
}

/** @brief The remedy of a request compute-sanitizer did not start: the wrap, by hand or by bench
 * run. */
std::string wrapRemedy(const std::string& tool, const std::string& profileArgs) {
  const std::string REQUEST = requestText(profileArgs);
  return "Wrap it: compute-sanitizer --tool=" + tool + " --error-exitcode " +
         std::to_string(FINDINGS_EXIT_CODE) + " --log-file=" + byHandLog() + " <this-binary> " +
         REQUEST + " [...]; or run it with bench run " + REQUEST +
         ", which wraps it and reads the report.";
}

/** @brief The doctor's decision: whether compute-sanitizer runs here. */
ReadinessResult probeTool(const std::string& tool, const ReadinessContext& ctx) {
  const auto RESOLVED = resolveExecutable("compute-sanitizer", ctx);
  if (!RESOLVED) {
    return missingTool();
  }
  if (!RESOLVED->executable) {
    return readinessResult(ReadinessCause::UNUSABLE, RESOLVED->path + " is not an executable file",
                           "Reinstall it, or fix PATH so it finds a working compute-sanitizer.");
  }
  const ProbeResult VERSION =
      runBoundedProbe({RESOLVED->path, "--version"}, VERSION_PROBE_TIMEOUT_MS, ctx);
  if (!VERSION.succeeded()) {
    const std::string TAIL = outputTail(VERSION.output, 3);
    return readinessResult(ReadinessCause::UNUSABLE,
                           RESOLVED->path + " --version: " + VERSION.describe() +
                               (TAIL.empty() ? std::string{} : ": " + TAIL),
                           "Reinstall it, or fix PATH so it finds a working compute-sanitizer.");
  }
  return readinessResult(ReadinessCause::UNVERIFIED,
                         RESOLVED->path + " runs here (" + versionLine(VERSION.output) +
                             "); whether its " + tool +
                             " checks the benchmark's kernels is not checked before the run",
                         "");
}

/** @brief A run's decision: whether compute-sanitizer started this process. */
ReadinessResult decideRuntime(const std::string& tool, const std::string& profileArgs,
                              const ReadinessContext& ctx, const std::string& mapsText) {
  if (profiler_env::computeSanitizerSession(ctx) ||
      profiler_env::mapsShowComputeSanitizer(mapsText)) {
    return readinessResult(ReadinessCause::UNVERIFIED,
                           "compute-sanitizer started this process and reports when it exits; "
                           "which tool it runs, and what it finds, is not seen from inside the "
                           "process",
                           "");
  }
  if (!resolveExecutable("compute-sanitizer", ctx)) {
    return missingTool();
  }
  return readinessResult(ReadinessCause::MISSING,
                         "compute-sanitizer checks a process only when it starts it, and it did "
                         "not start this one",
                         wrapRemedy(tool, profileArgs));
}

ReadinessResult checkWith(const ReadinessRequest& request, const ReadinessContext& ctx,
                          const std::string* mapsText) {
  std::string tool;
  if (auto refused = parseSanitizerTool(request.profileArgs, tool)) {
    return *refused;
  }
  ReadinessResult result = request.scope == ReadinessScope::RUNTIME
                               ? decideRuntime(tool, request.profileArgs, ctx,
                                               mapsText != nullptr ? *mapsText : mapsOf(ctx.self()))
                               : probeTool(tool, ctx);
  if (!request.analyze) {
    return result;
  }
  return valgrind_tool::withAnalysis(
      std::move(result),
      readinessResult(ReadinessCause::UNSUPPORTED,
                      "--profile-analyze: compute-sanitizer has no analysis of its own; the "
                      "check still runs and its report is kept",
                      "Its report is the result: read it, or let bench run read it after the "
                      "process exits, and drop --profile-analyze.",
                      ReadinessStage::ANALYSIS));
}

} // namespace

std::optional<ReadinessResult> parseSanitizerTool(const std::string& profileArgs,
                                                  std::string& tool) {
  tool = "memcheck";
  std::optional<ReadinessResult> unknown;
  int tools = 0;
  for (const std::string& WORD : valgrind_tool::modeWords(profileArgs)) {
    if (std::find(sanitizerTools().begin(), sanitizerTools().end(), WORD) ==
        sanitizerTools().end()) {
      if (!unknown) {
        unknown = valgrind_tool::refusedWord("compute-sanitizer", WORD, sanitizerTools());
      }
    } else if (tools++ == 0) {
      tool = WORD;
    }
  }
  if (unknown) {
    return unknown;
  }
  if (tools > 1) {
    return readinessResult(ReadinessCause::CONFIGURATION,
                           "compute-sanitizer runs one tool at a time; choose one of memcheck, "
                           "racecheck, synccheck, initcheck",
                           "Name one tool in --profile-args.");
  }
  return std::nullopt;
}

ReadinessResult checkComputeSanitizerRequest(const ReadinessRequest& request,
                                             const ReadinessContext& ctx) {
  return checkWith(request, ctx, nullptr);
}

ReadinessResult checkComputeSanitizerRequestWithMaps(const ReadinessRequest& request,
                                                     const ReadinessContext& ctx,
                                                     const std::string& mapsText) {
  return checkWith(request, ctx, &mapsText);
}

namespace {

/**
 * @brief libbench's profiler for a request compute-sanitizer may check: it
 * names the backend and its folder and does nothing else. A CUDA build's
 * registration replaces it with the backend.
 */
std::unique_ptr<Profiler> makePassiveComputeSanitizerProfiler(const PerfConfig& cfg,
                                                              const std::string& testName,
                                                              const ReadinessResult& result) {
  if (!result.collectionReady()) {
    return nullptr;
  }
  return std::make_unique<detail::NoOpProfiler>(
      "compute-sanitizer", profiler_env::resolveArtifactDir(cfg.profileTool, cfg.artifactRoot,
                                                            testName, "compute-sanitizer"));
}

} // namespace

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_READINESS_FALLBACK("compute-sanitizer",
                                    ::vernier::bench::checkComputeSanitizerRequest,
                                    ::vernier::bench::makePassiveComputeSanitizerProfiler,
                                    "Install the CUDA toolkit; compute-sanitizer ships with it.",
                                    "NV_SANITIZER_INJECTION_PORT_BASE", "CUDA_INJECTION64_PATH")
