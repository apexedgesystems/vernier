/**
 * @file ValgrindTool.cpp
 * @brief The words, wrap arguments, start probe, process identity and
 * not-wrapped result the valgrind backends share.
 */

#include "src/bench/inc/ValgrindTool.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <utility>

namespace vernier {
namespace bench {
namespace valgrind_tool {

namespace {

/** @brief @p parts joined with @p separator. */
std::string join(const std::vector<std::string>& parts, const std::string& separator) {
  std::string out;
  for (const std::string& PART : parts) {
    out += (out.empty() ? "" : separator) + PART;
  }
  return out;
}

bool endsWith(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool startsWith(const std::string& text, const std::string& prefix) {
  return text.compare(0, prefix.size(), prefix) == 0;
}

/** @brief The request's --profile-args as the remedies repeat it: its words, comma-joined. */
std::string wordsText(const std::string& profileArgs) { return join(modeWords(profileArgs), ","); }

/** @brief `--profile <backend>` with the request's words, as a command line states it. */
std::string requestText(const std::string& backend, const std::string& profileArgs) {
  const std::string WORDS = wordsText(profileArgs);
  return "--profile " + backend + (WORDS.empty() ? std::string{} : " --profile-args " + WORDS);
}

} // namespace

/* ----------------------------- Mode ----------------------------- */

std::vector<std::string> modeWords(const std::string& profileArgs) {
  std::vector<std::string> words;
  std::string word;
  for (const char CH : profileArgs) {
    if (std::isspace(static_cast<unsigned char>(CH)) != 0 || CH == ',') {
      if (!word.empty()) {
        words.push_back(word);
        word.clear();
      }
    } else {
      word += CH;
    }
  }
  if (!word.empty()) {
    words.push_back(word);
  }
  return words;
}

ReadinessResult refusedWord(const std::string& backend, const std::string& word,
                            const std::vector<std::string>& accepted) {
  if (accepted.empty()) {
    return readinessResult(ReadinessCause::CONFIGURATION,
                           "'" + word + "' is not a mode of " + backend + ", which takes none",
                           "Drop --profile-args: " + backend + " takes no mode.");
  }
  const std::string MODES = join(accepted, ", ");
  return readinessResult(ReadinessCause::CONFIGURATION,
                         "'" + word + "' is not a mode of " + backend + "; its modes are " + MODES,
                         "Use one of " + MODES + " in --profile-args, or drop it.");
}

OutputFile outputFile(const std::string& backend) {
  if (backend == "callgrind") {
    return {"--callgrind-out-file", "callgrind.out"};
  }
  if (backend == "massif") {
    return {"--massif-out-file", "massif.out"};
  }
  if (backend == "memcheck") {
    return {"--log-file", "memcheck.log"};
  }
  if (backend == "helgrind") {
    return {"--log-file", "helgrind.log"};
  }
  return {};
}

std::vector<std::string> wrapArguments(const std::string& backend, const ValgrindMode& mode,
                                       const std::string& dir) {
  std::vector<std::string> args{"--tool=" + mode.tool};
  args.insert(args.end(), mode.options.begin(), mode.options.end());
  const OutputFile OUT = outputFile(backend);
  args.push_back(OUT.option + "=" + dir + "/" + OUT.name);
  return args;
}

std::string wrapRemedy(const std::string& backend, const ValgrindMode& mode,
                       const std::string& profileArgs,
                       const std::vector<std::string>& extraOptions) {
  std::vector<std::string> args{"--tool=" + mode.tool};
  args.insert(args.end(), extraOptions.begin(), extraOptions.end());
  const std::vector<std::string> ROUTE = wrapArguments(backend, mode, ".");
  args.insert(args.end(), ROUTE.begin() + 1, ROUTE.end());
  const std::string REQUEST = requestText(backend, profileArgs);
  return "Wrap it: valgrind " + join(args, " ") + " <this-binary> " + REQUEST +
         " [...]; or run it with bench run " + REQUEST + ", which wraps it.";
}

/* ----------------------------- Start Probe ----------------------------- */

ReadinessResult valgrindMissing() {
  return readinessResult(ReadinessCause::MISSING, "valgrind not found on PATH",
                         "Install valgrind (apt install valgrind).");
}

ReadinessResult probeTool(const std::string& backend, const ValgrindMode& mode,
                          const ReadinessContext& ctx, std::string& valgrindPath) {
  const auto TOOL = resolveExecutable("valgrind", ctx);
  if (!TOOL) {
    return valgrindMissing();
  }
  if (!TOOL->executable) {
    return readinessResult(ReadinessCause::UNUSABLE, TOOL->path + " is not an executable file",
                           "Reinstall valgrind, or fix PATH so it finds a working valgrind.");
  }
  valgrindPath = TOOL->path;
  const auto TRUE_PROGRAM = resolveExecutable("/bin/true", ctx);
  if (!TRUE_PROGRAM || !TRUE_PROGRAM->executable) {
    return readinessResult(ReadinessCause::MISSING_HELPER,
                           "/bin/true, the program the start probe runs under valgrind, is missing",
                           "Restore /bin/true (coreutils).");
  }

  // The tool starts, not only valgrind: `valgrind --version` succeeds without
  // starting any tool. The output goes to /dev/null, so nothing is written.
  std::vector<std::string> argv{TOOL->path, "--tool=" + mode.tool};
  argv.insert(argv.end(), mode.options.begin(), mode.options.end());
  argv.push_back(outputFile(backend).option + "=/dev/null");
  argv.push_back(TRUE_PROGRAM->path);
  const std::string COMMAND = join(argv, " ");
  const ProbeResult PROBE = runBoundedProbe(argv, PROBE_TIMEOUT_MS, ctx);
  if (!PROBE.succeeded()) {
    const std::string TAIL = outputTail(PROBE.output);
    return readinessResult(ReadinessCause::UNUSABLE,
                           "valgrind does not start " + mode.tool + ": " + COMMAND + ": " +
                               PROBE.describe() + (TAIL.empty() ? std::string{} : ": " + TAIL),
                           "Run that command by hand to see why; reinstall valgrind if the tool "
                           "is missing.");
  }
  const std::string WITH = mode.unseen.empty() ? std::string{} : " with " + join(mode.unseen, " ");
  return readinessResult(ReadinessCause::READY,
                         "valgrind starts " + mode.tool + WITH + " (probe: " + COMMAND + ")", "");
}

/* ----------------------------- Identity ----------------------------- */

std::string ValgrindIdentity::tool() const {
  if (!underValgrind || executables.size() != 1) {
    return {};
  }
  for (const std::string& PRELOAD : preloads) {
    if (PRELOAD != executables.front()) {
      return {};
    }
  }
  return executables.front();
}

std::string ValgrindIdentity::describe() const {
  if (!underValgrind) {
    return "no valgrind preload";
  }
  std::string text = "valgrind's core preload";
  text += executables.empty() ? ", no tool executable"
                              : ", the executable of " + join(executables, " and ");
  if (!preloads.empty()) {
    text += ", the preload of " + join(preloads, " and ");
  }
  return text;
}

ValgrindIdentity identityFromMaps(const std::string& mapsText) {
  // Every mapped file once, split into its directory and its name.
  std::set<std::pair<std::string, std::string>> files;
  std::istringstream lines(mapsText);
  std::string line;
  while (std::getline(lines, line)) {
    const std::size_t SLASH = line.find('/');
    if (SLASH == std::string::npos) {
      continue;
    }
    std::string path = line.substr(SLASH);
    while (!path.empty() && std::isspace(static_cast<unsigned char>(path.back())) != 0) {
      path.pop_back();
    }
    const std::size_t LAST = path.rfind('/');
    files.emplace(path.substr(0, LAST), path.substr(LAST + 1));
  }

  // The core preload names valgrind's directory and platform; the tools'
  // executables and preloads count only beside it.
  const std::string CORE = "vgpreload_core-";
  const std::string SO = ".so";
  ValgrindIdentity identity;
  std::set<std::string> executables;
  std::set<std::string> preloads;
  for (const auto& [DIR, NAME] : files) {
    if (!startsWith(NAME, CORE) || !endsWith(NAME, SO) || NAME.size() <= CORE.size() + SO.size()) {
      continue;
    }
    identity.underValgrind = true;
    const std::string PLATFORM = NAME.substr(CORE.size(), NAME.size() - CORE.size() - SO.size());
    const std::string SUFFIX = "-" + PLATFORM;
    for (const auto& [OTHER_DIR, OTHER] : files) {
      if (OTHER_DIR != DIR || OTHER == NAME) {
        continue;
      }
      if (startsWith(OTHER, "vgpreload_") && endsWith(OTHER, SUFFIX + SO)) {
        const std::size_t FROM = std::string{"vgpreload_"}.size();
        const std::size_t LENGTH = OTHER.size() - FROM - SUFFIX.size() - SO.size();
        if (LENGTH > 0) {
          preloads.insert(OTHER.substr(FROM, LENGTH));
        }
      } else if (endsWith(OTHER, SUFFIX) && OTHER.size() > SUFFIX.size()) {
        executables.insert(OTHER.substr(0, OTHER.size() - SUFFIX.size()));
      }
    }
  }
  identity.executables.assign(executables.begin(), executables.end());
  identity.preloads.assign(preloads.begin(), preloads.end());
  return identity;
}

ValgrindIdentity identityOf(pid_t pid) {
#ifdef __linux__
  std::ifstream maps("/proc/" + std::to_string(static_cast<long>(pid)) + "/maps");
  std::stringstream text;
  text << maps.rdbuf();
  return identityFromMaps(text.str());
#else
  (void)pid;
  return {};
#endif
}

/* ----------------------------- Decisions ----------------------------- */

ReadinessResult decideRuntime(const std::string& backend, const ValgrindMode& mode,
                              const std::string& profileArgs, const ValgrindIdentity& identity,
                              LaunchContext launch, const std::string& remedy,
                              const ReadinessContext& ctx) {
  auto plan = std::make_shared<ValgrindPlan>();
  plan->backend = backend;
  plan->mode = mode;
  ReadinessResult result;
  if (!identity.underValgrind) {
    plan->launch = LaunchContext::NOT_WRAPPED;
    if (!resolveExecutable("valgrind", ctx)) {
      result = valgrindMissing();
    } else {
      result = readinessResult(ReadinessCause::MISSING,
                               backend + " collects only when valgrind's " + mode.tool +
                                   " runs the process, and valgrind does not run this one",
                               remedy);
    }
    result.plan = std::move(plan);
    return result;
  }

  const bool BY_RUNNER = launch == LaunchContext::RUNNER_WRAPPED;
  plan->launch = BY_RUNNER ? LaunchContext::RUNNER_WRAPPED : LaunchContext::MANUALLY_WRAPPED;
  const std::string RUNNING = identity.tool();
  if (RUNNING.empty()) {
    result = readinessResult(ReadinessCause::UNVERIFIED,
                             "valgrind runs this process, but its memory map does not show which "
                             "tool: " +
                                 identity.describe(),
                             remedy);
  } else if (RUNNING != mode.tool) {
    result = readinessResult(ReadinessCause::UNSUPPORTED,
                             requestText(backend, profileArgs) + " needs valgrind's " + mode.tool +
                                 ", and this process runs under valgrind's " + RUNNING,
                             remedy);
  } else if (BY_RUNNER) {
    result =
        readinessResult(ReadinessCause::READY,
                        "valgrind's " + RUNNING + " runs this process, under bench run's wrap", "");
  } else if (!mode.unseen.empty()) {
    // The wrap was started by hand: its options are the user's command's,
    // which the process cannot read back.
    result = readinessResult(ReadinessCause::UNVERIFIED,
                             "valgrind's " + RUNNING + " runs this process; whether with " +
                                 join(mode.unseen, " ") + " cannot be seen from inside it",
                             remedy);
  } else {
    result = readinessResult(
        ReadinessCause::READY,
        "valgrind's " + RUNNING + " runs this process, under a wrap started by hand", "");
  }
  result.plan = std::move(plan);
  return result;
}

ReadinessResult decideCollection(const std::string& backend, const ValgrindMode& mode,
                                 const ReadinessRequest& request, const ReadinessContext& ctx,
                                 const std::string& remedy, const ValgrindIdentity* identity) {
  if (request.scope == ReadinessScope::RUNTIME) {
    return decideRuntime(backend, mode, request.profileArgs,
                         identity != nullptr ? *identity : identityOf(ctx.self()), request.launch,
                         remedy, ctx);
  }
  auto plan = std::make_shared<ValgrindPlan>();
  plan->backend = backend;
  plan->mode = mode;
  ReadinessResult result = probeTool(backend, mode, ctx, plan->valgrind);
  result.plan = std::move(plan);
  return result;
}

ReadinessResult withAnalysis(ReadinessResult collection, ReadinessResult analysis) {
  if (!collection.collectionReady() || analysis.report.status != EnvReport::Status::Error) {
    return collection;
  }
  analysis.plan = std::move(collection.plan);
  analysis.context = std::move(collection.context);
  return analysis;
}

} // namespace valgrind_tool
} // namespace bench
} // namespace vernier
