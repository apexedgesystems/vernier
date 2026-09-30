/**
 * @file ProfilerCallgrind.cpp
 * @brief Implementation of Valgrind Callgrind profiler backend.
 *
 * Under a manual `valgrind --tool=callgrind --instr-atstart=no` wrap, switches
 * instrumentation on for each measured window and off after it with
 * callgrind_control. A recording made by bench run's wrap covers the whole
 * process and is left alone. Not under valgrind, the backend prints how to
 * wrap and the measurement runs normally.
 */

#include "src/bench/inc/ProfilerCallgrind.hpp"

#ifdef __linux__
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#endif

#include "src/bench/inc/ProfilerEnv.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- Helpers ----------------------------- */

namespace {

#ifdef __linux__
bool isValgrindAvailable() { return (std::system("command -v valgrind >/dev/null 2>&1") == 0); }

bool isCallgrindControlAvailable() {
  return (std::system("command -v callgrind_control >/dev/null 2>&1") == 0);
}

bool isRunningUnderValgrind() { return profiler_env::isRunningUnderValgrind(); }

/// True when bench run wrapped this process with callgrind: that recording
/// covers the whole process, and switching instrumentation off after a
/// measured window would cut it short.
bool isWrappedByRunner(const PerfConfig& cfg) {
  return !cfg.profileTool.empty() && profiler_env::externalWrapTool() == cfg.profileTool;
}

/// Switch callgrind's instrumentation for this process on or off.
/// callgrind_control takes the process id as a trailing argument, not as an
/// option. It exits 0 whether or not the command reached the process, so its
/// status is not read.
void switchInstrumentation(const char* state) {
  const std::string CMD = std::string{"callgrind_control -i "} + state + " " +
                          std::to_string(::getpid()) + " >/dev/null 2>&1";
  [[maybe_unused]] const int RC = std::system(CMD.c_str());
}
#endif

} // namespace

/* ----------------------------- CallgrindProfiler Methods ----------------------------- */

CallgrindProfiler::CallgrindProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
#ifdef __linux__
  artifactDir_ =
      profiler_env::resolveArtifactDir(cfg_.profileTool, cfg_.artifactRoot, testName_, "callgrind");

  runningUnderValgrind_ = isRunningUnderValgrind();
  const bool HAS_CONTROL = isCallgrindControlAvailable();
  canToggle_ = runningUnderValgrind_ && HAS_CONTROL && !isWrappedByRunner(cfg_);

  if (!runningUnderValgrind_) {
    // Without callgrind_control the window cannot be switched on, so the hint
    // leaves instrumentation on from the start and records the whole process.
    std::fprintf(stderr,
                 "\n[callgrind] not running under valgrind; instrumentation skipped.\n"
                 "[callgrind] To collect a profile, wrap externally:\n"
                 "[callgrind]   valgrind --tool=callgrind%s \\\n"
                 "[callgrind]     --callgrind-out-file=%s/callgrind.out \\\n"
                 "[callgrind]     <this-binary> --profile callgrind [...]\n%s\n",
                 HAS_CONTROL ? " --instr-atstart=no" : "", artifactDir_.c_str(),
                 HAS_CONTROL ? ""
                             : "[callgrind] callgrind_control is not on PATH, so the profile "
                               "covers the whole process.\n");
  } else if (!HAS_CONTROL && !isWrappedByRunner(cfg_)) {
    std::fprintf(stderr,
                 "\n[callgrind] running under valgrind, but callgrind_control is not on PATH:\n"
                 "[callgrind] instrumentation cannot be switched on for the measured window,\n"
                 "[callgrind] so a run started with --instr-atstart=no records nothing.\n\n");
  }
#else
  (void)cfg_;
  (void)testName_;
#endif
}

void CallgrindProfiler::beforeMeasure() {
#ifdef __linux__
  if (canToggle_) {
    switchInstrumentation("on");
  }
#endif
}

void CallgrindProfiler::afterMeasure(const Stats& /*s*/) {
#ifdef __linux__
  if (!runningUnderValgrind_) {
    return; // Not wrapped at all -- nothing to do.
  }
  if (canToggle_) {
    switchInstrumentation("off");
  }

  // Valgrind writes the profile when the process exits, to the file its
  // --callgrind-out-file names: callgrind.out in artifactDir_ under bench
  // run's wrap and under the hint's. Any other manual wrap names its own.
  // The profile is complete only then, so nothing here reads it: bench run
  // checks it, and annotates it for --profile-analyze, after the exit.
  const std::string outFile = artifactDir_ + "/callgrind.out";
  const bool BY_RUNNER = isWrappedByRunner(cfg_);
  const bool KNOWN_FILE = canToggle_ || BY_RUNNER;

  std::printf("\n=== Callgrind Profile ===\n");
  std::printf("Output: %s%s\n", artifactDir_.c_str(),
              KNOWN_FILE ? ""
                         : " (or where --callgrind-out-file points; by default "
                           "callgrind.out.<pid> in the working directory)");
  if (BY_RUNNER) {
    std::printf("   bench run checks the profile after valgrind has written it%s\n",
                cfg_.profileAnalyze ? ", then annotates it" : "");
  } else {
    std::printf("   valgrind writes the profile when this process exits; read it then with\n");
    std::printf("   callgrind_annotate %s (or kcachegrind)\n",
                KNOWN_FILE ? outFile.c_str() : "<profile>");
  }
  std::printf("\n");
#endif
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeCallgrindProfiler(const PerfConfig& cfg,
                                                const std::string& testName) {
#ifdef __linux__
  if (!isValgrindAvailable()) {
    return nullptr;
  }
  return std::make_unique<CallgrindProfiler>(cfg, testName);
#else
  (void)cfg;
  (void)testName;
  return nullptr;
#endif
}

} // namespace bench
} // namespace vernier

namespace vernier {
namespace bench {

EnvReport checkCallgrindEnvironment() {
  if (std::system("command -v valgrind >/dev/null 2>&1") != 0) {
    return EnvReport{EnvReport::Status::Error, "valgrind binary not found on PATH",
                     "apt install valgrind."};
  }
  // Under a Docker PID namespace, callgrind_control attach is unreliable; run
  // callgrind by wrapping valgrind directly instead.
  if (std::system("grep -q docker /proc/1/cgroup 2>/dev/null") == 0) {
    return EnvReport{EnvReport::Status::Warning,
                     "valgrind available; running in Docker (PID namespace)",
                     "Run via 'bench run', which wraps valgrind directly (no attach needed)."};
  }
  return EnvReport{EnvReport::Status::Ok, "valgrind available", ""};
}

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_PROFILER_BACKEND("callgrind", ::vernier::bench::makeCallgrindProfiler,
                                  ::vernier::bench::checkCallgrindEnvironment,
                                  "Install valgrind: apt install valgrind.")
