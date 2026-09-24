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
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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
  const std::string outFile = artifactDir_ + "/callgrind.out";
  const bool KNOWN_FILE = canToggle_ || isWrappedByRunner(cfg_);

  std::printf("\n=== Callgrind Profile ===\n");
  std::printf("Output: %s%s\n", artifactDir_.c_str(),
              KNOWN_FILE ? ""
                         : " (or where --callgrind-out-file points; by default "
                           "callgrind.out.<pid> in the working directory)");

  if (cfg_.profileAnalyze) {
    runAnnotateAnalysis();
  } else if (KNOWN_FILE) {
    std::printf("   Run with --profile-analyze for automatic annotation\n");
    std::printf("   Or manually: callgrind_annotate %s\n", outFile.c_str());
    std::printf("   Or: kcachegrind %s\n", outFile.c_str());
  }
  std::printf("\n");
#endif
}

void CallgrindProfiler::runAnnotateAnalysis() const {
#ifdef __linux__
  bool hasAnnotate = (std::system("command -v callgrind_annotate >/dev/null 2>&1") == 0);
  if (!hasAnnotate) {
    std::fprintf(stderr, "[INFO] callgrind_annotate not found. Install valgrind.\n");
    return;
  }

  // Find the most recent callgrind.out file in the artifact directory
  std::string latestFile;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(artifactDir_, ec)) {
    const std::string NAME = entry.path().filename().string();
    if (NAME.find("callgrind.out") != std::string::npos) {
      latestFile = entry.path().string();
    }
  }

  if (latestFile.empty()) {
    std::fprintf(stderr, "[WARN] No callgrind output file found in %s\n", artifactDir_.c_str());
    return;
  }

  std::printf("\n--- Callgrind Annotation (top functions) ---\n\n");

  std::string cmd = "callgrind_annotate --auto=yes '" + latestFile + "' 2>/dev/null | head -40";
  [[maybe_unused]] int rc = std::system(cmd.c_str());

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
