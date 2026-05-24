/**
 * @file ProfilerCallgrind.cpp
 * @brief Implementation of Valgrind Callgrind profiler backend.
 *
 * Uses callgrind_control to toggle instrumentation around the measured window.
 * When not running under valgrind, the toggle commands are harmless no-ops.
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

bool isRunningUnderValgrind() {
  // Use the shared detection that does NOT depend on callgrind_control
  // (which can't reach the valgrind process inside a Docker PID namespace).
  return profiler_env::isRunningUnderValgrind();
}
#endif

} // namespace

/* ----------------------------- CallgrindProfiler Methods ----------------------------- */

CallgrindProfiler::CallgrindProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
#ifdef __linux__
  if (!cfg_.artifactRoot.empty()) {
    artifactDir_ = cfg_.artifactRoot + "/" + testName_ + ".callgrind";
  } else {
    artifactDir_ = "./" + testName_ + ".callgrind";
  }
  std::error_code ec;
  std::filesystem::create_directories(artifactDir_, ec);

  // Parse mode from profileArgs
  std::string args = cfg_.profileArgs;
  wantCache_ = (args.find("cache") != std::string::npos);
  wantBranch_ = (args.find("branch") != std::string::npos);

  runningUnderValgrind_ = isRunningUnderValgrind();
  canToggle_ = runningUnderValgrind_ && isCallgrindControlAvailable() &&
               !profiler_env::isInContainer();

  if (!runningUnderValgrind_) {
    std::fprintf(stderr, "\n[callgrind] not running under valgrind; instrumentation skipped.\n"
                         "[callgrind] To collect a profile, wrap externally:\n"
                         "[callgrind]   valgrind --tool=callgrind --instr-atstart=no \\\n"
                         "[callgrind]     --callgrind-out-file=%s/callgrind.out \\\n"
                         "[callgrind]     <this-binary> --profile callgrind [...]\n\n",
                 artifactDir_.c_str());
  } else if (!canToggle_) {
    std::fprintf(stderr, "\n[callgrind] running under valgrind; callgrind_control cannot reach\n"
                         "[callgrind] this PID (likely Docker PID namespace). Recording will run\n"
                         "[callgrind] for the whole process; output written at exit.\n\n");
  }
#else
  (void)cfg_;
  (void)testName_;
#endif
}

void CallgrindProfiler::beforeMeasure() {
#ifdef __linux__
  if (!canToggle_) {
    return; // Either not under valgrind, or in a PID namespace -- skip toggling.
  }

  // Zero counters and enable instrumentation for the measured window
  std::string pid = std::to_string(::getpid());
  std::string cmd = "callgrind_control --pid=" + pid + " -z >/dev/null 2>&1";
  [[maybe_unused]] int rc = std::system(cmd.c_str());

  cmd = "callgrind_control --pid=" + pid + " -i on >/dev/null 2>&1";
  rc = std::system(cmd.c_str());
#endif
}

void CallgrindProfiler::afterMeasure(const Stats& /*s*/) {
#ifdef __linux__
  if (!runningUnderValgrind_) {
    return; // Not wrapped at all -- nothing to do.
  }

  // dumpPath is always defined so the post-section can reference it; the
  // file only exists when canToggle_ is true.
  const std::string dumpPath = artifactDir_ + "/callgrind.out";
  if (canToggle_) {
    // Disable instrumentation and dump results
    std::string pid = std::to_string(::getpid());
    std::string cmd = "callgrind_control --pid=" + pid + " -i off >/dev/null 2>&1";
    [[maybe_unused]] int rc = std::system(cmd.c_str());

    cmd = "callgrind_control --pid=" + pid + " -d '" + dumpPath + "' >/dev/null 2>&1";
    rc = std::system(cmd.c_str());
  }
  // If we can't toggle (Docker), valgrind writes callgrind.out.<pid> in the
  // CWD on process exit; the user has to point callgrind_annotate at that.

  std::printf("\n=== Callgrind Profile ===\n");
  std::printf("Output: %s%s\n", artifactDir_.c_str(),
              canToggle_ ? "" : " (or callgrind.out.<pid> in CWD if not toggled)");

  if (cfg_.profileAnalyze) {
    runAnnotateAnalysis();
  } else if (canToggle_) {
    std::printf("   Run with --profile-analyze for automatic annotation\n");
    std::printf("   Or manually: callgrind_annotate %s\n", dumpPath.c_str());
    std::printf("   Or: kcachegrind %s\n", dumpPath.c_str());
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
    return EnvReport{EnvReport::Status::Error,
                     "valgrind binary not found on PATH",
                     "apt install valgrind."};
  }
  // Docker PID namespace breaks callgrind_control attach; warn so users know to
  // expect the wrap-mode fallback (handled by the backend in a later commit).
  if (std::system("grep -q docker /proc/1/cgroup 2>/dev/null") == 0) {
    return EnvReport{EnvReport::Status::Warning,
                     "valgrind available; running in Docker (PID namespace)",
                     "callgrind_control attach will be replaced by direct valgrind wrap."};
  }
  return EnvReport{EnvReport::Status::Ok, "valgrind available", ""};
}

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_PROFILER_BACKEND(
    "callgrind",
    ::vernier::bench::makeCallgrindProfiler,
    ::vernier::bench::checkCallgrindEnvironment,
    "Install valgrind: apt install valgrind.")
