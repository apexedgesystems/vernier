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
  // Check the RUNNING_ON_VALGRIND environment variable or /proc/self/status
  // Valgrind sets the parent name visible in /proc
  if (std::getenv("RUNNING_ON_VALGRIND") != nullptr) {
    return true;
  }

  // Alternative: check if callgrind_control can talk to us
  std::string cmd = "callgrind_control --pid=" + std::to_string(::getpid()) + " -s >/dev/null 2>&1";
  return (std::system(cmd.c_str()) == 0);
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

  runningUnderValgrind_ = isCallgrindControlAvailable() && isRunningUnderValgrind();

  if (!runningUnderValgrind_) {
    std::fprintf(stderr, "\n[INFO] Callgrind profiler: not running under valgrind.\n"
                         "   For instruction-level profiling, run:\n"
                         "     valgrind --tool=callgrind --instr-atstart=no \\\n"
                         "       ./<binary> --profile callgrind [other flags]\n"
                         "   Measurements will proceed normally without instrumentation.\n\n");
  }
#else
  (void)cfg_;
  (void)testName_;
#endif
}

void CallgrindProfiler::beforeMeasure() {
#ifdef __linux__
  if (!runningUnderValgrind_) {
    return;
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
    return;
  }

  // Disable instrumentation and dump results
  std::string pid = std::to_string(::getpid());
  std::string cmd = "callgrind_control --pid=" + pid + " -i off >/dev/null 2>&1";
  [[maybe_unused]] int rc = std::system(cmd.c_str());

  // Dump to artifact directory
  std::string dumpPath = artifactDir_ + "/callgrind.out";
  cmd = "callgrind_control --pid=" + pid + " -d '" + dumpPath + "' >/dev/null 2>&1";
  rc = std::system(cmd.c_str());

  std::printf("\n=== Callgrind Profile ===\n");
  std::printf("Output: %s\n", artifactDir_.c_str());

  if (cfg_.profileAnalyze) {
    runAnnotateAnalysis();
  } else {
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
