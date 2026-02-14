/**
 * @file ProfilerPerf.cpp
 * @brief Implementation of Linux perf profiler backend.
 */

#include "src/bench/inc/ProfilerPerf.hpp"

#ifdef __linux__
#include <array>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <thread>
#include <chrono>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace vernier {
namespace bench {

PerfStatProfiler::PerfStatProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
#ifdef __linux__
  // Resolve artifact directory: <artifactRoot>/<Suite.Case>.perf/
  if (!cfg_.artifactRoot.empty()) {
    artifactDir_ = cfg_.artifactRoot + "/" + testName_ + ".perf";
  } else {
    artifactDir_ = "./" + testName_ + ".perf";
  }
  std::error_code ec;
  std::filesystem::create_directories(artifactDir_, ec);
#else
  (void)cfg_;
  (void)testName_;
#endif
}

void PerfStatProfiler::beforeMeasure() {
#ifdef __linux__
  if (!isPerfAvailable()) {
    return;
  }

  pid_t targetPid = ::getpid();
  bool useRecord = startsWithTrim(cfg_.profileArgs, "record");

  if (useRecord) {
    // perf record mode
    dataPath_ = artifactDir_ + "/perf.data";
    errPath_ = artifactDir_ + "/record.err.txt";
    // Build: perf record <args> -p PID
    std::string cmd = "perf record ";
    if (!cfg_.profileArgs.empty()) {
      // strip leading "record"
      auto i = cfg_.profileArgs.find_first_not_of(" \t", 6);
      auto rest = (i == std::string::npos) ? std::string{} : cfg_.profileArgs.substr(i);
      cmd += rest + " ";
    }
    cmd += "-p " + std::to_string(targetPid) + " -o '" + dataPath_ + "'";
    launchBackground(cmd, /*stdoutPath*/ "", errPath_);
  } else {
    // perf stat mode
    statPath_ = artifactDir_ + "/stat.txt";
    std::string events = "cpu-cycles,instructions,branches,branch-misses,cache-misses";
    std::string cmd = "perf stat -e " + events + " -p " + std::to_string(targetPid);
    if (!cfg_.profileArgs.empty()) {
      cmd += " " + cfg_.profileArgs;
    }
    // perf stat writes to stderr -> redirect to file
    launchBackground(cmd, /*stdoutPath*/ "", statPath_);
  }

  // Grace period to ensure perf attaches properly before measurement starts
  // Critical for short benchmarks where measurement might start before perf is ready
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
#endif
}

void PerfStatProfiler::afterMeasure(const Stats& /*s*/) {
#ifdef __linux__
  if (childPid_ <= 0) {
    return;
  }

  // ============================================================================
  // Proper perf termination with data flush
  // ============================================================================

  // Step 1: Send SIGINT to allow perf to finalize data gracefully
  if (::kill(childPid_, 0) == 0) { // Check if process exists
    ::kill(childPid_, SIGINT);
  } else {
    childPid_ = -1;
    return; // Process already exited
  }

  // Step 2: Give perf initial time to start shutdown (CRITICAL for perf record)
  // Perf needs time to stop sampling, write buffers, and finalize the data file header
  // This is the most critical step - perf.data header is written during shutdown
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  // Step 3: Wait for perf to finish writing data (with timeout)
  constexpr int TIMEOUT_MS = 5000; // 5 second total timeout
  constexpr int POLL_INTERVAL_MS = 100;
  int elapsed = 1000; // Already waited 1000ms
  int status = 0;

  while (elapsed < TIMEOUT_MS) {
    pid_t result = ::waitpid(childPid_, &status, WNOHANG);

    if (result == childPid_) {
      // Process exited - give filesystem time to flush buffers
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      childPid_ = -1;

      // Verify perf.data was written (for record mode)
      if (!dataPath_.empty()) {
        if (std::filesystem::exists(dataPath_)) {
          auto fileSize = std::filesystem::file_size(dataPath_);
          if (fileSize > 0) {
            // Success: perf.data written and has data
            return;
          } else {
            std::fprintf(stderr,
                         "Warning: perf.data exists but is empty (size=%zu) - perf may not have "
                         "flushed data\n",
                         fileSize);
          }
        } else {
          std::fprintf(stderr,
                       "Warning: perf.data not found - perf may have terminated abnormally\n");
        }
      }
      return;
    } else if (result == -1) {
      // Error in waitpid (process may have been reaped)
      childPid_ = -1;
      return;
    }

    // Process still running, wait a bit more
    std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
    elapsed += POLL_INTERVAL_MS;
  }

  // Step 3: Timeout reached - force kill
  std::fprintf(stderr, "Warning: perf did not exit after %dms - forcing termination\n", TIMEOUT_MS);

  if (::kill(childPid_, 0) == 0) {
    ::kill(childPid_, SIGKILL);
    ::waitpid(childPid_, &status, 0); // Block until killed
  }

  childPid_ = -1;
#endif
}

bool PerfStatProfiler::isPerfAvailable() const {
#ifdef __linux__
  return (std::system("command -v perf >/dev/null 2>&1") == 0);
#else
  return false;
#endif
}

bool PerfStatProfiler::startsWithTrim(const std::string& s, const char* prefix) {
  std::size_t i = 0;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) {
    ++i;
  }
  auto plen = std::strlen(prefix);
  return (s.size() >= i + plen && s.compare(i, plen, prefix) == 0);
}

void PerfStatProfiler::launchBackground(const std::string& cmdCore, const std::string& stdoutPath,
                                        const std::string& stderrPath) {
#ifdef __linux__
  // Build: sh -c "<cmd> >STDOUT 2>STDERR & echo $!"
  std::string cmd = cmdCore;
  std::string redirs;
  if (!stdoutPath.empty()) {
    redirs += " >'" + stdoutPath + "'";
  }
  if (!stderrPath.empty()) {
    redirs += " 2>'" + stderrPath + "'";
  }
  std::string shellCmd = "sh -c '" + cmd + redirs + " & echo $!'";

  FILE* pipe = ::popen(shellCmd.c_str(), "r");
  if (!pipe) {
    return;
  }

  std::array<char, 64> buf{};
  if (::fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
    childPid_ = static_cast<pid_t>(std::strtol(buf.data(), nullptr, 10));
  }
  ::pclose(pipe);
#else
  (void)cmdCore;
  (void)stdoutPath;
  (void)stderrPath;
#endif
}

bool PerfStatProfiler::killChild(int sig) noexcept {
#ifdef __linux__
  if (childPid_ <= 0) {
    return false;
  }
  if (::kill(childPid_, 0) != 0) {
    return false;
  }
  ::kill(childPid_, sig);
  return true;
#else
  (void)sig;
  return false;
#endif
}

// Factory implementation
std::unique_ptr<Profiler> makePerfProfiler(const PerfConfig& cfg, const std::string& testName) {
#ifdef __linux__
  return std::make_unique<PerfStatProfiler>(cfg, testName);
#else
  (void)cfg;
  (void)testName;
  return nullptr;
#endif
}

} // namespace bench
} // namespace vernier