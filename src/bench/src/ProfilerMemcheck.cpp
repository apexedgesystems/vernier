/**
 * @file ProfilerMemcheck.cpp
 * @brief Valgrind Memcheck implementation.
 *
 * Same wrap-externally pattern as callgrind / massif.
 */

#include "src/bench/inc/ProfilerMemcheck.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "src/bench/inc/ProfilerRegistry.hpp"

namespace vernier {
namespace bench {

namespace {

bool isValgrindAvailable() { return std::system("command -v valgrind >/dev/null 2>&1") == 0; }

bool isRunningUnderValgrind() { return std::getenv("RUNNING_ON_VALGRIND") != nullptr; }

} // namespace

/* ----------------------------- MemcheckProfiler ----------------------------- */

MemcheckProfiler::MemcheckProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  artifactDir_ = cfg_.artifactRoot.empty() ? "./" + testName_ + ".memcheck"
                                           : cfg_.artifactRoot + "/" + testName_ + ".memcheck";
  std::error_code ec;
  std::filesystem::create_directories(artifactDir_, ec);
  (void)ec;

  runningUnderValgrind_ = isRunningUnderValgrind();
  if (!runningUnderValgrind_) {
    const bool leakFull = cfg_.profileArgs.find("leak-full") != std::string::npos;
    const bool trackOrigins = cfg_.profileArgs.find("track-origins") != std::string::npos;
    std::fprintf(stderr,
                 "\n[memcheck] NOT running under valgrind; measurement will execute normally but\n"
                 "[memcheck] no memory checking happens. To check:\n"
                 "[memcheck]   valgrind --tool=memcheck%s%s \\\n"
                 "[memcheck]       --log-file=%s/memcheck.log \\\n"
                 "[memcheck]       <this-binary> --profile memcheck --cycles 5 [...]\n\n",
                 leakFull ? " --leak-check=full" : " --leak-check=summary",
                 trackOrigins ? " --track-origins=yes" : "", artifactDir_.c_str());
  }
}

void MemcheckProfiler::beforeMeasure() {
  // Memcheck runs continuously; nothing to toggle. Logs at process exit.
}

void MemcheckProfiler::afterMeasure(const Stats& /*s*/) {
  // Memcheck writes its log at process exit when running under valgrind.
}

/* ----------------------------- Env check ----------------------------- */

EnvReport checkMemcheckEnvironment() {
  if (!isValgrindAvailable()) {
    return EnvReport{EnvReport::Status::Error, "valgrind binary not found on PATH",
                     "apt install valgrind."};
  }
  return EnvReport{EnvReport::Status::Ok, "valgrind available (memcheck is the default tool)", ""};
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeMemcheckProfiler(const PerfConfig& cfg, const std::string& testName) {
  if (!isValgrindAvailable())
    return nullptr;
  return std::make_unique<MemcheckProfiler>(cfg, testName);
}

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_PROFILER_BACKEND("memcheck", ::vernier::bench::makeMemcheckProfiler,
                                  ::vernier::bench::checkMemcheckEnvironment,
                                  "apt install valgrind (memcheck is its default tool).")
