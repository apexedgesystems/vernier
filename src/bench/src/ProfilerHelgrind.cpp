/**
 * @file ProfilerHelgrind.cpp
 * @brief Valgrind Helgrind / DRD thread-error detector implementation.
 *
 * Same wrap-externally pattern as memcheck / massif. Valgrind presence is
 * detected by scanning /proc/self/maps for the vgpreload library, which is
 * mapped whenever the process actually runs under valgrind (an env-var check
 * is unreliable -- valgrind exposes RUNNING_ON_VALGRIND as a client request,
 * not an environment variable).
 */

#include "src/bench/inc/ProfilerHelgrind.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#include "src/bench/inc/ProfilerRegistry.hpp"

namespace vernier {
namespace bench {

namespace {

bool isValgrindAvailable() { return std::system("command -v valgrind >/dev/null 2>&1") == 0; }

// Detect a live valgrind instrumentation by its preload library in the process
// memory map -- present only when the binary actually runs under valgrind.
bool detectUnderValgrind() {
  const char* preload = std::getenv("LD_PRELOAD");
  if (preload && std::strstr(preload, "vgpreload"))
    return true;
  std::FILE* fp = std::fopen("/proc/self/maps", "r");
  if (!fp)
    return false;
  char line[512];
  bool found = false;
  while (std::fgets(line, sizeof(line), fp)) {
    if (std::strstr(line, "vgpreload") || std::strstr(line, "/valgrind/")) {
      found = true;
      break;
    }
  }
  std::fclose(fp);
  return found;
}

// Helgrind by default; DRD when the user passes --profile-args drd.
const char* selectTool(const PerfConfig& cfg) {
  return cfg.profileArgs.find("drd") != std::string::npos ? "drd" : "helgrind";
}

} // namespace

/* ----------------------------- HelgrindProfiler ----------------------------- */

HelgrindProfiler::HelgrindProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  artifactDir_ = cfg_.artifactRoot.empty() ? "./" + testName_ + ".helgrind"
                                           : cfg_.artifactRoot + "/" + testName_ + ".helgrind";
  std::error_code ec;
  std::filesystem::create_directories(artifactDir_, ec);
  (void)ec;

  runningUnderValgrind_ = detectUnderValgrind();
  if (!runningUnderValgrind_) {
    const char* tool = selectTool(cfg_);
    std::fprintf(stderr,
                 "\n[helgrind] NOT running under valgrind; measurement will execute normally but\n"
                 "[helgrind] no thread-error checking happens. To check:\n"
                 "[helgrind]   valgrind --tool=%s \\\n"
                 "[helgrind]       --log-file=%s/helgrind.log \\\n"
                 "[helgrind]       <this-binary> --profile helgrind --cycles 5 [...]\n\n",
                 tool, artifactDir_.c_str());
  }
}

void HelgrindProfiler::beforeMeasure() {
  // Helgrind/DRD instrument continuously; nothing to toggle. Logs at exit.
}

void HelgrindProfiler::afterMeasure(const Stats& /*s*/) {
  // Valgrind writes its log at process exit when running under the tool.
}

/* ----------------------------- Env check ----------------------------- */

EnvReport checkHelgrindEnvironment() {
  if (!isValgrindAvailable()) {
    return EnvReport{EnvReport::Status::Error, "valgrind binary not found on PATH",
                     "apt install valgrind."};
  }
  return EnvReport{EnvReport::Status::Ok,
                   "valgrind available (helgrind + drd thread-error detectors ship with it)", ""};
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeHelgrindProfiler(const PerfConfig& cfg, const std::string& testName) {
  if (!isValgrindAvailable())
    return nullptr;
  return std::make_unique<HelgrindProfiler>(cfg, testName);
}

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_PROFILER_BACKEND("helgrind", ::vernier::bench::makeHelgrindProfiler,
                                  ::vernier::bench::checkHelgrindEnvironment,
                                  "apt install valgrind (helgrind + drd ship with it).")
