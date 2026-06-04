/**
 * @file ProfilerMassif.cpp
 * @brief Valgrind Massif heap profiler implementation.
 *
 * Same wrap-externally pattern as callgrind: the binary running with
 * --profile massif is passive when not under valgrind; under valgrind it
 * cooperates by setting up the artifact dir for ms_print output.
 */

#include "src/bench/inc/ProfilerMassif.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#include "src/bench/inc/ProfilerRegistry.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- Helpers ----------------------------- */

namespace {

bool isValgrindAvailable() { return std::system("command -v valgrind >/dev/null 2>&1") == 0; }

// Detect a live valgrind by scanning /proc/self/maps for the vgpreload module;
// this is reliable regardless of how the tool was launched. RUNNING_ON_VALGRIND
// is a valgrind client request rather than an environment variable, so a
// getenv() check cannot stand in for the maps scan.
bool isRunningUnderValgrind() {
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

} // namespace

/* ----------------------------- MassifProfiler ----------------------------- */

MassifProfiler::MassifProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  artifactDir_ = cfg_.artifactRoot.empty() ? "./" + testName_ + ".massif"
                                           : cfg_.artifactRoot + "/" + testName_ + ".massif";
  std::error_code ec;
  std::filesystem::create_directories(artifactDir_, ec);
  (void)ec;

  runningUnderValgrind_ = isRunningUnderValgrind();
  if (!runningUnderValgrind_) {
    const bool PAGES = cfg_.profileArgs.find("pages") != std::string::npos;
    const bool STACKS = cfg_.profileArgs.find("stacks") != std::string::npos;
    std::fprintf(stderr,
                 "\n[massif] NOT running under valgrind; measurement will execute normally but\n"
                 "[massif] no heap profile will be collected. To collect:\n"
                 "[massif]   valgrind --tool=massif%s%s \\\n"
                 "[massif]       --massif-out-file=%s/massif.out \\\n"
                 "[massif]       <this-binary> --profile massif [...]\n"
                 "[massif] Then: ms_print %s/massif.out | head -40\n\n",
                 PAGES ? " --pages-as-heap=yes" : "", STACKS ? " --stacks=yes" : "",
                 artifactDir_.c_str(), artifactDir_.c_str());
  }
}

void MassifProfiler::beforeMeasure() {
  // Massif samples continuously while running under valgrind; nothing to toggle.
}

void MassifProfiler::afterMeasure(const Stats& /*s*/) {
  // Massif writes its output file at process exit when running under valgrind.
}

/* ----------------------------- Env check ----------------------------- */

EnvReport checkMassifEnvironment() {
  if (!isValgrindAvailable()) {
    return EnvReport{EnvReport::Status::Error, "valgrind binary not found on PATH",
                     "apt install valgrind."};
  }
  return EnvReport{EnvReport::Status::Ok, "valgrind available (massif tool ships with it)", ""};
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeMassifProfiler(const PerfConfig& cfg, const std::string& testName) {
  if (!isValgrindAvailable())
    return nullptr;
  return std::make_unique<MassifProfiler>(cfg, testName);
}

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_PROFILER_BACKEND("massif", ::vernier::bench::makeMassifProfiler,
                                  ::vernier::bench::checkMassifEnvironment,
                                  "apt install valgrind (massif ships with it).")
