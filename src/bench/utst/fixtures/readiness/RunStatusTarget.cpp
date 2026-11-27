/**
 * @file RunStatusTarget.cpp
 * @brief A benchmark that ends as its first argument says, for the tests of
 * how bench run reads a benchmark's end under heaptrack.
 *
 * Built for the tests only, never installed. bench run takes the benchmark's
 * exit status from heaptrack when heaptrack passes it through, and from the
 * shell that waits on the benchmark when it does not; this program ends in
 * each of the ways a benchmark can, so the tests can read how each run ends.
 * One mode per run:
 *
 *   finish <status> [<profile>]  returns ProfilerRegistry::finishRun() for a
 *                                run whose tests ended with <status>, under
 *                                --profile <profile> when one is given
 *   abi                          calls the ABI check with a version this
 *                                libbench was not built with, which ends the
 *                                process with status 3
 *   abort                        aborts, as a crash ends a benchmark
 *   finish-abort-local <status>  returns finishRun()'s status from main(), and a
 *                                local object's destructor aborts on the way out
 *   finish-abort-static <status> returns finishRun()'s status from main(), and a
 *                                static object's destructor aborts in the
 *                                process's cleanup after main() has returned
 *   finish-exit-local <code>     returns finishRun()'s status for a run whose
 *                                tests passed, and a local object's destructor
 *                                ends the process with <code> on the way out
 *   finish-exit-late <code>      returns finishRun()'s status for a run whose
 *                                tests passed, and a static object's destructor
 *                                ends the process with <code> in its cleanup
 *   return <status>              returns <status> from main() without
 *                                finishRun(), as a main() of its own can
 *
 * When the first argument is none of these, as when bench run starts the
 * program with its own options first, the mode's words come from
 * RUN_STATUS_TARGET_MODE (e.g. "finish 1"), and the arguments are ignored.
 */

#include "src/bench/inc/PerfAbi.hpp"
#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

/// The run's end for a run whose tests ended with @p testStatus.
int finish(int testStatus, const char* profile) {
  vernier::bench::PerfConfig cfg;
  if (profile != nullptr) {
    cfg.profileTool = profile;
  }
  return vernier::bench::ProfilerRegistry::instance().finishRun(cfg, testStatus, 1);
}

/// An object whose destructor aborts the process.
struct AbortOnDestruction {
  AbortOnDestruction() = default;
  AbortOnDestruction(const AbortOnDestruction&) = delete;
  AbortOnDestruction& operator=(const AbortOnDestruction&) = delete;
  ~AbortOnDestruction() { std::abort(); }
};

/// An object whose destructor ends the process with the status it was given.
struct ExitOnDestruction {
  int code = 0;
  ~ExitOnDestruction() { std::_Exit(code); }
};

} // namespace

int main(int argc, char** argv) {
  static const std::set<std::string> MODES = {"finish",
                                              "abi",
                                              "abort",
                                              "finish-abort-local",
                                              "finish-abort-static",
                                              "finish-exit-local",
                                              "finish-exit-late",
                                              "return"};
  std::vector<std::string> words(argv + 1, argv + argc);
  const char* FROM_ENVIRONMENT = std::getenv("RUN_STATUS_TARGET_MODE");
  if ((words.empty() || MODES.count(words.front()) == 0) && FROM_ENVIRONMENT != nullptr) {
    words.clear();
    std::istringstream in(FROM_ENVIRONMENT);
    for (std::string word; in >> word;) {
      words.push_back(word);
    }
  }
  const std::string MODE = words.empty() ? "" : words[0];
  const char* SECOND = words.size() > 1 ? words[1].c_str() : nullptr;
  if (MODE == "finish" && SECOND != nullptr) {
    return finish(std::atoi(SECOND), words.size() > 2 ? words[2].c_str() : nullptr);
  }
  if (MODE == "abi") {
    vernier::bench::checkBenchAbi(vernier::bench::BENCH_ABI_VERSION + 1,
                                  sizeof(vernier::bench::PerfConfig),
                                  sizeof(vernier::bench::Stats));
    return 0;
  }
  if (MODE == "abort") {
    std::abort();
  }
  if (MODE == "finish-abort-local" && SECOND != nullptr) {
    const AbortOnDestruction GUARD;
    return finish(std::atoi(SECOND), nullptr);
  }
  if (MODE == "finish-abort-static" && SECOND != nullptr) {
    static const AbortOnDestruction GUARD;
    return finish(std::atoi(SECOND), nullptr);
  }
  if (MODE == "finish-exit-local" && SECOND != nullptr) {
    const ExitOnDestruction GUARD{std::atoi(SECOND)};
    return finish(0, nullptr);
  }
  if (MODE == "finish-exit-late" && SECOND != nullptr) {
    static ExitOnDestruction late;
    late.code = std::atoi(SECOND);
    return finish(0, nullptr);
  }
  if (MODE == "return" && SECOND != nullptr) {
    return std::atoi(SECOND);
  }
  std::fprintf(stderr, "RunStatusTarget: unknown mode '%s'\n", MODE.c_str());
  return 64;
}
