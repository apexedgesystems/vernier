/**
 * @file RunStatusTarget.cpp
 * @brief A benchmark that ends as its first argument says, for the tests of
 * the exit status record libbench keeps for bench run.
 *
 * Built for the tests only, never installed. bench run asks for the record
 * when the tool wrapping the benchmark does not pass the benchmark's exit
 * status through; this program ends in each of the ways a benchmark can, so
 * the tests can read what each leaves. One mode per run:
 *
 *   finish <status> [<profile>]  returns ProfilerRegistry::finishRun() for a
 *                                run whose tests ended with <status>, under
 *                                --profile <profile> when one is given
 *   abi                          calls the ABI check with a version this
 *                                libbench was not built with, which ends the
 *                                process with status 3
 *   abort                        aborts, as a crash ends a benchmark
 *   return <status>              returns <status> from main() without
 *                                finishRun(), as a main() of its own can
 *   fork                         a child of fork() returns finishRun() for a
 *                                run whose tests ended with 7; this process
 *                                waits for it, then ends with status 9 without
 *                                finishRun()
 *   spawn                        runs this program again as a child, in
 *                                "finish 7", waits for it, then ends with
 *                                status 9 without finishRun()
 *   environment                  prints "token: set" or "token: unset" for the
 *                                record's variable, then returns finishRun()
 *                                for a run whose tests passed
 *
 * When the first argument is none of these, as when bench run starts the
 * program with its own options first, the mode's words come from
 * RUN_STATUS_TARGET_MODE (e.g. "finish 1"), and the arguments are ignored.
 */

#include "src/bench/inc/PerfAbi.hpp"
#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
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

/// The status a child ended with, or -1 when it did not exit.
int childStatus(pid_t child) {
  int raw = 0;
  if (::waitpid(child, &raw, 0) != child || !WIFEXITED(raw)) {
    return -1;
  }
  return WEXITSTATUS(raw);
}

} // namespace

int main(int argc, char** argv) {
  std::vector<std::string> words(argv + 1, argv + argc);
  const std::string FIRST = words.empty() ? "" : words.front();
  const char* FROM_ENVIRONMENT = std::getenv("RUN_STATUS_TARGET_MODE");
  if (FIRST != "finish" && FIRST != "abi" && FIRST != "abort" && FIRST != "return" &&
      FIRST != "fork" && FIRST != "spawn" && FIRST != "environment" &&
      FROM_ENVIRONMENT != nullptr) {
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
  if (MODE == "return" && SECOND != nullptr) {
    return std::atoi(SECOND);
  }
  if (MODE == "fork") {
    const pid_t CHILD = ::fork();
    if (CHILD == 0) {
      std::exit(finish(7, nullptr));
    }
    std::printf("child ended with status %d\n", childStatus(CHILD));
    std::fflush(stdout);
    ::_exit(9);
  }
  if (MODE == "spawn") {
    const pid_t CHILD = ::fork();
    if (CHILD == 0) {
      ::execl("/proc/self/exe", argv[0], "finish", "7", static_cast<char*>(nullptr));
      ::_exit(127);
    }
    std::printf("child ended with status %d\n", childStatus(CHILD));
    std::fflush(stdout);
    ::_exit(9);
  }
  if (MODE == "environment") {
    std::printf("token: %s\n", std::getenv("VERNIER_EXTERNAL_WRAP_STATUS") ? "set" : "unset");
    std::fflush(stdout);
    return finish(0, nullptr);
  }
  std::fprintf(stderr, "RunStatusTarget: unknown mode '%s'\n", MODE.c_str());
  return 64;
}
