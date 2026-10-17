/**
 * @file AbiProfilerClient.cpp
 * @brief A benchmark built from this tree's headers that runs the start-up
 * check, then calls Profiler::abortMeasure() on a profiler its library built.
 *
 * Linked once against a library built from the same headers (the matching
 * pair) and once against one built before the abortMeasure() slot existed
 * (the mixed pair), which the check must refuse before the profiler is built:
 * past the check, the call goes through a slot the library's profiler does
 * not have. A crash in that call is caught and reported as such.
 *
 * Usage: AbiProfilerClient
 */

#include "src/bench/inc/PerfAbi.hpp"
#include "src/bench/inc/Profiler.hpp"

#include <unistd.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>

extern "C" vernier::bench::Profiler* abiLibraryProfiler();

namespace {

void onCrash(int /*sig*/) {
  static const char MESSAGE[] = "client: crashed in the call through the profiler's vtable\n";
  (void)::write(STDERR_FILENO, MESSAGE, sizeof(MESSAGE) - 1);
  std::_Exit(77);
}

} // namespace

int main() {
  (void)std::signal(SIGSEGV, onCrash);
  (void)std::signal(SIGBUS, onCrash);
  (void)std::signal(SIGILL, onCrash);
  vernier::bench::ensureBenchAbi();
  std::fprintf(stderr, "client: the start-up check passed\n");
  vernier::bench::Profiler* profiler = abiLibraryProfiler();
  std::fprintf(stderr, "client: constructed '%s'\n", profiler->toolName().c_str());
  profiler->abortMeasure();
  std::fprintf(stderr, "client: abortMeasure() returned\n");
  delete profiler;
  return 0;
}
