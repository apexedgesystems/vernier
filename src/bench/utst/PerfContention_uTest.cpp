/**
 * @file PerfContention_uTest.cpp
 * @brief Unit tests for how contentionRun() ends when a worker throws or a
 * worker thread cannot be started: as measured() ends when its callback
 * throws, through the abort hook, with nothing published and the exception at
 * the caller.
 */

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfHarness.hpp"
#include "src/bench/inc/PerfRegistry.hpp"
#include "src/bench/inc/PerfStats.hpp"

#include <gtest/gtest.h>

#include <pthread.h>
#include <sys/resource.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace {

using vernier::bench::PerfCase;
using vernier::bench::PerfConfig;
using vernier::bench::PerfRegistry;
using vernier::bench::Stats;

/// True in a build with the address or the thread sanitizer, whose runtimes
/// map memory as they go and fail on their own under a limit on the address
/// space.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
constexpr bool SANITIZER_MAPS_MEMORY = true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
constexpr bool SANITIZER_MAPS_MEMORY = true;
#else
constexpr bool SANITIZER_MAPS_MEMORY = false;
#endif
#else
constexpr bool SANITIZER_MAPS_MEMORY = false;
#endif

/** @brief This process's address space in bytes (VmSize), 0 when it cannot be read. */
std::uint64_t addressSpaceBytes() {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("VmSize:", 0) == 0) {
      return std::stoull(line.substr(7)) * 1024;
    }
  }
  return 0;
}

/** @brief The stack size a new thread gets, 0 when it cannot be read. */
std::uint64_t defaultStackBytes() {
  pthread_attr_t attr;
  if (::pthread_getattr_default_np(&attr) != 0) {
    return 0;
  }
  std::size_t size = 0;
  const int RC = ::pthread_attr_getstacksize(&attr, &size);
  ::pthread_attr_destroy(&attr);
  return RC == 0 ? size : 0;
}

/** @brief What a worker throws, with a mark to recognise it by. */
struct WorkerFailed {
  int mark = 0;
};

/**
 * @brief A case of @p threads workers making 20 calls a repeat, 3 repeats,
 * whose hooks append their names to @p hooks.
 */
PerfCase hookedCase(const std::string& name, int threads, std::vector<std::string>& hooks) {
  PerfConfig cfg;
  cfg.cycles = 20;
  cfg.repeats = 3;
  cfg.threads = threads;
  PerfCase pc{name, cfg};
  pc.setBeforeMeasureHook([&hooks](const PerfCase&) { hooks.emplace_back("before"); });
  pc.setAfterMeasureHook([&hooks](const PerfCase&, const Stats&) { hooks.emplace_back("after"); });
  pc.setAbortMeasureHook([&hooks](const PerfCase&) { hooks.emplace_back("abort"); });
  return pc;
}

} // namespace

/**
 * @test A worker that throws ends the run on the calling thread: every worker
 * joined, the abort hook in place of the after hook, no row published, and
 * the exception at the caller with its mark. The case then runs again,
 * through both hooks, and publishes its row.
 */
TEST(PerfContentionTest, WorkerThatThrowsEndsTheRunThroughTheAbortHook) {
  (void)PerfRegistry::instance().takeAll();
  std::vector<std::string> hooks;
  PerfCase pc = hookedCase("Contention.Throws", 4, hooks);
  std::atomic<int> calls{0};
  try {
    (void)pc.contentionRun([&calls] {
      if (calls.fetch_add(1) == 10) {
        throw WorkerFailed{7};
      }
    });
    ADD_FAILURE() << "the worker's exception did not reach the caller";
  } catch (const WorkerFailed& failed) {
    EXPECT_EQ(failed.mark, 7) << "another exception reached the caller";
  }
  EXPECT_EQ(hooks, (std::vector<std::string>{"before", "abort"}));
  EXPECT_FALSE(PerfRegistry::instance().take().has_value()) << "the failed run published a row";

  (void)pc.contentionRun([] {});
  EXPECT_EQ(hooks, (std::vector<std::string>{"before", "abort", "before", "after"}));
  EXPECT_TRUE(PerfRegistry::instance().take().has_value()) << "the next run published no row";
}

/**
 * @test A worker thread that cannot be started (RLIMIT_NPROC 0) ends the run
 * as a worker's exception does: the abort hook, no row, and std::system_error
 * with EAGAIN at the caller. With threads allowed again the case runs through
 * both hooks. Skipped where a thread starts past the limit (root, or
 * CAP_SYS_RESOURCE).
 */
TEST(PerfContentionTest, WorkerThatCannotStartEndsTheRunThroughTheAbortHook) {
  (void)PerfRegistry::instance().takeAll();
  std::vector<std::string> hooks;
  PerfCase pc = hookedCase("Contention.NoThread", 2, hooks);
  struct rlimit original{};
  ASSERT_EQ(::getrlimit(RLIMIT_NPROC, &original), 0);
  struct rlimit none = original;
  none.rlim_cur = 0;
  ASSERT_EQ(::setrlimit(RLIMIT_NPROC, &none), 0);
  bool threadStarted = false;
  try {
    std::thread probe([] {});
    probe.join();
    threadStarted = true;
  } catch (const std::system_error&) {
    // No thread starts: what the case needs.
  }
  std::optional<std::error_code> code;
  if (!threadStarted) {
    try {
      (void)pc.contentionRun([] {});
    } catch (const std::system_error& error) {
      code = error.code();
    }
  }
  ASSERT_EQ(::setrlimit(RLIMIT_NPROC, &original), 0);
  if (threadStarted) {
    GTEST_SKIP() << "a thread started past RLIMIT_NPROC 0 (root, or CAP_SYS_RESOURCE)";
  }
  ASSERT_TRUE(code.has_value()) << "contentionRun() returned though no thread could start";
  EXPECT_EQ(*code, std::errc::resource_unavailable_try_again) << code->message();
  EXPECT_EQ(hooks, (std::vector<std::string>{"before", "abort"}));
  EXPECT_FALSE(PerfRegistry::instance().take().has_value()) << "the failed run published a row";

  (void)pc.contentionRun([] {});
  EXPECT_EQ(hooks, (std::vector<std::string>{"before", "abort", "before", "after"}));
  EXPECT_TRUE(PerfRegistry::instance().take().has_value()) << "the next run published no row";
}

/**
 * @test A run that starts some of its workers and then cannot start the next
 * ends as one that starts none: the workers started make their calls and are
 * joined (left joinable, they would end the process), then the abort hook
 * runs and std::system_error reaches the caller. The address space is limited
 * to what holds three more thread stacks and a margin, so between 3 and about
 * 8 of the 64 workers start (glibc reuses up to 40 MiB of cached stacks).
 * Skipped under the address and thread sanitizers.
 */
TEST(PerfContentionTest, WorkerThatCannotStartAfterOthersEndsTheRunThroughTheAbortHook) {
  if (SANITIZER_MAPS_MEMORY) {
    GTEST_SKIP() << "the sanitizer's runtime maps memory under the limit this case sets";
  }
  (void)PerfRegistry::instance().takeAll();
  std::vector<std::string> hooks;
  PerfCase pc = hookedCase("Contention.SomeThreads", 64, hooks);
  std::atomic<int> calls{0};
  const std::uint64_t STACK = defaultStackBytes();
  const std::uint64_t SPACE = addressSpaceBytes();
  ASSERT_GT(STACK, 0U);
  ASSERT_GT(SPACE, 0U);
  struct rlimit original{};
  ASSERT_EQ(::getrlimit(RLIMIT_AS, &original), 0);
  struct rlimit limited = original;
  limited.rlim_cur = SPACE + 3 * (STACK + (std::uint64_t{1} << 20)) + (std::uint64_t{6} << 20);
  ASSERT_EQ(::setrlimit(RLIMIT_AS, &limited), 0);
  std::optional<std::error_code> code;
  try {
    (void)pc.contentionRun([&calls] { calls.fetch_add(1, std::memory_order_relaxed); });
  } catch (const std::system_error& error) {
    code = error.code();
  }
  ASSERT_EQ(::setrlimit(RLIMIT_AS, &original), 0);
  ASSERT_TRUE(code.has_value()) << "all 64 workers started under the limit";
  EXPECT_EQ(*code, std::errc::resource_unavailable_try_again) << code->message();
  EXPECT_GT(calls.load(), 0) << "no worker started before the one that could not";
  EXPECT_EQ(calls.load() % 20, 0) << "a started worker did not make all its calls";
  EXPECT_EQ(hooks, (std::vector<std::string>{"before", "abort"}));
  EXPECT_FALSE(PerfRegistry::instance().take().has_value()) << "the failed run published a row";
}
