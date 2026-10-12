/**
 * @file PerfMetadata_uTest.cpp
 * @brief The run's metadata: captureMetadata is safe under concurrent
 *        publishers, and the git it starts runs without a profiler's wrap.
 *
 * The GPU harness publishes results from more than one thread (main plus
 * the worker CUPTI flushes ride on), and every publish captures metadata.
 * The cache must therefore initialize exactly once under contention and
 * hand out identical, immutable values afterwards. Found as a real
 * cycles-dependent segfault on aarch64: the old shape assigned bare
 * statics behind an unguarded flag while captureGitHash's subprocess
 * held the window open. Run under TSan (make tsan) this test convicts
 * that shape; on the magic-static shape it is silent.
 *
 * A benchmark started under a tool's wrap carries the variables that put
 * the tool into a process (bench run's jemalloc wrap sets LD_PRELOAD and
 * MALLOC_CONF); the `git describe` behind the gitHash column must start
 * without them, or the tool records git too.
 */
#include <gtest/gtest.h>

#include <sys/stat.h>

#include <barrier>
#include <cstdlib>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "src/bench/inc/PerfHarness.hpp"
#include "src/bench/utst/ReadinessFixtures.hpp"

namespace {

using vernier::bench::test::FakeToolDir;
using vernier::bench::test::ScopedEnv;

/** @brief The value of @p name in @p env (one NAME=value per line), if it is set. */
std::optional<std::string> envValue(const std::string& env, const std::string& name) {
  const std::string PREFIX = name + "=";
  std::istringstream in(env);
  std::string line;
  while (std::getline(in, line)) {
    if (line.compare(0, PREFIX.size(), PREFIX) == 0) {
      return line.substr(PREFIX.size());
    }
  }
  return std::nullopt;
}

TEST(PerfMetadataTest, ConcurrentCaptureIsSafeAndStable) {
  constexpr int THREADS = 8;
  constexpr int ITERS = 32;

  std::vector<std::string> firstHash(THREADS);
  std::vector<std::string> firstHost(THREADS);
  std::barrier startLine(THREADS);
  std::vector<std::thread> pool;
  pool.reserve(THREADS);

  for (int t = 0; t < THREADS; ++t) {
    pool.emplace_back([&, t] {
      startLine.arrive_and_wait();
      for (int i = 0; i < ITERS; ++i) {
        const auto [TS, HASH, HOST, PLATFORM] = vernier::bench::captureMetadata(true);
        (void)TS;
        (void)PLATFORM;
        if (i == 0) {
          firstHash[t] = HASH;
          firstHost[t] = HOST;
        } else {
          // Cached values are immutable after first capture: any drift
          // means a caller observed a torn or re-assigned cache.
          ASSERT_EQ(HASH, firstHash[t]);
          ASSERT_EQ(HOST, firstHost[t]);
        }
      }
    });
  }
  for (auto& th : pool) {
    th.join();
  }

  // Every thread saw the same cache, not per-thread copies of a race.
  for (int t = 1; t < THREADS; ++t) {
    EXPECT_EQ(firstHash[t], firstHash[0]);
    EXPECT_EQ(firstHost[t], firstHost[0]);
  }
}

TEST(PerfMetadataTest, UncachedPathStaysFresh) {
  // cacheMetadata=false must not touch or depend on the cache.
  const auto [TS1, HASH1, HOST1, PLATFORM1] = vernier::bench::captureMetadata(false);
  const auto [TS2, HASH2, HOST2, PLATFORM2] = vernier::bench::captureMetadata(false);
  EXPECT_EQ(HASH1, HASH2);
  EXPECT_EQ(HOST1, HOST2);
  EXPECT_EQ(PLATFORM1, PLATFORM2);
  (void)TS1;
  (void)TS2;
}

/**
 * @test git describe runs without the variables that put a profiler into a
 *       process.
 *
 * They are set in this process as a wrap sets them, and a stand-in git first
 * on PATH writes the environment it was given to its log and answers
 * describe. None of them reaches it, an ordinary variable does, and its answer
 * is the hash.
 */
TEST(PerfMetadataTest, GitDescribeRunsWithoutTheWrap) {
  const FakeToolDir DIR;
  ASSERT_TRUE(DIR.ok());
  const std::string GIT = DIR.writeFile("git", "#!/bin/sh\n"
                                               "/usr/bin/env > \"$FAKE_LOG\"\n"
                                               "case \"${1:-}\" in\n"
                                               "describe) echo v9.8.7-stand-in ;;\n"
                                               "*) exit 1 ;;\n"
                                               "esac\n");
  ASSERT_EQ(::chmod(GIT.c_str(), 0755), 0);
  const char* PATH_NOW = std::getenv("PATH");
  const ScopedEnv PATH("PATH",
                       DIR.path() + ":" + (PATH_NOW != nullptr ? PATH_NOW : "/usr/bin:/bin"));
  const ScopedEnv LOG("FAKE_LOG", DIR.logPath());
  const ScopedEnv ORDINARY("VERNIER_TEST_ORDINARY", "kept");

  // What a wrap puts into the process it starts: bench run's jemalloc wrap
  // the first two, and the variables of the other tools' wraps. LD_PRELOAD
  // names the C library, which every process maps already, so the stand-in
  // runs the same whether or not the variable reaches it.
  const std::vector<std::pair<std::string, std::string>> WRAP = {
      {"LD_PRELOAD", "libc.so.6"},
      {"MALLOC_CONF",
       "prof:true,prof_final:true,prof_accum:true,prof_prefix:" + DIR.path() + "/jeprof"},
      {"HEAPTRACK_OUTPUT", DIR.path() + "/run"},
      {"CUDA_INJECTION64_PATH", DIR.path() + "/injection.so"},
      {"ROCP_TOOL_LIB", DIR.path() + "/tool.so"},
      {"ROCPROFILER_LIBRARY", DIR.path() + "/rocprofiler.so"},
      {"VERNIER_EXTERNAL_WRAP", "jemalloc"},
      {"VERNIER_EXTERNAL_WRAP_DIR", DIR.path()}};
  std::vector<std::unique_ptr<ScopedEnv>> wrapped;
  for (const auto& [NAME, VALUE] : WRAP) {
    wrapped.push_back(std::make_unique<ScopedEnv>(NAME.c_str(), VALUE));
  }

  EXPECT_EQ(vernier::bench::captureGitHash(), "v9.8.7-stand-in");

  const std::string ENV = DIR.log();
  ASSERT_FALSE(ENV.empty()) << "the stand-in git did not run";
  for (const auto& [NAME, VALUE] : WRAP) {
    const std::optional<std::string> SEEN = envValue(ENV, NAME);
    EXPECT_FALSE(SEEN.has_value()) << NAME << " reached git as '" << SEEN.value_or("") << "'";
  }
  EXPECT_EQ(envValue(ENV, "VERNIER_TEST_ORDINARY").value_or("<unset>"), "kept")
      << "an ordinary variable did not reach git";
}

} // namespace
