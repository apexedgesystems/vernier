/**
 * @file PerfMetadata_uTest.cpp
 * @brief captureMetadata must be safe under concurrent publishers.
 *
 * The GPU harness publishes results from more than one thread (main plus
 * the worker CUPTI flushes ride on), and every publish captures metadata.
 * The cache must therefore initialize exactly once under contention and
 * hand out identical, immutable values afterwards. Found as a real
 * cycles-dependent segfault on aarch64: the old shape assigned bare
 * statics behind an unguarded flag while captureGitHash's subprocess
 * held the window open. Run under TSan (make tsan) this test convicts
 * that shape; on the magic-static shape it is silent.
 */
#include <gtest/gtest.h>

#include <barrier>
#include <string>
#include <thread>
#include <vector>

#include "src/bench/inc/PerfHarness.hpp"

namespace {

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

} // namespace
