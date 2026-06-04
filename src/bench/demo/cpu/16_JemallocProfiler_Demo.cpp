/**
 * @file 16_JemallocProfiler_Demo.cpp
 * @brief Demo 16: jemalloc prof sampling -- find the allocation hotspot.
 *
 * jemalloc's heap profiler samples allocations (sampled by bytes, so it stays
 * cheap on allocation-heavy code) and attributes them to call stacks. `jeprof`
 * then ranks the sites by sampled bytes, surfacing the allocation hotspot the
 * same way a CPU profiler surfaces a compute hotspot.
 *
 * Two variants exercise the same work at different allocation churn:
 *   Slow: ChurningStrings -- a fresh std::string built per iteration (heap)
 *   Fast: ReusedBuffer    -- one string reused across iterations (no churn)
 *
 * Story: building a transient string by repeated append allocates and grows on
 * the heap every iteration. jemalloc's sampled profile shows the append/string
 * growth as the dominant site; reusing one buffer (clear + append into reserved
 * capacity) makes the sampled bytes collapse.
 *
 * Usage:
 *   @code{.sh}
 *   # Resolve libjemalloc, enable prof sampling, and run the slow path:
 *   LD_PRELOAD=$(ldconfig -p | awk '/libjemalloc.so / {print $4; exit}') \
 *   MALLOC_CONF=prof:true,prof_prefix:/tmp/jeprof.slow \
 *       ./build/native-linux-debug/bin/ptests/BenchDemo_16_JemallocProfiler \
 *       --profile jemalloc --cycles 200 --gtest_filter='Jemalloc.ChurningStrings'
 *   jeprof --text \
 *       ./build/native-linux-debug/bin/ptests/BenchDemo_16_JemallocProfiler \
 *       /tmp/jeprof.slow.*.heap | head -20
 *
 *   # Fast path -- reused buffer, far fewer sampled bytes:
 *   LD_PRELOAD=$(ldconfig -p | awk '/libjemalloc.so / {print $4; exit}') \
 *   MALLOC_CONF=prof:true,prof_prefix:/tmp/jeprof.fast \
 *       ./build/native-linux-debug/bin/ptests/BenchDemo_16_JemallocProfiler \
 *       --profile jemalloc --cycles 200 --gtest_filter='Jemalloc.ReusedBuffer'
 *   jeprof --text \
 *       ./build/native-linux-debug/bin/ptests/BenchDemo_16_JemallocProfiler \
 *       /tmp/jeprof.fast.*.heap | head -20
 *   @endcode
 *
 * @see docs/22_JEMALLOC_PROFILER.md for the step-by-step walkthrough.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include "src/bench/inc/Perf.hpp"

static constexpr std::size_t APPENDS = 4096;

/**
 * @test Slow: a fresh string built by repeated append each iteration.
 *
 * Each iteration allocates a new std::string and grows it on the heap as it is
 * appended to. jemalloc's sampled profile attributes the bulk of sampled bytes
 * to this string growth, ranking it at the top of the `jeprof --text` output.
 */
PERF_THROUGHPUT(Jemalloc, ChurningStrings) {
  UB_PERF_GUARD(perf);

  perf.warmup([&] {
    std::string s;
    for (std::size_t i = 0; i < APPENDS; ++i)
      s.append("payload");
  });

  volatile std::size_t sink = 0;
  auto result = perf.throughputLoop(
      [&] {
        // Fresh string each iteration -- repeated heap growth is the
        // allocation churn jemalloc's sampled profile reveals.
        std::string s;
        for (std::size_t i = 0; i < APPENDS; ++i)
          s.append("payload");
        sink = s.size();
      },
      "churning_strings");

  EXPECT_GT(result.callsPerSecond, 1.0);
  (void)sink;
}

/**
 * @test Fast: one string reserved up-front, cleared and reused per iteration.
 *
 * Same characters appended; the single up-front reserve plus clear()-keeps-
 * capacity removes the per-iteration allocation and growth. jemalloc's sampled
 * bytes for the loop collapse because no allocation happens inside it.
 */
PERF_THROUGHPUT(Jemalloc, ReusedBuffer) {
  UB_PERF_GUARD(perf);

  // Allocate ONCE outside the measured loop, with the final capacity.
  std::string s;
  s.reserve(APPENDS * 7); // "payload" is 7 chars

  perf.warmup([&] {
    s.clear();
    for (std::size_t i = 0; i < APPENDS; ++i)
      s.append("payload");
  });

  volatile std::size_t sink = 0;
  auto result = perf.throughputLoop(
      [&] {
        // clear() keeps capacity, so append reuses the existing buffer --
        // no allocation happens inside the measured loop.
        s.clear();
        for (std::size_t i = 0; i < APPENDS; ++i)
          s.append("payload");
        sink = s.size();
      },
      "reused_buffer");

  EXPECT_GT(result.callsPerSecond, 1.0);
  (void)sink;
}

PERF_MAIN()
