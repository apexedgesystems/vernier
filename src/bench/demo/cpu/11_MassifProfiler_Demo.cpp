/**
 * @file 11_MassifProfiler_Demo.cpp
 * @brief Demo 11: Valgrind Massif heap profiler -- peak heap, and who owns it
 *
 * Measures the two versions of the shared join example at 20,000 words, where
 * the strings are large enough to dominate the heap, and guards what massif
 * shows about them:
 *  1. One measurement per test, so each writes its own CSV row
 *  2. Run one measuring test under massif to see the peak and who holds it
 *  3. JoinPeakHeap fails unless V0 holds more than three times V1's heap at
 *     its peak
 *
 * Usage:
 *   @code{.sh}
 *   # Measure
 *   ./BenchDemo_11_MassifProfiler --target-time 50ms --repeats 10 --csv run.csv
 *
 *   # Profile one version; --time-unit=B puts the bytes allocated and freed,
 *   # not instructions, on the graph's x-axis
 *   valgrind --tool=massif --time-unit=B --massif-out-file=massif.v0.out \
 *     ./BenchDemo_11_MassifProfiler --profile massif --cycles 1 --repeats 1 \
 *     --gtest_filter=Massif.JoinV0
 *   ms_print massif.v0.out
 *   @endcode
 *
 * @see docs/14_MASSIF_PROFILER.md for the step-by-step walkthrough
 */

#include <gtest/gtest.h>

#include <malloc.h>

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"

namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

/// Words per join. Large enough that the joined string and the words
/// themselves hold almost all of the heap, small enough that V0, whose cost
/// grows with the square of this number, stays under half a second per call
/// on the reference rig. The walkthrough records how it was chosen.
static constexpr std::size_t PART_COUNT = 20000;

/// Fixed seed: every run joins the same words.
static constexpr unsigned PART_SEED = 42;

static constexpr char SEPARATOR = ',';

/// V0 must hold more than this many times the heap V1 holds at its peak. V1
/// holds one buffer, its result. V0 keeps the string built so far while it
/// builds the next one, and with libstdc++ the next one outgrows its first
/// buffer on the way, so at the last word three buffers are live: five times
/// what V1 holds (four with libc++; both measured while choosing this
/// number). Three fails if V0 ever holds two strings' worth or less, when
/// what massif shows would no longer be what the walkthrough explains.
static constexpr double MIN_PEAK_RATIO = 3.0;

/* ----------------------------- File Helpers ----------------------------- */

// This binary replaces the global allocation functions (below) to count what
// operator new has handed out and not taken back, and the most held at once
// since peakHeapDuring() last reset it. A delete is not always told the size,
// so both sides count the allocator's usable size and agree. Valgrind
// replaces these functions with its own, so nothing is counted under it.

namespace {

std::atomic<std::size_t> heapLive{0};
std::atomic<std::size_t> heapPeak{0};

void countAllocation(void* block) noexcept {
  const std::size_t SIZE = malloc_usable_size(block);
  const std::size_t LIVE = heapLive.fetch_add(SIZE, std::memory_order_relaxed) + SIZE;
  std::size_t peak = heapPeak.load(std::memory_order_relaxed);
  while (LIVE > peak && !heapPeak.compare_exchange_weak(peak, LIVE, std::memory_order_relaxed)) {
  }
}

void countRelease(void* block) noexcept {
  heapLive.fetch_sub(malloc_usable_size(block), std::memory_order_relaxed);
}

/// Most bytes held at once while op runs, beyond what was held when it
/// started: the heap one call owns at its peak.
template <typename Op> std::size_t peakHeapDuring(Op&& op) {
  const std::size_t BASE = heapLive.load(std::memory_order_relaxed);
  heapPeak.store(BASE, std::memory_order_relaxed);
  op();
  return heapPeak.load(std::memory_order_relaxed) - BASE;
}

/// Bytes a joined string holds: every part plus one separator each.
std::size_t joinedSize(const std::vector<std::string>& parts) {
  std::size_t total = 0;
  for (const std::string& part : parts) {
    total += part.size() + 1;
  }
  return total;
}

} // namespace

/* ----------------------------- Global Allocation Functions ----------------------------- */

void* operator new(std::size_t size) {
  for (;;) {
    if (void* block = std::malloc(size == 0 ? 1 : size)) {
      countAllocation(block);
      return block;
    }
    const std::new_handler HANDLER = std::get_new_handler();
    if (HANDLER == nullptr) {
      throw std::bad_alloc();
    }
    HANDLER();
  }
}

void* operator new[](std::size_t size) { return ::operator new(size); }

void* operator new(std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
  try {
    return ::operator new(size);
  } catch (...) {
    return nullptr;
  }
}

void* operator new[](std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
  try {
    return ::operator new(size);
  } catch (...) {
    return nullptr;
  }
}

void operator delete(void* block) noexcept {
  if (block != nullptr) {
    countRelease(block);
    std::free(block);
  }
}

void operator delete[](void* block) noexcept { ::operator delete(block); }

void operator delete(void* block, std::size_t /*size*/) noexcept { ::operator delete(block); }

void operator delete[](void* block, std::size_t /*size*/) noexcept { ::operator delete(block); }

void operator delete(void* block, const std::nothrow_t& /*tag*/) noexcept {
  ::operator delete(block);
}

void operator delete[](void* block, const std::nothrow_t& /*tag*/) noexcept {
  ::operator delete(block);
}

/* ----------------------------- Tests ----------------------------- */

/** @test Throughput of the one-liner at PART_COUNT words. */
PERF_THROUGHPUT(Massif, JoinV0) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR).size(), joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); }, "join_v0");
}

/** @test Throughput of the reserving version at PART_COUNT words. */
PERF_THROUGHPUT(Massif, JoinV1) {
  PERF_GUARD(perf);

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV1(PARTS, SEPARATOR).size(), joinedSize(PARTS));

  volatile std::size_t sink = 0;
  perf.warmup([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); });
  perf.throughputLoop([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); }, "join_v1");
}

/**
 * @test V0 holds more than MIN_PEAK_RATIO times the heap V1 holds at its peak.
 *
 * Counts what one call of each version holds at once, beyond what was live
 * before it, and writes no CSV row. Massif shows this peak from outside the
 * process; under valgrind this binary's counting is bypassed, so the test
 * skips itself there.
 */
PERF_TEST(Massif, JoinPeakHeap) {
  if (vernier::bench::profiler_env::isRunningUnderValgrind()) {
    GTEST_SKIP() << "valgrind replaces this binary's operator new, so the heap is not counted; "
                    "run this test without valgrind";
  }

  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(demo::joinV0(PARTS, SEPARATOR), demo::joinV1(PARTS, SEPARATOR));

  volatile std::size_t sink = 0;
  const std::size_t V0_PEAK = peakHeapDuring([&] { sink = demo::joinV0(PARTS, SEPARATOR).size(); });
  const std::size_t V1_PEAK = peakHeapDuring([&] { sink = demo::joinV1(PARTS, SEPARATOR).size(); });
  const std::size_t JOINED = joinedSize(PARTS);

  // V1 holds its result at its peak; less means the counting missed it.
  ASSERT_GE(V1_PEAK, JOINED) << "V1 held " << V1_PEAK << " bytes, less than its own " << JOINED
                             << "-byte result: the heap is not being counted";

  std::printf("[Massif.JoinPeakHeap]  joined %zu bytes  V0 holds %zu  V1 holds %zu  %.1fx\n",
              JOINED, V0_PEAK, V1_PEAK,
              static_cast<double>(V0_PEAK) / static_cast<double>(V1_PEAK));
  EXPECT_GT(static_cast<double>(V0_PEAK), MIN_PEAK_RATIO * static_cast<double>(V1_PEAK))
      << "V0 held " << V0_PEAK << " bytes at its peak, not " << MIN_PEAK_RATIO << "x the "
      << V1_PEAK << " bytes V1 held: the demo has stopped demonstrating";
}

PERF_MAIN()
