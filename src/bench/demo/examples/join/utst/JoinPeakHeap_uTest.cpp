/**
 * @file JoinPeakHeap_uTest.cpp
 * @brief The check behind walkthrough 14: at 20,000 words, one call of joinV0
 *        holds more than three times the heap one call of joinV1 holds at its
 *        peak.
 *
 * Notes:
 *  - Massif shows each call's peak heap from outside the process. This
 *    program counts the same peak from inside: it replaces the global
 *    allocation functions with versions that count the bytes operator new has
 *    handed out and not taken back, and the most held at once.
 *  - A program of its own, so the demo massif profiles carries no replacement
 *    and is the program a reader would write. The example's unit tests keep
 *    their own replacement, which counts calls, not bytes.
 *  - It counts bytes, not time, so a busy machine does not change its answer
 *    and ctest runs it (labels demo and massif).
 *  - A thread-sanitizer build cannot replace the allocation functions
 *    (AllocationCounting.hpp), so there it replaces nothing and the test
 *    skips, saying why.
 */

#include "src/bench/demo/examples/join/inc/Join.hpp"

#include "src/bench/demo/examples/join/utst/AllocationCounting.hpp"

#include <gtest/gtest.h>

#include <malloc.h>

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <new>

using vernier::bench::demo::test::ALLOCATIONS_NOT_COUNTED;

#if VERNIER_DEMO_REPLACES_ALLOCATION

using vernier::bench::demo::joinedSize;
using vernier::bench::demo::joinV0;
using vernier::bench::demo::joinV1;
using vernier::bench::demo::makeParts;

/* ----------------------------- Constants ----------------------------- */

/// Words per join: demo 11's input, where the strings dominate the heap.
constexpr std::size_t PART_COUNT = 20000;

/// Fixed seed: every run joins the same words.
constexpr unsigned PART_SEED = 42;

constexpr char SEPARATOR = ',';

/// V0 must hold more than this many times the heap V1 holds at its peak. V1
/// holds one buffer, its result. V0 keeps the string built so far while it
/// builds the next one, and with libstdc++ the next one outgrows its first
/// buffer on the way, so at the last word three buffers are live: five times
/// what V1 holds (four with libc++; both measured while choosing this
/// number). Three fails if V0 ever holds two strings' worth or less, when
/// what massif shows would no longer be what the walkthrough explains.
constexpr double MIN_PEAK_RATIO = 3.0;

/* ----------------------------- Heap Counting ----------------------------- */

// The global allocation functions below count what operator new has handed
// out and not taken back, and the most held at once since peakHeapDuring()
// last reset it. A delete is not always told the size, so both sides count
// the allocator's usable size and agree. Valgrind replaces these functions
// with its own, so under valgrind nothing is counted and the test fails on
// its first assertion.

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

// Every delete is out of line: inlined into a caller, free() would meet a
// pointer the compiler knows came from operator new, or the scalar delete one
// from new[], and it warns about the pair.
[[gnu::noinline]] void operator delete(void* block) noexcept {
  if (block != nullptr) {
    countRelease(block);
    std::free(block);
  }
}

[[gnu::noinline]] void operator delete[](void* block) noexcept { ::operator delete(block); }

[[gnu::noinline]] void operator delete(void* block, std::size_t /*size*/) noexcept {
  ::operator delete(block);
}

[[gnu::noinline]] void operator delete[](void* block, std::size_t /*size*/) noexcept {
  ::operator delete(block);
}

[[gnu::noinline]] void operator delete(void* block, const std::nothrow_t& /*tag*/) noexcept {
  ::operator delete(block);
}

[[gnu::noinline]] void operator delete[](void* block, const std::nothrow_t& /*tag*/) noexcept {
  ::operator delete(block);
}

#endif // VERNIER_DEMO_REPLACES_ALLOCATION

/* ----------------------------- API Tests ----------------------------- */

/** @test V0 holds over MIN_PEAK_RATIO times V1's peak heap, and joinedSize holds none */
TEST(Massif, JoinPeakHeap) {
#if VERNIER_DEMO_REPLACES_ALLOCATION
  const auto PARTS = makeParts(PART_COUNT, PART_SEED);
  ASSERT_EQ(joinV0(PARTS, SEPARATOR), joinV1(PARTS, SEPARATOR));

  volatile std::size_t sink = 0;
  const std::size_t V0_PEAK = peakHeapDuring([&] { sink = joinV0(PARTS, SEPARATOR).size(); });
  const std::size_t V1_PEAK = peakHeapDuring([&] { sink = joinV1(PARTS, SEPARATOR).size(); });
  std::size_t joined = 0;
  const std::size_t SIZE_PEAK = peakHeapDuring([&] { joined = joinedSize(PARTS); });
  const std::size_t JOINED = joined;

  // V1 holds its result at its peak; less means the counting missed it.
  ASSERT_GE(V1_PEAK, JOINED) << "V1 held " << V1_PEAK << " bytes, less than its own " << JOINED
                             << "-byte result: the heap is not being counted";
  EXPECT_EQ(SIZE_PEAK, 0u) << "joinedSize held " << SIZE_PEAK
                           << " bytes: it is meant to compute the length without allocating";

  std::printf("[Massif.JoinPeakHeap]  joined %zu bytes  V0 holds %zu  V1 holds %zu  %.1fx\n",
              JOINED, V0_PEAK, V1_PEAK,
              static_cast<double>(V0_PEAK) / static_cast<double>(V1_PEAK));
  EXPECT_GT(static_cast<double>(V0_PEAK), MIN_PEAK_RATIO * static_cast<double>(V1_PEAK))
      << "V0 held " << V0_PEAK << " bytes at its peak, not " << MIN_PEAK_RATIO << "x the "
      << V1_PEAK << " bytes V1 held: the demo has stopped demonstrating";
#else
  GTEST_SKIP() << ALLOCATIONS_NOT_COUNTED;
#endif
}
