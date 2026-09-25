/**
 * @file Join_uTest.cpp
 * @brief Unit tests for vernier::bench::demo join.
 *
 * Notes:
 *  - Every version must return the same string for the same arguments; that
 *    is what lets a walkthrough compare their timings.
 *  - How often each version allocates is what the heap-profiler walkthroughs
 *    read, so it is tested too: this binary replaces the global operator new
 *    to count the calls each version makes.
 *  - The counts are those of a plain run. A tool preloaded into this binary
 *    that allocates through operator new on the test's thread adds its own
 *    calls: under heaptrack, a count taken through a call stack it has not
 *    met before reads several calls high. The counter and a heap profiler
 *    each disturb the other, which is why this guard is not in the demo.
 *  - Tests are platform-agnostic and independent of execution order.
 */

#include "src/bench/demo/examples/join/inc/Join.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>

#include <new>
#include <string>
#include <vector>

using vernier::bench::demo::joinedSize;
using vernier::bench::demo::joinV0;
using vernier::bench::demo::joinV1;
using vernier::bench::demo::makeParts;

/* ----------------------------- Allocation Counting ----------------------------- */

namespace {

/// Calls to operator new made by the calling thread. Thread-local, so a count
/// taken on one thread never includes another thread's allocations.
thread_local std::size_t newCallsOnThisThread = 0;

/// Calls to operator new that @p op makes on the calling thread.
template <typename Op> std::size_t countNewCalls(Op&& op) {
  const std::size_t before = newCallsOnThisThread;
  op();
  return newCallsOnThisThread - before;
}

/// The one allocation every replaced operator new makes, counted once.
void* countedAllocate(std::size_t size) noexcept {
  ++newCallsOnThisThread;
  return std::malloc(size == 0 ? 1 : size);
}

} // namespace

// Replacing the global allocation functions is how a test sees the allocations
// std::string makes. Memory from one form of operator new may be released by
// any delete of its family, so the whole scalar family is replaced together,
// on malloc and free: throwing and nothrow new, plain, sized and nothrow
// delete. Nothing then crosses between these and a runtime's own forms (a
// sanitizer's, for one). Array and aligned forms pair only within their own
// families and stay the runtime's.
void* operator new(std::size_t size) {
  if (void* p = countedAllocate(size)) {
    return p;
  }
  throw std::bad_alloc();
}

void* operator new(std::size_t size, const std::nothrow_t& /*tag*/) noexcept {
  return countedAllocate(size);
}

// Out of line: inlined into a caller, free() would meet a pointer the compiler
// knows came from operator new, and it warns about the pair.
[[gnu::noinline]] void operator delete(void* p) noexcept { std::free(p); }

[[gnu::noinline]] void operator delete(void* p, std::size_t /*size*/) noexcept { std::free(p); }

[[gnu::noinline]] void operator delete(void* p, const std::nothrow_t& /*tag*/) noexcept {
  std::free(p);
}

/* ----------------------------- API Tests ----------------------------- */

/** @test Every version joins parts with a separator after each part */
TEST(JoinTest, KnownAnswer) {
  const std::vector<std::string> parts = {"alpha", "beta", "gamma"};

  EXPECT_EQ(joinV0(parts, ','), "alpha,beta,gamma,");
  EXPECT_EQ(joinV1(parts, ','), "alpha,beta,gamma,");
}

/** @test Every version returns an empty string for no parts */
TEST(JoinTest, NoPartsGiveEmptyString) {
  const std::vector<std::string> parts;

  EXPECT_EQ(joinV0(parts, ','), "");
  EXPECT_EQ(joinV1(parts, ','), "");
}

/** @test Every version separates empty parts as it separates other parts */
TEST(JoinTest, EmptyPartsKeepTheirSeparators) {
  const std::vector<std::string> one = {"x"};
  const std::vector<std::string> empties = {"", ""};

  EXPECT_EQ(joinV0(one, ';'), "x;");
  EXPECT_EQ(joinV1(one, ';'), "x;");
  EXPECT_EQ(joinV0(empties, '-'), "--");
  EXPECT_EQ(joinV1(empties, '-'), "--");
}

/* ----------------------------- Version Agreement Tests ----------------------------- */

class JoinSizesTest : public ::testing::TestWithParam<std::size_t> {
protected:
  std::vector<std::string> parts_;

  void SetUp() override { parts_ = makeParts(GetParam(), 42); }
};

/** @test Every version returns the same string at every input size */
TEST_P(JoinSizesTest, VersionsAgree) { EXPECT_EQ(joinV1(parts_, ','), joinV0(parts_, ',')); }

/** @test The joined length is every part plus one separator each */
TEST_P(JoinSizesTest, JoinedLengthCoversEveryPart) {
  std::size_t expected = 0;
  for (const std::string& part : parts_) {
    expected += part.size() + 1;
  }

  EXPECT_EQ(joinV0(parts_, ',').size(), expected);
  EXPECT_EQ(joinV1(parts_, ',').size(), expected);
}

/** @test joinedSize is the length every version returns at every input size */
TEST_P(JoinSizesTest, JoinedSizeIsEveryVersionsLength) {
  EXPECT_EQ(joinedSize(parts_), joinV0(parts_, ',').size());
  EXPECT_EQ(joinedSize(parts_), joinV1(parts_, ',').size());
}

INSTANTIATE_TEST_SUITE_P(Counts, JoinSizesTest, ::testing::Values(0, 1, 10, 100, 1000, 10000));

/* ----------------------------- joinedSize Tests ----------------------------- */

/** @test joinedSize counts every part and one separator after each */
TEST(JoinedSizeTest, CountsEveryPartAndOneSeparatorEach) {
  const std::vector<std::string> none;
  const std::vector<std::string> words = {"alpha", "beta", "gamma"};
  const std::vector<std::string> empties = {"", ""};

  EXPECT_EQ(joinedSize(none), 0u);
  EXPECT_EQ(joinedSize(words), std::string("alpha,beta,gamma,").size());
  EXPECT_EQ(joinedSize(empties), std::string("--").size());
}

/* ----------------------------- Allocation Tests ----------------------------- */

/// Parts per join in the walkthroughs that read allocation counts.
constexpr std::size_t PROFILED_PART_COUNT = 1000;

/// At PROFILED_PART_COUNT parts, joinV0 must make at least this many
/// allocations per call for each one joinV1 makes. joinV0 makes about two per
/// part, four times this; joinV0 building like joinV1, or joinV1 losing its
/// reserve, falls below it.
constexpr std::size_t MIN_ALLOCATION_RATIO = 500;

class JoinAllocationSizesTest : public ::testing::TestWithParam<std::size_t> {
protected:
  std::vector<std::string> parts_;

  void SetUp() override { parts_ = makeParts(GetParam(), 42); }
};

/** @test joinV1 allocates once per call, for the final size, at every size */
TEST_P(JoinAllocationSizesTest, V1AllocatesOnce) {
  std::string joined;
  const std::size_t calls = countNewCalls([&] { joined = joinV1(parts_, ','); });

  EXPECT_EQ(calls, 1u) << "joinV1 of " << GetParam() << " parts";
  EXPECT_EQ(joined, joinV0(parts_, ','));
}

// Sizes whose joined string is too long for the string's inline buffer, so
// the one allocation is always needed.
INSTANTIATE_TEST_SUITE_P(Counts, JoinAllocationSizesTest, ::testing::Values(10, 100, 1000, 10000));

/** @test joinV0 allocates at least MIN_ALLOCATION_RATIO times as often as joinV1 */
TEST(JoinAllocationTest, V0AllocatesFarMoreOftenThanV1) {
  const auto parts = makeParts(PROFILED_PART_COUNT, 42);

  std::string joined;
  const std::size_t v1Calls = countNewCalls([&] { joined = joinV1(parts, ','); });
  const std::size_t v0Calls = countNewCalls([&] { joined = joinV0(parts, ','); });

  ASSERT_EQ(v1Calls, 1u) << "the ratio below is only meaningful against one allocation";
  EXPECT_GE(v0Calls, MIN_ALLOCATION_RATIO * v1Calls)
      << "joinV0 made " << v0Calls << " allocations for " << PROFILED_PART_COUNT
      << " parts, joinV1 " << v1Calls << ": the heap-profiler demos have stopped demonstrating";
}

/* ----------------------------- makeParts Tests ----------------------------- */

/** @test makeParts returns the requested number of parts */
TEST(MakePartsTest, ReturnsRequestedCount) {
  EXPECT_EQ(makeParts(0, 42).size(), 0u);
  EXPECT_EQ(makeParts(7, 42).size(), 7u);
}

/** @test Part lengths stay inside the documented range */
TEST(MakePartsTest, PartLengthsAreInRange) {
  const auto parts = makeParts(200, 42);

  for (const std::string& part : parts) {
    EXPECT_GE(part.size(), 3u) << "part '" << part << "' is shorter than 3 characters";
    EXPECT_LE(part.size(), 10u) << "part '" << part << "' is longer than 10 characters";
  }
}

/* ----------------------------- Determinism Tests ----------------------------- */

/** @test The same seed gives the same parts, a different seed does not */
TEST(MakePartsTest, SeedDecidesTheParts) {
  EXPECT_EQ(makeParts(50, 42), makeParts(50, 42));
  EXPECT_NE(makeParts(50, 43), makeParts(50, 42));
}
