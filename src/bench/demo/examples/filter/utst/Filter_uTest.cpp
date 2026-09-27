/**
 * @file Filter_uTest.cpp
 * @brief Unit tests for vernier::bench::demo filter.
 *
 * Notes:
 *  - Every version must keep the same values in the same order for the same
 *    arguments; that is what lets a walkthrough compare their counters.
 *  - Whether filterBranchy() still branches on the data, and whether
 *    filterBranchless() removes what that branch costs, is what the
 *    hardware-counter walkthrough teaches, so both are tested too, through a
 *    hardware counter (utst/HardwareCounter.hpp). Where the counter cannot
 *    count (it did not open, or was not on the PMU for the counted calls),
 *    the tests skip and say why. Only the counter's own state makes them
 *    skip: a count of zero from a counter that ran is a result, and fails
 *    them.
 *  - Tests are platform-agnostic and independent of execution order.
 */

#include "src/bench/demo/examples/filter/inc/Filter.hpp"

#include "src/bench/demo/examples/utst/HardwareCounter.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdio>

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

using vernier::bench::demo::filterBranchless;
using vernier::bench::demo::filterBranchy;
using vernier::bench::demo::makeSortedValues;
using vernier::bench::demo::makeValues;
using vernier::bench::demo::test::HardwareCounter;
using vernier::bench::demo::test::HardwareEvent;
using vernier::bench::demo::test::joinReasons;
using vernier::bench::demo::test::Reading;
using vernier::bench::demo::test::scalingNote;

/* ----------------------------- File Helpers ----------------------------- */

namespace {

/// The values above threshold, in order, by the standard library.
std::vector<double> expectedAbove(const std::vector<double>& values, double threshold) {
  std::vector<double> kept;
  std::copy_if(values.begin(), values.end(), std::back_inserter(kept),
               [threshold](double value) { return value > threshold; });
  return kept;
}

/// What one version left in out[0, count).
template <typename Filter>
std::vector<double> keptBy(Filter filter, const std::vector<double>& values, double threshold) {
  std::vector<double> out(values.size());
  const std::size_t count = filter(values, threshold, out);
  out.resize(count);
  return out;
}

} // namespace

/* ----------------------------- API Tests ----------------------------- */

/** @test Every version keeps the values above the threshold, in their order */
TEST(FilterTest, KnownAnswer) {
  const std::vector<double> values = {0.1, 0.9, 0.5, 0.7, 0.3};
  const std::vector<double> expected = {0.9, 0.7};

  EXPECT_EQ(keptBy(filterBranchy, values, 0.5), expected);
  EXPECT_EQ(keptBy(filterBranchless, values, 0.5), expected);
}

/** @test A value equal to the threshold is not above it */
TEST(FilterTest, EqualIsNotAbove) {
  const std::vector<double> values = {0.5, 0.5, 0.5};

  EXPECT_TRUE(keptBy(filterBranchy, values, 0.5).empty());
  EXPECT_TRUE(keptBy(filterBranchless, values, 0.5).empty());
}

/** @test Every version keeps nothing from no values */
TEST(FilterTest, NoValuesKeepNothing) {
  const std::vector<double> values;

  EXPECT_TRUE(keptBy(filterBranchy, values, 0.5).empty());
  EXPECT_TRUE(keptBy(filterBranchless, values, 0.5).empty());
}

/** @test A threshold below the range keeps everything, one at its top keeps nothing */
TEST(FilterTest, AllOrNothing) {
  const std::vector<double> values = makeValues(100, 42);

  EXPECT_EQ(keptBy(filterBranchy, values, -1.0), values);
  EXPECT_EQ(keptBy(filterBranchless, values, -1.0), values);
  EXPECT_TRUE(keptBy(filterBranchy, values, 1.0).empty());
  EXPECT_TRUE(keptBy(filterBranchless, values, 1.0).empty());
}

/* ----------------------------- Version Agreement Tests ----------------------------- */

class FilterSizesTest : public ::testing::TestWithParam<std::size_t> {
protected:
  std::vector<double> random_;
  std::vector<double> sorted_;

  void SetUp() override {
    random_ = makeValues(GetParam(), 42);
    sorted_ = makeSortedValues(GetParam(), 42);
  }
};

/** @test Every version agrees with the standard library on random and on sorted input, at every
 * size */
TEST_P(FilterSizesTest, VersionsAgree) {
  for (const double threshold : {0.25, 0.5, 0.75}) {
    EXPECT_EQ(keptBy(filterBranchy, random_, threshold), expectedAbove(random_, threshold))
        << "threshold " << threshold;
    EXPECT_EQ(keptBy(filterBranchless, random_, threshold), expectedAbove(random_, threshold))
        << "threshold " << threshold;
    EXPECT_EQ(keptBy(filterBranchy, sorted_, threshold), expectedAbove(sorted_, threshold))
        << "threshold " << threshold;
    EXPECT_EQ(keptBy(filterBranchless, sorted_, threshold), expectedAbove(sorted_, threshold))
        << "threshold " << threshold;
  }
}

INSTANTIATE_TEST_SUITE_P(Counts, FilterSizesTest, ::testing::Values(0, 1, 10, 1000, 100000));

/* ----------------------------- makeValues Tests ----------------------------- */

/** @test makeValues returns the requested number of values, each in [0, 1) */
TEST(MakeValuesTest, ReturnsRequestedCountInRange) {
  EXPECT_EQ(makeValues(0, 42).size(), 0u);

  const std::vector<double> values = makeValues(1000, 42);
  EXPECT_EQ(values.size(), 1000u);
  for (const double value : values) {
    EXPECT_GE(value, 0.0);
    EXPECT_LT(value, 1.0);
  }
}

/** @test makeSortedValues holds the same values as makeValues, in ascending order */
TEST(MakeValuesTest, SortedValuesAreTheSameValuesAscending) {
  const std::vector<double> sorted = makeSortedValues(1000, 42);
  EXPECT_TRUE(std::is_sorted(sorted.begin(), sorted.end()));

  std::vector<double> reference = makeValues(1000, 42);
  std::sort(reference.begin(), reference.end());
  EXPECT_EQ(sorted, reference);
}

/* ----------------------------- Determinism Tests ----------------------------- */

/** @test The same seed gives the same values, a different seed does not */
TEST(MakeValuesTest, SeedDecidesTheValues) {
  EXPECT_EQ(makeValues(50, 42), makeValues(50, 42));
  EXPECT_NE(makeValues(50, 43), makeValues(50, 42));
}

/* ----------------------------- Branch Tests ----------------------------- */

/// Values per call in the branch test: enough that the loop's own branches
/// dwarf a call's fixed cost.
constexpr std::size_t BRANCH_TEST_SIZE = 100000;

/// Calls counted per input.
constexpr int BRANCH_TEST_CALLS = 10;

/// A branch that depends on random data, taken half the time, is mispredicted
/// about half the time by any predictor; a loop whose test was turned into a
/// conditional select has no such branch and mispredicts almost nothing. The
/// floor sits far below the first and far above the second.
constexpr double MIN_MISPREDICTS_PER_VALUE = 0.1;

/// On sorted input the same branch turns once per call, so its mispredictions
/// are a handful, and filterBranchless has no branch on the data at all; the
/// branchy filter on random input must mispredict at least this many times as
/// often as either. On the reference rig both ratios are in the thousands.
constexpr double MIN_MISPREDICT_RATIO = 10.0;

/** @test filterBranchy's test of each value stays a branch the processor has to predict */
TEST(FilterBranchTest, ConditionalStoreKeepsItsBranch) {
  HardwareCounter misses(HardwareEvent::BRANCH_MISSES);
  if (!misses.isOpen()) {
    GTEST_SKIP() << "cannot count branch-misses here: " << misses.failure();
  }

  const std::vector<double> random = makeValues(BRANCH_TEST_SIZE, 42);
  const std::vector<double> sorted = makeSortedValues(BRANCH_TEST_SIZE, 42);
  std::vector<double> out(BRANCH_TEST_SIZE);
  volatile std::size_t sink = 0;

  const Reading random_ =
      misses.perCall(BRANCH_TEST_CALLS, [&] { sink = filterBranchy(random, 0.5, out); });
  const Reading sorted_ =
      misses.perCall(BRANCH_TEST_CALLS, [&] { sink = filterBranchy(sorted, 0.5, out); });
  // Whether the branch still mispredicts is what this test checks, so a zero
  // from a counter that ran is a result for the assertions, not a reason to skip.
  const std::string notCounted =
      joinReasons({random_.whyNotCounted(false), sorted_.whyNotCounted(false)});
  if (!notCounted.empty()) {
    GTEST_SKIP() << "could not count branch-misses here: " << notCounted;
  }
  const double perValue = random_.value / static_cast<double>(BRANCH_TEST_SIZE);

  std::printf("[FilterBranchTest.ConditionalStoreKeepsItsBranch]  random %.0f%s  sorted %.0f%s  "
              "branch-misses/call\n",
              random_.value, scalingNote(random_).c_str(), sorted_.value,
              scalingNote(sorted_).c_str());
  EXPECT_GE(perValue, MIN_MISPREDICTS_PER_VALUE)
      << "filterBranchy mispredicted " << perValue << " branches per value on random input"
      << scalingNote(random_) << ": its test of each value is no longer a branch";
  EXPECT_GE(random_.value, MIN_MISPREDICT_RATIO * sorted_.value)
      << "filterBranchy mispredicted " << random_.value << " branches per call on random input"
      << scalingNote(random_) << " and " << sorted_.value << " on sorted input"
      << scalingNote(sorted_) << ": its branch no longer depends on the data";
}

/** @test filterBranchless removes the mispredictions filterBranchy makes on random input */
TEST(FilterBranchTest, BranchlessFormRemovesTheMisses) {
  HardwareCounter misses(HardwareEvent::BRANCH_MISSES);
  if (!misses.isOpen()) {
    GTEST_SKIP() << "cannot count branch-misses here: " << misses.failure();
  }

  const std::vector<double> random = makeValues(BRANCH_TEST_SIZE, 42);
  std::vector<double> out(BRANCH_TEST_SIZE);
  volatile std::size_t sink = 0;

  const Reading branchy =
      misses.perCall(BRANCH_TEST_CALLS, [&] { sink = filterBranchy(random, 0.5, out); });
  const Reading branchless =
      misses.perCall(BRANCH_TEST_CALLS, [&] { sink = filterBranchless(random, 0.5, out); });
  // As above: only the counter's own state refuses a reading. The branchless
  // filter may count as few mispredictions as it likes, zero included.
  const std::string notCounted =
      joinReasons({branchy.whyNotCounted(false), branchless.whyNotCounted(false)});
  if (!notCounted.empty()) {
    GTEST_SKIP() << "could not count branch-misses here: " << notCounted;
  }

  std::printf(
      "[FilterBranchTest.BranchlessFormRemovesTheMisses]  branchy %.0f%s  branchless %.0f%s  "
      "branch-misses/call\n",
      branchy.value, scalingNote(branchy).c_str(), branchless.value,
      scalingNote(branchless).c_str());
  const double branchyPerValue = branchy.value / static_cast<double>(BRANCH_TEST_SIZE);
  ASSERT_GE(branchyPerValue, MIN_MISPREDICTS_PER_VALUE)
      << "filterBranchy mispredicted " << branchyPerValue << " branches per value on random input"
      << scalingNote(branchy) << ": there are no mispredictions for the branchless version to "
      << "remove, and the comparison below would mean nothing";
  EXPECT_GE(branchy.value, MIN_MISPREDICT_RATIO * branchless.value)
      << "filterBranchy mispredicted " << branchy.value << " branches per call on random input"
      << scalingNote(branchy) << " and filterBranchless " << branchless.value
      << scalingNote(branchless) << ": the branchless version no longer removes them";
}
