/**
 * @file PerfStats_uTest.cpp
 * @brief Unit tests for vernier::Stats and summarize().
 *
 * Tests statistical calculations: percentiles, mean, stddev, CV, and edge cases.
 */

#include "src/bench/inc/PerfStats.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using vernier::bench::Stats;
using vernier::bench::summarize;

/* ----------------------------- Empty Data Tests ----------------------------- */

/** @test Empty vector returns zero-initialized Stats. */
TEST(PerfStatsTest, EmptyVectorReturnsZeroStats) {
  std::vector<double> values;
  const Stats S = summarize(values);

  EXPECT_DOUBLE_EQ(S.median, 0.0);
  EXPECT_DOUBLE_EQ(S.p10, 0.0);
  EXPECT_DOUBLE_EQ(S.p90, 0.0);
  EXPECT_DOUBLE_EQ(S.min, 0.0);
  EXPECT_DOUBLE_EQ(S.max, 0.0);
  EXPECT_DOUBLE_EQ(S.mean, 0.0);
  EXPECT_DOUBLE_EQ(S.stddev, 0.0);
  EXPECT_DOUBLE_EQ(S.cv, 0.0);
}

/* ----------------------------- Single Value Tests ----------------------------- */

/** @test Single value: all stats equal that value, stddev=0, cv=0. */
TEST(PerfStatsTest, SingleValueAllStatsMatch) {
  std::vector<double> values = {42.0};
  const Stats S = summarize(values);

  EXPECT_DOUBLE_EQ(S.median, 42.0);
  EXPECT_DOUBLE_EQ(S.p10, 42.0);
  EXPECT_DOUBLE_EQ(S.p90, 42.0);
  EXPECT_DOUBLE_EQ(S.min, 42.0);
  EXPECT_DOUBLE_EQ(S.max, 42.0);
  EXPECT_DOUBLE_EQ(S.mean, 42.0);
  EXPECT_DOUBLE_EQ(S.stddev, 0.0);
  EXPECT_DOUBLE_EQ(S.cv, 0.0);
}

/* ----------------------------- Two Value Tests ----------------------------- */

/** @test Two values: median is average, min/max correct. */
TEST(PerfStatsTest, TwoValuesMedianIsAverage) {
  std::vector<double> values = {10.0, 20.0};
  const Stats S = summarize(values);

  EXPECT_DOUBLE_EQ(S.min, 10.0);
  EXPECT_DOUBLE_EQ(S.max, 20.0);
  EXPECT_DOUBLE_EQ(S.mean, 15.0);
  EXPECT_DOUBLE_EQ(S.median, 15.0); // Linear interpolation at 0.5
}

/* ----------------------------- Identical Values Tests ----------------------------- */

/** @test Identical values: stddev=0, cv=0. */
TEST(PerfStatsTest, IdenticalValuesZeroVariance) {
  std::vector<double> values = {5.0, 5.0, 5.0, 5.0, 5.0};
  const Stats S = summarize(values);

  EXPECT_DOUBLE_EQ(S.median, 5.0);
  EXPECT_DOUBLE_EQ(S.min, 5.0);
  EXPECT_DOUBLE_EQ(S.max, 5.0);
  EXPECT_DOUBLE_EQ(S.mean, 5.0);
  EXPECT_DOUBLE_EQ(S.stddev, 0.0);
  EXPECT_DOUBLE_EQ(S.cv, 0.0);
}

/* ----------------------------- Percentile Tests ----------------------------- */

/** @test Percentiles with known distribution (1-10). */
TEST(PerfStatsTest, PercentilesKnownDistribution) {
  std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
  const Stats S = summarize(values);

  // 10 values: indices 0-9
  // p10 = quantile(0.10) = index 0.9 -> interpolate between 1.0 and 2.0
  // p50 = quantile(0.50) = index 4.5 -> interpolate between 5.0 and 6.0
  // p90 = quantile(0.90) = index 8.1 -> interpolate between 9.0 and 10.0

  EXPECT_NEAR(S.p10, 1.9, 0.01);
  EXPECT_NEAR(S.median, 5.5, 0.01);
  EXPECT_NEAR(S.p90, 9.1, 0.01);
  EXPECT_DOUBLE_EQ(S.min, 1.0);
  EXPECT_DOUBLE_EQ(S.max, 10.0);
}

/** @test Median of odd count is middle value. */
TEST(PerfStatsTest, MedianOddCount) {
  std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
  const Stats S = summarize(values);

  // 5 values: index 2 is middle
  // quantile(0.5) = index 2.0 -> value 3.0
  EXPECT_DOUBLE_EQ(S.median, 3.0);
}

/* ----------------------------- Mean and Stddev Tests ----------------------------- */

/** @test Mean calculation is correct. */
TEST(PerfStatsTest, MeanCalculation) {
  std::vector<double> values = {2.0, 4.0, 6.0, 8.0, 10.0};
  const Stats S = summarize(values);

  // Mean = (2+4+6+8+10)/5 = 30/5 = 6.0
  EXPECT_DOUBLE_EQ(S.mean, 6.0);
}

/** @test Standard deviation (population formula). */
TEST(PerfStatsTest, StddevPopulationFormula) {
  std::vector<double> values = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
  const Stats S = summarize(values);

  // Mean = 40/8 = 5.0
  // Variance = sum((x-5)^2)/8 = (9+1+1+1+0+0+4+16)/8 = 32/8 = 4.0
  // Stddev = sqrt(4.0) = 2.0
  EXPECT_DOUBLE_EQ(S.mean, 5.0);
  EXPECT_DOUBLE_EQ(S.stddev, 2.0);
}

/* ----------------------------- CV (Coefficient of Variation) Tests -----------------------------
 */

/** @test CV is stddev/mean. */
TEST(PerfStatsTest, CvIsStddevOverMean) {
  std::vector<double> values = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
  const Stats S = summarize(values);

  // Mean = 5.0, Stddev = 2.0
  // CV = 2.0 / 5.0 = 0.4
  EXPECT_DOUBLE_EQ(S.cv, 0.4);
}

/** @test CV is 0 when mean is 0 (avoid division by zero). */
TEST(PerfStatsTest, CvZeroWhenMeanZero) {
  std::vector<double> values = {0.0, 0.0, 0.0};
  const Stats S = summarize(values);

  EXPECT_DOUBLE_EQ(S.mean, 0.0);
  EXPECT_DOUBLE_EQ(S.cv, 0.0); // Protected against division by zero
}

/* ----------------------------- Min/Max Tests ----------------------------- */

/** @test Min and max are correct with unsorted input. */
TEST(PerfStatsTest, MinMaxWithUnsortedInput) {
  std::vector<double> values = {5.0, 1.0, 9.0, 3.0, 7.0};
  const Stats S = summarize(values);

  EXPECT_DOUBLE_EQ(S.min, 1.0);
  EXPECT_DOUBLE_EQ(S.max, 9.0);
}

/** @test Negative values handled correctly. */
TEST(PerfStatsTest, NegativeValuesHandled) {
  std::vector<double> values = {-5.0, -1.0, 0.0, 1.0, 5.0};
  const Stats S = summarize(values);

  EXPECT_DOUBLE_EQ(S.min, -5.0);
  EXPECT_DOUBLE_EQ(S.max, 5.0);
  EXPECT_DOUBLE_EQ(S.mean, 0.0);
  EXPECT_DOUBLE_EQ(S.median, 0.0);
}

/* ----------------------------- Large Dataset Tests ----------------------------- */

/** @test Large dataset has finite values. */
TEST(PerfStatsTest, LargeDatasetFiniteValues) {
  std::vector<double> values;
  values.reserve(10000);
  for (int i = 1; i <= 10000; ++i) {
    values.push_back(static_cast<double>(i));
  }

  const Stats S = summarize(values);

  EXPECT_TRUE(std::isfinite(S.median));
  EXPECT_TRUE(std::isfinite(S.mean));
  EXPECT_TRUE(std::isfinite(S.stddev));
  EXPECT_TRUE(std::isfinite(S.cv));
  EXPECT_DOUBLE_EQ(S.min, 1.0);
  EXPECT_DOUBLE_EQ(S.max, 10000.0);
  EXPECT_DOUBLE_EQ(S.mean, 5000.5);
}
