/**
 * @file Join_uTest.cpp
 * @brief Unit tests for vernier::bench::demo join.
 *
 * Notes:
 *  - Every version must return the same string for the same arguments; that
 *    is what lets a walkthrough compare their timings.
 *  - Tests are platform-agnostic and independent of execution order.
 */

#include "src/bench/demo/examples/join/inc/Join.hpp"

#include <gtest/gtest.h>

#include <cstddef>

#include <string>
#include <vector>

using vernier::bench::demo::joinV0;
using vernier::bench::demo::joinV1;
using vernier::bench::demo::makeParts;

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

INSTANTIATE_TEST_SUITE_P(Counts, JoinSizesTest, ::testing::Values(0, 1, 10, 100, 1000, 10000));

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
