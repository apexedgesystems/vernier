/**
 * @file MonitorTag_uTest.cpp
 * @brief Unit tests for vernier::monitor::MonitorTag.
 *
 * Notes:
 *  - Tests are platform-agnostic: assert invariants, not exact values.
 */

#include "src/monitor/inc/MonitorTag.hpp"

#include <gtest/gtest.h>

#include <cstring>

using vernier::monitor::MonitorTag;
using vernier::monitor::nowNs;
using vernier::monitor::Sample;
using vernier::monitor::SampleKind;

/* ----------------------------- Default Construction ----------------------------- */

/** @test Default MonitorTag is zero-initialized */
TEST(MonitorTagDefaultTest, DefaultConstructed) {
  const MonitorTag TAG{};
  EXPECT_EQ(TAG.id, 0);
  EXPECT_STREQ(TAG.name, "");
}

/** @test Sample default construction */
TEST(SampleDefaultTest, DefaultConstructed) {
  const Sample S{};
  EXPECT_EQ(S.timestampNs, 0u);
  EXPECT_EQ(S.durationNs, 0u);
  EXPECT_EQ(S.kind, SampleKind::SCOPE);
  EXPECT_DOUBLE_EQ(S.value, 0.0);
}

/* ----------------------------- MonitorTag Method Tests ----------------------------- */

/** @test MonitorTag stores name and id */
TEST(MonitorTagTest, NameAndId) {
  const MonitorTag TAG("decoder", 42);
  EXPECT_STREQ(TAG.name, "decoder");
  EXPECT_EQ(TAG.id, 42);
}

/** @test Long name is truncated to 31 chars (null-terminated) */
TEST(MonitorTagTest, LongNameTruncated) {
  const MonitorTag TAG("this_is_a_very_long_name_that_exceeds_32_characters", 1);
  EXPECT_EQ(std::strlen(TAG.name), 31u);
  EXPECT_EQ(TAG.name[31], '\0');
}

/** @test Null name is handled gracefully */
TEST(MonitorTagTest, NullName) {
  const MonitorTag TAG(nullptr, 5);
  EXPECT_STREQ(TAG.name, "");
  EXPECT_EQ(TAG.id, 5);
}

/* ----------------------------- API Tests ----------------------------- */

/** @test nowNs returns a non-zero monotonic value */
TEST(NowNsTest, Monotonic) {
  const auto T1 = nowNs();
  const auto T2 = nowNs();
  EXPECT_GT(T1, 0u);
  EXPECT_GE(T2, T1);
}

/* ----------------------------- Constants Tests ----------------------------- */

/** @test MonitorTag is trivially copyable (required for lock-free queue) */
TEST(MonitorTagTest, TriviallyCopyable) { EXPECT_TRUE(std::is_trivially_copyable_v<MonitorTag>); }

/** @test Sample is trivially copyable */
TEST(SampleTest, TriviallyCopyable) { EXPECT_TRUE(std::is_trivially_copyable_v<Sample>); }
