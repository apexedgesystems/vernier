/**
 * @file PerfAbi_uTest.cpp
 * @brief Unit tests for the layout libbench shares with its consumers, and
 * for the start-up check that compares both sides of it.
 *
 * Notes:
 *  - libbench reads PerfConfig and Stats through references handed over by
 *    header-inline code compiled into the consumer, so member offsets are part
 *    of the shared library's ABI even though no symbol names them.
 *  - The layout rule is "append only". Each pinned member must start exactly
 *    where the previous pinned member ends (rounded up to its alignment), and
 *    must keep its type. That holds on any platform, and it fails as soon as a
 *    member is inserted, removed, reordered or retyped ahead of the tail.
 *  - A failure here means: move the new member to the end of the struct. If
 *    the layout truly has to change, raise the library's SOVERSION with it.
 */

#include "src/bench/inc/PerfAbi.hpp"

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfGpuConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"

#include <gtest/gtest.h>

#include <cstddef>

#include <optional>
#include <string>
#include <type_traits>
#include <vector>

using vernier::bench::BENCH_ABI_MISMATCH_EXIT_CODE;
using vernier::bench::BENCH_ABI_VERSION;
using vernier::bench::checkBenchAbi;
using vernier::bench::PerfConfig;
using vernier::bench::PerfGpuConfig;
using vernier::bench::Stats;
using vernier::bench::detail::AbiField;
using vernier::bench::detail::abiMismatchMessage;
using vernier::bench::detail::AbiRule;

// offsetof on a struct with std::string members is conditionally supported;
// every toolchain this project builds with supports it, and the alternative
// (member-pointer arithmetic on a live object) says the same thing less clearly.
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif

namespace {

/// Where a member of alignment @p align starts when it directly follows a
/// member that ends at @p prevEnd.
constexpr std::size_t startAfter(std::size_t prevEnd, std::size_t align) {
  return (prevEnd + align - 1) / align * align;
}

/// One pinned member: its name, where it is, how big it is, how it aligns.
struct Member {
  const char* name;
  std::size_t offset;
  std::size_t size;
  std::size_t align;
};

// Records a member and refuses to compile if its type changed.
#define VERNIER_PINNED_MEMBER(Struct, member, Type)                                                \
  (static_cast<void>(sizeof(char[std::is_same_v<decltype(Struct::member), Type> ? 1 : -1])),       \
   Member{#member, offsetof(Struct, member), sizeof(Type), alignof(Type)})

/// Expects @p members to be packed back to back from offset 0, in this order.
template <std::size_t N> void expectContiguousFromZero(const Member (&members)[N]) {
  std::size_t prevEnd = 0;
  const char* prevName = "(start of struct)";
  for (const Member& m : members) {
    EXPECT_EQ(m.offset, startAfter(prevEnd, m.align))
        << "'" << m.name << "' does not directly follow '" << prevName
        << "': a member was inserted, removed or reordered ahead of the append-only tail";
    prevEnd = m.offset + m.size;
    prevName = m.name;
  }
}

} // namespace

/* ----------------------------- PerfConfig Layout Tests ----------------------------- */

/** @test Members the 1.0.3 library knows keep their order, types and offsets */
TEST(PerfConfigLayoutTest, ReleasedMembersStayWhereTheLibraryReadsThem) {
  const Member RELEASED[] = {
      VERNIER_PINNED_MEMBER(PerfConfig, cycles, int),
      VERNIER_PINNED_MEMBER(PerfConfig, repeats, int),
      VERNIER_PINNED_MEMBER(PerfConfig, warmup, int),
      VERNIER_PINNED_MEMBER(PerfConfig, threads, int),
      VERNIER_PINNED_MEMBER(PerfConfig, msgBytes, int),
      VERNIER_PINNED_MEMBER(PerfConfig, console, bool),
      VERNIER_PINNED_MEMBER(PerfConfig, nonBlocking, bool),
      VERNIER_PINNED_MEMBER(PerfConfig, minLevel, std::string),
      VERNIER_PINNED_MEMBER(PerfConfig, csv, std::optional<std::string>),
      VERNIER_PINNED_MEMBER(PerfConfig, profileTool, std::string),
      VERNIER_PINNED_MEMBER(PerfConfig, profileArgs, std::string),
      VERNIER_PINNED_MEMBER(PerfConfig, bpfScripts, std::vector<std::string>),
      VERNIER_PINNED_MEMBER(PerfConfig, artifactRoot, std::string),
      VERNIER_PINNED_MEMBER(PerfConfig, profileFrequency, int),
      VERNIER_PINNED_MEMBER(PerfConfig, profileAnalyze, bool),
      VERNIER_PINNED_MEMBER(PerfConfig, profileTestTimeoutSecs, int),
      VERNIER_PINNED_MEMBER(PerfConfig, quickMode, bool),
  };
  expectContiguousFromZero(RELEASED);
}

/** @test Members added since 1.0.3 follow the released ones, in the order they were added */
TEST(PerfConfigLayoutTest, NewMembersAreAppended) {
  const Member TAIL[] = {
      VERNIER_PINNED_MEMBER(PerfConfig, quickMode, bool),
      VERNIER_PINNED_MEMBER(PerfConfig, targetTimeUs, int),
  };
  EXPECT_EQ(TAIL[1].offset, startAfter(TAIL[0].offset + TAIL[0].size, TAIL[1].align))
      << "'targetTimeUs' must directly follow 'quickMode', the last 1.0.3 member";
}

#if defined(__GLIBCXX__) && defined(__LP64__)
/** @test On 64-bit libstdc++ the library-read members sit at the offsets measured in 1.0.3 */
TEST(PerfConfigLayoutTest, MatchesOffsetsMeasuredInReleasedLibrary) {
  EXPECT_EQ(offsetof(PerfConfig, minLevel), 24U);
  EXPECT_EQ(offsetof(PerfConfig, csv), 56U);
  EXPECT_EQ(offsetof(PerfConfig, profileTool), 96U);
  EXPECT_EQ(offsetof(PerfConfig, profileArgs), 128U);
  EXPECT_EQ(offsetof(PerfConfig, bpfScripts), 160U);
  EXPECT_EQ(offsetof(PerfConfig, artifactRoot), 184U);
  EXPECT_EQ(offsetof(PerfConfig, profileFrequency), 216U);
  EXPECT_EQ(offsetof(PerfConfig, profileAnalyze), 220U);
}
#endif

/* ----------------------------- Stats Layout Tests ----------------------------- */

/** @test Stats members keep their order and types */
TEST(StatsLayoutTest, MembersStayInOrder) {
  // libbench reads only `median`; libbench_cuda embeds the whole struct in the
  // results it hands back, so every member's place matters there.
  const Member PINNED[] = {
      VERNIER_PINNED_MEMBER(Stats, median, double), VERNIER_PINNED_MEMBER(Stats, p10, double),
      VERNIER_PINNED_MEMBER(Stats, p90, double),    VERNIER_PINNED_MEMBER(Stats, p99, double),
      VERNIER_PINNED_MEMBER(Stats, p999, double),   VERNIER_PINNED_MEMBER(Stats, min, double),
      VERNIER_PINNED_MEMBER(Stats, max, double),    VERNIER_PINNED_MEMBER(Stats, mean, double),
      VERNIER_PINNED_MEMBER(Stats, stddev, double), VERNIER_PINNED_MEMBER(Stats, cv, double),
  };
  expectContiguousFromZero(PINNED);
}

/* ----------------------------- PerfGpuConfig Layout Tests ----------------------------- */

/** @test PerfGpuConfig members keep their order and types */
TEST(PerfGpuConfigLayoutTest, MembersStayInOrder) {
  const Member PINNED[] = {
      VERNIER_PINNED_MEMBER(PerfGpuConfig, gpuWarmup, int),
      VERNIER_PINNED_MEMBER(PerfGpuConfig, memStrategy, PerfGpuConfig::MemoryStrategy),
      VERNIER_PINNED_MEMBER(PerfGpuConfig, captureOccupancy, bool),
      VERNIER_PINNED_MEMBER(PerfGpuConfig, captureClockSpeeds, bool),
      VERNIER_PINNED_MEMBER(PerfGpuConfig, captureMemoryBandwidth, bool),
      VERNIER_PINNED_MEMBER(PerfGpuConfig, captureUnifiedMemory, bool),
      VERNIER_PINNED_MEMBER(PerfGpuConfig, deviceId, int),
      VERNIER_PINNED_MEMBER(PerfGpuConfig, useHighPriorityStream, bool),
      VERNIER_PINNED_MEMBER(PerfGpuConfig, minSpeedupVsCpu, double),
      VERNIER_PINNED_MEMBER(PerfGpuConfig, maxTransferOverhead, double),
  };
  expectContiguousFromZero(PINNED);
}

/* ----------------------------- API Tests ----------------------------- */

class BenchAbiCheckTest : public ::testing::Test {
protected:
  std::string savedStyle_;

  // The failing path ends the process; the child re-executes the binary so
  // threads left behind by other tests in this process cannot leak into it.
  void SetUp() override {
    savedStyle_ = GTEST_FLAG_GET(death_test_style);
    GTEST_FLAG_SET(death_test_style, "threadsafe");
  }

  void TearDown() override { GTEST_FLAG_SET(death_test_style, savedStyle_); }
};

/** @test Returns for the values this test binary and libbench were both built with */
TEST_F(BenchAbiCheckTest, MatchingBuildPasses) {
  checkBenchAbi(BENCH_ABI_VERSION, sizeof(PerfConfig), sizeof(Stats));
  vernier::bench::ensureBenchAbi();
  vernier::bench::ensureBenchAbi();
  SUCCEED() << "both calls returned";
}

/** @test Accepts a benchmark whose structs are larger: members are appended */
TEST_F(BenchAbiCheckTest, LargerBenchmarkStructsPass) {
  checkBenchAbi(BENCH_ABI_VERSION, sizeof(PerfConfig) + sizeof(int),
                sizeof(Stats) + sizeof(double));
  SUCCEED() << "returned";
}

/** @test A different ABI version prints both values and exits with the mismatch status */
TEST_F(BenchAbiCheckTest, VersionMismatchReportsBothSidesAndExits) {
  const std::string EXPECTED = "^\\[bench\\] ABI mismatch: this benchmark and the libbench it "
                               "loaded were built from different vernier headers \\(ABI version: "
                               "benchmark " +
                               std::to_string(BENCH_ABI_VERSION + 1) + ", library " +
                               std::to_string(BENCH_ABI_VERSION) +
                               "\\)\\. Rebuild the benchmark against this libbench, or load the "
                               "libbench that matches the benchmark's headers\\. Exiting\\.\n$";
  EXPECT_EXIT(checkBenchAbi(BENCH_ABI_VERSION + 1, sizeof(PerfConfig), sizeof(Stats)),
              ::testing::ExitedWithCode(BENCH_ABI_MISMATCH_EXIT_CODE), EXPECTED);
}

/** @test A benchmark built with a smaller PerfConfig than the library reads is refused */
TEST_F(BenchAbiCheckTest, SmallerPerfConfigIsRefused) {
  const std::string EXPECTED = "sizeof\\(PerfConfig\\): benchmark " +
                               std::to_string(sizeof(PerfConfig) - sizeof(int)) + ", library " +
                               std::to_string(sizeof(PerfConfig)) + "\\)";
  EXPECT_EXIT(checkBenchAbi(BENCH_ABI_VERSION, sizeof(PerfConfig) - sizeof(int), sizeof(Stats)),
              ::testing::ExitedWithCode(BENCH_ABI_MISMATCH_EXIT_CODE), EXPECTED);
}

/** @test Every violated field appears in the one message, in order; satisfied fields do not */
TEST(AbiMismatchMessageTest, ListsEachViolatedFieldOnce) {
  const AbiField FIELDS[] = {
      {"ABI version", 2, 1, AbiRule::EQUAL},
      {"sizeof(A)", 48, 40, AbiRule::BENCHMARK_AT_LEAST_LIBRARY},
      {"sizeof(B)", 32, 40, AbiRule::BENCHMARK_AT_LEAST_LIBRARY},
      {"sizeof(C)", 88, 80, AbiRule::EQUAL},
  };
  const std::string MESSAGE = abiMismatchMessage("libbench_cuda", FIELDS, 4);
  EXPECT_NE(MESSAGE.find("(ABI version: benchmark 2, library 1; sizeof(B): benchmark 32, "
                         "library 40; sizeof(C): benchmark 88, library 80)"),
            std::string::npos)
      << MESSAGE;
  EXPECT_EQ(MESSAGE.find("sizeof(A)"), std::string::npos) << MESSAGE;
  EXPECT_NE(MESSAGE.find("the libbench_cuda it loaded"), std::string::npos) << MESSAGE;
  EXPECT_EQ(MESSAGE.back(), '\n');
  EXPECT_EQ(MESSAGE.find('\n'), MESSAGE.size() - 1) << "one line: " << MESSAGE;
}

/** @test No violated field yields no message */
TEST(AbiMismatchMessageTest, EmptyWhenEveryFieldHolds) {
  const AbiField FIELDS[] = {
      {"ABI version", 1, 1, AbiRule::EQUAL},
      {"sizeof(A)", 48, 40, AbiRule::BENCHMARK_AT_LEAST_LIBRARY},
  };
  EXPECT_TRUE(abiMismatchMessage("libbench", FIELDS, 2).empty());
  EXPECT_TRUE(abiMismatchMessage("libbench", nullptr, 0).empty());
}
