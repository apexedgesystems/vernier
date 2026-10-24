/**
 * @file ProfilerGperfSampling_uTest.cpp
 * @brief Unit tests for gperf's sampling rate: the decision a request and the
 * environment make together, and the reader of a finished profile.
 *
 * decideGperfSampling() and readGperfProfile() need no gperftools, so their
 * rules are tested on every build, against the table both languages read
 * (tools/rust/tests/fixtures/profile_frequency.tsv). The profiles the reader
 * is given are written here slot by slot. The gperf check and the registry's
 * decisions are tested where gperftools is compiled in, and skip without it.
 */

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/ProfilerGperf.hpp"
#include "src/bench/inc/ProfilerReadiness.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/utst/ReadinessFixtures.hpp"
#include "src/bench/utst/SamplingRateTable.hpp"
#include "src/bench/utst/StderrCapture.hpp"

#include <gtest/gtest.h>

#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using vernier::bench::checkGperfRequest;
using vernier::bench::decideGperfSampling;
using vernier::bench::EnvReport;
using vernier::bench::GperfModes;
using vernier::bench::GperfPlan;
using vernier::bench::GperfProfileSummary;
using vernier::bench::GperfRateSource;
using vernier::bench::GperfSampling;
using vernier::bench::parseProfileFrequency;
using vernier::bench::PerfConfig;
using vernier::bench::ProfileFailure;
using vernier::bench::ProfilerRegistry;
using vernier::bench::readGperfProfile;
using vernier::bench::ReadinessCause;
using vernier::bench::ReadinessContext;
using vernier::bench::ReadinessRequest;
using vernier::bench::readinessRequestFor;
using vernier::bench::ReadinessResult;
using vernier::bench::ReadinessScope;
using vernier::bench::ReadinessStage;
using vernier::bench::test::FakeToolDir;
using vernier::bench::test::samplingRateRows;
using vernier::bench::test::StderrCapture;
using vernier::bench::test::tableText;

/// The variable a rate is applied through.
const std::string VARIABLE = vernier::bench::SAMPLING_RATE_VARIABLE;

const GperfModes CPU{true, false};
const GperfModes HEAP{false, true};
const GperfModes BOTH{true, true};

/** @brief A snapshot whose CPUPROFILE_FREQUENCY is @p value; unset when it is nullopt. */
ReadinessContext withVariable(const std::optional<std::string>& value) {
  std::map<std::string, std::string> env{{"PATH", "/nonexistent"}};
  if (value) {
    env[VARIABLE] = *value;
  }
  return ReadinessContext(::geteuid(), ::getpid(), std::move(env));
}

/** @brief A table field that names a value of the variable: '-' is unset. */
std::optional<std::string> variableField(const std::string& field) {
  return field == "-" ? std::nullopt : std::optional<std::string>{field};
}

/** @brief A table field that names a request: '-' is none (0). */
int requestField(const std::string& field) {
  if (field == "-") {
    return 0;
  }
  const std::optional<int> HZ = parseProfileFrequency(field);
  EXPECT_TRUE(HZ.has_value()) << "the table's request '" << field << "' is not a rate";
  return HZ.value_or(0);
}

/** @brief This program as a command line names it: its argv[0], as one shell word. */
std::string programWord() {
#if defined(__GLIBC__) && defined(_GNU_SOURCE)
  if (program_invocation_name != nullptr && *program_invocation_name != '\0') {
    return vernier::bench::detail::shellQuote(program_invocation_name);
  }
#endif
  return "<this-binary>";
}

/** @brief The remedy of a rate @p hz this process did not start with; @p args after gperf. */
std::string startRemedy(int hz, const std::string& args = "") {
  const std::string HZ = std::to_string(hz);
  const std::string REQUEST = "--profile gperf" + args + " --profile-frequency " + HZ;
  return "Run it through bench run " + programWord() + " " + REQUEST +
         ", which sets CPUPROFILE_FREQUENCY before the benchmark starts, or start it as "
         "CPUPROFILE_FREQUENCY=" +
         HZ + " " + programWord() + " " + REQUEST + ".";
}

/** @brief The READY clause of a rate taken from the environment. */
std::string inheritedClause(int hz, int periodUs) {
  return "sampling at CPUPROFILE_FREQUENCY=" + std::to_string(hz) +
         " from the environment: expected period " + std::to_string(periodUs) +
         " us, checked in each finished profile";
}

/** @brief The READY clause of a rate asked for, which the environment carries. */
std::string requestedClause(int hz, int periodUs) {
  return std::to_string(hz) +
         " Hz requested; the current environment matches the request (CPUPROFILE_FREQUENCY=" +
         std::to_string(hz) + "); expected period " + std::to_string(periodUs) +
         " us, checked in each finished profile";
}

/** @brief Expect @p s to be a CONFIGURATION error with @p message and @p hint, and nothing else. */
void expectRefused(const GperfSampling& s, const std::string& message, const std::string& hint) {
  ASSERT_TRUE(s.error.has_value()) << "not refused; expected: " << message;
  EXPECT_EQ(s.error->report.status, EnvReport::Status::Error);
  EXPECT_EQ(s.error->cause, ReadinessCause::CONFIGURATION);
  EXPECT_EQ(s.error->stage, ReadinessStage::COLLECTION);
  EXPECT_EQ(s.error->report.message, "configuration: " + message);
  EXPECT_EQ(s.error->report.hint, hint);
  EXPECT_FALSE(s.expectedPeriodUs.has_value()) << "a refusal expects a period";
  EXPECT_EQ(s.clause, "");
}

} // namespace

/* ----------------------------- The Decision ----------------------------- */

/**
 * @test No rate asked for and none in the environment: gperftools' default,
 * a period of 10000 us expected, and nothing added to the READY message.
 */
TEST(GperfSamplingDecision, NoRateAndNoVariableIsTheDefault) {
  for (const GperfModes& MODES : {CPU, BOTH}) {
    const GperfSampling S = decideGperfSampling(MODES, 0, withVariable(std::nullopt));
    EXPECT_FALSE(S.error.has_value()) << S.error->report.message;
    EXPECT_EQ(S.expectedPeriodUs, 10000);
    EXPECT_EQ(S.source, GperfRateSource::DEFAULT);
    EXPECT_EQ(S.hz, 0);
    EXPECT_EQ(S.clause, "") << "the default adds to the READY message";
  }
}

/**
 * @test No rate asked for and a whole number in the environment: that rate,
 * its period expected, and the clause says where it comes from.
 */
TEST(GperfSamplingDecision, VariableAloneIsTheRate) {
  const GperfSampling S = decideGperfSampling(CPU, 0, withVariable("250"));
  EXPECT_FALSE(S.error.has_value()) << S.error->report.message;
  EXPECT_EQ(S.expectedPeriodUs, 4000);
  EXPECT_EQ(S.source, GperfRateSource::INHERITED);
  EXPECT_EQ(S.hz, 250);
  EXPECT_EQ(S.clause, inheritedClause(250, 4000));
  const GperfSampling ZEROS = decideGperfSampling(CPU, 0, withVariable("0250"));
  EXPECT_EQ(ZEROS.expectedPeriodUs, 4000) << "leading zeros are digits";
  EXPECT_EQ(ZEROS.clause, inheritedClause(250, 4000));
}

/**
 * @test No rate asked for and a value in the environment that is not a whole
 * number from 1 to 4000: refused, whatever gperftools would make of it.
 */
TEST(GperfSamplingDecision, VariableThatIsNoRateIsRefused) {
  const std::vector<std::string> VALUES = {"abc", "0", "4001", "10000", "+250", " 250", "250 ", ""};
  for (const std::string& VALUE : VALUES) {
    expectRefused(decideGperfSampling(CPU, 0, withVariable(VALUE)),
                  "CPUPROFILE_FREQUENCY='" + VALUE +
                      "' in the environment is not a whole number of Hz from 1 to 4000: "
                      "gperftools reads such a value without saying what rate it takes from it",
                  "Unset it, or set it to a whole number from 1 to 4000 before the benchmark "
                  "starts.");
  }
}

/**
 * @test A rate asked for that the environment carries, compared as numbers:
 * that rate, its period expected, and the clause says the environment
 * matches; with heap profiling too, it is the CPU capture's.
 */
TEST(GperfSamplingDecision, RequestTheEnvironmentCarriesIsTheRate) {
  const GperfSampling S = decideGperfSampling(CPU, 1000, withVariable("1000"));
  EXPECT_FALSE(S.error.has_value()) << S.error->report.message;
  EXPECT_EQ(S.expectedPeriodUs, 1000);
  EXPECT_EQ(S.source, GperfRateSource::REQUESTED);
  EXPECT_EQ(S.hz, 1000);
  EXPECT_EQ(S.clause, requestedClause(1000, 1000));
  const GperfSampling ZEROS = decideGperfSampling(BOTH, 250, withVariable("0250"));
  EXPECT_FALSE(ZEROS.error.has_value()) << ZEROS.error->report.message;
  EXPECT_EQ(ZEROS.expectedPeriodUs, 4000);
  EXPECT_EQ(ZEROS.clause, requestedClause(250, 4000));
  const GperfSampling QUANTIZED = decideGperfSampling(CPU, 3000, withVariable("3000"));
  EXPECT_EQ(QUANTIZED.expectedPeriodUs, 333) << "1000000 / 3000, rounded down, as gperftools";
}

/**
 * @test A rate asked for that this process did not start with (no variable,
 * or another value, a refused one included) is refused: gperftools reads the
 * rate only as the process starts. The remedy is the command that starts it
 * with the rate, the modes asked for included.
 */
TEST(GperfSamplingDecision, RequestTheProcessDidNotStartWithIsRefused) {
  expectRefused(decideGperfSampling(CPU, 1000, withVariable(std::nullopt)),
                "1000 Hz requested, and this process's environment has no "
                "CPUPROFILE_FREQUENCY: gperftools reads the rate only when the process starts",
                startRemedy(1000));
  const std::vector<std::string> VALUES = {"250", "abc", "", "1000 "};
  for (const std::string& VALUE : VALUES) {
    expectRefused(decideGperfSampling(CPU, 1000, withVariable(VALUE)),
                  "1000 Hz requested, and this process's environment has CPUPROFILE_FREQUENCY='" +
                      VALUE + "': gperftools reads the rate only when the process starts",
                  startRemedy(1000));
  }
  const GperfSampling BOTH_MODES = decideGperfSampling(BOTH, 250, withVariable(std::nullopt));
  ASSERT_TRUE(BOTH_MODES.error.has_value());
  EXPECT_EQ(BOTH_MODES.error->report.hint, startRemedy(250, " --profile-args both"));
}

/**
 * @test A rate outside 1 to 4000 (only a source value can be one, the 10000
 * of headers before 1.0.4 among them) is refused whatever the environment
 * says, heap profiling alone included.
 */
TEST(GperfSamplingDecision, RateOutsideTheRangeIsRefused) {
  for (const int HZ : {-5, 4001, 10000}) {
    for (const GperfModes& MODES : {CPU, HEAP, BOTH}) {
      for (const std::optional<std::string>& VALUE :
           {std::optional<std::string>{}, std::optional<std::string>{std::to_string(HZ)},
            std::optional<std::string>{"250"}}) {
        expectRefused(decideGperfSampling(MODES, HZ, withVariable(VALUE)),
                      "a sampling rate of " + std::to_string(HZ) +
                          " Hz was asked for (--profile-frequency or "
                          "PerfConfig::profileFrequency): gperftools samples at 1 to 4000 Hz",
                      "Ask for a whole number from 1 to 4000, or leave "
                      "PerfConfig::profileFrequency at 0 for gperftools' default 100 Hz.");
      }
    }
  }
}

/**
 * @test Heap profiling alone takes no rate: a rate asked for is refused
 * before the environment is looked at, and without one the variable is not
 * read at all (a refused value passes) and no period is expected.
 */
TEST(GperfSamplingDecision, HeapAloneTakesNoRate) {
  for (const std::optional<std::string>& VALUE :
       {std::optional<std::string>{}, std::optional<std::string>{"1000"},
        std::optional<std::string>{"abc"}}) {
    expectRefused(decideGperfSampling(HEAP, 1000, withVariable(VALUE)),
                  "--profile-frequency 1000 sets the CPU sampling rate, and --profile-args heap "
                  "profiles the heap only",
                  "Profile the CPU too (--profile-args both), or drop --profile-frequency.");
    const GperfSampling NONE = decideGperfSampling(HEAP, 0, withVariable(VALUE));
    EXPECT_FALSE(NONE.error.has_value()) << NONE.error->report.message;
    EXPECT_FALSE(NONE.expectedPeriodUs.has_value()) << "heap profiling expects a CPU period";
    EXPECT_EQ(NONE.clause, "");
  }
}

/* ----------------------------- The Shared Table ----------------------------- */

/**
 * @test Every rate of the shared table, as a request the environment carries
 * and as the variable alone: the rates it accepts give its period, the ones
 * it refuses are refused.
 */
TEST(GperfSamplingTable, RatesAsTheTableSays) {
  const auto ROWS = samplingRateRows("rate");
  ASSERT_GE(ROWS.size(), 20U) << "the shared table was not read";
  for (const auto& row : ROWS) {
    const std::string INPUT = tableText(row[1]);
    const GperfSampling ALONE = decideGperfSampling(CPU, 0, withVariable(INPUT));
    if (row[2] == "ok") {
      const int PERIOD = std::stoi(row[4]);
      EXPECT_EQ(ALONE.expectedPeriodUs, PERIOD) << "the variable '" << INPUT << "'";
      const GperfSampling ASKED = decideGperfSampling(CPU, std::stoi(row[3]), withVariable(INPUT));
      EXPECT_EQ(ASKED.expectedPeriodUs, PERIOD) << "a request of '" << INPUT << "'";
      EXPECT_EQ(ASKED.source, GperfRateSource::REQUESTED) << INPUT;
    } else {
      EXPECT_TRUE(ALONE.error.has_value()) << "the variable '" << INPUT << "' was taken";
    }
  }
  const auto VARIABLE_ROWS = samplingRateRows("variable");
  ASSERT_EQ(VARIABLE_ROWS.size(), 1U);
  EXPECT_EQ(VARIABLE_ROWS[0][1], VARIABLE);
}

/**
 * @test The benchmark's half of the table's precedence rows (given the
 * request and the value `bench run` starts it with) and of its direct rows
 * (started directly with that request and that value): the decision and the
 * expected period.
 */
TEST(GperfSamplingTable, PrecedenceAndDirectRowsAsTheTableSays) {
  const auto PRECEDENCE = samplingRateRows("precedence");
  const auto DIRECT = samplingRateRows("direct");
  ASSERT_GE(PRECEDENCE.size(), 8U) << "the shared table was not read";
  ASSERT_GE(DIRECT.size(), 8U);
  // precedence: kind request inherited child decision period
  // direct:     kind request variable decision period
  std::vector<std::vector<std::string>> cases;
  for (const auto& row : PRECEDENCE) {
    cases.push_back({row[1], row[3], row[4], row[5]});
  }
  for (const auto& row : DIRECT) {
    cases.push_back({row[1], row[2], row[3], row[4]});
  }
  for (const auto& c : cases) {
    const GperfSampling S =
        decideGperfSampling(CPU, requestField(c[0]), withVariable(variableField(c[1])));
    const std::string WHAT = "request '" + c[0] + "', variable '" + c[1] + "'";
    if (c[2] == "ready") {
      EXPECT_FALSE(S.error.has_value()) << WHAT << ": " << S.error->report.message;
      EXPECT_EQ(S.expectedPeriodUs, std::stoi(c[3])) << WHAT;
    } else {
      ASSERT_EQ(c[2], "configuration") << "the table's decision for " << WHAT;
      ASSERT_TRUE(S.error.has_value()) << WHAT;
      EXPECT_EQ(S.error->cause, ReadinessCause::CONFIGURATION) << WHAT;
    }
  }
}

/* ----------------------------- The Profile Reader ----------------------------- */

namespace {

using Slot = std::uintptr_t;

/// The size of one slot of a profile this build writes and reads.
constexpr std::size_t SLOT_BYTES = sizeof(Slot);

/** @brief A header recording @p periodUs: 0, 3, 0, the period, 0. */
std::vector<Slot> header(Slot periodUs) { return {0, 3, 0, periodUs, 0}; }

/** @brief @p a followed by @p b. */
std::vector<Slot> operator+(std::vector<Slot> a, const std::vector<Slot>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

/// The trailer every complete profile ends its binary part with.
const std::vector<Slot> TRAILER = {0, 1, 0};

/** @brief Profiles written slot by slot into a private folder. */
class GperfProfileReader : public ::testing::Test {
protected:
  void SetUp() override { ASSERT_TRUE(dir_.ok()); }

  /** @brief Write @p slots, then @p text, as profile @p name; returns its path. */
  std::string write(const std::string& name, const std::vector<Slot>& slots,
                    const std::string& text = "") const {
    const std::string PATH = dir_.path() + "/" + name;
    std::ofstream out(PATH, std::ios::binary | std::ios::trunc);
    for (const Slot VALUE : slots) {
      out.write(reinterpret_cast<const char*>(&VALUE), sizeof VALUE);
    }
    out << text;
    return PATH;
  }

  /** @brief Expect the profile of @p slots and @p text to be refused for @p problem. */
  void expectProblem(const std::vector<Slot>& slots, const std::string& problem,
                     const std::string& text = "") const {
    const GperfProfileSummary S = readGperfProfile(write("refused.prof", slots, text));
    EXPECT_FALSE(S.valid) << "taken: " << problem;
    EXPECT_EQ(S.problem, problem);
  }

  FakeToolDir dir_;
};

/** @brief @p slot as the byte offset it starts at, in text. */
std::string byteOf(std::size_t slot) { return std::to_string(slot * SLOT_BYTES); }

} // namespace

/**
 * @test A complete profile: its period, the samples its records count, and
 * where its binary part ends; the mapped-objects text after it is not read.
 */
TEST_F(GperfProfileReader, CompleteProfile) {
  const std::string PATH = write("cpu.prof",
                                 header(1000) + std::vector<Slot>{5, 2, 0x1000, 0x2000} +
                                     std::vector<Slot>{3, 1, 0x3000} + TRAILER,
                                 "00400000-00452000 r-xp 00000000 08:01 1 /bin/bench\n");
  const GperfProfileSummary S = readGperfProfile(PATH);
  EXPECT_TRUE(S.valid) << S.problem;
  EXPECT_EQ(S.periodUs, 1000);
  EXPECT_EQ(S.samples, 8U);
  EXPECT_EQ(S.binaryBytes, static_cast<long long>(15 * SLOT_BYTES));
  EXPECT_EQ(S.problem, "");
}

/** @test A profile without samples is complete: its period is still recorded. */
TEST_F(GperfProfileReader, ProfileWithoutSamples) {
  const GperfProfileSummary S = readGperfProfile(write("cpu.prof", header(10000) + TRAILER));
  EXPECT_TRUE(S.valid) << S.problem;
  EXPECT_EQ(S.periodUs, 10000);
  EXPECT_EQ(S.samples, 0U);
  EXPECT_EQ(S.binaryBytes, static_cast<long long>(8 * SLOT_BYTES));
}

/**
 * @test A header with more slots than three after its count is read past
 * them: the records start where the header says it ends.
 */
TEST_F(GperfProfileReader, LongerHeader) {
  const GperfProfileSummary S =
      readGperfProfile(write("cpu.prof", std::vector<Slot>{0, 4, 0, 4000, 0, 0} +
                                             std::vector<Slot>{2, 1, 0x10} + TRAILER));
  EXPECT_TRUE(S.valid) << S.problem;
  EXPECT_EQ(S.periodUs, 4000);
  EXPECT_EQ(S.samples, 2U);
  EXPECT_EQ(S.binaryBytes, static_cast<long long>(12 * SLOT_BYTES));
}

/** @test The periods of 4000 Hz and 1 Hz are the bounds of what gperftools writes. */
TEST_F(GperfProfileReader, PeriodBounds) {
  EXPECT_TRUE(readGperfProfile(write("fast.prof", header(250) + TRAILER)).valid);
  EXPECT_TRUE(readGperfProfile(write("slow.prof", header(1000000) + TRAILER)).valid);
  for (const Slot PERIOD : {Slot{0}, Slot{249}, Slot{1000001}}) {
    expectProblem(header(PERIOD) + TRAILER,
                  "its header records a sampling period of " + std::to_string(PERIOD) +
                      " us, outside the 250 to 1000000 us gperftools writes");
  }
}

/** @test A header that does not start 0, at least 3, 0 is not a gperftools CPU profile. */
TEST_F(GperfProfileReader, ForeignHeader) {
  const std::string WRITES = " where gperftools writes 0, then 3 or more, then 0";
  expectProblem(std::vector<Slot>{1, 3, 0, 1000, 0} + TRAILER, "its header starts 1,3,0" + WRITES);
  expectProblem(std::vector<Slot>{0, 2, 0, 1000, 0} + TRAILER, "its header starts 0,2,0" + WRITES);
  expectProblem(std::vector<Slot>{0, 3, 1, 1000, 0} + TRAILER, "its header starts 0,3,1" + WRITES);
}

/** @test A file too short for a header: empty, or one byte short of five slots. */
TEST_F(GperfProfileReader, NoHeader) {
  const std::string SLOTS =
      " bytes hold no header (5 slots of " + std::to_string(SLOT_BYTES) + " bytes)";
  expectProblem({}, "its 0" + SLOTS);
  expectProblem({0, 3, 0, 1000}, "its " + std::to_string(5 * SLOT_BYTES - 1) + SLOTS,
                std::string(SLOT_BYTES - 1, '\0'));
}

/**
 * @test A profile cut short: after the header alone, after a whole record,
 * inside a record, inside the trailer, or inside a header longer than the
 * file.
 */
TEST_F(GperfProfileReader, TruncatedProfile) {
  expectProblem(header(1000), "it ends at byte " + byteOf(5) + " with no trailer (0, 1, 0)");
  expectProblem(header(1000) + std::vector<Slot>{5, 1, 0x10},
                "it ends at byte " + byteOf(8) + " with no trailer (0, 1, 0)");
  expectProblem(header(1000) + std::vector<Slot>{5, 3, 0x10, 0x20},
                "the record at byte " + byteOf(5) + " runs past the end of the file");
  expectProblem(header(1000) + std::vector<Slot>{0, 1},
                "the record at byte " + byteOf(5) + " runs past the end of the file");
  expectProblem(std::vector<Slot>{0, 100, 0, 1000, 0} + TRAILER,
                "its header of 102 slots runs past the end of the file");
}

/** @test Records gperftools never writes: no program counters, or no samples unless the trailer. */
TEST_F(GperfProfileReader, RecordsGperftoolsNeverWrites) {
  expectProblem(header(1000) + std::vector<Slot>{5, 0} + TRAILER,
                "the record at byte " + byteOf(5) + " holds no program counters");
  expectProblem(header(1000) + std::vector<Slot>{0, 2, 0x10, 0x20} + TRAILER,
                "the record at byte " + byteOf(5) +
                    " counts no samples and is not the trailer (0, 1, 0)");
  expectProblem(header(1000) + std::vector<Slot>{0, 1, 0x10} + TRAILER,
                "the record at byte " + byteOf(5) +
                    " counts no samples and is not the trailer (0, 1, 0)");
}

/** @test A file that is not there cannot be read. */
TEST_F(GperfProfileReader, MissingFile) {
  const GperfProfileSummary S = readGperfProfile(dir_.path() + "/none.prof");
  EXPECT_FALSE(S.valid);
  EXPECT_EQ(S.problem, "it cannot be opened");
}

/* ----------------------------- The Check (gperftools builds) ----------------------------- */

namespace {

/** @brief gperf's check of a request for @p modes words and rate @p hz in @p ctx. */
ReadinessResult checkGperf(const std::string& profileArgs, int hz, const ReadinessContext& ctx) {
  ReadinessRequest request;
  request.backend = "gperf";
  request.profileArgs = profileArgs;
  request.profileFrequency = hz;
  request.scope = ReadinessScope::PREFLIGHT;
  return checkGperfRequest(request, ctx);
}

/** @brief What the check says of a CPU capture with no analyzer on PATH, before any rate. */
std::string readyMessage(const std::string& modes = "cpu") {
  return "gperftools profiles " + modes +
         (UB_HAS_GPERF_HEAP != 0 ? " (built: cpu, heap)" : " (built: cpu)") +
         "; no analyzer on PATH (only --profile-analyze needs one)";
}

} // namespace

/**
 * @test The check takes the sampling decision into its own: the default's
 * READY message is the one it had before rates were checked and its plan
 * expects 10000 us; an inherited or requested rate adds its clause and its
 * plan expects that period; a refusal carries no plan, so nothing is
 * collected.
 */
TEST(GperfSamplingCheck, CheckCarriesTheSamplingDecision) {
  if (UB_HAS_GPERF_CPU == 0) {
    GTEST_SKIP() << "gperftools CPU profiling is not compiled in";
  }
  const ReadinessResult DEFAULT = checkGperf("", 0, withVariable(std::nullopt));
  EXPECT_EQ(DEFAULT.report.status, EnvReport::Status::Ok) << DEFAULT.report.message;
  EXPECT_EQ(DEFAULT.report.message, readyMessage()) << "the default's doctor row changed";
  const auto DEFAULT_PLAN = std::dynamic_pointer_cast<const GperfPlan>(DEFAULT.plan);
  ASSERT_NE(DEFAULT_PLAN, nullptr);
  EXPECT_EQ(DEFAULT_PLAN->expectedPeriodUs, 10000);
  EXPECT_EQ(DEFAULT_PLAN->samplingSource, GperfRateSource::DEFAULT);

  const ReadinessResult INHERITED = checkGperf("", 0, withVariable("250"));
  EXPECT_EQ(INHERITED.report.message, readyMessage() + "; " + inheritedClause(250, 4000));
  const auto INHERITED_PLAN = std::dynamic_pointer_cast<const GperfPlan>(INHERITED.plan);
  ASSERT_NE(INHERITED_PLAN, nullptr);
  EXPECT_EQ(INHERITED_PLAN->expectedPeriodUs, 4000);
  EXPECT_EQ(INHERITED_PLAN->samplingSource, GperfRateSource::INHERITED);
  EXPECT_EQ(INHERITED_PLAN->samplingHz, 250);

  const ReadinessResult REQUESTED = checkGperf("", 1000, withVariable("1000"));
  EXPECT_EQ(REQUESTED.report.message, readyMessage() + "; " + requestedClause(1000, 1000));
  const auto REQUESTED_PLAN = std::dynamic_pointer_cast<const GperfPlan>(REQUESTED.plan);
  ASSERT_NE(REQUESTED_PLAN, nullptr);
  EXPECT_EQ(REQUESTED_PLAN->expectedPeriodUs, 1000);
  EXPECT_EQ(REQUESTED_PLAN->samplingSource, GperfRateSource::REQUESTED);
  EXPECT_EQ(REQUESTED_PLAN->samplingHz, 1000);

  const ReadinessResult REFUSED = checkGperf("", 1000, withVariable(std::nullopt));
  EXPECT_EQ(REFUSED.cause, ReadinessCause::CONFIGURATION);
  EXPECT_FALSE(REFUSED.collectionReady());
  EXPECT_EQ(REFUSED.plan, nullptr) << "a refused rate carries a plan to collect with";
}

/**
 * @test The heap-only refusal comes after the build's: in a build without
 * heap profiling a heap request is refused for that, rate or not.
 */
TEST(GperfSamplingCheck, HeapRefusalsInTheBuildsOrder) {
  if (UB_HAS_GPERF_CPU == 0) {
    GTEST_SKIP() << "gperftools CPU profiling is not compiled in";
  }
  const ReadinessResult R = checkGperf("heap", 1000, withVariable("1000"));
  EXPECT_EQ(R.report.status, EnvReport::Status::Error);
  if (UB_HAS_GPERF_HEAP != 0) {
    EXPECT_EQ(R.report.message, "configuration: --profile-frequency 1000 sets the CPU sampling "
                                "rate, and --profile-args heap profiles the heap only");
  } else {
    EXPECT_EQ(R.cause, ReadinessCause::UNSUPPORTED) << R.report.message;
  }
}

/* ----------------------------- Through the Registry (gperftools builds)
 * ----------------------------- */

/**
 * @test The four requests of the registry's memo test get gperf's own
 * decisions, the doctor's row and the run alike: 250 Hz with 250 in the
 * environment and 1000 with 1000 are ready, each expecting its period; 1000
 * with 250 is refused, and the run records it and builds a profiler that
 * collects nothing; no rate with 250 takes the variable's rate.
 */
TEST(GperfSamplingRegistry, DoctorRowAndRunDecideAlike) {
  if (UB_HAS_GPERF_CPU == 0) {
    GTEST_SKIP() << "gperftools CPU profiling is not compiled in";
  }
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  ProfilerRegistry& reg = ProfilerRegistry::instance();
  reg.resetFailures();
  reg.resetReadiness();
  struct Case {
    int hz;
    std::string variable;
    std::string message; ///< The decision's message.
  };
  const std::vector<Case> CASES = {
      {250, "250", readyMessage() + "; " + requestedClause(250, 4000)},
      {1000, "250",
       "configuration: 1000 Hz requested, and this process's environment has "
       "CPUPROFILE_FREQUENCY='250': gperftools reads the rate only when the process starts"},
      {1000, "1000", readyMessage() + "; " + requestedClause(1000, 1000)},
      {0, "250", readyMessage() + "; " + inheritedClause(250, 4000)},
  };
  for (const Case& C : CASES) {
    const ReadinessContext CTX = dir.context({{VARIABLE, C.variable}});
    PerfConfig cfg;
    cfg.profileTool = "gperf";
    cfg.profileFrequency = C.hz;
    cfg.artifactRoot = dir.path();
    const std::string WHAT = std::to_string(C.hz) + " Hz, " + VARIABLE + "=" + C.variable;
    const ReadinessResult ROW =
        reg.checkRequest(readinessRequestFor(cfg, ReadinessScope::PREFLIGHT, CTX), CTX);
    EXPECT_EQ(ROW.report.message, C.message) << WHAT;
    StderrCapture quiet;
    reg.resetFailures();
    const auto PROFILER = reg.make("gperf", cfg, "Rate.Case", CTX);
    const std::vector<ProfileFailure> FAILED = reg.failures();
    if (ROW.report.status == EnvReport::Status::Ok) {
      EXPECT_EQ(PROFILER->toolName(), "gperf") << WHAT;
      EXPECT_TRUE(FAILED.empty()) << WHAT << ": " << FAILED.front().result.report.message;
    } else {
      EXPECT_EQ(PROFILER->toolName(), "") << WHAT << ": a profiler for a refused rate";
      ASSERT_EQ(FAILED.size(), 1U) << WHAT;
      EXPECT_EQ(FAILED[0].result.report.message, C.message)
          << WHAT << ": the run and the doctor differ";
      EXPECT_EQ(FAILED[0].result.report.hint, ROW.report.hint) << WHAT;
    }
  }
  reg.resetFailures();
  reg.resetReadiness();
}
