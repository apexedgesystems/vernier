/**
 * @file CuptiCollectorDecision_uTest.cpp
 * @brief The CUPTI collector applies the shared decision before it registers:
 *        a setting that stands it down, or that is rejected, registers nothing.
 *        And it checks what CUPTI answers: a refusal leaves it unavailable
 *        with the reason, and a window whose records are not known to be
 *        complete reports no launch and names the problem.
 *
 * Built against the counting stand-in in fake_cupti/ instead of CUPTI (first on
 * this target's include path), so it runs on any machine: it compiles the real
 * CuptiCollector.cu into this test, counts the CUPTI calls it makes and sets
 * what each call answers. Each test sets or clears the variables it depends on
 * and restores them, and starts from the stand-in's defaults.
 */

#include "src/bench/src/CuptiCollector.cu" // the real source, on the counting stand-in

#include "src/bench/utst/ScopedEnv.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using vernier::bench::CuptiCollector;
using vernier::bench::ReadinessCause;
using vernier::bench::ReadinessResult;
using vernier::bench::readinessResult;
using vernier::bench::profiler_env::CuptiDecision;
using vernier::bench::profiler_env::cuptiDecision;
using vernier::bench::test::ScopedEnv;

namespace {

/// Every value the common grammar reads as true, false, or neither.
const char* const TRUE_SPELLINGS[] = {"1", "true", "TRUE", "True", "yes", "YES", "on", "On"};
const char* const FALSE_SPELLINGS[] = {"0", "false", "FALSE", "no", "No", "off", "OFF", ""};
const char* const INVALID_VALUES[] = {"2", "-1", "maybe", "disable", "truee", "y", " 1", "1 "};

} // namespace

/** @brief No session, no override, and no registration counted yet. */
class CuptiCollectorDecision : public ::testing::Test {
protected:
  void SetUp() override {
    for (const char* name : {"VERNIER_DISABLE_CUPTI", "VERNIER_EXTERNAL_WRAP",
                             "NSYS_PROFILING_SESSION_ID", "NV_NSIGHT_INJECTION_PORT_BASE"}) {
      scrub_[count_++].emplace(name, nullptr);
    }
    fake_cupti::reset();
  }

  void TearDown() override { fake_cupti::reset(); }

  /** @brief Registrations made while building one collector for @p value, or -1 when it threw. */
  static int registrationsFor(const char* value, bool forceDisabled = false) {
    const ScopedEnv SETTING("VERNIER_DISABLE_CUPTI", value);
    fake_cupti::calls() = {};
    try {
      const CuptiCollector COLLECTOR(forceDisabled);
      return fake_cupti::calls().registrations;
    } catch (const std::invalid_argument&) {
      return fake_cupti::calls().registrations == 0 ? -1 : -2;
    }
  }

private:
  std::optional<ScopedEnv> scrub_[4];
  int count_ = 0;
};

/** @test Without a session or an override the collector registers once, and is available. */
TEST_F(CuptiCollectorDecision, RegistersWithoutASessionOrOverride) {
  const CuptiCollector COLLECTOR;
  EXPECT_TRUE(COLLECTOR.isAvailable());
  EXPECT_EQ(fake_cupti::calls().registrations, 1);
}

/** @test Every false spelling, in any case, and the empty value leave it registering. */
TEST_F(CuptiCollectorDecision, EveryFalseSpellingKeepsItRegistering) {
  for (const char* value : FALSE_SPELLINGS) {
    EXPECT_EQ(registrationsFor(value), 1) << "VERNIER_DISABLE_CUPTI='" << value << "'";
  }
}

/** @test Every true spelling, in any case, stands it down before it registers. */
TEST_F(CuptiCollectorDecision, EveryTrueSpellingStandsItDownFirst) {
  for (const char* value : TRUE_SPELLINGS) {
    EXPECT_EQ(registrationsFor(value), 0) << "VERNIER_DISABLE_CUPTI='" << value << "'";
  }
}

/** @test Inside a session a false value does not keep it on: nothing registers. */
TEST_F(CuptiCollectorDecision, ASessionWinsOverFalse) {
  const ScopedEnv SESSION("NSYS_PROFILING_SESSION_ID", "1017521");
  for (const char* value : FALSE_SPELLINGS) {
    EXPECT_EQ(registrationsFor(value), 0) << "VERNIER_DISABLE_CUPTI='" << value << "'";
  }
}

/**
 * @test Any other value is a configuration error, raised before anything
 *       registers, in the words of its CONFIGURATION readiness report
 */
TEST_F(CuptiCollectorDecision, AnInvalidValueIsRejectedBeforeRegistering) {
  for (const char* value : INVALID_VALUES) {
    EXPECT_EQ(registrationsFor(value), -1)
        << "VERNIER_DISABLE_CUPTI='" << value << "' was not rejected before registering";
  }
  const ScopedEnv SETTING("VERNIER_DISABLE_CUPTI", "maybe");
  try {
    const CuptiCollector COLLECTOR;
    FAIL() << "an invalid value was accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_EQ(std::string(e.what()).rfind("configuration: VERNIER_DISABLE_CUPTI='maybe' is not a "
                                          "boolean. Use 1, true, yes or on",
                                          0),
              0U)
        << e.what();
    const CuptiDecision DECISION = cuptiDecision();
    const ReadinessResult REPORT =
        readinessResult(ReadinessCause::CONFIGURATION, DECISION.error, DECISION.remedy);
    EXPECT_EQ(std::string(e.what()), REPORT.report.message + ". " + REPORT.report.hint);
  }
}

/** @test An invalid value is still rejected inside a session: the error is never silent. */
TEST_F(CuptiCollectorDecision, AnInvalidValueIsRejectedInsideASession) {
  const ScopedEnv SESSION("NV_NSIGHT_INJECTION_PORT_BASE", "49152");
  EXPECT_EQ(registrationsFor("maybe"), -1);
}

/** @test An explicit forceDisabled wins without reading the setting: no error, no registration. */
TEST_F(CuptiCollectorDecision, ForceDisabledWinsWithoutReadingTheSetting) {
  EXPECT_EQ(registrationsFor("maybe", /*forceDisabled=*/true), 0);
  EXPECT_EQ(registrationsFor(nullptr, /*forceDisabled=*/true), 0);
}

/* ----------------------------- Collection Checked ----------------------------- */

namespace {

constexpr std::size_t LAUNCHES = 12;     ///< Kernel records one flush delivers
constexpr std::uint16_t REGISTERS = 16;  ///< Each record's registers per thread
constexpr std::int32_t STATIC_SMEM = 48; ///< Each record's static shared memory (bytes)

/** @brief Records a flush delivers: LAUNCHES kernels of REGISTERS and STATIC_SMEM each. */
void deliverLaunches() {
  fake_cupti::behaviour().recordsPerFlush = LAUNCHES;
  fake_cupti::behaviour().registersPerThread = REGISTERS;
  fake_cupti::behaviour().staticSharedMemory = STATIC_SMEM;
}

/** @brief One start/stop window of @p collector. */
void window(CuptiCollector& collector) {
  collector.start();
  collector.stop();
}

} // namespace

/** @test A window's records give its count and medians; nothing is reported missing. */
TEST_F(CuptiCollectorDecision, CountsTheWindowsRecords) {
  deliverLaunches();
  CuptiCollector collector;
  window(collector);
  EXPECT_TRUE(collector.isAvailable());
  EXPECT_TRUE(collector.unavailableReason().empty()) << collector.unavailableReason();
  EXPECT_TRUE(collector.windowProblem().empty()) << collector.windowProblem();
  EXPECT_EQ(collector.stats().kernelLaunches, LAUNCHES);
  EXPECT_EQ(collector.stats().registersMedian, REGISTERS);
  EXPECT_EQ(collector.stats().registersMax, REGISTERS);
  EXPECT_EQ(collector.stats().staticSmemBytes, static_cast<std::uint32_t>(STATIC_SMEM));
  EXPECT_EQ(collector.stats().dynamicSmemBytes, 0U);
  EXPECT_EQ(collector.stats().firstKernelName, "fakeKernel");
}

/**
 * @test A window switches on KERNEL records alone, once, and switches them off:
 *       with KERNEL on, CUPTI refuses CONCURRENT_KERNEL
 */
TEST_F(CuptiCollectorDecision, EnablesKernelRecordsOnly) {
  CuptiCollector collector;
  collector.start();
  EXPECT_EQ(fake_cupti::calls().enabled,
            std::vector<CUpti_ActivityKind>{CUPTI_ACTIVITY_KIND_KERNEL});
  collector.stop();
  EXPECT_EQ(fake_cupti::calls().disables, 1);
  EXPECT_EQ(fake_cupti::calls().flushes, 1);
}

/** @test A refused registration leaves it unavailable, says why, and nothing is switched on. */
TEST_F(CuptiCollectorDecision, ARefusedRegistrationIsStated) {
  fake_cupti::behaviour().registerResult = CUPTI_ERROR_UNKNOWN;
  deliverLaunches();
  CuptiCollector collector;
  window(collector);
  EXPECT_FALSE(collector.isAvailable());
  EXPECT_EQ(collector.unavailableReason(),
            "CUPTI refused the collector's activity callbacks (CUPTI_ERROR_UNKNOWN)");
  EXPECT_EQ(fake_cupti::calls().enables, 0);
  EXPECT_EQ(collector.stats().kernelLaunches, 0U);
  EXPECT_TRUE(collector.windowProblem().empty()) << "a window it did not collect has no problem";
}

/** @test A result CUPTI has no name for is quoted by its number. */
TEST_F(CuptiCollectorDecision, AnUnnamedResultIsQuotedByNumber) {
  fake_cupti::behaviour().registerResult = 42;
  const CuptiCollector COLLECTOR;
  EXPECT_EQ(COLLECTOR.unavailableReason(),
            "CUPTI refused the collector's activity callbacks (CUPTI result 42)");
}

/**
 * @test A refused enable leaves it unavailable with the reason; the window and
 *       every later one collect nothing and switch nothing on
 */
TEST_F(CuptiCollectorDecision, ARefusedEnableIsStated) {
  fake_cupti::behaviour().enableResult = CUPTI_ERROR_NOT_COMPATIBLE;
  deliverLaunches();
  CuptiCollector collector;
  window(collector);
  EXPECT_FALSE(collector.isAvailable());
  EXPECT_EQ(collector.unavailableReason(),
            "CUPTI refused to record kernel activity (CUPTI_ERROR_NOT_COMPATIBLE)");
  EXPECT_EQ(collector.stats().kernelLaunches, 0U);
  EXPECT_EQ(fake_cupti::calls().flushes, 0) << "a window that never started flushes nothing";
  window(collector);
  EXPECT_EQ(fake_cupti::calls().enables, 1) << "an unavailable collector asks CUPTI nothing";
  EXPECT_EQ(collector.stats().kernelLaunches, 0U);
}

/**
 * @test A failed flush leaves the window without stats, though it delivered
 *       records, and names the result; the collector stays available
 */
TEST_F(CuptiCollectorDecision, AFailedFlushEmptiesTheWindow) {
  deliverLaunches();
  fake_cupti::behaviour().flushResult = CUPTI_ERROR_UNKNOWN;
  CuptiCollector collector;
  window(collector);
  EXPECT_EQ(collector.stats().kernelLaunches, 0U);
  EXPECT_EQ(collector.stats().registersMedian, 0U);
  EXPECT_EQ(collector.windowProblem(),
            "CUPTI failed to flush its activity buffers (CUPTI_ERROR_UNKNOWN)");
  EXPECT_TRUE(collector.isAvailable());
}

/** @test Dropped records leave the window without stats and are counted in the problem. */
TEST_F(CuptiCollectorDecision, DroppedRecordsEmptyTheWindow) {
  deliverLaunches();
  fake_cupti::behaviour().droppedPerFlush = 3;
  CuptiCollector collector;
  window(collector);
  EXPECT_EQ(collector.stats().kernelLaunches, 0U);
  EXPECT_EQ(collector.windowProblem(), "CUPTI dropped 3 activity records");

  fake_cupti::behaviour().droppedPerFlush = 1;
  window(collector);
  EXPECT_EQ(collector.windowProblem(), "CUPTI dropped 1 activity record");
  EXPECT_TRUE(collector.isAvailable());
}

/**
 * @test Records CUPTI dropped without a buffer to hand over, which reach no
 *       completion callback, still empty the window
 */
TEST_F(CuptiCollectorDecision, RecordsDroppedWithoutABufferCount) {
  deliverLaunches();
  fake_cupti::behaviour().droppedPerFlush = 2;
  fake_cupti::behaviour().dropsReachABuffer = false;
  CuptiCollector collector;
  window(collector);
  EXPECT_EQ(collector.stats().kernelLaunches, 0U);
  EXPECT_EQ(collector.windowProblem(), "CUPTI dropped 2 activity records");
}

/** @test A dropped-record count CUPTI cannot give leaves the window without stats. */
TEST_F(CuptiCollectorDecision, AnUncountedDropEmptiesTheWindow) {
  deliverLaunches();
  fake_cupti::behaviour().droppedCountResult = CUPTI_ERROR_UNKNOWN;
  CuptiCollector collector;
  window(collector);
  EXPECT_EQ(collector.stats().kernelLaunches, 0U);
  EXPECT_EQ(collector.windowProblem(),
            "CUPTI could not count its dropped records (CUPTI_ERROR_UNKNOWN)");
}

/** @test A window with no kernel record says so. */
TEST_F(CuptiCollectorDecision, AWindowWithoutARecordSaysSo) {
  CuptiCollector collector;
  window(collector);
  EXPECT_EQ(collector.stats().kernelLaunches, 0U);
  EXPECT_EQ(collector.windowProblem(), "CUPTI recorded no kernel launch");
  EXPECT_TRUE(collector.isAvailable());
}

/**
 * @test A refused disable keeps the window's stats, which were flushed first,
 *       and ends collection: records still on would reach the next window
 */
TEST_F(CuptiCollectorDecision, ARefusedDisableEndsCollection) {
  deliverLaunches();
  fake_cupti::behaviour().disableResult = CUPTI_ERROR_UNKNOWN;
  CuptiCollector collector;
  window(collector);
  EXPECT_EQ(collector.stats().kernelLaunches, LAUNCHES);
  EXPECT_FALSE(collector.isAvailable());
  EXPECT_EQ(collector.unavailableReason(),
            "CUPTI did not stop recording kernel activity (CUPTI_ERROR_UNKNOWN)");

  window(collector);
  EXPECT_EQ(collector.stats().kernelLaunches, 0U) << "the next window must not repeat the last";
  EXPECT_EQ(fake_cupti::calls().enables, 1);
  EXPECT_EQ(fake_cupti::calls().flushes, 1);
}

/** @test A window after one with a problem is judged on its own records. */
TEST_F(CuptiCollectorDecision, AWindowAfterAProblemCountsAgain) {
  deliverLaunches();
  fake_cupti::behaviour().droppedPerFlush = 3;
  CuptiCollector collector;
  window(collector);
  ASSERT_FALSE(collector.windowProblem().empty());

  fake_cupti::behaviour().droppedPerFlush = 0;
  window(collector);
  EXPECT_TRUE(collector.windowProblem().empty()) << collector.windowProblem();
  EXPECT_EQ(collector.stats().kernelLaunches, LAUNCHES);
}

/** @test reset() clears the last window's stats and problem. */
TEST_F(CuptiCollectorDecision, ResetClearsTheWindow) {
  CuptiCollector collector;
  window(collector);
  ASSERT_FALSE(collector.windowProblem().empty());
  collector.reset();
  EXPECT_TRUE(collector.windowProblem().empty()) << collector.windowProblem();

  deliverLaunches();
  window(collector);
  ASSERT_EQ(collector.stats().kernelLaunches, LAUNCHES);
  collector.reset();
  EXPECT_EQ(collector.stats().kernelLaunches, 0U);
}

/** @test A collector that stood down says so, whichever way it was asked to. */
TEST_F(CuptiCollectorDecision, AStoodDownCollectorSaysWhy) {
  const std::string STOOD_DOWN =
      "the collector stood down (an Nsight session, VERNIER_DISABLE_CUPTI or its caller)";
  EXPECT_EQ(CuptiCollector(true).unavailableReason(), STOOD_DOWN);
  const ScopedEnv SETTING("VERNIER_DISABLE_CUPTI", "1");
  EXPECT_EQ(CuptiCollector(false).unavailableReason(), STOOD_DOWN);
}
