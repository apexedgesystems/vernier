/**
 * @file CuptiCollectorDecision_uTest.cpp
 * @brief The CUPTI collector applies the shared decision before it registers:
 *        a setting that stands it down, or that is rejected, registers nothing.
 *        And it checks what CUPTI answers: a refusal leaves it unavailable
 *        with the reason, and a window whose records are not known to be
 *        complete reports no launch and names the problem. Asked for the
 *        unified-memory counters, a window configures and reads what
 *        UnifiedMemoryProfile documents, and nothing without the request.
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
  EXPECT_EQ(fake_cupti::calls().configures, 0) << "unified-memory counters configured unasked";
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

/* ----------------------------- Record Walk ----------------------------- */

namespace {

/// What a window reports when a walk of its records ends in CUPTI_ERROR_INVALID_KIND.
constexpr const char* INVALID_KIND_PROBLEM =
    "CUPTI could not read all of its activity records (CUPTI_ERROR_INVALID_KIND)";

/** @brief A buffer's walk hands over @p readable records, then answers CUPTI_ERROR_INVALID_KIND. */
void failWalkAfter(std::size_t readable) {
  fake_cupti::behaviour().readableRecords = readable;
  fake_cupti::behaviour().nextRecordResult = CUPTI_ERROR_INVALID_KIND;
}

} // namespace

/**
 * @test Control: a walk CUPTI ends at the buffer's end
 *       (CUPTI_ERROR_MAX_LIMIT_REACHED) has read every record, and the window
 *       counts them all
 */
TEST_F(CuptiCollectorDecision, AWalkToTheBuffersEndCountsEveryRecord) {
  deliverLaunches();
  CuptiCollector collector;
  window(collector);
  EXPECT_EQ(fake_cupti::calls().walkEnds, std::vector<CUptiResult>{CUPTI_ERROR_MAX_LIMIT_REACHED});
  EXPECT_TRUE(collector.windowProblem().empty()) << collector.windowProblem();
  EXPECT_EQ(collector.stats().kernelLaunches, LAUNCHES);
  EXPECT_EQ(collector.stats().registersMedian, REGISTERS);
}

/**
 * @test A walk that fails after a valid record leaves the window without stats,
 *       so nothing of the record read before the failure is published (the
 *       harness leaves the cupti* cells of a window without launches empty),
 *       and names CUPTI's answer; the collector stays available
 */
TEST_F(CuptiCollectorDecision, AWalkThatFailsAfterARecordEmptiesTheWindow) {
  deliverLaunches();
  failWalkAfter(1);
  CuptiCollector collector;
  window(collector);
  ASSERT_EQ(fake_cupti::calls().walkEnds, std::vector<CUptiResult>{CUPTI_ERROR_INVALID_KIND});
  EXPECT_EQ(collector.windowProblem(), INVALID_KIND_PROBLEM);
  EXPECT_EQ(collector.stats().kernelLaunches, 0U);
  EXPECT_EQ(collector.stats().registersMedian, 0U);
  EXPECT_EQ(collector.stats().registersMax, 0U);
  EXPECT_EQ(collector.stats().staticSmemBytes, 0U);
  EXPECT_EQ(collector.stats().dynamicSmemBytes, 0U);
  EXPECT_TRUE(collector.stats().firstKernelName.empty()) << collector.stats().firstKernelName;
  EXPECT_TRUE(collector.isAvailable()) << "a window that failed is not a collector that cannot";
  EXPECT_TRUE(collector.unavailableReason().empty()) << collector.unavailableReason();
}

/** @test A walk that fails at its first record says so, not that the window had no launch. */
TEST_F(CuptiCollectorDecision, AWalkThatFailsAtTheFirstRecordSaysSo) {
  deliverLaunches();
  failWalkAfter(0);
  CuptiCollector collector;
  window(collector);
  EXPECT_EQ(collector.windowProblem(), INVALID_KIND_PROBLEM);
  EXPECT_EQ(collector.stats().kernelLaunches, 0U);
}

/**
 * @test Control: a window after one whose walk failed is judged on its own
 *       records; the failure ended that window, not collection
 */
TEST_F(CuptiCollectorDecision, AWindowAfterAFailedWalkCountsAgain) {
  deliverLaunches();
  failWalkAfter(1);
  CuptiCollector collector;
  window(collector);
  ASSERT_EQ(fake_cupti::calls().walkEnds, std::vector<CUptiResult>{CUPTI_ERROR_INVALID_KIND});

  fake_cupti::behaviour().nextRecordResult = CUPTI_SUCCESS;
  window(collector);
  EXPECT_TRUE(collector.windowProblem().empty()) << collector.windowProblem();
  EXPECT_EQ(collector.stats().kernelLaunches, LAUNCHES);
  EXPECT_TRUE(collector.isAvailable());
  EXPECT_EQ(fake_cupti::calls().enables, 2) << "the second window switches kernel records on again";
}

/* ----------------------------- Unified-Memory Counters ----------------------------- */

namespace {

using vernier::bench::UnifiedMemoryProfile;

/// The CPU fault records' value: the faulting instruction's address, never a count.
constexpr std::uint64_t FAULTING_PC = 0x7f00deadbeefULL;

/**
 * @brief Records a flush delivers after the kernel records: two host-to-device
 *        transfers (4096 and 61440 bytes, 2592 and 7424 ns), one device-to-host
 *        (8192 bytes, 1280 ns), three GPU fault records of 12, 4 and 1 fault
 *        groups, five CPU faults and two thrashing records.
 */
void deliverUnifiedMemory() {
  using fake_cupti::unifiedMemoryRecord;
  fake_cupti::behaviour().unifiedMemoryRecords = {
      unifiedMemoryRecord(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD, 4096,
                          1000, 3592),
      unifiedMemoryRecord(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD, 61440,
                          3592, 11016),
      unifiedMemoryRecord(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH, 8192,
                          20000, 21280),
      unifiedMemoryRecord(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT, 12, 500, 900),
      unifiedMemoryRecord(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT, 4, 950, 990),
      unifiedMemoryRecord(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT, 1, 995, 999),
      unifiedMemoryRecord(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT,
                          FAULTING_PC, 19000),
      unifiedMemoryRecord(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT,
                          FAULTING_PC, 19100),
      unifiedMemoryRecord(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT,
                          FAULTING_PC, 19200),
      unifiedMemoryRecord(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT,
                          FAULTING_PC, 19300),
      unifiedMemoryRecord(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT,
                          FAULTING_PC, 19400),
      unifiedMemoryRecord(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING, 65536, 30000),
      unifiedMemoryRecord(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING, 65536, 31000),
  };
}

/** @brief A collector asked for the unified-memory counters. */
struct AskingCollector {
  CuptiCollector collector;
  AskingCollector() { collector.collectUnifiedMemory(true); }
};

} // namespace

/**
 * @test Asked for the counters, a window configures the five kinds it reads
 *       (each enabled), switches their kind on after the kernel records, and
 *       at stop flushes before it switches either kind off: switched off
 *       first, the unified-memory kind loses the window's device-side records
 */
TEST_F(CuptiCollectorDecision, ARequestConfiguresTheFiveCountersThenFlushesFirst) {
  AskingCollector asking;
  window(asking.collector);
  ASSERT_EQ(fake_cupti::calls().configures, 1);
  const std::vector<CUpti_ActivityUnifiedMemoryCounterConfig>& CONFIG =
      fake_cupti::calls().configuration;
  ASSERT_EQ(CONFIG.size(), 5U);
  const CUpti_ActivityUnifiedMemoryCounterKind KINDS[] = {
      CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD,
      CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH,
      CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT,
      CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT,
      CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING};
  for (std::size_t i = 0; i < CONFIG.size(); ++i) {
    EXPECT_EQ(CONFIG[i].kind, KINDS[i]) << "configuration entry " << i;
    EXPECT_EQ(CONFIG[i].enable, 1U) << "configuration entry " << i;
    EXPECT_EQ(CONFIG[i].scope, CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE);
  }
  EXPECT_EQ(fake_cupti::calls().log,
            (std::vector<std::string>{"enable 3", "configure", "enable 25", "flush", "disable 3",
                                      "disable 25"}));
  EXPECT_TRUE(asking.collector.unifiedMemoryRefusal().empty())
      << asking.collector.unifiedMemoryRefusal();
}

/**
 * @test Each record adds to its field: the GPU fault groups and the CPU faults
 *       (one each, whatever their value) to pageFaults; each transfer to its
 *       direction's count and bytes, and its end minus start to the time; each
 *       thrashing record to thrashingEvents. The kernel records beside them
 *       keep their stats
 */
TEST_F(CuptiCollectorDecision, EachUnifiedMemoryRecordAddsToItsField) {
  deliverLaunches();
  deliverUnifiedMemory();
  AskingCollector asking;
  window(asking.collector);
  ASSERT_TRUE(asking.collector.windowProblem().empty()) << asking.collector.windowProblem();
  ASSERT_TRUE(asking.collector.unifiedMemory().has_value());
  const UnifiedMemoryProfile& UM = *asking.collector.unifiedMemory();
  EXPECT_EQ(UM.pageFaults, 17U + 5U);
  EXPECT_EQ(UM.h2dMigrations, 2U);
  EXPECT_EQ(UM.h2dBytes, 65536U);
  EXPECT_EQ(UM.d2hMigrations, 1U);
  EXPECT_EQ(UM.d2hBytes, 8192U);
  EXPECT_DOUBLE_EQ(UM.migrationTimeUs, (2592.0 + 7424.0 + 1280.0) / 1000.0);
  EXPECT_EQ(UM.thrashingEvents, 2U);
  EXPECT_EQ(asking.collector.stats().kernelLaunches, LAUNCHES);
  EXPECT_EQ(asking.collector.stats().registersMedian, REGISTERS);
}

/**
 * @test Records of kinds the counters do not read (configured elsewhere in the
 *       process: CUPTI keeps a kind's configuration) add nothing
 */
TEST_F(CuptiCollectorDecision, RecordsOfOtherKindsAddNothing) {
  deliverLaunches();
  using fake_cupti::unifiedMemoryRecord;
  fake_cupti::behaviour().unifiedMemoryRecords = {
      unifiedMemoryRecord(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING, 0, 100, 700),
      unifiedMemoryRecord(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP, 4096, 800),
      unifiedMemoryRecord(CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD, 4096, 900,
                          950)};
  AskingCollector asking;
  window(asking.collector);
  ASSERT_TRUE(asking.collector.unifiedMemory().has_value());
  const UnifiedMemoryProfile& UM = *asking.collector.unifiedMemory();
  EXPECT_EQ(UM.pageFaults + UM.h2dMigrations + UM.d2hMigrations + UM.thrashingEvents, 0U);
  EXPECT_EQ(UM.h2dBytes + UM.d2hBytes, 0U);
  EXPECT_DOUBLE_EQ(UM.migrationTimeUs, 0.0);
}

/** @test A window that records the counters and gets no record observes zeros. */
TEST_F(CuptiCollectorDecision, AWindowWithoutUnifiedMemoryRecordsObservesZeros) {
  deliverLaunches();
  AskingCollector asking;
  window(asking.collector);
  ASSERT_TRUE(asking.collector.unifiedMemory().has_value());
  EXPECT_EQ(asking.collector.unifiedMemory()->pageFaults, 0U);
  EXPECT_EQ(asking.collector.unifiedMemory()->h2dMigrations, 0U);
  EXPECT_EQ(asking.collector.stats().kernelLaunches, LAUNCHES);
}

/**
 * @test A window without a kernel launch says so and keeps its counters: they
 *       need complete records, not a launch
 */
TEST_F(CuptiCollectorDecision, AWindowWithoutALaunchKeepsItsCounters) {
  deliverUnifiedMemory();
  AskingCollector asking;
  window(asking.collector);
  EXPECT_EQ(asking.collector.windowProblem(), "CUPTI recorded no kernel launch");
  EXPECT_EQ(asking.collector.stats().kernelLaunches, 0U);
  ASSERT_TRUE(asking.collector.unifiedMemory().has_value());
  EXPECT_EQ(asking.collector.unifiedMemory()->h2dMigrations, 2U);
}

/**
 * @test A refused configuration leaves the window without counters and says so
 *       in CUPTI's words; the kind is never switched on, the kernel records are
 *       collected, and the next window asks again
 */
TEST_F(CuptiCollectorDecision, ARefusedConfigurationIsStated) {
  deliverLaunches();
  deliverUnifiedMemory();
  fake_cupti::behaviour().configureResult = CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED;
  AskingCollector asking;
  window(asking.collector);
  EXPECT_FALSE(asking.collector.unifiedMemory().has_value());
  EXPECT_EQ(asking.collector.unifiedMemoryRefusal(),
            "CUPTI refused to configure the unified-memory counters "
            "(CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED)");
  EXPECT_EQ(fake_cupti::calls().enabled,
            std::vector<CUpti_ActivityKind>{CUPTI_ACTIVITY_KIND_KERNEL});
  EXPECT_EQ(fake_cupti::calls().disabled,
            std::vector<CUpti_ActivityKind>{CUPTI_ACTIVITY_KIND_KERNEL});
  EXPECT_EQ(asking.collector.stats().kernelLaunches, LAUNCHES);
  EXPECT_TRUE(asking.collector.isAvailable());
  EXPECT_TRUE(asking.collector.windowProblem().empty()) << asking.collector.windowProblem();
  window(asking.collector);
  EXPECT_EQ(fake_cupti::calls().configures, 2) << "each window asks again";
  EXPECT_FALSE(asking.collector.unifiedMemoryRefusal().empty());
}

/**
 * @test A refused kind leaves the window without counters, though records
 *       reach the buffer, and says so in CUPTI's words; it is not switched off
 *       at stop, and the kernel records are collected
 */
TEST_F(CuptiCollectorDecision, ARefusedKindIsStated) {
  deliverLaunches();
  deliverUnifiedMemory();
  fake_cupti::behaviour().unifiedMemoryEnableResult = CUPTI_ERROR_NOT_READY;
  AskingCollector asking;
  window(asking.collector);
  EXPECT_FALSE(asking.collector.unifiedMemory().has_value());
  EXPECT_EQ(asking.collector.unifiedMemoryRefusal(),
            "CUPTI refused to record unified-memory counter activity (CUPTI_ERROR_NOT_READY)");
  EXPECT_EQ(fake_cupti::calls().disabled,
            std::vector<CUpti_ActivityKind>{CUPTI_ACTIVITY_KIND_KERNEL});
  EXPECT_EQ(asking.collector.stats().kernelLaunches, LAUNCHES);
  EXPECT_TRUE(asking.collector.isAvailable());
}

/**
 * @test A kind CUPTI does not switch off keeps that window's counters, which
 *       were flushed first; no later window switches it on or reads its records,
 *       each says why, and kernel windows go on
 */
TEST_F(CuptiCollectorDecision, AKindThatStaysOnEndsTheCounters) {
  deliverLaunches();
  deliverUnifiedMemory();
  fake_cupti::behaviour().unifiedMemoryDisableResult = CUPTI_ERROR_UNKNOWN;
  AskingCollector asking;
  window(asking.collector);
  ASSERT_TRUE(asking.collector.unifiedMemory().has_value());
  EXPECT_EQ(asking.collector.unifiedMemory()->h2dMigrations, 2U);

  window(asking.collector);
  EXPECT_EQ(fake_cupti::calls().configures, 1) << "a later window configured the counters again";
  EXPECT_FALSE(asking.collector.unifiedMemory().has_value())
      << "a later window read the records of a kind it did not switch on";
  EXPECT_EQ(asking.collector.unifiedMemoryRefusal(),
            "CUPTI did not stop recording unified-memory counter activity (CUPTI_ERROR_UNKNOWN)");
  EXPECT_EQ(asking.collector.stats().kernelLaunches, LAUNCHES);
  EXPECT_TRUE(asking.collector.isAvailable());
}

/**
 * @test Each problem that leaves a window's records incomplete leaves it
 *       without counters, as it leaves it without kernel stats
 */
TEST_F(CuptiCollectorDecision, IncompleteRecordsLeaveNoCounters) {
  struct Case {
    const char* name;
    void (*set)();
  };
  const Case CASES[] = {
      {"a failed flush", [] { fake_cupti::behaviour().flushResult = CUPTI_ERROR_UNKNOWN; }},
      {"dropped records", [] { fake_cupti::behaviour().droppedPerFlush = 2; }},
      {"an uncounted drop",
       [] { fake_cupti::behaviour().droppedCountResult = CUPTI_ERROR_UNKNOWN; }},
      {"a walk ended early", [] { failWalkAfter(1); }},
  };
  for (const Case& c : CASES) {
    fake_cupti::reset();
    deliverLaunches();
    deliverUnifiedMemory();
    c.set();
    AskingCollector asking;
    window(asking.collector);
    EXPECT_FALSE(asking.collector.windowProblem().empty()) << c.name;
    EXPECT_FALSE(asking.collector.unifiedMemory().has_value()) << c.name;
    EXPECT_EQ(asking.collector.stats().kernelLaunches, 0U) << c.name;
  }
}

/**
 * @test A transfer that ends before it starts leaves the window without
 *       counters and kernel stats, and says so: its time would wrap
 */
TEST_F(CuptiCollectorDecision, ATransferEndingBeforeItStartsLeavesNoCounters) {
  deliverLaunches();
  fake_cupti::behaviour().unifiedMemoryRecords = {fake_cupti::unifiedMemoryRecord(
      CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD, 4096, 5000, 4000)};
  AskingCollector asking;
  window(asking.collector);
  EXPECT_EQ(asking.collector.windowProblem(),
            "CUPTI recorded 1 unified-memory transfer that ends before it starts");
  EXPECT_FALSE(asking.collector.unifiedMemory().has_value());
  EXPECT_EQ(asking.collector.stats().kernelLaunches, 0U);
}

/** @test A window that was not asked for the counters reads none of their records. */
TEST_F(CuptiCollectorDecision, AWindowNotAskedReadsNoUnifiedMemoryRecord) {
  deliverLaunches();
  deliverUnifiedMemory();
  CuptiCollector collector;
  window(collector);
  EXPECT_FALSE(collector.unifiedMemory().has_value());
  EXPECT_TRUE(collector.unifiedMemoryRefusal().empty()) << collector.unifiedMemoryRefusal();
  EXPECT_EQ(fake_cupti::calls().configures, 0);
  EXPECT_EQ(collector.stats().kernelLaunches, LAUNCHES);
}

/** @test The request can be withdrawn: the next window configures and switches on nothing. */
TEST_F(CuptiCollectorDecision, AWithdrawnRequestRecordsNoCounters) {
  deliverLaunches();
  deliverUnifiedMemory();
  AskingCollector asking;
  window(asking.collector);
  ASSERT_TRUE(asking.collector.unifiedMemory().has_value());
  asking.collector.collectUnifiedMemory(false);
  fake_cupti::calls() = {};
  window(asking.collector);
  EXPECT_FALSE(asking.collector.unifiedMemory().has_value());
  EXPECT_EQ(fake_cupti::calls().configures, 0);
  EXPECT_EQ(fake_cupti::calls().enabled,
            std::vector<CUpti_ActivityKind>{CUPTI_ACTIVITY_KIND_KERNEL});
}

/** @test reset() clears the last window's counters. */
TEST_F(CuptiCollectorDecision, ResetClearsTheCounters) {
  deliverUnifiedMemory();
  AskingCollector asking;
  window(asking.collector);
  ASSERT_TRUE(asking.collector.unifiedMemory().has_value());
  asking.collector.reset();
  EXPECT_FALSE(asking.collector.unifiedMemory().has_value());
}
