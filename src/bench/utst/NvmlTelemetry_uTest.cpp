/**
 * @file NvmlTelemetry_uTest.cpp
 * @brief The GPU harness's NVML readings: each one checked, a cell filled
 *        only from readings NVML reported, and the statement naming the
 *        readings it did not report and the cells that leaves empty.
 *
 * Built against the scripted stand-in in fake_nvml/ instead of NVML (first on
 * this target's include path), so it runs on any machine: it compiles the
 * real NvmlTelemetry.hpp into this test and scripts what each NVML reading
 * answers at a window's start and at its end. Each test starts from the
 * stand-in's defaults.
 */

#include "src/bench/src/NvmlTelemetry.hpp" // the real header, on the scripted stand-in

#include <gtest/gtest.h>

#include <string>

namespace fn = fake_nvml;
using vernier::bench::ClockSpeedProfile;
using vernier::bench::PowerThermalProfile;
using vernier::bench::nvml_telemetry::absenceStatement;
using vernier::bench::nvml_telemetry::Cells;
using vernier::bench::nvml_telemetry::cellsOf;
using vernier::bench::nvml_telemetry::fillProfiles;
using vernier::bench::nvml_telemetry::missingReadingsStatement;
using vernier::bench::nvml_telemetry::Session;
using vernier::bench::nvml_telemetry::throttlingWarning;
using vernier::bench::nvml_telemetry::WindowReadings;

namespace {

constexpr fn::Answer NOT_SUPPORTED{NVML_ERROR_NOT_SUPPORTED, 0};

/** @brief A result that is @p first at a window's start and @p later at its end. */
fn::Script startThenEnd(fn::Answer first, fn::Answer later) { return fn::Script{first, later, 0}; }

/**
 * @brief A device that reports every reading: SM clock 1800 then 1700 MHz
 *        (maximum 2000), power 20 then 30 W (limit 150 W), 50 then 53 C.
 */
fn::Device reportingDevice() {
  fn::Device d;
  d.smClock = startThenEnd({NVML_SUCCESS, 1800}, {NVML_SUCCESS, 1700});
  d.memClock = fn::Script::always({NVML_SUCCESS, 7001});
  d.maxSmClock = fn::Script::always({NVML_SUCCESS, 2000});
  d.powerMw = startThenEnd({NVML_SUCCESS, 20000}, {NVML_SUCCESS, 30000});
  d.powerLimitMw = fn::Script::always({NVML_SUCCESS, 150000});
  d.temperatureC = startThenEnd({NVML_SUCCESS, 50}, {NVML_SUCCESS, 53});
  return d;
}

/** @brief A device that answers "Not Supported" to every reading. */
fn::Device silentDevice() {
  fn::Device d;
  d.smClock = fn::Script::always(NOT_SUPPORTED);
  d.memClock = fn::Script::always(NOT_SUPPORTED);
  d.maxSmClock = fn::Script::always(NOT_SUPPORTED);
  d.powerMw = fn::Script::always(NOT_SUPPORTED);
  d.powerLimitMw = fn::Script::always(NOT_SUPPORTED);
  d.temperatureC = fn::Script::always(NOT_SUPPORTED);
  return d;
}

/** @brief One measured window's readings from device 0. */
WindowReadings windowOf(const fn::Device& device) {
  fn::behaviour().devices = {device};
  const Session SESSION(true, 0);
  EXPECT_TRUE(SESSION.ready()) << SESSION.unavailableReason();
  WindowReadings w;
  SESSION.readStart(w);
  SESSION.readEnd(w);
  return w;
}

/// Every NVML cell stays empty, as a statement's tail names them.
constexpr const char* ALL_CELLS = "smClockMHz, throttling, powerDrawW, powerLimitW, temperatureC "
                                  "and temperatureDeltaC stay empty.";

} // namespace

/** @brief The stand-in's defaults before and after each test. */
class NvmlTelemetry : public ::testing::Test {
protected:
  void SetUp() override { fn::reset(); }
  void TearDown() override { fn::reset(); }
};

/* ----------------------------- Cells ----------------------------- */

/** @test Every reading reported fills every cell, and nothing is stated. */
TEST_F(NvmlTelemetry, EveryReadingReportedFillsEveryCell) {
  const WindowReadings W = windowOf(reportingDevice());
  const Cells CELLS = cellsOf(W);
  EXPECT_EQ(CELLS.smClockMHz, 1700);
  EXPECT_EQ(CELLS.throttling, false) << "1800 to 1700 MHz is under the 10% drop";
  EXPECT_EQ(CELLS.powerDrawW, 25.0);
  EXPECT_EQ(CELLS.powerLimitW, 150.0);
  EXPECT_EQ(CELLS.temperatureC, 53);
  EXPECT_EQ(CELLS.temperatureDeltaC, 3);
  EXPECT_EQ(missingReadingsStatement(W), "");
  EXPECT_EQ(throttlingWarning(W), "");
}

/** @test Readings answered "Not Supported" leave every cell empty, and say which and why. */
TEST_F(NvmlTelemetry, NotSupportedReadingsAreEmptyAndStated) {
  const WindowReadings W = windowOf(silentDevice());
  const Cells CELLS = cellsOf(W);
  EXPECT_FALSE(CELLS.smClockMHz.has_value());
  EXPECT_FALSE(CELLS.throttling.has_value());
  EXPECT_FALSE(CELLS.powerDrawW.has_value());
  EXPECT_FALSE(CELLS.powerLimitW.has_value());
  EXPECT_FALSE(CELLS.temperatureC.has_value());
  EXPECT_FALSE(CELLS.temperatureDeltaC.has_value());
  EXPECT_EQ(missingReadingsStatement(W),
            std::string("[gpu] NVML reported no SM clock, maximum SM clock, power draw, power "
                        "limit or GPU temperature (Not Supported): ") +
                ALL_CELLS);
}

/** @test One reading missing empties only the cells that need it. */
TEST_F(NvmlTelemetry, OneMissingReadingEmptiesOnlyItsCell) {
  fn::Device device = reportingDevice();
  device.powerLimitMw = fn::Script::always(NOT_SUPPORTED);
  const WindowReadings W = windowOf(device);
  const Cells CELLS = cellsOf(W);
  EXPECT_FALSE(CELLS.powerLimitW.has_value());
  EXPECT_EQ(CELLS.smClockMHz, 1700);
  EXPECT_EQ(CELLS.powerDrawW, 25.0);
  EXPECT_EQ(CELLS.temperatureC, 53);
  EXPECT_EQ(missingReadingsStatement(W),
            "[gpu] NVML reported no power limit (Not Supported): powerLimitW stays empty.");
}

/**
 * @test A failed end sample empties every cell that needs it, keeps the one
 *       read at the start, and gives no throttling warning from a clock never
 *       read
 */
TEST_F(NvmlTelemetry, AFailedEndSampleEmptiesTheCellsThatNeedIt) {
  fn::Device device = reportingDevice();
  const fn::Answer FAILED{NVML_ERROR_UNKNOWN, 0};
  device.smClock = startThenEnd({NVML_SUCCESS, 1800}, FAILED);
  device.powerMw = startThenEnd({NVML_SUCCESS, 20000}, FAILED);
  device.temperatureC = startThenEnd({NVML_SUCCESS, 50}, FAILED);
  const WindowReadings W = windowOf(device);
  const Cells CELLS = cellsOf(W);
  EXPECT_FALSE(CELLS.smClockMHz.has_value());
  EXPECT_FALSE(CELLS.throttling.has_value());
  EXPECT_FALSE(CELLS.powerDrawW.has_value()) << "a mean needs both ends";
  EXPECT_FALSE(CELLS.temperatureC.has_value());
  EXPECT_FALSE(CELLS.temperatureDeltaC.has_value());
  EXPECT_EQ(CELLS.powerLimitW, 150.0);
  EXPECT_EQ(missingReadingsStatement(W),
            "[gpu] NVML reported no SM clock, power draw or GPU temperature (Unknown Error): "
            "smClockMHz, throttling, powerDrawW, temperatureC and temperatureDeltaC stay empty.");
  EXPECT_EQ(throttlingWarning(W), "") << "an end clock never read is not a drop to 0 MHz";
}

/**
 * @test A failed start sample empties the cells that need both ends and keeps
 *       the ones read at the end
 */
TEST_F(NvmlTelemetry, AFailedStartSampleEmptiesTheCellsThatNeedBothEnds) {
  fn::Device device = reportingDevice();
  const fn::Answer FAILED{NVML_ERROR_UNKNOWN, 0};
  device.smClock = startThenEnd(FAILED, {NVML_SUCCESS, 1700});
  device.powerMw = startThenEnd(FAILED, {NVML_SUCCESS, 30000});
  device.temperatureC = startThenEnd(FAILED, {NVML_SUCCESS, 53});
  const WindowReadings W = windowOf(device);
  const Cells CELLS = cellsOf(W);
  EXPECT_EQ(CELLS.smClockMHz, 1700);
  EXPECT_FALSE(CELLS.throttling.has_value());
  EXPECT_FALSE(CELLS.powerDrawW.has_value()) << "a mean needs both ends";
  EXPECT_EQ(CELLS.temperatureC, 53);
  EXPECT_FALSE(CELLS.temperatureDeltaC.has_value()) << "a difference needs both ends";
  EXPECT_EQ(CELLS.powerLimitW, 150.0);
  EXPECT_EQ(missingReadingsStatement(W),
            "[gpu] NVML reported no SM clock, power draw or GPU temperature (Unknown Error): "
            "throttling, powerDrawW and temperatureDeltaC stay empty.");
}

/** @test Throttling needs the clock at both ends and the maximum, and warns only then. */
TEST_F(NvmlTelemetry, ThrottlingNeedsBothClocksAndTheMaximum) {
  fn::Device device = reportingDevice();
  device.smClock = startThenEnd({NVML_SUCCESS, 1800}, {NVML_SUCCESS, 1500});
  WindowReadings w = windowOf(device);
  EXPECT_EQ(cellsOf(w).throttling, true);
  EXPECT_EQ(throttlingWarning(w), "Warning: GPU throttling detected (1800 -> 1500 MHz)");

  fn::reset();
  device = reportingDevice();
  device.smClock = startThenEnd({NVML_SUCCESS, 1800}, {NVML_SUCCESS, 1500});
  device.maxSmClock = fn::Script::always(NOT_SUPPORTED);
  w = windowOf(device);
  EXPECT_FALSE(cellsOf(w).throttling.has_value());
  EXPECT_EQ(cellsOf(w).smClockMHz, 1500);
  EXPECT_EQ(throttlingWarning(w), "");
  EXPECT_EQ(missingReadingsStatement(w),
            "[gpu] NVML reported no maximum SM clock (Not Supported): throttling stays empty.");
}

/** @test Readings that fail differently are each named with their own text. */
TEST_F(NvmlTelemetry, DifferentFailuresAreNamedEach) {
  fn::Device device = reportingDevice();
  device.smClock = fn::Script::always(NOT_SUPPORTED);
  device.temperatureC = fn::Script::always({NVML_ERROR_NO_PERMISSION, 0});
  const WindowReadings W = windowOf(device);
  EXPECT_EQ(missingReadingsStatement(W),
            "[gpu] NVML reported no SM clock (Not Supported) or GPU temperature (Insufficient "
            "Permissions): smClockMHz, throttling, temperatureC and temperatureDeltaC stay "
            "empty.");
}

/** @test The public profiles carry each reading, and 0 where NVML reported none. */
TEST_F(NvmlTelemetry, ProfilesKeepZeroWhereNotReported) {
  fn::Device device = reportingDevice();
  device.smClock = startThenEnd({NVML_SUCCESS, 1800}, {NVML_ERROR_UNKNOWN, 0});
  device.powerLimitMw = fn::Script::always(NOT_SUPPORTED);
  const WindowReadings W = windowOf(device);
  ClockSpeedProfile clocks{};
  PowerThermalProfile powerThermal{};
  fillProfiles(W, clocks, powerThermal);
  EXPECT_EQ(clocks.smClockMHzStart, 1800);
  EXPECT_EQ(clocks.smClockMHzEnd, 0);
  EXPECT_EQ(clocks.memClockMHzStart, 7001);
  EXPECT_EQ(clocks.memClockMHzEnd, 7001);
  EXPECT_EQ(clocks.boostClockMHz, 2000);
  EXPECT_EQ(powerThermal.powerDrawWStart, 20.0);
  EXPECT_EQ(powerThermal.powerDrawWEnd, 30.0);
  EXPECT_EQ(powerThermal.powerLimitW, 0.0);
  EXPECT_EQ(powerThermal.temperatureCStart, 50);
  EXPECT_EQ(powerThermal.temperatureCEnd, 53);
}

/* ----------------------------- Session ----------------------------- */

/** @test A window reads the maximum clock and the power limit once, the rest at both ends. */
TEST_F(NvmlTelemetry, AWindowReadsTheLimitsOnce) {
  static_cast<void>(windowOf(reportingDevice()));
  const fn::Device& device = fn::behaviour().devices.front();
  EXPECT_EQ(device.smClock.calls, 2);
  EXPECT_EQ(device.powerMw.calls, 2);
  EXPECT_EQ(device.temperatureC.calls, 2);
  EXPECT_EQ(device.maxSmClock.calls, 1);
  EXPECT_EQ(device.powerLimitMw.calls, 1);
}

/** @test NVML that does not initialize is the reason, in NVML's words, and is not shut down. */
TEST_F(NvmlTelemetry, InitFailureIsTheReason) {
  fn::behaviour().initResult = NVML_ERROR_DRIVER_NOT_LOADED;
  fn::behaviour().devices = {reportingDevice()};
  {
    const Session SESSION(true, 0);
    EXPECT_FALSE(SESSION.ready());
    EXPECT_EQ(SESSION.unavailableReason(), "NVML did not initialize (Driver Not Loaded)");
    WindowReadings w;
    SESSION.readStart(w);
    SESSION.readEnd(w);
    EXPECT_EQ(fn::behaviour().devices.front().smClock.calls, 0) << "nothing is read";
    EXPECT_EQ(absenceStatement(SESSION.unavailableReason()),
              std::string("[gpu] NVML did not initialize (Driver Not Loaded): ") + ALL_CELLS);
  }
  EXPECT_EQ(fn::calls().shutdowns, 0);
}

/** @test No device at the index is the reason; NVML, which did initialize, is shut down. */
TEST_F(NvmlTelemetry, NoDeviceAtTheIndexIsTheReason) {
  fn::behaviour().devices = {reportingDevice()};
  {
    const Session SESSION(true, 1);
    EXPECT_FALSE(SESSION.ready());
    EXPECT_EQ(SESSION.unavailableReason(), "NVML found no device at index 1 (Invalid Argument)");
  }
  EXPECT_EQ(fn::calls().inits, 1);
  EXPECT_EQ(fn::calls().shutdowns, 1);
}

/** @test A ready session shuts NVML down once. */
TEST_F(NvmlTelemetry, AReadySessionShutsNvmlDownOnce) {
  fn::behaviour().devices = {reportingDevice()};
  {
    const Session SESSION(true, 0);
    EXPECT_TRUE(SESSION.ready());
    EXPECT_TRUE(SESSION.unavailableReason().empty()) << SESSION.unavailableReason();
  }
  EXPECT_EQ(fn::calls().inits, 1);
  EXPECT_EQ(fn::calls().shutdowns, 1);
}

/** @test With capture off NVML is not opened, and that is the reason. */
TEST_F(NvmlTelemetry, CaptureOffOpensNothing) {
  {
    const Session SESSION(false, 0);
    EXPECT_FALSE(SESSION.ready());
    EXPECT_EQ(SESSION.unavailableReason(),
              "clock and power capture is off (PerfGpuConfig::captureClockSpeeds)");
  }
  EXPECT_EQ(fn::calls().inits, 0);
  EXPECT_EQ(fn::calls().shutdowns, 0);
}
