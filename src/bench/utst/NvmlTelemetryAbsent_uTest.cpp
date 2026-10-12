/**
 * @file NvmlTelemetryAbsent_uTest.cpp
 * @brief The GPU harness's NVML readings in a build without NVML: the header
 *        compiles without any nvml.h, samples nothing, and says the build has
 *        no NVML.
 *
 * Compiles the real NvmlTelemetry.hpp into this test with
 * COMPAT_NVML_AVAILABLE=0 (this target's definition), which is what a GPU
 * build without NVML, or configured with -DVERNIER_USE_NVML=OFF, compiles.
 * Needs no CUDA, so every configuration runs it.
 */

#include "src/bench/src/NvmlTelemetry.hpp" // the real header, built without NVML

#include <gtest/gtest.h>

#include <string>

using vernier::bench::nvml_telemetry::absenceStatement;
using vernier::bench::nvml_telemetry::Cells;
using vernier::bench::nvml_telemetry::cellsOf;
using vernier::bench::nvml_telemetry::Session;
using vernier::bench::nvml_telemetry::WindowReadings;

namespace {

/// A device UUID as NVML spells it; this build never looks it up.
constexpr const char* ANY_UUID = "GPU-11111111-2222-3333-4444-555555555555";

} // namespace

/** @test The session is never ready and says the build has no NVML */
TEST(NvmlTelemetryAbsent, SaysTheBuildHasNoNvml) {
  const Session SESSION(true, ANY_UUID);
  EXPECT_FALSE(SESSION.ready());
  EXPECT_EQ(SESSION.unavailableReason(), "this build has no NVML");
  EXPECT_EQ(absenceStatement(SESSION.unavailableReason()),
            "[gpu] this build has no NVML: smClockMHz, throttling, powerDrawW, powerLimitW, "
            "temperatureC and temperatureDeltaC stay empty.");
}

/** @test A window reads nothing, so every cell is empty */
TEST(NvmlTelemetryAbsent, FillsNoCell) {
  const Session SESSION(true, ANY_UUID);
  WindowReadings w;
  SESSION.readStart(w);
  SESSION.readEnd(w);
  const Cells CELLS = cellsOf(w);
  EXPECT_FALSE(CELLS.smClockMHz.has_value());
  EXPECT_FALSE(CELLS.throttling.has_value());
  EXPECT_FALSE(CELLS.powerDrawW.has_value());
  EXPECT_FALSE(CELLS.powerLimitW.has_value());
  EXPECT_FALSE(CELLS.temperatureC.has_value());
  EXPECT_FALSE(CELLS.temperatureDeltaC.has_value());
}

/** @test With capture off, that is the reason, in this build too */
TEST(NvmlTelemetryAbsent, CaptureOffIsStillTheReason) {
  const Session SESSION(false, ANY_UUID);
  EXPECT_EQ(SESSION.unavailableReason(),
            "clock and power capture is off (PerfGpuConfig::captureClockSpeeds)");
}
