/**
 * @file CuptiCollectorAbsent_uTest.cpp
 * @brief The CUPTI collector in a build without CUPTI: it compiles, never
 *        collects, and takes the same stand-down decision as with CUPTI.
 *
 * Compiles the real CuptiCollector.cu into this test with
 * COMPAT_CUPTI_AVAILABLE=0 (this target's definition), which is what a GPU
 * build configured with -DVERNIER_USE_CUPTI=OFF, or against a toolkit without
 * CUPTI, compiles. Needs no CUDA, so every configuration runs it. Each test
 * sets or clears the variables it depends on and restores them.
 */

#include "src/bench/src/CuptiCollector.cu" // the real source, built without CUPTI

#include "src/bench/utst/ScopedEnv.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <stdexcept>
#include <string>

using vernier::bench::CuptiCollector;
using vernier::bench::ReadinessCause;
using vernier::bench::ReadinessResult;
using vernier::bench::readinessResult;
using vernier::bench::profiler_env::CuptiDecision;
using vernier::bench::profiler_env::cuptiDecision;
using vernier::bench::test::ScopedEnv;

/** @brief No session and no override for each test. */
class CuptiCollectorAbsent : public ::testing::Test {
protected:
  void SetUp() override {
    for (const char* name : {"VERNIER_DISABLE_CUPTI", "VERNIER_EXTERNAL_WRAP",
                             "NSYS_PROFILING_SESSION_ID", "NV_NSIGHT_INJECTION_PORT_BASE"}) {
      scrub_[count_++].emplace(name, nullptr);
    }
  }

private:
  std::optional<ScopedEnv> scrub_[4];
  int count_ = 0;
};

/* ----------------------------- API Tests ----------------------------- */

/** @test Without CUPTI the collector is never available, whatever it is asked */
TEST_F(CuptiCollectorAbsent, IsNeverAvailable) {
  EXPECT_FALSE(CuptiCollector().isAvailable());
  EXPECT_FALSE(CuptiCollector(true).isAvailable());
  const ScopedEnv SETTING("VERNIER_DISABLE_CUPTI", "0");
  EXPECT_FALSE(CuptiCollector(false).isAvailable()) << "a false override enables nothing here";
}

/** @test It says why it does not collect; a collector that stood down says that first */
TEST_F(CuptiCollectorAbsent, SaysWhy) {
  EXPECT_EQ(CuptiCollector().unavailableReason(), "this build has no CUPTI");
  EXPECT_EQ(CuptiCollector(true).unavailableReason(),
            "the collector stood down (an Nsight session, VERNIER_DISABLE_CUPTI or its caller)");
}

/** @test start, stop and reset collect nothing: the stats stay empty, with no window problem */
TEST_F(CuptiCollectorAbsent, CollectsNothing) {
  CuptiCollector collector;
  collector.start();
  collector.stop();
  EXPECT_EQ(collector.stats().kernelLaunches, 0U);
  EXPECT_EQ(collector.stats().registersMedian, 0U);
  EXPECT_EQ(collector.stats().registersMax, 0U);
  EXPECT_EQ(collector.stats().staticSmemBytes, 0U);
  EXPECT_EQ(collector.stats().dynamicSmemBytes, 0U);
  EXPECT_TRUE(collector.stats().firstKernelName.empty());
  EXPECT_TRUE(collector.windowProblem().empty()) << collector.windowProblem();
  collector.reset();
  EXPECT_EQ(collector.stats().kernelLaunches, 0U);
}

/**
 * @test An invalid VERNIER_DISABLE_CUPTI is the same configuration error as in
 *       a build with CUPTI, in the words of its CONFIGURATION readiness report
 */
TEST_F(CuptiCollectorAbsent, AnInvalidSettingIsStillAConfigurationError) {
  const ScopedEnv SETTING("VERNIER_DISABLE_CUPTI", "maybe");
  try {
    const CuptiCollector COLLECTOR;
    FAIL() << "an invalid value was accepted";
  } catch (const std::invalid_argument& e) {
    const CuptiDecision DECISION = cuptiDecision();
    const ReadinessResult REPORT =
        readinessResult(ReadinessCause::CONFIGURATION, DECISION.error, DECISION.remedy);
    EXPECT_EQ(std::string(e.what()), REPORT.report.message + ". " + REPORT.report.hint);
  }
}

/** @test An explicit forceDisabled wins without reading the setting */
TEST_F(CuptiCollectorAbsent, ForceDisabledSkipsTheSetting) {
  const ScopedEnv SETTING("VERNIER_DISABLE_CUPTI", "maybe");
  EXPECT_NO_THROW(CuptiCollector{true});
}
