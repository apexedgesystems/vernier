/**
 * @file PerfRegistry_uTest.cpp
 * @brief Unit tests for vernier::bench::PerfRegistry.
 *
 * Tests singleton access, set/take operations, and profile metadata updates.
 */

#include "src/bench/inc/PerfRegistry.hpp"
#include "src/bench/inc/PerfConfig.hpp"

#include <gtest/gtest.h>

#include <thread>
#include <vector>

using vernier::bench::globalPerfConfig;
using vernier::bench::PerfConfig;
using vernier::bench::PerfRegistry;
using vernier::bench::PerfRow;
using vernier::bench::setGlobalPerfConfig;
using vernier::bench::Stats;

namespace {

PerfRow makeTestRow(const std::string& name) {
  PerfRow row;
  row.testName = name;
  row.cycles = 1000;
  row.repeats = 10;
  row.warmup = 1;
  row.threads = 1;
  row.msgBytes = 64;
  row.console = false;
  row.nonBlocking = false;
  row.minLevel = "INFO";
  row.stats = Stats{.median = 100.0,
                    .p10 = 90.0,
                    .p90 = 110.0,
                    .min = 80.0,
                    .max = 120.0,
                    .mean = 100.0,
                    .stddev = 10.0,
                    .cv = 0.1};
  row.callsPerSecond = 10000.0;
  return row;
}

} // namespace

/* ----------------------------- Singleton Tests ----------------------------- */

/** @test Registry is a singleton. */
TEST(PerfRegistryTest, IsSingleton) {
  PerfRegistry& r1 = PerfRegistry::instance();
  PerfRegistry& r2 = PerfRegistry::instance();

  EXPECT_EQ(&r1, &r2);
}

/* ----------------------------- Set/Take Tests ----------------------------- */

/** @test Set and take returns the row. */
TEST(PerfRegistryTest, SetAndTakeReturnsRow) {
  PerfRegistry& registry = PerfRegistry::instance();

  // Clear any existing state
  registry.take();

  PerfRow row = makeTestRow("SetTakeTest");
  registry.set(std::move(row));

  const auto RESULT = registry.take();

  ASSERT_TRUE(RESULT.has_value());
  EXPECT_EQ(RESULT->testName, "SetTakeTest");
  EXPECT_EQ(RESULT->cycles, 1000);
  EXPECT_DOUBLE_EQ(RESULT->stats.median, 100.0);
}

/** @test Take on empty registry returns nullopt. */
TEST(PerfRegistryTest, TakeOnEmptyReturnsNullopt) {
  PerfRegistry& registry = PerfRegistry::instance();

  // Ensure empty
  registry.take();

  const auto RESULT = registry.take();

  EXPECT_FALSE(RESULT.has_value());
}

/** @test Take clears the registry. */
TEST(PerfRegistryTest, TakeClearsRegistry) {
  PerfRegistry& registry = PerfRegistry::instance();

  registry.set(makeTestRow("ClearTest"));

  // First take succeeds
  const auto FIRST = registry.take();
  ASSERT_TRUE(FIRST.has_value());

  // Second take returns empty
  const auto SECOND = registry.take();
  EXPECT_FALSE(SECOND.has_value());
}

/** @test Set overwrites previous value. */
TEST(PerfRegistryTest, SetOverwritesPrevious) {
  PerfRegistry& registry = PerfRegistry::instance();

  registry.set(makeTestRow("First"));
  registry.set(makeTestRow("Second"));

  const auto RESULT = registry.take();

  ASSERT_TRUE(RESULT.has_value());
  EXPECT_EQ(RESULT->testName, "Second");
}

/* ----------------------------- Profile Metadata Tests ----------------------------- */

/** @test Update profile metadata on existing row. */
TEST(PerfRegistryTest, UpdateProfileMetadata) {
  PerfRegistry& registry = PerfRegistry::instance();

  PerfRow row = makeTestRow("ProfileTest");
  registry.set(std::move(row));

  registry.updateProfileMeta("perf", "/tmp/artifacts");

  const auto RESULT = registry.take();

  ASSERT_TRUE(RESULT.has_value());
  ASSERT_TRUE(RESULT->profileTool.has_value());
  EXPECT_EQ(RESULT->profileTool.value(), "perf");
  ASSERT_TRUE(RESULT->profileDir.has_value());
  EXPECT_EQ(RESULT->profileDir.value(), "/tmp/artifacts");
}

/** @test Update profile metadata on empty registry is safe. */
TEST(PerfRegistryTest, UpdateProfileMetadataOnEmptyIsSafe) {
  PerfRegistry& registry = PerfRegistry::instance();

  // Ensure empty
  registry.take();

  // Should not crash
  registry.updateProfileMeta("tool", "dir");

  const auto RESULT = registry.take();
  EXPECT_FALSE(RESULT.has_value());
}

/* ----------------------------- Global Config Tests ----------------------------- */

/** @test Global config is initially null. */
TEST(PerfRegistryTest, GlobalConfigInitiallyNull) {
  // Reset to known state
  setGlobalPerfConfig(nullptr);

  const PerfConfig* CFG = globalPerfConfig();
  EXPECT_EQ(CFG, nullptr);
}

/** @test Set and get global config. */
TEST(PerfRegistryTest, SetAndGetGlobalConfig) {
  PerfConfig cfg;
  cfg.cycles = 5000;
  cfg.repeats = 5;

  setGlobalPerfConfig(&cfg);

  const PerfConfig* RESULT = globalPerfConfig();

  ASSERT_NE(RESULT, nullptr);
  EXPECT_EQ(RESULT->cycles, 5000);
  EXPECT_EQ(RESULT->repeats, 5);

  // Cleanup
  setGlobalPerfConfig(nullptr);
}

/* ----------------------------- PerfRow Field Tests ----------------------------- */

/** @test PerfRow default construction has empty optionals. */
TEST(PerfRowTest, DefaultConstructionEmptyOptionals) {
  const PerfRow ROW;

  EXPECT_FALSE(ROW.profileTool.has_value());
  EXPECT_FALSE(ROW.profileDir.has_value());
  EXPECT_FALSE(ROW.gpuModel.has_value());
  EXPECT_FALSE(ROW.kernelTimeUs.has_value());
  EXPECT_FALSE(ROW.deviceId.has_value());
  EXPECT_FALSE(ROW.umPageFaults.has_value());
}

/** @test PerfRow GPU fields can be set. */
TEST(PerfRowTest, GpuFieldsCanBeSet) {
  PerfRow row;
  row.gpuModel = "NVIDIA RTX 4090";
  row.computeCapability = "8.9";
  row.kernelTimeUs = 42.5;
  row.speedupVsCpu = 100.0;

  ASSERT_TRUE(row.gpuModel.has_value());
  EXPECT_EQ(row.gpuModel.value(), "NVIDIA RTX 4090");
  ASSERT_TRUE(row.kernelTimeUs.has_value());
  EXPECT_DOUBLE_EQ(row.kernelTimeUs.value(), 42.5);
}

/** @test PerfRow multi-GPU fields can be set. */
TEST(PerfRowTest, MultiGpuFieldsCanBeSet) {
  PerfRow row;
  row.deviceId = 0;
  row.deviceCount = 4;
  row.multiGpuEfficiency = 0.95;
  row.p2pBandwidthGBs = 25.0;

  ASSERT_TRUE(row.deviceId.has_value());
  EXPECT_EQ(row.deviceId.value(), 0);
  ASSERT_TRUE(row.deviceCount.has_value());
  EXPECT_EQ(row.deviceCount.value(), 4);
  ASSERT_TRUE(row.multiGpuEfficiency.has_value());
  EXPECT_DOUBLE_EQ(row.multiGpuEfficiency.value(), 0.95);
}

/** @test PerfRow unified memory fields can be set. */
TEST(PerfRowTest, UnifiedMemoryFieldsCanBeSet) {
  PerfRow row;
  row.umPageFaults = 1000;
  row.umH2DMigrations = 500;
  row.umD2HMigrations = 300;
  row.umMigrationTimeUs = 1500.0;
  row.umThrashing = true;

  ASSERT_TRUE(row.umPageFaults.has_value());
  EXPECT_EQ(row.umPageFaults.value(), 1000U);
  ASSERT_TRUE(row.umThrashing.has_value());
  EXPECT_TRUE(row.umThrashing.value());
}

/* ----------------------------- Thread Safety Tests ----------------------------- */

/** @test Concurrent set/take doesn't crash. */
TEST(PerfRegistryTest, ConcurrentAccessDoesNotCrash) {
  PerfRegistry& registry = PerfRegistry::instance();

  constexpr int NUM_THREADS = 10;
  constexpr int ITERATIONS = 100;

  std::vector<std::thread> threads;
  threads.reserve(NUM_THREADS);

  for (int t = 0; t < NUM_THREADS; ++t) {
    threads.emplace_back([&registry, t]() {
      for (int i = 0; i < ITERATIONS; ++i) {
        if (i % 2 == 0) {
          registry.set(makeTestRow("Thread" + std::to_string(t)));
        } else {
          registry.take();
        }
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  // If we get here without crashing, the test passes
  SUCCEED();
}
