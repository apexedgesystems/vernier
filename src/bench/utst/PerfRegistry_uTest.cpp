/**
 * @file PerfRegistry_uTest.cpp
 * @brief Unit tests for vernier::bench::PerfRegistry.
 *
 * Tests singleton access, publication and takes, the names a test's rows are
 * given, and profile metadata updates.
 *
 * Notes:
 *  - The registry is process-wide: every test that publishes first takes
 *    whatever an earlier test left waiting.
 */

#include "src/bench/inc/PerfRegistry.hpp"
#include "src/bench/inc/PerfConfig.hpp"

#include <gtest/gtest.h>

#include <cstddef>

#include <future>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using vernier::bench::globalPerfConfig;
using vernier::bench::PerfConfig;
using vernier::bench::PerfRegistry;
using vernier::bench::PerfRow;
using vernier::bench::setGlobalPerfConfig;
using vernier::bench::Stats;
using vernier::bench::detail::rowIdentities;

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

PerfRow labelledRow(const std::string& caseName, const std::string& label) {
  PerfRow row = makeTestRow(caseName);
  row.label = label;
  return row;
}

/** @brief Names takeAll() gives rows published as (case name, label), in order. */
std::vector<std::string> namesTaken(const std::vector<std::pair<std::string, std::string>>& rows) {
  PerfRegistry& registry = PerfRegistry::instance();
  static_cast<void>(registry.takeAll());
  for (const auto& [caseName, label] : rows) {
    registry.set(labelledRow(caseName, label));
  }
  std::vector<std::string> names;
  for (const PerfRow& row : registry.takeAll()) {
    names.push_back(row.testName);
  }
  return names;
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

/** @test Set keeps every row; the next take returns them all, in publication order */
TEST(PerfRegistryTest, SetKeepsEveryRow) {
  PerfRegistry& registry = PerfRegistry::instance();
  static_cast<void>(registry.takeAll());

  registry.set(makeTestRow("First"));
  registry.set(makeTestRow("Second"));

  const std::vector<PerfRow> ROWS = registry.takeAll();

  ASSERT_EQ(ROWS.size(), 2U);
  EXPECT_EQ(ROWS[0].testName, "First");
  EXPECT_EQ(ROWS[1].testName, "Second");
}

/** @test Takes nothing a second time: a row leaves with exactly one take */
TEST(PerfRegistryTest, TakeAllLeavesNothingWaiting) {
  PerfRegistry& registry = PerfRegistry::instance();
  static_cast<void>(registry.takeAll());

  registry.set(makeTestRow("Suite.One"));
  registry.set(makeTestRow("Suite.Two"));

  EXPECT_EQ(registry.takeAll().size(), 2U);
  EXPECT_TRUE(registry.takeAll().empty());
  EXPECT_FALSE(registry.take().has_value());
}

/** @test Take returns the most recent row, named, and takes the others with it */
TEST(PerfRegistryTest, TakeReturnsTheMostRecentRowAndTakesTheRest) {
  PerfRegistry& registry = PerfRegistry::instance();
  static_cast<void>(registry.takeAll());

  registry.set(labelledRow("Suite.Case", "first"));
  registry.set(labelledRow("Suite.Case", "second"));

  const auto RESULT = registry.take();

  ASSERT_TRUE(RESULT.has_value());
  EXPECT_EQ(RESULT->testName, "Suite.Case/second");
  EXPECT_EQ(RESULT->label, "second");
  EXPECT_TRUE(registry.takeAll().empty());
}

/** @test Gives the end-of-run summary entries of taken rows the rows' names */
TEST(PerfRegistryTest, TakeAllGivesSummaryEntriesTheRowNames) {
  PerfRegistry& registry = PerfRegistry::instance();
  static_cast<void>(registry.takeAll());
  const std::size_t FIRST = registry.summary().size();

  registry.set(labelledRow("Suite.Summary", "a"));
  registry.set(labelledRow("Suite.Summary", "b"));
  registry.set(labelledRow("Suite.Alone", "c"));
  ASSERT_EQ(registry.summary().size(), FIRST + 3);
  EXPECT_EQ(registry.summary()[FIRST].testName, "Suite.Summary");

  const std::vector<PerfRow> ROWS = registry.takeAll();

  ASSERT_EQ(ROWS.size(), 3U);
  for (std::size_t i = 0; i < ROWS.size(); ++i) {
    EXPECT_EQ(registry.summary()[FIRST + i].testName, ROWS[i].testName) << "entry " << i;
  }
  EXPECT_EQ(registry.summary()[FIRST].testName, "Suite.Summary/a");
  EXPECT_EQ(registry.summary()[FIRST + 2].testName, "Suite.Alone");
}

/* ----------------------------- Row Identity Tests ----------------------------- */

/** @test A test's only measurement keeps its case name, whatever its label */
TEST(PerfRegistryTest, OneMeasurementKeepsItsCaseName) {
  EXPECT_EQ(namesTaken({{"Suite.Case", "throughput"}}), (std::vector<std::string>{"Suite.Case"}));
  EXPECT_EQ(namesTaken({{"Suite.Case", ""}}), (std::vector<std::string>{"Suite.Case"}));
}

/** @test Separately named cases in one test keep their names */
TEST(PerfRegistryTest, SeparateCasesKeepTheirNames) {
  EXPECT_EQ(namesTaken({{"Sweep.Sizes/64", "throughput"},
                        {"Sweep.Sizes/256", "throughput"},
                        {"Sweep.Sizes/1024", "throughput"}}),
            (std::vector<std::string>{"Sweep.Sizes/64", "Sweep.Sizes/256", "Sweep.Sizes/1024"}));
}

/** @test Names each row of a case measured more than once by its label */
TEST(PerfRegistryTest, CaseMeasuredMoreThanOnceNamesEachRowByItsLabel) {
  EXPECT_EQ(namesTaken({{"Suite.Case", "64"},
                        {"Other.Case", "throughput"},
                        {"Suite.Case", "256"},
                        {"Suite.Case", "1024"}}),
            (std::vector<std::string>{"Suite.Case/64", "Other.Case", "Suite.Case/256",
                                      "Suite.Case/1024"}));
}

/** @test Numbers a repeated label's rows from 1 in completion order */
TEST(PerfRegistryTest, RepeatedLabelIsNumberedInCompletionOrder) {
  EXPECT_EQ(namesTaken({{"Suite.Case", "x"}, {"Suite.Case", "y"}, {"Suite.Case", "x"}}),
            (std::vector<std::string>{"Suite.Case/x#1", "Suite.Case/y", "Suite.Case/x#2"}));
  EXPECT_EQ(namesTaken({{"Suite.Case", "x"}, {"Suite.Case", "x"}}),
            (std::vector<std::string>{"Suite.Case/x#1", "Suite.Case/x#2"}));
}

/** @test Numbers an empty label, alone or repeated */
TEST(PerfRegistryTest, EmptyLabelIsNumbered) {
  EXPECT_EQ(namesTaken({{"Suite.Case", ""}, {"Suite.Case", ""}}),
            (std::vector<std::string>{"Suite.Case/#1", "Suite.Case/#2"}));
  EXPECT_EQ(namesTaken({{"Suite.Case", ""}, {"Suite.Case", "x"}}),
            (std::vector<std::string>{"Suite.Case/#1", "Suite.Case/x"}));
}

/** @test A row name equal to a case name in the test takes the next free number */
TEST(PerfRegistryTest, NameEqualToACaseNameTakesTheNextNumber) {
  // "A/b" is a case of its own and keeps its name, wherever it completes.
  EXPECT_EQ(namesTaken({{"A", "b"}, {"A/b", "t"}, {"A", "c"}}),
            (std::vector<std::string>{"A/b#2", "A/b", "A/c"}));
}

/** @test Equal row names of different cases, or through a '#' in a label, are told apart */
TEST(PerfRegistryTest, EqualRowNamesAreNumbered) {
  EXPECT_EQ(namesTaken({{"A", "b/c"}, {"A", "d"}, {"A/b", "c"}, {"A/b", "e"}}),
            (std::vector<std::string>{"A/b/c", "A/d", "A/b/c#2", "A/b/e"}));
  EXPECT_EQ(namesTaken({{"C", "x"}, {"C", "x"}, {"C", "x#1"}}),
            (std::vector<std::string>{"C/x#1", "C/x#2", "C/x#1#2"}));
}

/** @test Gives unique names for every mix of colliding case names and labels */
TEST(PerfRegistryTest, RowNamesAreUniqueForEveryMix) {
  const std::vector<std::string> CASES = {"A", "A/x", "A/x#1"};
  const std::vector<std::string> LABELS = {"", "x", "x#1", "x#2", "#1"};
  std::vector<std::pair<std::string, std::string>> items;
  for (const std::string& caseName : CASES) {
    for (const std::string& label : LABELS) {
      items.emplace_back(caseName, label);
    }
  }

  std::size_t checked = 0;
  const std::size_t N = items.size();
  for (std::size_t a = 0; a < N; ++a) {
    for (std::size_t b = 0; b < N; ++b) {
      for (std::size_t c = 0; c < N; ++c) {
        std::vector<PerfRow> rows;
        for (const std::size_t pick : {a, b, c}) {
          rows.push_back(labelledRow(items[pick].first, items[pick].second));
        }
        const std::vector<std::string> NAMES = rowIdentities(rows);
        ASSERT_EQ(NAMES.size(), rows.size());
        EXPECT_EQ(std::set<std::string>(NAMES.begin(), NAMES.end()).size(), NAMES.size())
            << "duplicate among " << NAMES[0] << " | " << NAMES[1] << " | " << NAMES[2];
        for (std::size_t i = 0; i < rows.size(); ++i) {
          std::size_t sameCase = 0;
          for (const PerfRow& other : rows) {
            sameCase += (other.testName == rows[i].testName) ? 1U : 0U;
          }
          if (sameCase == 1) {
            EXPECT_EQ(NAMES[i], rows[i].testName);
          } else {
            EXPECT_EQ(NAMES[i].rfind(rows[i].testName + "/" + rows[i].label, 0), 0U)
                << NAMES[i] << " does not start with its case and label";
          }
        }
        ++checked;
      }
    }
  }
  EXPECT_EQ(checked, N * N * N);
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

/** @test Stamps the row the calling thread published, not a newer one from another thread */
TEST(PerfRegistryTest, StampReachesTheCallingThreadsOwnRow) {
  PerfRegistry& registry = PerfRegistry::instance();
  static_cast<void>(registry.takeAll());

  std::promise<void> published;
  std::promise<void> overtaken;
  std::future<void> publishedSignal = published.get_future();
  std::future<void> overtakenSignal = overtaken.get_future();
  std::thread profiled([&] {
    registry.set(makeTestRow("Suite.Profiled"));
    published.set_value();
    overtakenSignal.wait();
    registry.updateProfileMeta("perf", "/tmp/Suite.Profiled.perf");
  });
  publishedSignal.wait();
  registry.set(makeTestRow("Suite.Newer"));
  overtaken.set_value();
  profiled.join();

  const std::vector<PerfRow> ROWS = registry.takeAll();

  ASSERT_EQ(ROWS.size(), 2U);
  EXPECT_EQ(ROWS[0].testName, "Suite.Profiled");
  ASSERT_TRUE(ROWS[0].profileDir.has_value());
  EXPECT_EQ(*ROWS[0].profileDir, "/tmp/Suite.Profiled.perf");
  EXPECT_EQ(ROWS[1].testName, "Suite.Newer");
  EXPECT_FALSE(ROWS[1].profileDir.has_value()) << "the stamp went to the newest row";
}

/** @test Stamps nothing once the calling thread's row has been taken */
TEST(PerfRegistryTest, StampAfterTheRowIsTakenChangesNothing) {
  PerfRegistry& registry = PerfRegistry::instance();
  static_cast<void>(registry.takeAll());

  registry.set(makeTestRow("Suite.Taken"));
  const std::vector<PerfRow> TAKEN = registry.takeAll();
  registry.updateProfileMeta("perf", "/tmp/late");
  registry.set(makeTestRow("Suite.Next"));

  const std::vector<PerfRow> ROWS = registry.takeAll();

  ASSERT_EQ(TAKEN.size(), 1U);
  EXPECT_FALSE(TAKEN[0].profileDir.has_value());
  ASSERT_EQ(ROWS.size(), 1U);
  EXPECT_FALSE(ROWS[0].profileDir.has_value()) << "a later row took a stamp meant for a taken one";
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
