/**
 * @file PerfListener_uTest.cpp
 * @brief Unit tests for the CSV listener's row handoff.
 *
 * Notes:
 *  - A row describes what its case ran. The cases here deliberately run
 *    with a config that differs from the process-wide one.
 *  - Columns are located by header name, so column order is free to change.
 */

#include "src/bench/inc/PerfListener.hpp"

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfHarness.hpp"
#include "src/bench/inc/PerfRegistry.hpp"
#include "src/bench/inc/PerfStats.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using vernier::bench::buildPerfRow;
using vernier::bench::PerfCase;
using vernier::bench::PerfConfig;
using vernier::bench::PerfRegistry;
using vernier::bench::setGlobalPerfConfig;
using vernier::bench::Stats;
using vernier::bench::detail::CsvListener;

namespace {

std::vector<std::string> splitCsvLine(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string cell;
  while (std::getline(ss, cell, ',')) {
    out.push_back(cell);
  }
  return out;
}

} // namespace

/**
 * @brief Drives a CsvListener the way GoogleTest does and reads back the row.
 *
 * The process-wide config is installed for the life of each test and
 * cleared afterwards so other suites see the default (none).
 */
class CsvListenerTest : public ::testing::Test {
protected:
  PerfConfig global_{};
  std::string path_;

  void SetUp() override {
    global_.cycles = 10000;
    global_.repeats = 10;
    global_.threads = 4;
    global_.msgBytes = 64;
    global_.console = false;
    global_.nonBlocking = false;
    global_.minLevel = "INFO";
    setGlobalPerfConfig(&global_);

    path_ =
        "/tmp/vernier_listener_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".csv";
    (void)PerfRegistry::instance().take();
  }

  void TearDown() override {
    setGlobalPerfConfig(nullptr);
    (void)PerfRegistry::instance().take();
    std::remove(path_.c_str());
  }

  /** @brief Emit the pending registry row through a listener; return column -> cell. */
  std::map<std::string, std::string> emitAndRead() {
    {
      CsvListener listener(path_, /*includeProfile=*/false, /*includeGpu=*/false);
      listener.OnTestEnd(*::testing::UnitTest::GetInstance()->current_test_info());
    }
    std::ifstream in(path_);
    std::string header;
    std::string row;
    std::getline(in, header);
    std::getline(in, row);
    const std::vector<std::string> names = splitCsvLine(header);
    const std::vector<std::string> cells = splitCsvLine(row);
    std::map<std::string, std::string> out;
    for (std::size_t i = 0; i < names.size() && i < cells.size(); ++i) {
      out[names[i]] = cells[i];
    }
    return out;
  }
};

/* ----------------------------- API Tests ----------------------------- */

/** @test Writes the case's own config columns when they differ from the process-wide config */
TEST_F(CsvListenerTest, RowKeepsCaseConfig) {
  PerfConfig caseCfg = global_;
  caseCfg.cycles = 666;
  caseCfg.repeats = 7;
  caseCfg.threads = 1;
  caseCfg.msgBytes = 256;
  caseCfg.console = true;
  caseCfg.nonBlocking = true;
  caseCfg.minLevel = "DEBUG";

  Stats stats{};
  stats.median = 1.5;
  PerfRegistry::instance().set(buildPerfRow("Listener.CaseConfig", caseCfg, /*actualWarmup=*/1,
                                            caseCfg.threads, stats, 1e6));

  const std::map<std::string, std::string> cols = emitAndRead();

  ASSERT_EQ(cols.count("test"), 1u);
  EXPECT_EQ(cols.at("test"), "Listener.CaseConfig");
  EXPECT_EQ(cols.at("cycles"), "666");
  EXPECT_EQ(cols.at("repeats"), "7");
  EXPECT_EQ(cols.at("threads"), "1");
  EXPECT_EQ(cols.at("msgBytes"), "256");
  EXPECT_EQ(cols.at("console"), "1");
  EXPECT_EQ(cols.at("nonBlocking"), "1");
  EXPECT_EQ(cols.at("minLevel"), "DEBUG");
}

/** @test Writes the cycle count a target-time case calibrated, not the process-wide cycles */
TEST_F(CsvListenerTest, RowKeepsCalibratedCycles) {
  PerfConfig caseCfg = global_;
  caseCfg.targetTimeUs = 2000;
  caseCfg.repeats = 2;
  caseCfg.threads = 1;

  volatile std::uint64_t sink = 0;
  PerfCase perf{"Listener.Calibrated", caseCfg};
  (void)perf.throughputLoop([&] { sink = sink + 1; });

  // Precondition: calibration moved the case away from the process-wide value.
  ASSERT_NE(perf.cycles(), global_.cycles);

  const std::map<std::string, std::string> cols = emitAndRead();

  ASSERT_EQ(cols.count("cycles"), 1u);
  EXPECT_EQ(cols.at("cycles"), std::to_string(perf.cycles()));
  EXPECT_EQ(cols.at("repeats"), "2");
  EXPECT_EQ(cols.at("threads"), "1");
}

/** @test Writes nothing beyond the header when no case left a row */
TEST_F(CsvListenerTest, NoRowWithoutResult) {
  const std::map<std::string, std::string> cols = emitAndRead();
  EXPECT_TRUE(cols.empty());
}
