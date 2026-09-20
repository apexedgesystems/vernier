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
using vernier::bench::PerfRow;
using vernier::bench::setGlobalPerfConfig;
using vernier::bench::Stats;
using vernier::bench::detail::CsvListener;

namespace {

/** @brief Split a CSV line, keeping empty cells including a trailing one. */
std::vector<std::string> splitCsvLine(const std::string& line) {
  std::vector<std::string> out;
  std::string cell;
  for (const char C : line) {
    if (C == ',') {
      out.push_back(cell);
      cell.clear();
    } else {
      cell += C;
    }
  }
  out.push_back(cell);
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
    if (!std::getline(in, row) || row.empty()) {
      return {};
    }
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

/* ----------------------------- Row Width Tests ----------------------------- */

/**
 * @brief Writes several rows of different kinds into one GPU CSV and reads
 *        every cell back by header name.
 *
 * A GPU binary's file holds both kinds of row: the GPU harness's own rows and
 * the plain CPU rows of a baseline case in the same binary. Whether a row
 * carries profile metadata is decided per row, while the header is written
 * once for the whole file.
 */
class GpuCsvListenerTest : public ::testing::Test {
protected:
  std::string path_;

  void SetUp() override {
    path_ = "/tmp/vernier_gpu_listener_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)) +
            ".csv";
    (void)PerfRegistry::instance().take();
  }

  void TearDown() override {
    (void)PerfRegistry::instance().take();
    std::remove(path_.c_str());
  }

  /** @brief A row as the CPU harness leaves it: no GPU cells, no profile metadata. */
  static PerfRow cpuRow() {
    PerfRow row;
    row.testName = "GpuSuite.CpuBaseline";
    row.cycles = 1000;
    row.repeats = 10;
    row.warmup = 1;
    row.threads = 1;
    row.msgBytes = 64;
    row.minLevel = "INFO";
    row.stats.median = 166.0;
    row.stats.cv = 0.02;
    row.callsPerSecond = 6024.0;
    row.timestamp = "2026-09-20T12:00:00Z";
    row.gitHash = "0123abc";
    row.hostname = "rig-host";
    row.platform = "aarch64";
    row.stable = true;
    row.cvThreshold = 0.10;
    return row;
  }

  /** @brief A row as the GPU harness leaves it: GPU cells, no profile metadata. */
  static PerfRow gpuRow() {
    PerfRow row = cpuRow();
    row.testName = "GpuSuite.GpuKernelOnly";
    row.stats.median = 21.18;
    row.callsPerSecond = 47214.0;
    row.gpuModel = "Test Device";
    row.computeCapability = "11.0";
    row.kernelTimeUs = 21.18;
    row.transferTimeUs = 0.0;
    row.h2dBytes = 0U;
    row.d2hBytes = 0U;
    row.memBandwidthGBs = 0.0;
    row.occupancy = 0.667;
    row.smClockMHz = 1400;
    row.throttling = false;
    row.deviceId = 0;
    row.deviceCount = 1;
    return row;
  }

  struct Csv {
    std::vector<std::string> names;
    std::vector<std::vector<std::string>> rows;

    /** @brief Cell of row @p r under column @p name; "<no such column>" if absent. */
    [[nodiscard]] std::string at(std::size_t r, const std::string& name) const {
      for (std::size_t i = 0; i < names.size(); ++i) {
        if (names[i] == name) {
          return (i < rows[r].size()) ? rows[r][i] : std::string("<short row>");
        }
      }
      return "<no such column>";
    }
  };

  /** @brief Emit @p rows through one listener and read the file back. */
  Csv emit(bool includeProfile, const std::vector<PerfRow>& rows, bool includeGpu = true) {
    {
      CsvListener listener(path_, includeProfile, includeGpu);
      for (const auto& row : rows) {
        PerfRegistry::instance().set(row);
        listener.OnTestEnd(*::testing::UnitTest::GetInstance()->current_test_info());
      }
    }

    Csv out;
    std::ifstream in(path_);
    std::string line;
    if (std::getline(in, line)) {
      out.names = splitCsvLine(line);
    }
    while (std::getline(in, line)) {
      out.rows.push_back(splitCsvLine(line));
    }
    return out;
  }
};

/** @test Under a profile, a GPU row and a CPU row both fill the header's columns */
TEST_F(GpuCsvListenerTest, ProfiledFileRowsMatchHeaderColumns) {
  const Csv CSV = emit(/*includeProfile=*/true, {gpuRow(), cpuRow()});

  ASSERT_EQ(CSV.rows.size(), 2U);
  EXPECT_EQ(CSV.rows[0].size(), CSV.names.size());
  EXPECT_EQ(CSV.rows[1].size(), CSV.names.size());

  // The GPU row carries no profile metadata of its own: those two cells are
  // empty, and every later value still stands under its own header.
  EXPECT_EQ(CSV.at(0, "test"), "GpuSuite.GpuKernelOnly");
  EXPECT_EQ(CSV.at(0, "profileTool"), "");
  EXPECT_EQ(CSV.at(0, "profileDir"), "");
  EXPECT_EQ(CSV.at(0, "timestamp"), "2026-09-20T12:00:00Z");
  EXPECT_EQ(CSV.at(0, "hostname"), "rig-host");
  EXPECT_EQ(CSV.at(0, "gpuModel"), "Test Device");
  EXPECT_EQ(CSV.at(0, "computeCapability"), "11.0");
  EXPECT_EQ(CSV.at(0, "deviceId"), "0");
  EXPECT_EQ(CSV.at(0, "deviceCount"), "1");
  EXPECT_EQ(CSV.at(0, "umThrashing"), "");

  // The CPU row has no GPU values: its metadata still sits under the metadata
  // headers, and the GPU cells are empty rather than absent.
  EXPECT_EQ(CSV.at(1, "test"), "GpuSuite.CpuBaseline");
  EXPECT_EQ(CSV.at(1, "timestamp"), "2026-09-20T12:00:00Z");
  EXPECT_EQ(CSV.at(1, "gitHash"), "0123abc");
  EXPECT_EQ(CSV.at(1, "platform"), "aarch64");
  EXPECT_EQ(CSV.at(1, "gpuModel"), "");
  EXPECT_EQ(CSV.at(1, "kernelTimeUs"), "");
  EXPECT_EQ(CSV.at(1, "deviceCount"), "");
}

/** @test Without a profile, a GPU row and a CPU row both fill the header's columns */
TEST_F(GpuCsvListenerTest, UnprofiledFileRowsMatchHeaderColumns) {
  const Csv CSV = emit(/*includeProfile=*/false, {gpuRow(), cpuRow()});

  ASSERT_EQ(CSV.rows.size(), 2U);
  EXPECT_EQ(CSV.rows[0].size(), CSV.names.size());
  EXPECT_EQ(CSV.rows[1].size(), CSV.names.size());

  // No profile columns exist at all in this file.
  EXPECT_EQ(CSV.at(0, "profileTool"), "<no such column>");

  EXPECT_EQ(CSV.at(0, "timestamp"), "2026-09-20T12:00:00Z");
  EXPECT_EQ(CSV.at(0, "gpuModel"), "Test Device");
  EXPECT_EQ(CSV.at(0, "occupancy"), "0.667000");
  EXPECT_EQ(CSV.at(1, "hostname"), "rig-host");
  EXPECT_EQ(CSV.at(1, "gpuModel"), "");
  EXPECT_EQ(CSV.at(1, "umThrashing"), "");
}

/** @test In a CPU file with profile columns, a row without profile metadata fills them too */
TEST_F(GpuCsvListenerTest, CpuFileRowWithoutProfileMetadataKeepsItsWidth) {
  PerfRow profiled = cpuRow();
  profiled.testName = "CpuSuite.Profiled";
  profiled.profileTool = "callgrind";
  profiled.profileDir = "./CpuSuite.Profiled.callgrind";

  const Csv CSV = emit(/*includeProfile=*/true, {cpuRow(), profiled}, /*includeGpu=*/false);

  // 22 result columns + 2 profile + 4 metadata.
  ASSERT_EQ(CSV.rows.size(), 2U);
  EXPECT_EQ(CSV.names.size(), 28U);
  EXPECT_EQ(CSV.rows[0].size(), CSV.names.size());
  EXPECT_EQ(CSV.rows[1].size(), CSV.names.size());

  EXPECT_EQ(CSV.at(0, "test"), "GpuSuite.CpuBaseline");
  EXPECT_EQ(CSV.at(0, "profileTool"), "");
  EXPECT_EQ(CSV.at(0, "profileDir"), "");
  EXPECT_EQ(CSV.at(0, "timestamp"), "2026-09-20T12:00:00Z");
  EXPECT_EQ(CSV.at(0, "platform"), "aarch64");

  EXPECT_EQ(CSV.at(1, "profileTool"), "callgrind");
  EXPECT_EQ(CSV.at(1, "profileDir"), "./CpuSuite.Profiled.callgrind");
  EXPECT_EQ(CSV.at(1, "hostname"), "rig-host");
}

/** @test A row that does carry profile metadata keeps it under its own headers */
TEST_F(GpuCsvListenerTest, ProfileMetadataStaysUnderItsHeaders) {
  PerfRow row = gpuRow();
  row.profileTool = "nsight";
  row.profileDir = "/tmp/bench-out/GpuSuite.GpuKernelOnly.nsight";

  const Csv CSV = emit(/*includeProfile=*/true, {row});

  ASSERT_EQ(CSV.rows.size(), 1U);
  EXPECT_EQ(CSV.rows[0].size(), CSV.names.size());
  EXPECT_EQ(CSV.at(0, "profileTool"), "nsight");
  EXPECT_EQ(CSV.at(0, "profileDir"), "/tmp/bench-out/GpuSuite.GpuKernelOnly.nsight");
  EXPECT_EQ(CSV.at(0, "gpuModel"), "Test Device");
}
