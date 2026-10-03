/**
 * @file PerfListener_uTest.cpp
 * @brief Unit tests for the CSV listener's row handoff.
 *
 * Notes:
 *  - A row describes what its case ran. The cases here deliberately run
 *    with a config that differs from the process-wide one.
 *  - Columns are located by header name, so column order is free to change.
 *  - A test that measures more than once checks each row against the result
 *    of its own measurement, formatted the way the writer formats it.
 */

#include "src/bench/inc/PerfListener.hpp"

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfHarness.hpp"
#include "src/bench/inc/PerfRegistry.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"
#include "src/bench/inc/ProfilerReadiness.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <array>
#include <atomic>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using vernier::bench::buildPerfRow;
using vernier::bench::makePerfCaseWithProfiler;
using vernier::bench::PerfCase;
using vernier::bench::PerfConfig;
using vernier::bench::PerfRegistry;
using vernier::bench::PerfRow;
using vernier::bench::PerfSummaryEntry;
using vernier::bench::printSummaryTable;
using vernier::bench::Profiler;
using vernier::bench::ProfilerRegistry;
using vernier::bench::ReadinessCause;
using vernier::bench::ReadinessContext;
using vernier::bench::ReadinessRequest;
using vernier::bench::ReadinessResult;
using vernier::bench::readinessResult;
using vernier::bench::setGlobalPerfConfig;
using vernier::bench::Stats;
using vernier::bench::detail::CsvListener;

namespace {

/** @brief A number as the CSV writer formats it (the stream's default format). */
std::string csvNumber(double value) {
  std::ostringstream out;
  out << value;
  return out.str();
}

/** @brief Work in proportion to @p n, kept from the optimizer by a volatile sink. */
void spin(int n) {
  static volatile std::uint64_t sink = 0;
  for (int i = 0; i < n; ++i) {
    sink = sink + static_cast<std::uint64_t>(i);
  }
}

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

  /** @brief Emit every pending registry row through one listener; read every row back. */
  std::vector<std::map<std::string, std::string>> emitAndReadAll(bool includeProfile = false) {
    {
      CsvListener listener(path_, includeProfile, /*includeGpu=*/false);
      listener.OnTestEnd(*::testing::UnitTest::GetInstance()->current_test_info());
    }
    return readRows();
  }

  /** @brief Every data row of the file, column -> cell, in file order. */
  std::vector<std::map<std::string, std::string>> readRows() const {
    std::ifstream in(path_);
    std::string header;
    std::getline(in, header);
    const std::vector<std::string> names = splitCsvLine(header);
    std::vector<std::map<std::string, std::string>> rows;
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) {
        continue;
      }
      const std::vector<std::string> cells = splitCsvLine(line);
      std::map<std::string, std::string> row;
      for (std::size_t i = 0; i < names.size() && i < cells.size(); ++i) {
        row[names[i]] = cells[i];
      }
      rows.push_back(std::move(row));
    }
    return rows;
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

/* ----------------------------- Thread Count Tests ----------------------------- */

/**
 * @test Writes one thread for throughputLoop() and measured(), whose calls run
 *       on the calling thread, however many threads the case is configured for
 */
TEST_F(CsvListenerTest, SingleThreadRowRecordsOneThread) {
  PerfConfig caseCfg = global_;
  caseCfg.cycles = 10;
  caseCfg.repeats = 2;
  ASSERT_EQ(caseCfg.threads, 4);

  int calls = 0;
  PerfCase loop{"Listener.SingleThreadLoop", caseCfg};
  (void)loop.throughputLoop([&] { ++calls; });
  EXPECT_EQ(calls, caseCfg.cycles * caseCfg.repeats);
  std::map<std::string, std::string> cols = emitAndRead();
  ASSERT_EQ(cols.count("threads"), 1u);
  EXPECT_EQ(cols.at("threads"), "1") << "throughputLoop() row";

  calls = 0;
  PerfCase body{"Listener.SingleThreadMeasured", caseCfg};
  (void)body.measured([&] { ++calls; });
  EXPECT_EQ(calls, caseCfg.repeats);
  cols = emitAndRead();
  ASSERT_EQ(cols.count("threads"), 1u);
  EXPECT_EQ(cols.at("threads"), "1") << "measured() row";
}

/** @test Writes the number of workers contentionRun() started, each making every call */
TEST_F(CsvListenerTest, ContentionRowRecordsItsWorkers) {
  PerfConfig caseCfg = global_;
  caseCfg.cycles = 10;
  caseCfg.repeats = 2;
  ASSERT_EQ(caseCfg.threads, 4);

  std::atomic<int> calls{0};
  PerfCase perf{"Listener.Contention", caseCfg};
  (void)perf.contentionRun([&] { calls.fetch_add(1, std::memory_order_relaxed); });
  EXPECT_EQ(calls.load(), caseCfg.threads * caseCfg.cycles * caseCfg.repeats);

  const std::map<std::string, std::string> cols = emitAndRead();
  ASSERT_EQ(cols.count("threads"), 1u);
  EXPECT_EQ(cols.at("threads"), "4");
}

/* ----------------------------- Measurement Row Tests ----------------------------- */

namespace {

/**
 * @brief A profiler whose folder is its case's and, like a backend that keeps
 *        one capture per measurement, a new one for every measurement.
 */
class RowFixtureProfiler final : public Profiler {
public:
  explicit RowFixtureProfiler(std::string testName) : testName_(std::move(testName)) {}
  std::string toolName() const noexcept override { return "rows-fixture"; }
  std::string artifactDir() const noexcept override {
    return testName_ + ".capture" + std::to_string(measurements_);
  }
  void afterMeasure(const Stats& /*s*/) override { ++measurements_; }

private:
  std::string testName_;
  int measurements_{0};
};

/** @brief Registers, for one test, a backend whose request always runs. */
class RowFixtureBackend {
public:
  static constexpr const char* NAME = "rows-fixture";
  RowFixtureBackend() {
    ProfilerRegistry::instance().registerReadinessBackend(
        NAME,
        [](const ReadinessRequest&, const ReadinessContext&) {
          return readinessResult(ReadinessCause::READY, "always ready", "");
        },
        [](const PerfConfig&, const std::string& testName, const ReadinessResult&) {
          return std::unique_ptr<Profiler>(std::make_unique<RowFixtureProfiler>(testName));
        },
        "");
  }
  ~RowFixtureBackend() { ProfilerRegistry::instance().unregisterBackend(NAME); }
  RowFixtureBackend(const RowFixtureBackend&) = delete;
  RowFixtureBackend& operator=(const RowFixtureBackend&) = delete;
};

} // namespace

/** @test Writes one row per measurement of a case measured three times, each with its median */
TEST_F(CsvListenerTest, ThreeMeasurementsOfOneCaseWriteThreeRows) {
  PerfConfig caseCfg = global_;
  caseCfg.cycles = 20;
  caseCfg.repeats = 3;
  PerfCase perf{"Listener.Sweep", caseCfg};
  std::vector<std::string> medians;
  for (const int size : {64, 256, 1024}) {
    medians.push_back(
        csvNumber(perf.throughputLoop([size] { spin(size); }, std::to_string(size)).stats.median));
  }

  const std::vector<std::map<std::string, std::string>> rows = emitAndReadAll();

  const std::array<const char*, 3> names = {"Listener.Sweep/64", "Listener.Sweep/256",
                                            "Listener.Sweep/1024"};
  ASSERT_EQ(rows.size(), names.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    EXPECT_EQ(rows[i].at("test"), names[i]);
    EXPECT_EQ(rows[i].at("wallMedian"), medians[i]) << names[i];
    EXPECT_EQ(rows[i].at("cycles"), "20") << names[i];
  }
}

/** @test Writes separately named cases of one test under their own names and config */
TEST_F(CsvListenerTest, SeparateCasesWriteTheirOwnConfig) {
  constexpr std::array<int, 3> SIZES = {64, 256, 1024};
  for (const int size : SIZES) {
    PerfConfig caseCfg = global_;
    caseCfg.msgBytes = size;
    caseCfg.cycles = size / 8;
    caseCfg.repeats = 2;
    PerfCase perf{"Listener.Size/" + std::to_string(size), caseCfg};
    (void)perf.throughputLoop([size] { spin(size); });
  }

  const std::vector<std::map<std::string, std::string>> rows = emitAndReadAll();

  ASSERT_EQ(rows.size(), SIZES.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    EXPECT_EQ(rows[i].at("test"), "Listener.Size/" + std::to_string(SIZES[i]));
    EXPECT_EQ(rows[i].at("msgBytes"), std::to_string(SIZES[i]));
    EXPECT_EQ(rows[i].at("cycles"), std::to_string(SIZES[i] / 8));
  }
}

/** @test A second drain at the same test's end writes nothing more and leaves nothing waiting */
TEST_F(CsvListenerTest, SecondDrainWritesNothing) {
  PerfConfig caseCfg = global_;
  caseCfg.cycles = 10;
  caseCfg.repeats = 2;
  PerfCase perf{"Listener.Drained", caseCfg};
  (void)perf.throughputLoop([] { spin(8); }, "a");
  (void)perf.throughputLoop([] { spin(8); }, "b");
  {
    CsvListener listener(path_, /*includeProfile=*/false, /*includeGpu=*/false);
    listener.OnTestEnd(*::testing::UnitTest::GetInstance()->current_test_info());
    listener.OnTestEnd(*::testing::UnitTest::GetInstance()->current_test_info());
  }

  const std::vector<std::map<std::string, std::string>> rows = readRows();

  ASSERT_EQ(rows.size(), 2U);
  EXPECT_EQ(rows[0].at("test"), "Listener.Drained/a");
  EXPECT_EQ(rows[1].at("test"), "Listener.Drained/b");
  EXPECT_FALSE(PerfRegistry::instance().take().has_value());
}

/** @test A measurement whose body throws publishes no row; the completed one is written alone */
TEST_F(CsvListenerTest, MeasurementThatThrowsPublishesNothing) {
  PerfConfig caseCfg = global_;
  caseCfg.cycles = 10;
  caseCfg.repeats = 2;
  PerfCase perf{"Listener.Throws", caseCfg};
  const std::string completed =
      csvNumber(perf.throughputLoop([] { spin(8); }, "completed").stats.median);
  EXPECT_THROW(
      (void)perf.throughputLoop([] { throw std::runtime_error("measured body failed"); }, "failed"),
      std::runtime_error);

  const std::vector<std::map<std::string, std::string>> rows = emitAndReadAll();

  ASSERT_EQ(rows.size(), 1U);
  EXPECT_EQ(rows[0].at("test"), "Listener.Throws");
  EXPECT_EQ(rows[0].at("wallMedian"), completed);
}

/** @test Each row carries the profiler folder of its own case and its own measurement */
TEST_F(CsvListenerTest, ProfilerIdentityStaysWithItsMeasurement) {
  const RowFixtureBackend backend;
  PerfConfig caseCfg = global_;
  caseCfg.cycles = 10;
  caseCfg.repeats = 2;
  caseCfg.profileTool = RowFixtureBackend::NAME;
  PerfCase first = makePerfCaseWithProfiler("Listener.ProfiledA", caseCfg);
  PerfCase second = makePerfCaseWithProfiler("Listener.ProfiledB", caseCfg);

  (void)first.throughputLoop([] { spin(8); }, "one");
  (void)second.throughputLoop([] { spin(8); }, "only");
  (void)first.throughputLoop([] { spin(8); }, "two");

  const std::vector<std::map<std::string, std::string>> rows =
      emitAndReadAll(/*includeProfile=*/true);

  ASSERT_EQ(rows.size(), 3U);
  EXPECT_EQ(rows[0].at("test"), "Listener.ProfiledA/one");
  EXPECT_EQ(rows[0].at("profileDir"), "Listener.ProfiledA.capture1");
  EXPECT_EQ(rows[1].at("test"), "Listener.ProfiledB");
  EXPECT_EQ(rows[1].at("profileDir"), "Listener.ProfiledB.capture1");
  EXPECT_EQ(rows[2].at("test"), "Listener.ProfiledA/two");
  EXPECT_EQ(rows[2].at("profileDir"), "Listener.ProfiledA.capture2");
  for (const auto& row : rows) {
    EXPECT_EQ(row.at("profileTool"), "rows-fixture") << row.at("test");
  }
}

/* ----------------------------- Summary Table Tests ----------------------------- */

namespace {

/** @brief The table printSummaryTable() prints for @p entries and @p tests. */
std::string summaryTable(const std::vector<PerfSummaryEntry>& entries, std::size_t tests) {
  ::testing::internal::CaptureStdout();
  printSummaryTable(entries, tests);
  std::fflush(stdout);
  return ::testing::internal::GetCapturedStdout();
}

/** @brief The last line of @p text, without its line break. */
std::string lastLine(const std::string& text) {
  const std::string body = text.substr(0, text.find_last_not_of('\n') + 1);
  return body.substr(body.find_last_of('\n') + 1);
}

PerfSummaryEntry summaryEntry(const std::string& name, bool stable) {
  return PerfSummaryEntry{name, 1.5, stable ? 0.01 : 0.4, 1e6, stable, 0.05};
}

} // namespace

/** @test Keeps the footer of one row per test byte for byte */
TEST(PrintSummaryTableTest, FooterIsUnchangedWhenEachTestPublishedOneRow) {
  const std::vector<PerfSummaryEntry> entries = {summaryEntry("Suite.First", true),
                                                 summaryEntry("Suite.Second", false)};

  const std::string table = summaryTable(entries, 2);

  EXPECT_EQ(lastLine(table), "2 tests | 1 stable | 1 unstable");
  EXPECT_EQ(table, summaryTable(entries, 0)) << "a caller that passes no test count";
}

/** @test Names rows and tests when a test published more than one row */
TEST(PrintSummaryTableTest, FooterNamesRowsAndTestsWhenATestPublishedSeveral) {
  const std::vector<PerfSummaryEntry> entries = {
      summaryEntry("Suite.Sweep/64", true), summaryEntry("Suite.Sweep/256", true),
      summaryEntry("Suite.Sweep/1024", false), summaryEntry("Suite.Other", true),
      summaryEntry("Suite.Last", true)};

  EXPECT_EQ(lastLine(summaryTable(entries, 3)), "5 rows from 3 tests | 4 stable | 1 unstable");
  EXPECT_EQ(lastLine(summaryTable(entries, 1)), "5 rows from 1 test | 4 stable | 1 unstable");
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
