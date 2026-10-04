/**
 * @file 03_SharedMemoryOpt_BankConflicts_uTest.cpp
 * @brief The check behind walkthrough 12: Nsight Compute counts the bank
 *        conflicts of demo 03's tiled transpose, and none in the padded one.
 *
 * Not part of demo 03, whose source shows only what it teaches. This program
 * runs the demo binary (BenchDemo_Gpu_03_SharedMemoryOpt, whose path the
 * build passes in) under ncu with the three metrics the walkthrough reads
 * and holds every launch of every kernel to what the walkthrough says: the
 * naive kernel loads nothing from shared memory; the tiled kernels execute
 * one shared-load instruction per warp; the conflicting tile's reads take
 * 32 wavefronts per instruction and the padded tile's one, and ncu's
 * bank-conflict counter reads accordingly. The process plumbing is
 * walkthrough 15's (12_MemcheckProfiler_Check.hpp); the reading of ncu's
 * CSV and the verdict on its readings are here, each with tests that need
 * no GPU. ctest runs it under the demo and ncu labels.
 *
 * Usage:
 *   @code{.sh}
 *   ctest --test-dir build -L ncu             # root where the counters are administrators' only
 *   ./build/bin/tests/TestDemoBankConflicts   # the same, by hand
 *   @endcode
 */

#include "src/bench/demo/cpu/utst/12_MemcheckProfiler_Check.hpp"
#include "src/bench/demo/gpu/03_SharedMemoryOpt_Workload.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <filesystem>
#include <map>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace check = vernier::bench::demo::memcheck_check;
namespace sm = vernier::bench::demo::shared_memory_demo;
namespace fs = std::filesystem;

namespace {

/* ----------------------------- Constants ----------------------------- */

/// The demo binary the check runs; the build passes its path.
constexpr const char* DEMO_BINARY = VERNIER_DEMO_GPU_03_BINARY;

/// The three metrics, as the walkthrough reads them: shared-load
/// instructions per launch, the wavefronts those instructions took at the
/// SASS level, and ncu's headline bank-conflict counter for shared loads.
constexpr const char* METRIC_INSTRUCTIONS = "smsp__inst_executed_op_shared_ld.sum";
constexpr const char* METRIC_WAVEFRONTS =
    "smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ld.sum";
constexpr const char* METRIC_CONFLICTS = "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum";

/// The kernels, as ncu names them once their namespace and parameters are
/// stripped.
constexpr const char* NAIVE = "transposeNaive";
constexpr const char* CONFLICT = "transposeSharedConflict";
constexpr const char* PADDED = "transposeSharedPadded";

/// The lanes of a warp; the tile's row is one warp wide.
constexpr long WARP_SIZE = 32;

/// One shared-load instruction per warp per launch: every warp reads one
/// column of its block's tile, and the demo's matrix is a grid of tiles.
constexpr long WARPS_PER_LAUNCH = (static_cast<long>(sm::MATRIX_DIM) / sm::TILE_DIM) *
                                  (static_cast<long>(sm::MATRIX_DIM) / sm::TILE_DIM) *
                                  (static_cast<long>(sm::TILE_DIM) * sm::TILE_DIM / WARP_SIZE);

/// A column read of a TILE_DIM-wide tile lands every lane in one bank, so
/// the instruction is served once per lane: TILE_DIM wavefronts, one per row
/// of the tile, where the padded tile's read takes one.
constexpr long CONFLICTED_WAVEFRONTS_PER_LOAD = sm::TILE_DIM;

/// The floor for ncu's bank-conflict counter on the conflicted kernel: the
/// conflicts of one load are its wavefronts beyond the first, TILE_DIM - 1,
/// and the counter is allowed a little under that (the reference rig reads
/// 31.03 per load).
constexpr long MIN_CONFLICTS_PER_CONFLICTED_LOAD = sm::TILE_DIM - 2;

/// The ceiling for the counter on the padded kernel, as a share of the
/// conflicted kernel's smallest reading: the reference rig reads 0.04%, a
/// discrete GPU 0.4%.
constexpr double MAX_PADDED_CONFLICT_SHARE = 0.02;

/// ncu's words when the user may not read the GPU's performance counters,
/// and the value it prints for a metric it has no reading of on this GPU
/// or in this version (it prints no error for one and exits 0).
constexpr const char* NCU_NO_PERMISSION = "ERR_NVGPUCTRPERM";
constexpr const char* NCU_NO_VALUE = "n/a";

/* ----------------------------- Reading ncu's CSV ----------------------------- */

/// One launch's three readings; -1 where the report has none.
struct LaunchMetrics {
  long instructions = -1;
  long wavefronts = -1;
  long conflicts = -1;
};

/// Every launch's readings, by the kernel's short name, in launch order.
using Readings = std::map<std::string, std::map<long, LaunchMetrics>>;

/// What ncu's CSV log holds: the readings, and the first row in which ncu
/// had no value for one of the three metrics, as it printed it.
struct Report {
  Readings readings;
  std::string unavailableRow;
};

/// The fields of one CSV row as ncu writes it: every field quoted, a quote
/// inside a field doubled, commas inside quotes part of the field.
std::vector<std::string> splitCsvRow(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char CH = line[i];
    if (quoted) {
      if (CH == '"' && i + 1 < line.size() && line[i + 1] == '"') {
        field += '"';
        ++i;
      } else if (CH == '"') {
        quoted = false;
      } else {
        field += CH;
      }
    } else if (CH == '"') {
      quoted = true;
    } else if (CH == ',') {
      fields.push_back(field);
      field.clear();
    } else {
      field += CH;
    }
  }
  fields.push_back(field);
  return fields;
}

/// A count as ncu prints it, with or without thousands separators; -1 when
/// the text is not a whole number.
long parseCount(const std::string& text) {
  long value = 0;
  bool any = false;
  for (const char CH : text) {
    if (CH >= '0' && CH <= '9') {
      value = value * 10 + (CH - '0');
      any = true;
    } else if (CH != ',') {
      return -1;
    }
  }
  return any ? value : -1;
}

/// A kernel's name without its namespaces and parameter list:
/// "<unnamed>::transposeNaive(const float *, float *, int)" is "transposeNaive".
std::string kernelShortName(const std::string& kernelName) {
  const std::size_t PAREN = kernelName.find('(');
  const std::string QUALIFIED = kernelName.substr(0, PAREN);
  const std::size_t SCOPE = QUALIFIED.rfind("::");
  return SCOPE == std::string::npos ? QUALIFIED : QUALIFIED.substr(SCOPE + 2);
}

/// The readings in ncu's CSV log: the header names the columns, the
/// "==PROF==" lines above it and any metric the check does not read are
/// ignored, and each row files one metric of one launch under its kernel.
Report readReport(const std::string& csv) {
  Report report;
  Readings& readings = report.readings;
  std::istringstream in(csv);
  std::string line;
  long idCol = -1;
  long kernelCol = -1;
  long metricCol = -1;
  long valueCol = -1;
  while (std::getline(in, line)) {
    if (line.rfind("==", 0) == 0 || line.empty()) {
      continue;
    }
    const std::vector<std::string> FIELDS = splitCsvRow(line);
    if (idCol < 0) {
      for (std::size_t i = 0; i < FIELDS.size(); ++i) {
        if (FIELDS[i] == "ID") {
          idCol = static_cast<long>(i);
        } else if (FIELDS[i] == "Kernel Name") {
          kernelCol = static_cast<long>(i);
        } else if (FIELDS[i] == "Metric Name") {
          metricCol = static_cast<long>(i);
        } else if (FIELDS[i] == "Metric Value") {
          valueCol = static_cast<long>(i);
        }
      }
      if (idCol < 0 || kernelCol < 0 || metricCol < 0 || valueCol < 0) {
        return report; // not ncu's CSV
      }
      continue;
    }
    const std::size_t NEEDED =
        static_cast<std::size_t>(std::max({idCol, kernelCol, metricCol, valueCol}));
    if (FIELDS.size() <= NEEDED) {
      continue;
    }
    const long ID = parseCount(FIELDS[static_cast<std::size_t>(idCol)]);
    if (ID < 0) {
      continue;
    }
    const std::string& METRIC = FIELDS[static_cast<std::size_t>(metricCol)];
    const bool READ =
        METRIC == METRIC_INSTRUCTIONS || METRIC == METRIC_WAVEFRONTS || METRIC == METRIC_CONFLICTS;
    if (!READ) {
      continue;
    }
    const std::string& TEXT = FIELDS[static_cast<std::size_t>(valueCol)];
    if (TEXT == NCU_NO_VALUE && report.unavailableRow.empty()) {
      report.unavailableRow = line;
    }
    LaunchMetrics& launch =
        readings[kernelShortName(FIELDS[static_cast<std::size_t>(kernelCol)])][ID];
    const long VALUE = parseCount(TEXT);
    if (METRIC == METRIC_INSTRUCTIONS) {
      launch.instructions = VALUE;
    } else if (METRIC == METRIC_WAVEFRONTS) {
      launch.wavefronts = VALUE;
    } else {
      launch.conflicts = VALUE;
    }
  }
  return report;
}

/// The first line of @p text that contains @p marker, as ncu printed it;
/// empty when none does. A skip quotes this line, so what it rests on is
/// ncu's text, not the check's.
std::string ncuLineWith(const std::string& text, const char* marker) {
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    if (line.find(marker) != std::string::npos) {
      return line;
    }
  }
  return "";
}

/// One kernel's launches, in one line, for the check's output: "no launch"
/// where the readings name none.
std::string describeLaunches(const Readings& readings, const char* kernel) {
  const auto FOUND = readings.find(kernel);
  if (FOUND == readings.end() || FOUND->second.empty()) {
    return "no launch";
  }
  std::string out;
  for (const auto& entry : FOUND->second) {
    const LaunchMetrics& m = entry.second;
    out += (out.empty() ? "" : "; ") + std::to_string(m.instructions) + " loads, " +
           std::to_string(m.wavefronts) + " wavefronts, " + std::to_string(m.conflicts) +
           " conflicts";
  }
  return out;
}

/* ----------------------------- The Verdict ----------------------------- */

/**
 * @brief Holds every launch in @p readings to the counts walkthrough 12
 *        gives, each departure a failure of the running test.
 *
 * Every kernel has a launch; the naive kernel executes no shared-load
 * instruction and reports no wavefront and no conflict; each tiled kernel
 * executes one shared-load instruction per warp of the launch; the
 * conflicting kernel's SASS-level wavefronts are exactly TILE_DIM times its
 * instructions and its L1TEX counter at least TILE_DIM - 2 times them; the
 * padded kernel's wavefronts equal its instructions, and its counter is
 * read and stays below MAX_PADDED_CONFLICT_SHARE of the conflicting
 * kernel's smallest reading.
 *
 * A reading the report lacks, or gives as other than a whole number, is -1
 * (readReport). Each exact count and the floor fail it; the padded kernel's
 * ceiling would pass it, so that counter must be read before the ceiling
 * applies.
 */
void expectWalkthroughCounts(const Readings& readings) {
  for (const char* kernel : {NAIVE, CONFLICT, PADDED}) {
    ASSERT_TRUE(readings.count(kernel) != 0 && !readings.at(kernel).empty())
        << "ncu's report names no launch of " << kernel;
  }

  for (const auto& [id, m] : readings.at(NAIVE)) {
    EXPECT_EQ(m.instructions, 0) << NAIVE << " launch " << id << " loads from shared memory";
    EXPECT_EQ(m.wavefronts, 0) << NAIVE << " launch " << id;
    EXPECT_EQ(m.conflicts, 0) << NAIVE << " launch " << id << " reports bank conflicts";
  }

  long fewestConflicts = -1;
  for (const auto& [id, m] : readings.at(CONFLICT)) {
    EXPECT_EQ(m.instructions, WARPS_PER_LAUNCH)
        << CONFLICT << " launch " << id << " does not load one tile column per warp";
    EXPECT_EQ(m.wavefronts, CONFLICTED_WAVEFRONTS_PER_LOAD * m.instructions)
        << CONFLICT << " launch " << id << ": its column reads no longer take " << sm::TILE_DIM
        << " wavefronts each, so the tile no longer conflicts";
    EXPECT_GE(m.conflicts, MIN_CONFLICTS_PER_CONFLICTED_LOAD * m.instructions)
        << CONFLICT << " launch " << id << ": ncu counts fewer bank conflicts than a "
        << sm::TILE_DIM << "-way conflict makes";
    if (fewestConflicts < 0 || m.conflicts < fewestConflicts) {
      fewestConflicts = m.conflicts;
    }
  }

  for (const auto& [id, m] : readings.at(PADDED)) {
    EXPECT_EQ(m.instructions, WARPS_PER_LAUNCH)
        << PADDED << " launch " << id << " does not load one tile column per warp";
    EXPECT_EQ(m.wavefronts, m.instructions)
        << PADDED << " launch " << id << ": its column reads take more than one wavefront "
        << "each, so the padding no longer spreads them over the banks";
    if (m.conflicts < 0) {
      ADD_FAILURE() << PADDED << " launch " << id << ": ncu's report has no count of "
                    << METRIC_CONFLICTS << " for it (no row, or a value that is not a whole "
                    << "number), so nothing shows the padding took the conflicts away";
      continue;
    }
    EXPECT_LT(static_cast<double>(m.conflicts),
              MAX_PADDED_CONFLICT_SHARE * static_cast<double>(fewestConflicts))
        << PADDED << " launch " << id << ": ncu's counter is not far below the conflicting "
        << "kernel's " << fewestConflicts;
  }
}

} // namespace

/* ----------------------------- Tests ----------------------------- */

/**
 * @test Nsight Compute counts 32 wavefronts per shared-load instruction in
 *       the conflicting tile and one in the padded tile, and its
 *       bank-conflict counter reads accordingly.
 *
 * Runs the demo binary under ncu with the three metrics on the fewest
 * launches the harness makes (one warmup and one measured launch per kernel,
 * after the harness's own first), reads ncu's CSV, and checks every launch
 * (expectWalkthroughCounts): the naive kernel executes no shared-load
 * instruction and reports no conflict; the tiled kernels execute one
 * shared-load instruction per warp of the launch; the conflicting kernel's
 * SASS-level wavefronts are exactly TILE_DIM times its instructions and its
 * L1TEX counter at least TILE_DIM - 2 times them; the padded kernel's
 * wavefronts equal its instructions and its counter, which the report must
 * give as a whole number, stays below 2% of the conflicting kernel's
 * smallest reading. A kernel the report names no launch of fails the check.
 *
 * Skipped where ncu is not on PATH or no CUDA device is present, decided
 * before anything runs; where ncu refuses the counters to this user,
 * quoting its line; and where ncu prints n/a for one of the metrics on this
 * GPU or in this version, quoting the row, once the demo's tests have run to
 * their end under it. The demo's three tests must pass under ncu and ncu
 * must exit 0; anything else fails with the run's last lines.
 */
TEST(BankConflicts, CountedByNsightCompute) {
  if (!vernier::bench::profiler_env::isOnPath("ncu")) {
    GTEST_SKIP() << "ncu is not on PATH; this test runs the demo under Nsight Compute";
  }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP() << "no CUDA device available; this test runs the demo's kernels under ncu";
  }

  std::error_code found;
  const std::string DEMO = fs::canonical(DEMO_BINARY, found).string();
  ASSERT_FALSE(found) << "the demo binary is missing: " << DEMO_BINARY;

  std::string dirTemplate = (fs::temp_directory_path() / "vernier-demo-gpu03-XXXXXX").string();
  ASSERT_NE(::mkdtemp(dirTemplate.data()), nullptr) << "cannot create a temporary directory";
  const fs::path DIR = dirTemplate;
  const fs::path CSV = DIR / "metrics.csv";
  const fs::path OUTPUT = DIR / "program.txt";

  // ncu replays every launch, so the demo makes the fewest the harness
  // allows; its metrics go to the log file, the demo's output to another.
  const std::vector<std::string> ARGS = {"ncu",
                                         "--target-processes",
                                         "all",
                                         "--metrics",
                                         std::string(METRIC_INSTRUCTIONS) + "," +
                                             METRIC_WAVEFRONTS + "," + METRIC_CONFLICTS,
                                         "--csv",
                                         "--log-file",
                                         CSV.string(),
                                         DEMO,
                                         "--gtest_filter=SharedMemoryOpt.*",
                                         "--gtest_print_time=0",
                                         "--gpu-warmup",
                                         "1",
                                         "--cycles",
                                         "1",
                                         "--repeats",
                                         "1"};
  const check::ChildExit END = check::runLogged(ARGS, OUTPUT);
  const std::string PROGRAM = check::readText(OUTPUT);
  const std::string LOG = check::readText(CSV);

  const std::string REFUSED = ncuLineWith(LOG + PROGRAM, NCU_NO_PERMISSION);
  if (!REFUSED.empty()) {
    std::error_code ec;
    fs::remove_all(DIR, ec);
    GTEST_SKIP() << "ncu may not read the GPU's performance counters as this user (on a Jetson "
                    "they are for administrators: run this test as root). It printed:\n"
                 << REFUSED;
  }

  ASSERT_TRUE(check::testsStarted(PROGRAM))
      << "the demo did not start under ncu: ncu " << check::describe(END) << ". The run printed:\n"
      << check::lastLines(PROGRAM + LOG, 40);
  ASSERT_NE(PROGRAM.find("[  PASSED  ] 3 tests."), std::string::npos)
      << "the demo's three tests did not all pass under ncu (its output is in " << DIR << "):\n"
      << check::lastLines(PROGRAM, 40);
  ASSERT_TRUE(check::exitedWith(END, 0))
      << "ncu " << check::describe(END) << " (its log is " << CSV << "):\n"
      << check::lastLines(LOG + PROGRAM, 20);

  const Report REPORT = readReport(LOG);
  if (!REPORT.unavailableRow.empty()) {
    std::error_code ec;
    fs::remove_all(DIR, ec);
    GTEST_SKIP() << "ncu has no value for a metric the check reads, on this GPU or in this "
                    "version. Its row:\n"
                 << REPORT.unavailableRow;
  }
  const Readings& READINGS = REPORT.readings;
  std::printf("[BankConflicts.CountedByNsightCompute]  per launch: naive %s | conflicting %s | "
              "padded %s\n",
              describeLaunches(READINGS, NAIVE).c_str(),
              describeLaunches(READINGS, CONFLICT).c_str(),
              describeLaunches(READINGS, PADDED).c_str());
  expectWalkthroughCounts(READINGS);

  if (HasFailure()) {
    std::printf("ncu output kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
}

/* ----------------------------- Reading Tests ----------------------------- */

// What CountedByNsightCompute's reading rests on, on lines taken from real
// runs on the reference rig (ncu 2025.3.1, which prints thousands
// separators) and the laptop's container (ncu 2026.2.0, which does not).

namespace {

/// ncu's CSV header, as both versions print it.
constexpr const char* HEADER =
    "\"ID\",\"Process ID\",\"Process Name\",\"Host Name\",\"Kernel Name\",\"Context\","
    "\"Stream\",\"Block Size\",\"Grid Size\",\"Device\",\"CC\",\"Section Name\","
    "\"Metric Name\",\"Metric Unit\",\"Metric Value\"";

/// One row of ncu's CSV for @p kernel's launch @p id.
std::string row(long id, const char* kernel, const char* metric, const char* value) {
  return "\"" + std::to_string(id) + "\",\"50672\",\"BenchDemo_Gpu_03_SharedMemoryOpt\"," +
         "\"127.0.0.1\",\"<unnamed>::" + kernel +
         "(const float *, float *, int)\",\"1\",\"14\",\"(32, 32, 1)\",\"(32, 32, 1)\","
         "\"0\",\"11.0\",\"Command line profiler metrics\",\"" +
         metric + "\",\"\",\"" + value + "\"\n";
}

} // namespace

/** @test A row's quoted fields split on the commas between them, not inside them */
TEST(NcuCsvTest, SplitsQuotedFieldsWithCommasInside) {
  const std::string LINE = row(3, CONFLICT, METRIC_CONFLICTS, "1,016,888");
  const std::vector<std::string> FIELDS = splitCsvRow(LINE.substr(0, LINE.size() - 1));

  ASSERT_EQ(FIELDS.size(), 15U);
  EXPECT_EQ(FIELDS[4], "<unnamed>::transposeSharedConflict(const float *, float *, int)");
  EXPECT_EQ(FIELDS[7], "(32, 32, 1)");
  EXPECT_EQ(FIELDS[14], "1,016,888");
}

/** @test A count reads with or without thousands separators, and only a count reads */
TEST(NcuCsvTest, ParsesCountsWithAndWithoutSeparators) {
  EXPECT_EQ(parseCount("1,016,888"), 1016888);
  EXPECT_EQ(parseCount("1017299"), 1017299);
  EXPECT_EQ(parseCount("0"), 0);
  EXPECT_EQ(parseCount(""), -1);
  EXPECT_EQ(parseCount("n/a"), -1);
  EXPECT_EQ(parseCount("65,536.00"), -1);
}

/** @test A kernel's short name drops its namespaces and parameters, whatever they are */
TEST(NcuCsvTest, KernelShortNameDropsScopeAndParameters) {
  EXPECT_EQ(kernelShortName("<unnamed>::transposeNaive(const float *, float *, int)"),
            "transposeNaive");
  EXPECT_EQ(kernelShortName("vernier::bench::demo::shared_memory_demo::transposeSharedPadded("
                            "const float *, float *, int)"),
            "transposeSharedPadded");
  EXPECT_EQ(kernelShortName("saxpyKernel(float, const float *, float *, unsigned long)"),
            "saxpyKernel");
  EXPECT_EQ(kernelShortName("plain"), "plain");
}

/** @test Readings file each metric of each launch under its kernel, skipping ncu's own lines */
TEST(NcuCsvTest, ReadsEveryLaunchOfEveryKernel) {
  const std::string CSV =
      "==PROF== Connected to process 50672 (/b/bin/ptests/BenchDemo_Gpu_03_SharedMemoryOpt)\n"
      "==PROF== Disconnected from process 50672\n" +
      std::string(HEADER) + "\n" + row(0, NAIVE, METRIC_CONFLICTS, "0") +
      row(0, NAIVE, METRIC_INSTRUCTIONS, "0") + row(0, NAIVE, METRIC_WAVEFRONTS, "0") +
      row(3, CONFLICT, METRIC_CONFLICTS, "1,016,888") +
      row(3, CONFLICT, METRIC_INSTRUCTIONS, "32,768") +
      row(3, CONFLICT, METRIC_WAVEFRONTS, "1,048,576") +
      row(3, CONFLICT, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", "5,809") +
      row(4, CONFLICT, METRIC_CONFLICTS, "1016904") +
      row(4, CONFLICT, METRIC_INSTRUCTIONS, "32768") +
      row(4, CONFLICT, METRIC_WAVEFRONTS, "1048576") + row(6, PADDED, METRIC_CONFLICTS, "394") +
      row(6, PADDED, METRIC_INSTRUCTIONS, "32,768") + row(6, PADDED, METRIC_WAVEFRONTS, "32,768");

  const Report REPORT = readReport(CSV);
  const Readings& READINGS = REPORT.readings;

  EXPECT_TRUE(REPORT.unavailableRow.empty());
  ASSERT_EQ(READINGS.size(), 3U);
  ASSERT_EQ(READINGS.at(NAIVE).size(), 1U);
  ASSERT_EQ(READINGS.at(CONFLICT).size(), 2U);
  ASSERT_EQ(READINGS.at(PADDED).size(), 1U);
  EXPECT_EQ(READINGS.at(NAIVE).at(0).instructions, 0);
  EXPECT_EQ(READINGS.at(CONFLICT).at(3).conflicts, 1016888);
  EXPECT_EQ(READINGS.at(CONFLICT).at(3).instructions, 32768);
  EXPECT_EQ(READINGS.at(CONFLICT).at(3).wavefronts, 1048576);
  EXPECT_EQ(READINGS.at(CONFLICT).at(4).conflicts, 1016904);
  EXPECT_EQ(READINGS.at(PADDED).at(6).conflicts, 394);
  EXPECT_EQ(READINGS.at(PADDED).at(6).wavefronts, 32768);
}

/** @test A launch a metric is missing for reads -1 there, and a foreign CSV reads nothing */
TEST(NcuCsvTest, MissingMetricsReadAsAbsent) {
  const Report PARTIAL =
      readReport(std::string(HEADER) + "\n" + row(6, PADDED, METRIC_CONFLICTS, "394"));
  ASSERT_EQ(PARTIAL.readings.at(PADDED).size(), 1U);
  EXPECT_EQ(PARTIAL.readings.at(PADDED).at(6).conflicts, 394);
  EXPECT_EQ(PARTIAL.readings.at(PADDED).at(6).instructions, -1);
  EXPECT_EQ(PARTIAL.readings.at(PADDED).at(6).wavefronts, -1);
  EXPECT_TRUE(PARTIAL.unavailableRow.empty());

  EXPECT_TRUE(readReport("test,wallMedian,wallCV\nSharedMemoryOpt.SharedPadded,40.58,0.0004\n")
                  .readings.empty());
  EXPECT_TRUE(readReport("").readings.empty());
}

/** @test The row ncu prints n/a in, for one of the three metrics, is kept as printed */
TEST(NcuCsvTest, KeepsTheRowOfAMetricWithoutAValue) {
  const std::string NA_ROW = row(6, PADDED, METRIC_WAVEFRONTS, "n/a");
  const Report REPORT =
      readReport(std::string(HEADER) + "\n" + row(6, PADDED, METRIC_CONFLICTS, "394") + NA_ROW +
                 row(6, PADDED, "some__other_metric.sum", "n/a"));

  EXPECT_EQ(REPORT.unavailableRow, NA_ROW.substr(0, NA_ROW.size() - 1));
  EXPECT_EQ(REPORT.readings.at(PADDED).at(6).wavefronts, -1);
  EXPECT_EQ(REPORT.readings.at(PADDED).at(6).conflicts, 394);

  // A metric the check does not read is not the check's concern.
  EXPECT_TRUE(
      readReport(std::string(HEADER) + "\n" + row(6, PADDED, "some__other_metric.sum", "n/a"))
          .unavailableRow.empty());
}

/** @test ncu's refusal line is found as printed, and only when printed */
TEST(NcuCsvTest, FindsTheRefusalAsPrinted) {
  const std::string REFUSAL =
      "==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU "
      "Performance Counters on the target device 0. For instructions on enabling permissions "
      "and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM";
  const std::string LOG = "==PROF== Connected to process 50727 (/b/BenchDemo_Gpu_03)\n" + REFUSAL +
                          "\n==PROF== Disconnected from process 50727\n";

  EXPECT_EQ(ncuLineWith(LOG, NCU_NO_PERMISSION), REFUSAL);
  EXPECT_TRUE(ncuLineWith(std::string(HEADER) + "\n", NCU_NO_PERMISSION).empty());
}

/* ----------------------------- Verdict Tests ----------------------------- */

// What CountedByNsightCompute decides once it has read ncu's CSV, with no GPU
// and no ncu: expectWalkthroughCounts on the check's nine launches as one run
// on the reference rig read them (ncu 2025.3.1, as root), with the padded
// kernel's last bank-conflict reading left out, malformed or zero.

namespace {

/// ncu's CSV for the check's three launches of each kernel: every reading as
/// that run printed it but the padded kernel's last bank-conflict reading,
/// which is @p paddedConflicts; nullptr leaves its row out.
std::string reportWithPaddedConflicts(const char* paddedConflicts) {
  std::string csv = std::string(HEADER) + "\n";
  long id = 0; // ncu numbers the launches in order: three of each kernel
  for (; id < 3; ++id) {
    csv += row(id, NAIVE, METRIC_CONFLICTS, "0") + row(id, NAIVE, METRIC_INSTRUCTIONS, "0") +
           row(id, NAIVE, METRIC_WAVEFRONTS, "0");
  }
  for (const char* conflicts : {"1,016,843", "1,016,845", "1,016,890"}) {
    csv += row(id, CONFLICT, METRIC_CONFLICTS, conflicts) +
           row(id, CONFLICT, METRIC_INSTRUCTIONS, "32,768") +
           row(id, CONFLICT, METRIC_WAVEFRONTS, "1,048,576");
    ++id;
  }
  for (const char* conflicts : {"337", "380", paddedConflicts}) {
    if (conflicts != nullptr) {
      csv += row(id, PADDED, METRIC_CONFLICTS, conflicts);
    }
    csv += row(id, PADDED, METRIC_INSTRUCTIONS, "32,768") +
           row(id, PADDED, METRIC_WAVEFRONTS, "32,768");
    ++id;
  }
  return csv;
}

/// The verdict's words when the padded kernel's last launch has no count.
std::string noCountForTheLastPaddedLaunch() {
  return std::string(PADDED) + " launch 8: ncu's report has no count of " + METRIC_CONFLICTS +
         " for it";
}

} // namespace

/** @test A padded launch whose bank-conflict row the report leaves out fails the verdict */
TEST(BankConflictsVerdictTest, FailsAPaddedLaunchWithoutItsConflictRow) {
  const Report REPORT = readReport(reportWithPaddedConflicts(nullptr));
  ASSERT_TRUE(REPORT.unavailableRow.empty()); // nothing the check would skip on
  ASSERT_EQ(REPORT.readings.at(PADDED).at(8).conflicts, -1);

  EXPECT_NONFATAL_FAILURE(expectWalkthroughCounts(REPORT.readings),
                          noCountForTheLastPaddedLaunch());
}

/** @test A padded bank-conflict reading that is not a whole number fails the verdict */
TEST(BankConflictsVerdictTest, FailsAPaddedConflictReadingThatIsNotACount) {
  const Report REPORT = readReport(reportWithPaddedConflicts("376.00"));
  ASSERT_TRUE(REPORT.unavailableRow.empty());
  ASSERT_EQ(REPORT.readings.at(PADDED).at(8).conflicts, -1);

  EXPECT_NONFATAL_FAILURE(expectWalkthroughCounts(REPORT.readings),
                          noCountForTheLastPaddedLaunch());
}

/** @test A padded bank-conflict count of zero is a reading, and passes the verdict */
TEST(BankConflictsVerdictTest, PassesAPaddedConflictCountOfZero) {
  const Report REPORT = readReport(reportWithPaddedConflicts("0"));
  ASSERT_TRUE(REPORT.unavailableRow.empty());
  ASSERT_EQ(REPORT.readings.at(PADDED).at(8).conflicts, 0);

  expectWalkthroughCounts(REPORT.readings);
}
