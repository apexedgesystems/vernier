/**
 * @file 02_NsightProfiler_KernelColumns_uTest.cpp
 * @brief Walkthrough 19's check: each GPU column of demo 02's two kernel rows
 *        holds what its source gives it.
 *
 * KernelColumns.MatchTheirSources runs demo 02's kernel cases
 * (NsightProfiler.Kernel*: the SAXPY example's kernel at 1 and at 256 threads
 * per block) as a child with --csv and holds each row to the column's source:
 *  - kernelTimeUs: CUDA events around the measured launches, per launch. The
 *    cases declare no transfer, so it is also the row's wall median, and
 *    transferTimeUs, h2dBytes and d2hBytes are 0.
 *  - occupancy: the harness's estimate for the row's launch shape, taken from a
 *    measurement of the kernel in that shape in this process.
 *  - cupti*: CUPTI's records of the measured launches: one per launch (cycles
 *    times repeats; the warmup is outside the window), a median and a maximum
 *    register count that agree and read as one of the two forms recorded for
 *    this kernel (isRegisterForm), the kernel's static shared memory and no
 *    dynamic. In a build without CUPTI the five cells are empty and the run
 *    says so.
 *  - The cells the run states empty are empty; the NVML cells it does not state
 *    are readings; the cells these cases have nothing for (speedupVsCpu,
 *    memBandwidthGBs, the multi-GPU and unified-memory cells) are empty.
 *
 * No timing is compared, so the machine's load cannot change the outcome. The
 * check skips, saying why, only on what is decided before anything runs: no
 * CUDA device, or the check itself running inside an Nsight session, whose
 * injection would reach the demo. A failing run keeps its folder and names it.
 */

#include "src/bench/demo/gpu/utst/02_NsightProfiler_KernelColumns_Kernel.hpp"

#include "src/bench/demo/cpu/utst/12_MemcheckProfiler_Check.hpp"
#include "src/bench/demo/examples/saxpy/inc/Saxpy.hpp"
#include "src/bench/inc/PerfGpu.hpp"
#include "src/bench/inc/PerfRegistry.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/utst/ScopedEnv.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace ub = vernier::bench;
namespace ubd = vernier::bench::demo;
namespace child = vernier::bench::demo::memcheck_check;
namespace fs = std::filesystem;

using ub::test::ScopedEnv;

namespace {

/* ----------------------------- Constants ----------------------------- */

/// Measured launches per repeat, and repeats, of the child run.
constexpr int CYCLES = 4;
constexpr int REPEATS = 3;

/// Demo 02's kernel cases and the threads per block each launches with.
const std::map<std::string, int>& kernelCases() {
  static const std::map<std::string, int> CASES = {
      {"NsightProfiler.KernelOneThreadPerBlock", 1},
      {"NsightProfiler.Kernel256ThreadsPerBlock", 256},
  };
  return CASES;
}

/// The NVML cells, and those of them where 0 is not a reading.
const std::vector<std::string> NVML_CELLS = {"smClockMHz",  "throttling",   "powerDrawW",
                                             "powerLimitW", "temperatureC", "temperatureDeltaC"};
const std::set<std::string> NONZERO_NVML_CELLS = {"smClockMHz", "powerDrawW", "powerLimitW",
                                                  "temperatureC"};

/// The CUPTI cells.
const std::vector<std::string> CUPTI_CELLS = {"cuptiKernelLaunches", "cuptiRegistersMedian",
                                              "cuptiRegistersMax", "cuptiStaticSmemBytes",
                                              "cuptiDynamicSmemBytes"};

/// Cells a kernel-only, single-GPU case without unified memory has nothing for.
const std::vector<std::string> NOTHING_TO_FILL = {
    "speedupVsCpu",    "memBandwidthGBs", "multiGpuEfficiency", "p2pBandwidthGBs", "umPageFaults",
    "umH2DMigrations", "umD2HMigrations", "umMigrationTimeUs",  "umThrashing"};

/* ----------------------------- CSV ----------------------------- */

/// One CSV row: column name to cell.
using Row = std::map<std::string, std::string>;

/** @brief Splits one CSV line at its commas, keeping empty cells. */
std::vector<std::string> splitCsvLine(const std::string& line) {
  std::vector<std::string> cells;
  std::string cell;
  for (const char CH : line) {
    if (CH == ',') {
      cells.push_back(cell);
      cell.clear();
    } else {
      cell += CH;
    }
  }
  cells.push_back(cell);
  return cells;
}

/** @brief The header's column names and each row, keyed by its test name. */
struct Csv {
  std::vector<std::string> columns;
  std::map<std::string, Row> rows;
};

/** @brief Reads @p file; a row whose cell count is not the header's is skipped. */
Csv readCsv(const fs::path& file) {
  Csv csv;
  std::ifstream in(file);
  std::string line;
  if (!std::getline(in, line)) {
    return csv;
  }
  csv.columns = splitCsvLine(line);
  while (std::getline(in, line)) {
    const std::vector<std::string> CELLS = splitCsvLine(line);
    if (line.empty() || CELLS.size() != csv.columns.size()) {
      continue;
    }
    Row row;
    for (std::size_t i = 0; i < CELLS.size(); ++i) {
      row[csv.columns[i]] = CELLS[i];
    }
    csv.rows[row["test"]] = row;
  }
  return csv;
}

/** @brief A cell as a number; nullopt when it is empty or not a number. */
std::optional<double> number(const Row& row, const std::string& column) {
  const auto IT = row.find(column);
  if (IT == row.end() || IT->second.empty()) {
    return std::nullopt;
  }
  char* end = nullptr;
  const double VALUE = std::strtod(IT->second.c_str(), &end);
  if (end == IT->second.c_str() || *end != '\0') {
    return std::nullopt;
  }
  return VALUE;
}

/** @brief A cell's text; empty when the row has no such column. */
std::string cell(const Row& row, const std::string& column) {
  const auto IT = row.find(column);
  return (IT == row.end()) ? std::string() : IT->second;
}

/* ----------------------------- Statements ----------------------------- */

/** @brief One "[gpu] ... stay empty." line of the run and the columns it names. */
struct Statement {
  std::string line;
  std::vector<std::string> columns;
};

/**
 * @brief The run's statements of empty cells: lines that start "[gpu] " and end
 *        "<columns> stay empty." (or "stays empty."), the columns listed after
 *        the line's last ": " or "so its ", as "a", "a and b" or "a, b and c".
 */
std::vector<Statement> statementsIn(const std::string& output) {
  std::vector<Statement> statements;
  std::istringstream in(output);
  std::string line;
  while (std::getline(in, line)) {
    std::string tail;
    for (const char* ENDING : {" stays empty.", " stay empty."}) {
      const std::string END = ENDING;
      if (line.rfind("[gpu] ", 0) == 0 && line.size() > END.size() &&
          line.compare(line.size() - END.size(), END.size(), END) == 0) {
        tail = line.substr(0, line.size() - END.size());
        break;
      }
    }
    if (tail.empty()) {
      continue;
    }
    const std::size_t COLON = tail.rfind(": ");
    const std::size_t SO_ITS = tail.rfind("so its ");
    std::size_t from = 0;
    if (COLON != std::string::npos) {
      from = COLON + 2;
    }
    if (SO_ITS != std::string::npos && SO_ITS + 7 > from) {
      from = SO_ITS + 7;
    }
    std::string list = tail.substr(from);
    Statement statement{line, {}};
    std::size_t at = 0;
    while (at <= list.size()) {
      std::size_t next = list.find(", ", at);
      std::size_t skip = 2;
      const std::size_t AND = list.find(" and ", at);
      if (AND != std::string::npos && (next == std::string::npos || AND < next)) {
        next = AND;
        skip = 5;
      }
      statement.columns.push_back(list.substr(at, next - at));
      if (next == std::string::npos) {
        break;
      }
      at = next + skip;
    }
    statements.push_back(statement);
  }
  return statements;
}

/* ----------------------------- Sources ----------------------------- */

/** @brief @p registers rounded up to a multiple of 8. */
int roundedUpTo8(int registers) { return (registers + 7) / 8 * 8; }

/**
 * @brief Whether @p text, a register cell, is one of the two forms recorded
 *        for a kernel compiled to @p compiled registers per thread: the
 *        compiled count, or that count rounded up to a multiple of 8, written
 *        as the CSV writes an integer.
 *
 * Neither form is promised. CUDA documents numRegs as the registers each
 * thread of the function uses, and CUPTI documents registersPerThread as the
 * registers each thread executing the kernel requires; neither says how the
 * two relate. The forms are what runs of this kernel read: on the RTX 5000
 * Ada a Release build compiled to 10 and read 16, and a Debug build compiled
 * to 20 and read 20; the Jetson AGX Thor's Release build read 16 for 10. Any
 * other text fails: an empty cell, 0, a fraction, and a count between the two
 * forms or outside them. A build that reads another form fails the check
 * until a run of it qualifies that form here.
 */
bool isRegisterForm(const std::string& text, int compiled) {
  return compiled > 0 &&
         (text == std::to_string(compiled) || text == std::to_string(roundedUpTo8(compiled)));
}

/**
 * @brief The harness's occupancy estimate for @p threadsPerBlock threads per
 *        block, from one measurement of the kernel in that shape in this
 *        process; nullopt when its buffers cannot be allocated.
 *
 * The estimate depends on the launch configuration and the device, not on how
 * much the launch computes, so the kernel runs over one block's elements.
 */
std::optional<ub::OccupancyMetrics> harnessOccupancyEstimate(int threadsPerBlock) {
  float* x = nullptr;
  float* y = nullptr;
  const std::size_t BYTES = static_cast<std::size_t>(threadsPerBlock) * sizeof(float);
  if (cudaMalloc(&x, BYTES) != cudaSuccess || cudaMalloc(&y, BYTES) != cudaSuccess) {
    cudaFree(x);
    return std::nullopt;
  }
  ub::PerfConfig cfg{};
  cfg.cycles = 1;
  cfg.repeats = 1;
  cfg.warmup = 0;
  std::optional<ub::OccupancyMetrics> estimate;
  {
    ub::PerfGpuCase perf{"KernelColumns.Estimate" + std::to_string(threadsPerBlock), cfg};
    const auto LAUNCH = [x, y, threadsPerBlock](cudaStream_t s) {
      ubd::launchSaxpy(2.0F, x, y, static_cast<std::size_t>(threadsPerBlock), threadsPerBlock, s);
    };
    const ub::PerfGpuResult RESULT = perf.cudaKernel(LAUNCH, "estimate")
                                         .withLaunchConfig(dim3(1), dim3(threadsPerBlock))
                                         .measure();
    (void)ub::PerfRegistry::instance().take();
    estimate = RESULT.stats.occupancy;
  }
  cudaFree(x);
  cudaFree(y);
  return estimate;
}

} // namespace

/* ----------------------------- Fixture ----------------------------- */

/** @brief A folder of the run's own; kept and named when the check fails. */
class KernelColumns : public ::testing::Test {
protected:
  fs::path dir_;

  void TearDown() override {
    if (dir_.empty()) {
      return;
    }
    if (HasFailure()) {
      std::fprintf(stderr, "[KernelColumns] the run's files are kept in %s\n", dir_.c_str());
      return;
    }
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }
};

/* ----------------------------- Check ----------------------------- */

/**
 * @test Demo 02's two kernel rows: every GPU cell is what its source gives it,
 *       or empty and stated
 */
TEST_F(KernelColumns, MatchTheirSources) {
  int devices = 0;
  const cudaError_t COUNTED = cudaGetDeviceCount(&devices);
  if (COUNTED != cudaSuccess || devices == 0) {
    GTEST_SKIP() << "no CUDA device: "
                 << (COUNTED != cudaSuccess ? cudaGetErrorString(COUNTED) : "none found");
  }
  const std::string SESSION = ub::profiler_env::nsightSessionTool();
  if (!SESSION.empty()) {
    GTEST_SKIP() << "this check runs inside an Nsight session (" << SESSION
                 << "), whose injection would reach the demo and stand its CUPTI collector down";
  }
  ub::ensureBenchGpuAbi();

  // The demo decides on CUPTI from its environment: run it without the
  // override and without a runner's wrap, as a plain run is.
  const ScopedEnv NO_OVERRIDE("VERNIER_DISABLE_CUPTI", nullptr);
  const ScopedEnv NO_WRAP("VERNIER_EXTERNAL_WRAP", nullptr);

  cudaFuncAttributes attributes{};
  ASSERT_EQ(ubd::kernel_columns::saxpyKernelAttributes(attributes), cudaSuccess);

  dir_ = fs::temp_directory_path() / ("vernier_kernel_columns_" + std::to_string(::getpid()));
  fs::remove_all(dir_);
  fs::create_directories(dir_);
  const fs::path CSV_FILE = dir_ / "kernel_columns.csv";
  const fs::path LOG_FILE = dir_ / "run.log";
  const child::ChildExit END = child::runLogged(
      {VERNIER_DEMO_02_BINARY, "--gtest_filter=NsightProfiler.Kernel*", "--cycles",
       std::to_string(CYCLES), "--repeats", std::to_string(REPEATS), "--csv", CSV_FILE.string()},
      LOG_FILE);
  const std::string OUTPUT = child::readText(LOG_FILE);
  ASSERT_TRUE(child::exitedWith(END, 0)) << "demo 02 " << child::describe(END) << "; it said:\n"
                                         << child::lastLines(OUTPUT, 30);

  const Csv CSV = readCsv(CSV_FILE);
  const std::vector<Statement> STATEMENTS = statementsIn(OUTPUT);
  const std::set<std::string> COLUMNS(CSV.columns.begin(), CSV.columns.end());
  for (const Statement& statement : STATEMENTS) {
    for (const std::string& column : statement.columns) {
      EXPECT_EQ(COLUMNS.count(column), 1U)
          << "the run states a column the CSV has not: '" << column << "' in\n"
          << statement.line;
    }
  }

  for (const auto& kernelCase : kernelCases()) {
    const std::string& testName = kernelCase.first;
    const int THREADS = kernelCase.second;
    SCOPED_TRACE(testName);
    const auto FOUND = CSV.rows.find(testName);
    ASSERT_NE(FOUND, CSV.rows.end()) << "no row; the run said:\n" << child::lastLines(OUTPUT, 30);
    const Row& row = FOUND->second;

    // What the run states empty for this row: its process-wide statements and
    // those that name this test.
    std::set<std::string> stated;
    for (const Statement& statement : STATEMENTS) {
      bool namesAnotherCase = false;
      for (const auto& other : kernelCases()) {
        namesAnotherCase =
            namesAnotherCase ||
            (other.first != testName && statement.line.find(other.first) != std::string::npos);
      }
      if (!namesAnotherCase) {
        stated.insert(statement.columns.begin(), statement.columns.end());
      }
    }

    // The run's shape.
    EXPECT_EQ(cell(row, "cycles"), std::to_string(CYCLES));
    EXPECT_EQ(cell(row, "repeats"), std::to_string(REPEATS));

    // CUDA events: the per-launch kernel time is the wall median of a case
    // without transfers, to the precision the CSV prints.
    const std::optional<double> KERNEL_US = number(row, "kernelTimeUs");
    const std::optional<double> WALL_US = number(row, "wallMedian");
    ASSERT_TRUE(KERNEL_US.has_value() && WALL_US.has_value())
        << "kernelTimeUs '" << cell(row, "kernelTimeUs") << "', wallMedian '"
        << cell(row, "wallMedian") << "'";
    EXPECT_GT(*KERNEL_US, 0.0);
    EXPECT_NEAR(*KERNEL_US, *WALL_US, 1e-5 * *WALL_US);
    EXPECT_EQ(number(row, "transferTimeUs"), 0.0);
    EXPECT_EQ(cell(row, "h2dBytes"), "0");
    EXPECT_EQ(cell(row, "d2hBytes"), "0");

    // The occupancy estimate for the row's launch shape.
    const std::optional<ub::OccupancyMetrics> ESTIMATE = harnessOccupancyEstimate(THREADS);
    ASSERT_TRUE(ESTIMATE.has_value()) << "no device buffers for the estimate's measurement";
    ASSERT_EQ(ESTIMATE->blockSize, THREADS) << "the harness estimated another block than declared";
    ASSERT_TRUE(number(row, "occupancy").has_value())
        << "occupancy '" << cell(row, "occupancy") << "'";
    EXPECT_NEAR(*number(row, "occupancy"), ESTIMATE->achievedOccupancy, 1e-6);

    // CUPTI's records of the measured launches.
    if (VERNIER_DEMO_02_HAS_CUPTI != 0) {
      const int COMPILED = attributes.numRegs;
      EXPECT_EQ(cell(row, "cuptiKernelLaunches"), std::to_string(CYCLES * REPEATS));
      for (const char* COLUMN : {"cuptiRegistersMedian", "cuptiRegistersMax"}) {
        EXPECT_TRUE(isRegisterForm(cell(row, COLUMN), COMPILED))
            << COLUMN << " is '" << cell(row, COLUMN) << "'; compiled to " << COMPILED
            << " registers, the kernel reads " << COMPILED << " or " << roundedUpTo8(COMPILED);
      }
      EXPECT_EQ(cell(row, "cuptiRegistersMax"), cell(row, "cuptiRegistersMedian"))
          << "the measured launches are one kernel in one shape";
      EXPECT_EQ(cell(row, "cuptiStaticSmemBytes"), std::to_string(attributes.sharedSizeBytes));
      EXPECT_EQ(cell(row, "cuptiDynamicSmemBytes"), "0");
    } else {
      for (const std::string& column : CUPTI_CELLS) {
        EXPECT_EQ(stated.count(column), 1U) << column << " is not stated empty by a build without "
                                            << "CUPTI; the run said:\n"
                                            << child::lastLines(OUTPUT, 30);
      }
    }

    // Stated empty is empty.
    for (const std::string& column : stated) {
      EXPECT_EQ(cell(row, column), "") << column << " is stated empty and holds a value";
    }

    // An NVML cell the run does not state empty is a reading.
    for (const std::string& column : NVML_CELLS) {
      if (stated.count(column) != 0) {
        continue;
      }
      const std::optional<double> VALUE = number(row, column);
      EXPECT_TRUE(VALUE.has_value())
          << column << " is '" << cell(row, column) << "' and no statement of the run names it";
      if (VALUE.has_value() && NONZERO_NVML_CELLS.count(column) != 0) {
        EXPECT_NE(*VALUE, 0.0) << column << " is 0, which is not a reading";
      }
    }

    // Cells these cases have nothing for.
    for (const std::string& column : NOTHING_TO_FILL) {
      EXPECT_EQ(cell(row, column), "") << column;
    }
  }
}
