/**
 * @file 03_SharedMemoryOpt_BankConflictsReport_uTest.cpp
 * @brief What demo 03's bank-conflict check decides from ncu's CSV, tested
 *        with no GPU and no ncu.
 *
 * BankConflicts.CountedByNsightCompute (03_SharedMemoryOpt_BankConflicts_uTest.cpp)
 * needs a device and ncu, and is built only where the GPU demos are. The
 * reading of ncu's CSV and the verdict it applies are
 * 03_SharedMemoryOpt_BankConflicts_Check.hpp's, and these tests hold them to
 * logs as ncu prints them in every configuration. ctest runs them under the
 * check's labels, demo and ncu.
 *
 * Usage:
 *   @code{.sh}
 *   ctest --test-dir build -L ncu                   # these, and the check where it is built
 *   ./build/bin/tests/TestDemoBankConflictsReport   # these alone, by hand
 *   @endcode
 */

#include "src/bench/demo/gpu/utst/03_SharedMemoryOpt_BankConflicts_Check.hpp"

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace bc = vernier::bench::demo::bank_conflicts_check;

using bc::CONFLICT;
using bc::expectWalkthroughCounts;
using bc::kernelShortName;
using bc::METRIC_CONFLICTS;
using bc::METRIC_INSTRUCTIONS;
using bc::METRIC_WAVEFRONTS;
using bc::NAIVE;
using bc::NCU_NO_PERMISSION;
using bc::ncuLineWith;
using bc::PADDED;
using bc::parseCount;
using bc::Readings;
using bc::readReport;
using bc::Report;
using bc::splitCsvRow;

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
