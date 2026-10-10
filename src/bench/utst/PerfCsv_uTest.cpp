/**
 * @file PerfCsv_uTest.cpp
 * @brief Unit tests for vernier CSV generation functions.
 *
 * Tests header generation, row serialization, and optional field handling.
 */

#include "src/bench/inc/PerfCsv.hpp"
#include "src/bench/inc/PerfStats.hpp"

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstddef>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using vernier::bench::PerfRow;
using vernier::bench::Stats;
using vernier::bench::writeCsvHeader;
using vernier::bench::writeCsvRow;

namespace {

/** @brief Helper to capture CSV output to a string. */
class CsvCapture {
public:
  CsvCapture()
      : path_("/tmp/test_csv_" + std::to_string(::getpid()) + "_" +
              ::testing::UnitTest::GetInstance()->current_test_info()->name() + ".csv") {
    ofs_.open(path_);
  }

  ~CsvCapture() { std::remove(path_.c_str()); }

  std::ofstream& stream() { return ofs_; }

  std::string content() {
    ofs_.close();
    std::ifstream ifs(path_);
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
  }

private:
  std::string path_;
  std::ofstream ofs_;
};

Stats makeTestStats() {
  return Stats{
      .median = 100.0,
      .p10 = 90.0,
      .p90 = 110.0,
      .p99 = 118.0,
      .p999 = 119.5,
      .min = 80.0,
      .max = 120.0,
      .mean = 100.0,
      .stddev = 10.0,
      .cv = 0.1,
  };
}

PerfRow makeBasicRow(const Stats& s) {
  PerfRow row;
  row.testName = "Suite.Test";
  row.cycles = 1000;
  row.repeats = 10;
  row.warmup = 1;
  row.threads = 4;
  row.msgBytes = 64;
  row.console = true;
  row.nonBlocking = false;
  row.minLevel = "INFO";
  row.stats = s;
  row.callsPerSecond = 10000.0;
  return row;
}

} // namespace

/* ----------------------------- Header Tests ----------------------------- */

/** @test Basic header has required columns. */
TEST(PerfCsvTest, BasicHeaderHasRequiredColumns) {
  CsvCapture csv;
  writeCsvHeader(csv.stream(), false, false, false);

  const std::string CONTENT = csv.content();

  EXPECT_NE(CONTENT.find("test"), std::string::npos);
  EXPECT_NE(CONTENT.find("cycles"), std::string::npos);
  EXPECT_NE(CONTENT.find("repeats"), std::string::npos);
  EXPECT_NE(CONTENT.find("wallMedian"), std::string::npos);
  EXPECT_NE(CONTENT.find("wallP99"), std::string::npos);
  EXPECT_NE(CONTENT.find("wallP999"), std::string::npos);
  EXPECT_NE(CONTENT.find("wallCV"), std::string::npos);
  EXPECT_NE(CONTENT.find("callsPerSecond"), std::string::npos);
}

/** @test Header with profile columns. */
TEST(PerfCsvTest, HeaderWithProfileColumns) {
  CsvCapture csv;
  writeCsvHeader(csv.stream(), true, false, false);

  const std::string CONTENT = csv.content();

  EXPECT_NE(CONTENT.find("profileTool"), std::string::npos);
  EXPECT_NE(CONTENT.find("profileDir"), std::string::npos);
}

/** @test Header with metadata columns. */
TEST(PerfCsvTest, HeaderWithMetadataColumns) {
  CsvCapture csv;
  writeCsvHeader(csv.stream(), false, true, false);

  const std::string CONTENT = csv.content();

  EXPECT_NE(CONTENT.find("timestamp"), std::string::npos);
  EXPECT_NE(CONTENT.find("gitHash"), std::string::npos);
  EXPECT_NE(CONTENT.find("hostname"), std::string::npos);
  EXPECT_NE(CONTENT.find("platform"), std::string::npos);
}

/** @test Header with GPU columns. */
TEST(PerfCsvTest, HeaderWithGpuColumns) {
  CsvCapture csv;
  writeCsvHeader(csv.stream(), false, false, true);

  const std::string CONTENT = csv.content();

  // Original GPU columns
  EXPECT_NE(CONTENT.find("gpuModel"), std::string::npos);
  EXPECT_NE(CONTENT.find("kernelTimeUs"), std::string::npos);
  EXPECT_NE(CONTENT.find("speedupVsCpu"), std::string::npos);

  // Multi-GPU columns
  EXPECT_NE(CONTENT.find("deviceId"), std::string::npos);
  EXPECT_NE(CONTENT.find("deviceCount"), std::string::npos);
  EXPECT_NE(CONTENT.find("multiGpuEfficiency"), std::string::npos);

  // Unified Memory columns
  EXPECT_NE(CONTENT.find("umPageFaults"), std::string::npos);
  EXPECT_NE(CONTENT.find("umThrashing"), std::string::npos);
}

/** @test Header ends with newline. */
TEST(PerfCsvTest, HeaderEndsWithNewline) {
  CsvCapture csv;
  writeCsvHeader(csv.stream(), false, false, false);

  const std::string CONTENT = csv.content();

  EXPECT_FALSE(CONTENT.empty());
  EXPECT_EQ(CONTENT.back(), '\n');
}

/* ----------------------------- Row Tests ----------------------------- */

/** @test Basic row serialization. */
TEST(PerfCsvTest, BasicRowSerialization) {
  CsvCapture csv;
  const Stats S = makeTestStats();

  writeCsvRow(csv.stream(), makeBasicRow(S));

  const std::string CONTENT = csv.content();

  EXPECT_NE(CONTENT.find("Suite.Test"), std::string::npos);
  EXPECT_NE(CONTENT.find("1000"), std::string::npos);  // cycles
  EXPECT_NE(CONTENT.find("10000"), std::string::npos); // callsPerSecond
  EXPECT_NE(CONTENT.find("100"), std::string::npos);   // median
  EXPECT_NE(CONTENT.find("0.1"), std::string::npos);   // cv
}

/** @test Row with profile metadata. */
TEST(PerfCsvTest, RowWithProfileMetadata) {
  CsvCapture csv;
  const Stats S = makeTestStats();

  PerfRow row;
  row.testName = "Test";
  row.cycles = 100;
  row.repeats = 5;
  row.warmup = 1;
  row.threads = 1;
  row.msgBytes = 32;
  row.console = false;
  row.nonBlocking = false;
  row.minLevel = "DEBUG";
  row.stats = S;
  row.callsPerSecond = 1000.0;
  row.profileTool = "perf";
  row.profileDir = "/tmp/perf";

  writeCsvRow(csv.stream(), row);

  const std::string CONTENT = csv.content();

  EXPECT_NE(CONTENT.find("perf"), std::string::npos);
  EXPECT_NE(CONTENT.find("/tmp/perf"), std::string::npos);
}

/** @test Row with run metadata. */
TEST(PerfCsvTest, RowWithRunMetadata) {
  CsvCapture csv;
  const Stats S = makeTestStats();

  PerfRow row;
  row.testName = "Test";
  row.cycles = 100;
  row.repeats = 5;
  row.warmup = 1;
  row.threads = 1;
  row.msgBytes = 32;
  row.console = false;
  row.nonBlocking = false;
  row.minLevel = "INFO";
  row.stats = S;
  row.callsPerSecond = 1000.0;
  row.timestamp = "2024-12-20T10:00:00Z";
  row.gitHash = "abc123";
  row.hostname = "myhost";
  row.platform = "x86_64";

  writeCsvRow(csv.stream(), row);

  const std::string CONTENT = csv.content();

  EXPECT_NE(CONTENT.find("2024-12-20T10:00:00Z"), std::string::npos);
  EXPECT_NE(CONTENT.find("abc123"), std::string::npos);
  EXPECT_NE(CONTENT.find("myhost"), std::string::npos);
  EXPECT_NE(CONTENT.find("x86_64"), std::string::npos);
}

/** @test Row ends with newline. */
TEST(PerfCsvTest, RowEndsWithNewline) {
  CsvCapture csv;
  const Stats S = makeTestStats();

  PerfRow row;
  row.testName = "Test";
  row.cycles = 100;
  row.repeats = 5;
  row.warmup = 1;
  row.threads = 1;
  row.msgBytes = 32;
  row.console = false;
  row.nonBlocking = false;
  row.minLevel = "INFO";
  row.stats = S;
  row.callsPerSecond = 1000.0;

  writeCsvRow(csv.stream(), row);

  const std::string CONTENT = csv.content();

  EXPECT_FALSE(CONTENT.empty());
  EXPECT_EQ(CONTENT.back(), '\n');
}

/** @test Boolean fields serialize as 0/1. */
TEST(PerfCsvTest, BooleanFieldsSerializeAs01) {
  CsvCapture csv;
  const Stats S = makeTestStats();

  // console=true, nonBlocking=false
  writeCsvRow(csv.stream(), makeBasicRow(S));

  const std::string CONTENT = csv.content();

  // The row format includes: ...msgBytes,console,nonBlocking,minLevel...
  // So we expect: ...64,1,0,INFO...
  EXPECT_NE(CONTENT.find(",1,0,INFO"), std::string::npos);
}

/* ----------------------------- Field Count Tests ----------------------------- */

/** @test Basic header has 20 columns (18 base + 2 stability). */
TEST(PerfCsvTest, BasicHeaderHas22Columns) {
  CsvCapture csv;
  writeCsvHeader(csv.stream(), false, false, false);

  const std::string CONTENT = csv.content();

  // Count commas (columns = commas + 1)
  const size_t COMMA_COUNT = std::count(CONTENT.begin(), CONTENT.end(), ',');
  EXPECT_EQ(COMMA_COUNT, 21U); // 22 columns = 21 commas
}

/** @test Header with all options has expected column count. */
TEST(PerfCsvTest, FullHeaderColumnCount) {
  CsvCapture csv;
  writeCsvHeader(csv.stream(), true, true, true);

  const std::string CONTENT = csv.content();

  // Base: 22, Profile: +2, Metadata: +4, GPU: +29 (11 base + 4 power/thermal +
  // 5 CUPTI + 4 multi-GPU + 5 Unified Memory) = 57 columns.
  const size_t COMMA_COUNT = std::count(CONTENT.begin(), CONTENT.end(), ',');
  EXPECT_EQ(COMMA_COUNT, 56U);
}

/** @test Header includes stability columns. */
TEST(PerfCsvTest, HeaderHasStabilityColumns) {
  CsvCapture csv;
  writeCsvHeader(csv.stream(), false, false, false);

  const std::string CONTENT = csv.content();

  EXPECT_NE(CONTENT.find("stable"), std::string::npos);
  EXPECT_NE(CONTENT.find("cvThreshold"), std::string::npos);
}

/** @test Row includes stability fields. */
TEST(PerfCsvTest, RowHasStabilityFields) {
  CsvCapture csv;
  const Stats S = makeTestStats();

  PerfRow row = makeBasicRow(S);
  row.stable = false;
  row.cvThreshold = 0.10;

  writeCsvRow(csv.stream(), row);

  const std::string CONTENT = csv.content();

  // stable=0 (false) and cvThreshold=0.1 should appear
  EXPECT_NE(CONTENT.find(",0,0.1"), std::string::npos);
}

/* ----------------------------- Test Name Cell Tests ----------------------------- */

namespace {

/**
 * @brief Read CSV text into records of cells the way RFC 4180 readers do: a
 *        quoted cell may hold commas, doubled quotes and line breaks; outside
 *        quotes a line feed, a carriage return or both end the record.
 */
std::vector<std::vector<std::string>> readCsvRecords(const std::string& text) {
  std::vector<std::vector<std::string>> records;
  std::vector<std::string> record;
  std::string cell;
  bool quoted = false;
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char C = text[i];
    if (quoted) {
      if (C == '"' && i + 1 < text.size() && text[i + 1] == '"') {
        cell += '"';
        ++i;
      } else if (C == '"') {
        quoted = false;
      } else {
        cell += C;
      }
    } else if (C == '"' && cell.empty()) {
      quoted = true;
    } else if (C == ',') {
      record.push_back(cell);
      cell.clear();
    } else if (C == '\r' || C == '\n') {
      record.push_back(cell);
      cell.clear();
      records.push_back(record);
      record.clear();
      if (C == '\r' && i + 1 < text.size() && text[i + 1] == '\n') {
        ++i;
      }
    } else {
      cell += C;
    }
  }
  if (!cell.empty() || !record.empty()) {
    record.push_back(cell);
    records.push_back(record);
  }
  return records;
}

/** @brief A header and one row named @p testName: the file's text and its records. */
struct WrittenCsv {
  std::string text;
  std::vector<std::vector<std::string>> records;
};

WrittenCsv writeOneNamedRow(const std::string& testName) {
  CsvCapture csv;
  writeCsvHeader(csv.stream(), false, false, false);
  PerfRow row = makeBasicRow(makeTestStats());
  row.testName = testName;
  writeCsvRow(csv.stream(), row, false, false, false);
  WrittenCsv out;
  out.text = csv.content();
  out.records = readCsvRecords(out.text);
  return out;
}

} // namespace

/** @test Quotes a test name holding a comma, so its row keeps the header's width */
TEST(PerfCsvTest, TestNameWithCommaIsOneQuotedCell) {
  const std::string NAME = "Sweep.Sizes/64,256";
  const WrittenCsv CSV = writeOneNamedRow(NAME);

  EXPECT_NE(CSV.text.find("\n\"Sweep.Sizes/64,256\",1000,"), std::string::npos) << CSV.text;
  ASSERT_EQ(CSV.records.size(), 2U);
  EXPECT_EQ(CSV.records[1].size(), CSV.records[0].size());
  EXPECT_EQ(CSV.records[1][0], NAME);
  EXPECT_EQ(CSV.records[1][1], "1000"); // cycles, under its own header
}

/** @test Doubles each quote inside a test name and quotes the cell */
TEST(PerfCsvTest, TestNameWithQuoteDoublesIt) {
  const std::string NAME = "Parse.Input/say \"hi\"";
  const WrittenCsv CSV = writeOneNamedRow(NAME);

  EXPECT_NE(CSV.text.find("\n\"Parse.Input/say \"\"hi\"\"\",1000,"), std::string::npos) << CSV.text;
  ASSERT_EQ(CSV.records.size(), 2U);
  EXPECT_EQ(CSV.records[1].size(), CSV.records[0].size());
  EXPECT_EQ(CSV.records[1][0], NAME);
}

/** @test Quotes a test name holding a line feed or a carriage return, keeping one record */
TEST(PerfCsvTest, TestNameWithLineBreakIsQuoted) {
  for (const std::string& name : {std::string("Case.Name/a\nb"), std::string("Case.Name/a\rb")}) {
    SCOPED_TRACE(name == "Case.Name/a\nb" ? "line feed" : "carriage return");
    const WrittenCsv CSV = writeOneNamedRow(name);

    EXPECT_NE(CSV.text.find("\n\"" + name + "\",1000,"), std::string::npos);
    ASSERT_EQ(CSV.records.size(), 2U) << "the line break split the row";
    EXPECT_EQ(CSV.records[1].size(), CSV.records[0].size());
    EXPECT_EQ(CSV.records[1][0], name);
  }
}

/** @test Writes a test name without a comma, quote or line break exactly as it is */
TEST(PerfCsvTest, PlainTestNameIsUnquoted) {
  const std::string NAME = "Sweep.Sizes/x 64#2";
  const WrittenCsv CSV = writeOneNamedRow(NAME);

  EXPECT_NE(CSV.text.find("\nSweep.Sizes/x 64#2,1000,"), std::string::npos) << CSV.text;
  EXPECT_EQ(CSV.text.find('"'), std::string::npos);
  ASSERT_EQ(CSV.records.size(), 2U);
  EXPECT_EQ(CSV.records[1][0], NAME);
}
