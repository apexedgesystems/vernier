/**
 * @file MeasurementRows_uTest.cpp
 * @brief The measurement-rows fixture run as a child: its CSV, end-of-run
 *        table and footer checked against the measurements it reports.
 *
 * Notes:
 *  - The fixture (MEASUREMENT_ROWS_PROBE) prints "[rows-probe] median=<m>
 *    cycles=<c> msgBytes=<b>" after each completed measurement, in the CSV
 *    writer's number format, so each row is tied to its own measurement.
 *  - Each run's CSV and streams stay in publication/ under the working
 *    directory.
 *  - What bench summary and bench compare read from the same CSV is
 *    MeasurementRows.CliAgreement's, a test of the CLI's own suite.
 */

#include <sys/wait.h>

#include <cstddef>
#include <cstdlib>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#ifndef MEASUREMENT_ROWS_PROBE
#error "MEASUREMENT_ROWS_PROBE must name the measurement-rows fixture"
#endif

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

namespace fs = std::filesystem;

/* ----------------------------- Constants ----------------------------- */

namespace {

/// A row the fixture publishes at --threads 4.
struct ExpectedRow {
  const char* name;    ///< Its test cell
  const char* threads; ///< Its threads cell: 4 for the contention run only
};

/// Every row the fixture publishes, in completion order. A row's name starts
/// with the name of the test that published it; the throwing test's failed
/// measurement and the last test publish none.
constexpr ExpectedRow EXPECTED_ROWS[] = {
    {"Rows.SeparateCases/64", "1"},
    {"Rows.SeparateCases/256", "1"},
    {"Rows.SeparateCases/1024", "1"},
    {"Rows.OneCaseThreeLabels/64", "1"},
    {"Rows.OneCaseThreeLabels/256", "1"},
    {"Rows.OneCaseThreeLabels/1024", "1"},
    {"Rows.SingleThenContention/single", "1"},
    {"Rows.SingleThenContention/contention", "4"},
    {"Rows.RepeatedLabel/x#1", "1"},
    {"Rows.RepeatedLabel/y", "1"},
    {"Rows.RepeatedLabel/x#2", "1"},
    {"Rows.EmptyLabels/#1", "1"},
    {"Rows.EmptyLabels/#2", "1"},
    {"Rows.LabelWithComma/a,b", "1"},
    {"Rows.LabelWithComma/c", "1"},
    {"Rows.OneMeasurement", "1"},
    {"Rows.SecondMeasurementThrows", "1"},
};

/// The sizes, and the cycles, of Rows.SeparateCases's three cases.
constexpr const char* SEPARATE_CASE_SIZES[] = {"64", "256", "1024"};

} // namespace

/* ----------------------------- Child Runs ----------------------------- */

namespace {

/// What one run of the fixture left: its exit status (-1 when it did not
/// exit) and its two streams.
struct FixtureRun {
  int status = -1;
  std::string out;
  std::string err;
};

/// A whole file as text; empty when it cannot be read.
std::string readText(const fs::path& file) {
  std::ifstream in(file);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

/// @p text as one single-quoted shell word.
std::string shellWord(const std::string& text) {
  std::string word = "'";
  for (const char c : text) {
    word += (c == '\'') ? std::string("'\\''") : std::string(1, c);
  }
  return word + "'";
}

/// Runs the fixture with @p args; its streams stay in @p dir.
FixtureRun runFixture(const fs::path& dir, const std::string& name,
                      const std::vector<std::string>& args) {
  std::string command = shellWord(MEASUREMENT_ROWS_PROBE);
  for (const std::string& arg : args) {
    command += " " + shellWord(arg);
  }
  const fs::path out = dir / (name + ".stdout.txt");
  const fs::path err = dir / (name + ".stderr.txt");
  const int status =
      std::system((command + " >" + shellWord(out) + " 2>" + shellWord(err)).c_str());
  return {WIFEXITED(status) ? WEXITSTATUS(status) : -1, readText(out), readText(err)};
}

/// The fixture's settings for every run but the calibrated one, then @p extra.
std::vector<std::string> withCommon(const std::vector<std::string>& extra) {
  std::vector<std::string> args = {"--threads", "4", "--cycles", "50", "--repeats", "3"};
  args.insert(args.end(), extra.begin(), extra.end());
  return args;
}

} // namespace

/* ----------------------------- Reading a Run ----------------------------- */

namespace {

/// One CSV data row: column name to cell.
using Row = std::map<std::string, std::string>;

/// The cells of one CSV line; a quoted cell may hold commas.
std::vector<std::string> csvCells(const std::string& line) {
  std::vector<std::string> cells(1);
  bool inQuotes = false;
  for (const char c : line) {
    if (c == '"') {
      inQuotes = !inQuotes;
    } else if (c == ',' && !inQuotes) {
      cells.emplace_back();
    } else {
      cells.back() += c;
    }
  }
  return cells;
}

/// Every data row of the CSV @p file, in file order.
std::vector<Row> readRows(const fs::path& file) {
  std::ifstream in(file);
  std::string line;
  std::getline(in, line);
  const std::vector<std::string> header = csvCells(line);
  std::vector<Row> rows;
  while (std::getline(in, line)) {
    const std::vector<std::string> cells = csvCells(line);
    Row& row = rows.emplace_back();
    for (std::size_t i = 0; i < header.size() && i < cells.size(); ++i) {
      row[header[i]] = cells[i];
    }
  }
  return rows;
}

/// @p row's cell under @p column; "<none>" when it has none.
std::string cellOf(const Row& row, const std::string& column) {
  const auto it = row.find(column);
  return it == row.end() ? "<none>" : it->second;
}

/// The test cell of each of @p rows.
std::vector<std::string> namesOf(const std::vector<Row>& rows) {
  std::vector<std::string> names;
  for (const Row& row : rows) {
    names.push_back(cellOf(row, "test"));
  }
  return names;
}

/// The end-of-run table a run printed: its rows' names, how many rows it marks
/// OK and UNSTABLE, and its footer.
struct Table {
  std::vector<std::string> names;
  int stable = 0;
  int unstable = 0;
  std::string footer;
};

Table readTable(const std::string& out) {
  Table table;
  const std::size_t header = out.find("Median (us)");
  if (header == std::string::npos) {
    return table;
  }
  std::istringstream in(out.substr(header));
  std::string line;
  std::getline(in, line); // the rest of the header
  std::getline(in, line); // the rule under it
  while (std::getline(in, line) && line.find_first_not_of('-') != std::string::npos) {
    table.names.push_back(line.substr(0, line.find(' ')));
    table.stable += line.ends_with("  OK") ? 1 : 0;
    table.unstable += line.ends_with("  UNSTABLE") ? 1 : 0;
  }
  std::getline(in, table.footer);
  return table;
}

/// The footer @p table must end with when @p tests tests published its rows:
/// both counts when a test published several, else one test per row; stable
/// and unstable count the rows it marks.
std::string expectedFooter(const Table& table, std::size_t tests) {
  const std::string rows = std::to_string(table.names.size());
  const std::string marks =
      std::to_string(table.stable) + " stable | " + std::to_string(table.unstable) + " unstable";
  if (tests < table.names.size()) {
    return rows + " rows from " + std::to_string(tests) + " tests | " + marks;
  }
  return rows + " tests | " + marks;
}

/// The capture groups of every match of @p pattern in @p text, in order.
std::vector<std::vector<std::string>> matches(const std::string& text, const char* pattern) {
  const std::regex expression(pattern);
  std::vector<std::vector<std::string>> found;
  for (std::sregex_iterator it(text.begin(), text.end(), expression), end; it != end; ++it) {
    std::vector<std::string>& groups = found.emplace_back();
    for (std::size_t group = 1; group < it->size(); ++group) {
      groups.push_back((*it)[group].str());
    }
  }
  return found;
}

} // namespace

/* ----------------------------- API Tests ----------------------------- */

/** @test Publishes a row per completed measurement, with its own values, as the table names it */
TEST(MeasurementRows, Publication) {
  const fs::path dir = fs::current_path() / "publication";
  fs::remove_all(dir);
  fs::create_directories(dir);
  std::vector<std::string> names;
  std::set<std::string> tests;
  for (const ExpectedRow& row : EXPECTED_ROWS) {
    names.emplace_back(row.name);
    tests.insert(names.back().substr(0, names.back().find('/')));
  }

  // One run to a CSV: a row per completed measurement, holding that
  // measurement's median, its case's cycles and message size, and the number
  // of threads that ran it.
  const FixtureRun csvRun =
      runFixture(dir, "csv", withCommon({"--csv", (dir / "rows.csv").string()}));
  EXPECT_EQ(csvRun.status, 0) << csvRun.err;
  const std::vector<Row> rows = readRows(dir / "rows.csv");
  EXPECT_THAT(namesOf(rows), ElementsAreArray(names));
  const auto reported =
      matches(csvRun.out, R"(\[rows-probe\] median=(\S+) cycles=(\d+) msgBytes=(\d+))");
  EXPECT_EQ(reported.size(), names.size()) << "measurements the fixture completed";
  if (rows.size() == names.size() && reported.size() == names.size()) {
    for (std::size_t i = 0; i < names.size(); ++i) {
      SCOPED_TRACE(names[i]);
      EXPECT_EQ(cellOf(rows[i], "wallMedian"), reported[i][0]);
      EXPECT_EQ(cellOf(rows[i], "cycles"), reported[i][1]);
      EXPECT_EQ(cellOf(rows[i], "msgBytes"), reported[i][2]);
      EXPECT_EQ(cellOf(rows[i], "threads"), EXPECTED_ROWS[i].threads);
    }
    for (std::size_t i = 0; i < std::size(SEPARATE_CASE_SIZES); ++i) {
      EXPECT_EQ(cellOf(rows[i], "msgBytes"), SEPARATE_CASE_SIZES[i]) << names[i];
      EXPECT_EQ(cellOf(rows[i], "cycles"), SEPARATE_CASE_SIZES[i]) << names[i];
    }
  }
  const Table table = readTable(csvRun.out);
  EXPECT_THAT(table.names, ElementsAreArray(namesOf(rows))) << "the table against the CSV";
  EXPECT_EQ(table.footer, expectedFooter(table, tests.size()));

  // Two repetitions: each writes its own rows once.
  const FixtureRun repeat = runFixture(
      dir, "repeat", withCommon({"--gtest_repeat=2", "--csv", (dir / "repeat.csv").string()}));
  EXPECT_EQ(repeat.status, 0) << repeat.err;
  std::vector<std::string> twice = names;
  twice.insert(twice.end(), names.begin(), names.end());
  EXPECT_THAT(namesOf(readRows(dir / "repeat.csv")), ElementsAreArray(twice));
  const Table repeatTable = readTable(repeat.out);
  EXPECT_THAT(repeatTable.names, ElementsAreArray(twice));
  EXPECT_EQ(repeatTable.footer, expectedFooter(repeatTable, 2 * tests.size()));

  // Without --csv the table still names every row.
  const FixtureRun plain = runFixture(dir, "plain", withCommon({}));
  EXPECT_EQ(plain.status, 0) << plain.err;
  const Table plainTable = readTable(plain.out);
  EXPECT_THAT(plainTable.names, ElementsAreArray(names));
  EXPECT_EQ(plainTable.footer, expectedFooter(plainTable, tests.size()));

  // Two tests of one row each: the footer keeps its one-row-per-test form.
  const FixtureRun single =
      runFixture(dir, "single",
                 withCommon({"--gtest_filter=Rows.OneMeasurement:Rows.SecondMeasurementThrows"}));
  EXPECT_EQ(single.status, 0) << single.err;
  const Table singleTable = readTable(single.out);
  EXPECT_THAT(singleTable.names,
              ElementsAre("Rows.OneMeasurement", "Rows.SecondMeasurementThrows"));
  EXPECT_EQ(singleTable.footer, expectedFooter(singleTable, 2));

  // --target-time: each separately named case calibrates its own cycles.
  const FixtureRun target =
      runFixture(dir, "target",
                 {"--threads", "4", "--repeats", "3", "--target-time", "2ms",
                  "--gtest_filter=Rows.SeparateCases", "--csv", (dir / "target.csv").string()});
  EXPECT_EQ(target.status, 0) << target.err;
  const std::vector<Row> targetRows = readRows(dir / "target.csv");
  EXPECT_THAT(namesOf(targetRows),
              ElementsAreArray(names.begin(), names.begin() + std::size(SEPARATE_CASE_SIZES)));
  const auto calibrations = matches(target.err, R"(-> cycles=(\d+))");
  EXPECT_EQ(calibrations.size(), std::size(SEPARATE_CASE_SIZES)) << target.err;
  for (std::size_t i = 0; i < calibrations.size() && i < targetRows.size(); ++i) {
    EXPECT_EQ(cellOf(targetRows[i], "cycles"), calibrations[i][0]) << names[i];
  }
}
