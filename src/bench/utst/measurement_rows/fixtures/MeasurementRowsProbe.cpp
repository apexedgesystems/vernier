/**
 * @file MeasurementRowsProbe.cpp
 * @brief A benchmark whose tests measure in each shape that decides how a
 * test's rows are named: the fixture of the measurement-rows checks.
 *
 * After each completed measurement it prints one line to stdout:
 * "[rows-probe] median=<m> cycles=<c> msgBytes=<b>", from that measurement's
 * result and its case, with the number format the CSV writer uses, so a check
 * can tie every CSV row to the measurement that produced it.
 *
 * A test fixture with its own PERF_MAIN: never installed, and its Rows.*
 * cases are never registered; the checks run it as a child.
 */

#include "src/bench/inc/Perf.hpp"

#include <gtest/gtest.h>

#include <cstdint>

#include <atomic>
#include <iostream>
#include <stdexcept>
#include <string>

namespace vb = vernier::bench;

namespace {

volatile std::uint64_t gSink = 0;

/** @brief Work in proportion to @p n. */
void spin(int n) {
  for (int i = 0; i < n; ++i) {
    gSink = gSink + static_cast<std::uint64_t>(i);
  }
}

/** @brief Print the line a check matches with the measurement's row. */
void report(const vb::PerfCase& perf, const vb::PerfResult& result) {
  std::cout << "[rows-probe] median=" << result.stats.median << " cycles=" << perf.cycles()
            << " msgBytes=" << perf.config().msgBytes << std::endl;
}

} // namespace

/* ----------------------------- Probe Tests ----------------------------- */

/** @brief Three separately named cases in one test, each with its own size. */
PERF_TEST(Rows, SeparateCases) {
  for (const int size : {64, 256, 1024}) {
    vb::PerfConfig cfg = vb::detail::getPerfConfig();
    cfg.msgBytes = size;
    cfg.cycles = size;
    vb::PerfCase perf{"Rows.SeparateCases/" + std::to_string(size), cfg};
    report(perf, perf.throughputLoop([size] { spin(size); }));
  }
}

/** @brief One case measured under three labels. */
PERF_TEST(Rows, OneCaseThreeLabels) {
  UB_PERF_GUARD(perf);
  for (const int size : {64, 256, 1024}) {
    report(perf, perf.throughputLoop([size] { spin(size); }, std::to_string(size)));
  }
}

/** @brief A measurement on the calling thread, then a contention run of the same case. */
PERF_TEST(Rows, SingleThenContention) {
  UB_PERF_GUARD(perf);
  report(perf, perf.throughputLoop([] { spin(64); }, "single"));
  std::atomic<std::uint64_t> calls{0};
  report(perf,
         perf.contentionRun([&] { calls.fetch_add(1, std::memory_order_relaxed); }, "contention"));
}

/** @brief A label used twice among three measurements. */
PERF_TEST(Rows, RepeatedLabel) {
  UB_PERF_GUARD(perf);
  report(perf, perf.throughputLoop([] { spin(64); }, "x"));
  report(perf, perf.throughputLoop([] { spin(128); }, "y"));
  report(perf, perf.throughputLoop([] { spin(256); }, "x"));
}

/** @brief Two measurements without a label. */
PERF_TEST(Rows, EmptyLabels) {
  UB_PERF_GUARD(perf);
  report(perf, perf.throughputLoop([] { spin(64); }, ""));
  report(perf, perf.throughputLoop([] { spin(128); }, ""));
}

/** @brief A label holding a comma, which the CSV writer quotes. */
PERF_TEST(Rows, LabelWithComma) {
  UB_PERF_GUARD(perf);
  report(perf, perf.throughputLoop([] { spin(64); }, "a,b"));
  report(perf, perf.throughputLoop([] { spin(128); }, "c"));
}

/** @brief A test's only measurement, whose row keeps the test's name. */
PERF_TEST(Rows, OneMeasurement) {
  UB_PERF_GUARD(perf);
  report(perf, perf.throughputLoop([] { spin(64); }, "solo"));
}

/** @brief A second measurement whose body throws, which publishes nothing. */
PERF_TEST(Rows, SecondMeasurementThrows) {
  UB_PERF_GUARD(perf);
  report(perf, perf.throughputLoop([] { spin(64); }, "completed"));
  EXPECT_THROW(
      (void)perf.throughputLoop([] { throw std::runtime_error("measured body failed"); }, "failed"),
      std::runtime_error);
}

/** @brief Runs after the throwing test and measures nothing, so it writes no row. */
PERF_TEST(Rows, NoMeasurement) { SUCCEED(); }

PERF_MAIN()
