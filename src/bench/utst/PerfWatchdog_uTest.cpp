/**
 * @file PerfWatchdog_uTest.cpp
 * @brief Unit tests for vernier::bench::perf_watchdog.
 *
 * Notes:
 *  - The handler ends the process, so each test runs it in a death-test child.
 *  - The alarm is raised directly; no test waits for a real timeout.
 *  - A test that arms through a measured case reads and cancels the alarm at
 *    once, so a failing assertion cannot leave it to end the process later.
 */

#include "src/bench/inc/PerfHarness.hpp"

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfRegistry.hpp"

#include <gtest/gtest.h>

#include <unistd.h>

#include <csignal>

#include <cstddef>
#include <string>
#include <vector>

namespace perf_watchdog = vernier::bench::perf_watchdog;

/* ----------------------------- API Tests ----------------------------- */

class PerfWatchdogTest : public ::testing::Test {
protected:
  std::string savedStyle_;

  // The child re-executes the binary, so threads left behind by other tests
  // in this process cannot leak into it.
  void SetUp() override {
    savedStyle_ = GTEST_FLAG_GET(death_test_style);
    GTEST_FLAG_SET(death_test_style, "threadsafe");
  }

  void TearDown() override { GTEST_FLAG_SET(death_test_style, savedStyle_); }
};

/** @test Reports the test and the profiler by name, in order, then exits with status 2 */
TEST_F(PerfWatchdogTest, TimeoutNamesTestAndToolThenExits) {
  EXPECT_EXIT(
      {
        perf_watchdog::arm("Suite.SlowCase", "callgrind", 3600);
        std::raise(SIGALRM);
      },
      ::testing::ExitedWithCode(2),
      "\\[bench\\] watchdog timeout: test 'Suite\\.SlowCase' under --profile callgrind "
      "exceeded the configured profile-test-timeout\\.\n"
      "(.|\n)*or raise --profile-test-timeout\\. Aborting\\.\n$");
}

/** @test Substitutes a placeholder for a missing name instead of reading a null pointer */
TEST_F(PerfWatchdogTest, NullNamesBecomePlaceholders) {
  EXPECT_EXIT(
      {
        perf_watchdog::arm(nullptr, nullptr, 3600);
        std::raise(SIGALRM);
      },
      ::testing::ExitedWithCode(2), "watchdog timeout: test '\\?' under --profile \\? exceeded");
}

/** @test A disarmed watchdog leaves no pending alarm and reports itself disarmed */
TEST_F(PerfWatchdogTest, DisarmCancelsPendingAlarm) {
  struct sigaction previous{};
  ASSERT_EQ(::sigaction(SIGALRM, nullptr, &previous), 0);

  perf_watchdog::arm("Suite.Case", "perf", 3600);
  EXPECT_TRUE(perf_watchdog::g_armed.load());
  perf_watchdog::disarm();
  EXPECT_FALSE(perf_watchdog::g_armed.load());
  EXPECT_EQ(::alarm(0), 0U) << "an alarm was still pending after disarm()";

  // arm() installs a process-wide handler; hand the signal back as found.
  ASSERT_EQ(::sigaction(SIGALRM, &previous, nullptr), 0);
}

namespace {

/** @brief Whether the watchdog was armed while a measured loop of a case parsed from @p flags ran.
 */
bool armedDuringMeasure(std::vector<const char*> flags) {
  std::vector<std::string> storage{"prog"};
  storage.insert(storage.end(), flags.begin(), flags.end());
  std::vector<char*> argv;
  for (std::string& arg : storage) {
    argv.push_back(arg.data());
  }
  int argc = static_cast<int>(argv.size());
  vernier::bench::PerfConfig cfg;
  vernier::bench::parsePerfFlags(cfg, &argc, argv.data());
  cfg.cycles = 1;
  cfg.repeats = 1;
  vernier::bench::PerfCase perf{"Watchdog.Case", cfg};
  bool armed = false;
  perf.measured([&] { armed = perf_watchdog::g_armed.load(); });
  EXPECT_FALSE(perf_watchdog::g_armed.load()) << "the loop left the watchdog armed";
  return armed;
}

} // namespace

/**
 * @test An explicit --profile-test-timeout 0 under --profile arms nothing; the
 * omitted flag (300 s) and a given one arm the watchdog for the measured loop.
 */
TEST_F(PerfWatchdogTest, ExplicitZeroDoesNotArm) {
  struct sigaction previous{};
  ASSERT_EQ(::sigaction(SIGALRM, nullptr, &previous), 0);

  EXPECT_FALSE(armedDuringMeasure({"--profile", "perf", "--profile-test-timeout", "0"}));
  EXPECT_TRUE(armedDuringMeasure({"--profile", "perf"}));
  EXPECT_TRUE(armedDuringMeasure({"--profile", "perf", "--profile-test-timeout", "30"}));
  EXPECT_FALSE(armedDuringMeasure({"--profile-test-timeout", "30"})) << "armed without --profile";

  // arm() installs a process-wide handler; hand the signal back as found.
  ASSERT_EQ(::sigaction(SIGALRM, &previous, nullptr), 0);
}

namespace {

/** @brief The workload's own exception type and value, to see both arrive. */
struct WorkloadError {
  int code;
};

} // namespace

/**
 * @test A measured callback that throws: its exception reaches the caller
 * unchanged, the watchdog is off with no alarm pending, the after hook does
 * not run and no row is published; the next case of the process measures,
 * publishes and turns its watchdog off as usual.
 */
TEST_F(PerfWatchdogTest, ThrowingCallbackTurnsTheWatchdogOff) {
  using vernier::bench::PerfCase;
  using vernier::bench::PerfRegistry;
  struct sigaction previous{};
  ASSERT_EQ(::sigaction(SIGALRM, nullptr, &previous), 0);

  vernier::bench::PerfConfig cfg;
  cfg.cycles = 1;
  cfg.repeats = 3;
  cfg.profileTool = "perf"; // a requested profile arms the watchdog; no profiler is made here
  cfg.profileTestTimeoutSecs = 30;
  (void)PerfRegistry::instance().take();
  const std::size_t PUBLISHED = PerfRegistry::instance().summary().size();

  int befores = 0;
  int afters = 0;
  int code = 0;
  {
    PerfCase perf{"Watchdog.Throws", cfg};
    perf.setBeforeMeasureHook([&](const PerfCase&) { ++befores; });
    perf.setAfterMeasureHook([&](const PerfCase&, const vernier::bench::Stats&) { ++afters; });
    try {
      perf.measured([] { throw WorkloadError{7}; });
      ADD_FAILURE() << "the exception did not reach the caller";
    } catch (const WorkloadError& e) {
      code = e.code;
    }
  }
  // Read (and cancel) at once: an alarm left pending would end this process.
  const unsigned int PENDING = ::alarm(0);
  const bool ARMED = perf_watchdog::g_armed.load();
  perf_watchdog::disarm();
  EXPECT_EQ(PENDING, 0U) << "the watchdog's alarm was still pending after the exception";
  EXPECT_FALSE(ARMED) << "the watchdog still reports itself armed";
  EXPECT_EQ(code, 7) << "the workload's exception arrived changed";
  EXPECT_EQ(befores, 1);
  EXPECT_EQ(afters, 0) << "the after hook ran for a measurement that ended by an exception";
  EXPECT_FALSE(PerfRegistry::instance().take().has_value()) << "a row was published";
  EXPECT_EQ(PerfRegistry::instance().summary().size(), PUBLISHED);

  bool armedInside = false;
  {
    PerfCase next{"Watchdog.Next", cfg};
    next.setAfterMeasureHook([&](const PerfCase&, const vernier::bench::Stats&) { ++afters; });
    next.measured([&] { armedInside = perf_watchdog::g_armed.load(); });
  }
  const unsigned int PENDING_AFTER_NEXT = ::alarm(0);
  EXPECT_TRUE(armedInside) << "the next case measured without its watchdog";
  EXPECT_EQ(PENDING_AFTER_NEXT, 0U);
  EXPECT_FALSE(perf_watchdog::g_armed.load());
  EXPECT_EQ(afters, 1) << "the next case's after hook";
  const auto ROW = PerfRegistry::instance().take();
  ASSERT_TRUE(ROW.has_value()) << "the next case published no row";
  EXPECT_EQ(ROW->testName, "Watchdog.Next");

  ASSERT_EQ(::sigaction(SIGALRM, &previous, nullptr), 0);
}
