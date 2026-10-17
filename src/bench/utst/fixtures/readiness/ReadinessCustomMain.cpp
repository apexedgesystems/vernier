/**
 * @file ReadinessCustomMain.cpp
 * @brief A benchmark with its own main(), for the readiness CLI tests.
 *
 * Built for the tests only, never installed. Its main() is the expansion of
 * PERF_MAIN() the advanced guide shows, so its runs show that a benchmark with
 * its own main() ends as PERF_MAIN() does. The second case fails when
 * READINESS_FIXTURE_FAIL is set, so the tests' own status can be shown to win.
 */

#include "src/bench/inc/Perf.hpp"

#include <cstdint>
#include <cstdlib>

PERF_TEST(ReadinessCustomMain, Guarded) {
  UB_PERF_GUARD(perf);
  volatile std::uint64_t acc = 0;
  perf.throughputLoop([&] { acc = acc + 1; });
}

PERF_TEST(ReadinessCustomMain, FailsOnRequest) {
  EXPECT_TRUE(std::getenv("READINESS_FIXTURE_FAIL") == nullptr) << "failing, as asked";
}

int main(int argc, char** argv) {
  vernier::bench::ensureBenchAbi();
  auto& cfg = vernier::bench::detail::perfConfigSingleton();
  vernier::bench::parsePerfFlags(cfg, &argc, argv);
  vernier::bench::setGlobalPerfConfig(&cfg);
  vernier::bench::installPerfEventListener(cfg);
  ::testing::InitGoogleTest(&argc, argv);
  const int rc = RUN_ALL_TESTS();
  VERNIER_WARN_IF_NO_TESTS_RAN_UNDER_PROFILE(cfg);
  return vernier::bench::ProfilerRegistry::instance().finishRun(
      cfg, rc, ::testing::UnitTest::GetInstance()->test_to_run_count());
}
