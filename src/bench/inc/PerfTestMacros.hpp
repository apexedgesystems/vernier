#ifndef VERNIER_PERFTESTMACROS_HPP
#define VERNIER_PERFTESTMACROS_HPP
/**
 * @file PerfTestMacros.hpp
 * @brief Macros for writing GoogleTest-style performance tests.
 *
 * This header provides the primary API for writing perf tests:
 *
 * ## Quick Reference
 *
 * | Macro | Purpose |
 * |-------|---------|
 * | `PERF_TEST(Suite, Name)` | Declare a perf test (use this for most tests) |
 * | `PERF_TEST_F(Fixture, Name)` | Fixture-based perf test (alias of TEST_F) |
 * | `PERF_TEST_P(Fixture, Name)` | Parameterized perf test (alias of TEST_P) |
 * | `PERF_GUARD(perf)` | Create PerfCase with auto-attached profiler hooks |
 * | `PERF_MAIN()` | Drop-in main() with flag parsing and CSV support |
 *
 * `UB_PERF_GUARD` / `UB_PERF_GUARD_NOPROFILE` remain as compatibility
 * aliases of the PERF_-prefixed names.
 *
 * ## Semantic Aliases (for discoverability)
 *
 * These are identical to PERF_TEST but signal test intent:
 * - `PERF_THROUGHPUT` - Measures operations per second
 * - `PERF_LATENCY` - Measures single-operation latency
 * - `PERF_CONTENTION` - Measures multi-threaded performance
 * - `PERF_TAIL` - Measures tail latency (p99/p999 across repeat samples;
 *   tails only bite with high --repeats — a per-op sampling loop is future work)
 * - `PERF_ALLOC` - Measures allocation overhead
 * - `PERF_IO` - Measures I/O operations
 * - `PERF_RAMP` - Measures thread scaling
 *
 * ## Example Usage
 *
 * @code{.cpp}
 * #include "src/bench/inc/Perf.hpp"
 *
 * PERF_TEST(MyComponent, Throughput) {
 *   UB_PERF_GUARD(perf);  // Profiler hooks auto-attached
 *
 *   auto data = prepareData();
 *   perf.warmup([&]{ process(data); });
 *   auto result = perf.throughputLoop([&]{ process(data); }, "process");
 *
 *   EXPECT_GT(result.callsPerSecond, 10000.0);
 * }
 *
 * PERF_MAIN()
 * @endcode
 *
 * Run with:
 * @code{.sh}
 * ./MyTest --cycles 10000 --repeats 10 --csv results.csv
 * ./MyTest --profile perf  # Enable profiling
 * ./MyTest --quick         # Fast iteration mode
 * @endcode
 */

#include <cstdio>
#include <string>

// gtest itself is included by user test binaries (the existing PERF_TEST
// macro chain already requires it). The PERF_MAIN macro below references
// ::testing::UnitTest directly so the gtest header is in scope only where
// PERF_MAIN is expanded -- never in library TUs that pull this header
// transitively. This keeps bench's interface clean of gtest for downstream
// projects that FetchContent vernier alongside their own gtest setup.

#include "src/bench/inc/PerfHarness.hpp"
#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {
namespace detail {

/* -------------------------------- Detail -------------------------------- */

/** @brief Singleton access to the shared PerfConfig (parsed once). */
inline PerfConfig& perfConfigSingleton() {
  static PerfConfig cfg{};
  return cfg;
}

/**
 * @brief Parse perf flags into the shared config, preserving gtest args.
 * Call this once in your test binary's main() before InitGoogleTest.
 */
inline void initPerfConfig(int* argc, char** argv) {
  parsePerfFlags(perfConfigSingleton(), argc, argv);
}

/** @brief Const accessor for read-only use sites. */
inline const PerfConfig& getPerfConfig() { return perfConfigSingleton(); }

} // namespace detail

} // namespace bench
} // namespace vernier

/* -------------------------- Test Declaration Macros -------------------------- */

/**
 * @brief Define a general perf test (semantic alias of gtest TEST).
 * Provide a PerfCase with UB_PERF_GUARD inside the body when needed.
 */
#define PERF_TEST(Suite, Name) TEST(Suite, Name)

/** @brief Throughput-focused test (alias of TEST for discoverability). */
#define PERF_THROUGHPUT(Suite, Name) TEST(Suite, Name)

/** @brief Single-op latency-focused test (alias of TEST). */
#define PERF_LATENCY(Suite, Name) TEST(Suite, Name)

/** @brief Contention/scaling-focused test (alias of TEST). */
#define PERF_CONTENTION(Suite, Name) TEST(Suite, Name)

/** @brief Tail-latency-focused test (alias of TEST). */
#define PERF_TAIL(Suite, Name) TEST(Suite, Name)

/** @brief Allocation-pressure-focused test (alias of TEST). */
#define PERF_ALLOC(Suite, Name) TEST(Suite, Name)

/** @brief Syscall/file/network I/O microbench (alias of TEST). */
#define PERF_IO(Suite, Name) TEST(Suite, Name)

/** @brief Thread-ramp sweep test (alias of TEST). */
#define PERF_RAMP(Suite, Name) TEST(Suite, Name)

/** @brief Fixture-based perf test (alias of TEST_F; use PERF_GUARD inside). */
#define PERF_TEST_F(Fixture, Name) TEST_F(Fixture, Name)

/** @brief Parameterized perf test (alias of TEST_P; use PERF_GUARD inside). */
#define PERF_TEST_P(Fixture, Name) TEST_P(Fixture, Name)

/* ----------------------------- Scoped Guard ----------------------------- */

/**
 * @brief Create a scoped PerfCase with profiler hooks auto-attached.
 *
 * This is the recommended macro for all perf tests. Profiler hooks are
 * automatically attached based on --profile flags (no-op if not specified).
 *
 * Usage inside a PERF_* body:
 * @code
 * PERF_TEST(Suite, Case) {
 *   UB_PERF_GUARD(perf);  // Profiler hooks auto-attached
 *   perf.warmup([&]{ work(); });
 *   auto result = perf.throughputLoop([&]{ work(); }, "label");
 * }
 * @endcode
 */
#define UB_PERF_GUARD(varName)                                                                     \
  vernier::bench::PerfCase varName = vernier::bench::makePerfCaseWithProfiler(                     \
      ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name() +                 \
          std::string(".") + ::testing::UnitTest::GetInstance()->current_test_info()->name(),      \
      vernier::bench::detail::getPerfConfig())

/**
 * @brief Create a scoped PerfCase WITHOUT profiler hooks (lightweight).
 *
 * Use this variant when you explicitly don't want profiling overhead,
 * or when manually attaching profiler hooks with custom configuration.
 */
#define UB_PERF_GUARD_NOPROFILE(varName)                                                           \
  vernier::bench::PerfCase varName {                                                               \
    ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name() +                   \
        std::string(".") + ::testing::UnitTest::GetInstance()->current_test_info()->name(),        \
        vernier::bench::detail::getPerfConfig()                                                    \
  }

/** @brief Preferred names for the guards; UB_-prefixed forms stay as
 *  compatibility aliases. */
#define PERF_GUARD(varName) UB_PERF_GUARD(varName)
#define PERF_GUARD_NOPROFILE(varName) UB_PERF_GUARD_NOPROFILE(varName)

/* ------------------------------ Main Macro ------------------------------ */

/**
 * @brief Drop-in replacement for main() in perf test binaries.
 *
 * Handles flag parsing, config singleton, CSV listener installation, and gtest initialization.
 * Eliminates ~6 lines of boilerplate per test file.
 *
 * Usage:
 *   PERF_MAIN()  // Replaces entire main() function
 *
 * Expands to:
 *   - Parse perf flags (--cycles, --repeats, --csv, etc.)
 *   - Register global config for CSV listener
 *   - Install CSV listener if --csv provided
 *   - Initialize GoogleTest
 *   - Run all tests
 */
/// Inline filter-validation: a typo in --gtest_filter under --profile silently
/// produces empty artifacts; emit a loud warning at end-of-main instead.
/// Expanded only inside test binaries that already include gtest.
#define VERNIER_WARN_IF_NO_TESTS_RAN_UNDER_PROFILE(cfg)                                            \
  do {                                                                                             \
    if (!(cfg).profileTool.empty()) {                                                              \
      const auto* _vernier_ut = ::testing::UnitTest::GetInstance();                                \
      if (_vernier_ut && _vernier_ut->test_to_run_count() == 0) {                                  \
        std::fprintf(stderr,                                                                       \
                     "\n[profile] --profile %s requested but the gtest filter matched zero "       \
                     "tests.\n[profile] Any profiler artifacts written are empty. Run with\n"      \
                     "[profile] --gtest_list_tests to see available test names.\n\n",              \
                     (cfg).profileTool.c_str());                                                   \
      }                                                                                            \
    }                                                                                              \
  } while (0)

#define PERF_MAIN()                                                                                \
  int main(int argc, char** argv) {                                                                \
    auto& cfg = vernier::bench::detail::perfConfigSingleton();                                     \
    vernier::bench::parsePerfFlags(cfg, &argc, argv);                                              \
    vernier::bench::setGlobalPerfConfig(&cfg);                                                     \
    vernier::bench::installPerfEventListener(cfg);                                                 \
    ::testing::InitGoogleTest(&argc, argv);                                                        \
    const int _vernier_rc = RUN_ALL_TESTS();                                                       \
    VERNIER_WARN_IF_NO_TESTS_RAN_UNDER_PROFILE(cfg);                                               \
    return _vernier_rc;                                                                            \
  }

#endif // VERNIER_PERFTESTMACROS_HPP