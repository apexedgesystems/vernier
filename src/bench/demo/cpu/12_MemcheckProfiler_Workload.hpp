#ifndef VERNIER_DEMO_12_MEMCHECK_WORKLOAD_HPP
#define VERNIER_DEMO_12_MEMCHECK_WORKLOAD_HPP
/**
 * @file 12_MemcheckProfiler_Workload.hpp
 * @brief What demo 12 joins, and how often its wrong case calls the wrong
 *        join.
 *
 * One definition for the demo and for the program that checks what memcheck
 * reports for it, so both describe the same run.
 */

#include <cstddef>

namespace vernier {
namespace bench {
namespace demo {
namespace memcheck_demo {

/* ----------------------------- Constants ----------------------------- */

/// Parts per join, as in demo 01, so the walkthroughs measure the same call.
inline constexpr std::size_t PART_COUNT = 1000;

/// Fixed seed: every run joins the same words.
inline constexpr unsigned PART_SEED = 42;

inline constexpr char SEPARATOR = ',';

/// Calls Memcheck.JoinOffByOne makes. Each writes one byte past its buffer, so
/// memcheck counts this many of the same error and reports the context once.
inline constexpr long OFF_BY_ONE_CALLS = 3;

} // namespace memcheck_demo
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_12_MEMCHECK_WORKLOAD_HPP
