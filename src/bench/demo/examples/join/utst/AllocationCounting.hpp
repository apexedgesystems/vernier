#ifndef VERNIER_DEMO_JOIN_ALLOCATIONCOUNTING_HPP
#define VERNIER_DEMO_JOIN_ALLOCATIONCOUNTING_HPP
/**
 * @file AllocationCounting.hpp
 * @brief Whether this build can replace the global allocation functions, which
 *        the join example's allocation checks count through.
 *
 * Two of the join example's test programs replace operator new and delete:
 * the unit tests count the calls each version makes, and TestDemoJoinPeakHeap
 * counts the bytes each holds at its peak. The thread sanitizer's runtime
 * defines those functions itself, and clang links that runtime statically, so
 * a replacement cannot link beside it ("duplicate symbol ... operator new").
 * A thread-sanitizer build therefore replaces nothing, and each check that
 * counts skips with ALLOCATIONS_NOT_COUNTED. The address and
 * undefined-behaviour sanitizers link beside a replacement and keep counting.
 *
 * Test support, private to the join example's tests.
 */

// GCC announces the thread sanitizer with a macro; clang defines none and
// answers __has_feature instead.
#if defined(__SANITIZE_THREAD__)
#define VERNIER_DEMO_REPLACES_ALLOCATION 0
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define VERNIER_DEMO_REPLACES_ALLOCATION 0
#endif
#endif
#ifndef VERNIER_DEMO_REPLACES_ALLOCATION
#define VERNIER_DEMO_REPLACES_ALLOCATION 1
#endif

namespace vernier {
namespace bench {
namespace demo {
namespace test {

/* ----------------------------- Constants ----------------------------- */

/// The reason a check that counts allocations prints when it skips.
inline constexpr const char* ALLOCATIONS_NOT_COUNTED =
    "this binary is built with the thread sanitizer, whose runtime defines operator new and "
    "delete itself; clang's cannot link beside a replacement, so this build replaces neither "
    "and counts no allocation: run this test in a build without the thread sanitizer";

} // namespace test
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_JOIN_ALLOCATIONCOUNTING_HPP
