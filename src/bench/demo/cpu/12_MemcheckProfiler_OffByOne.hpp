#ifndef VERNIER_DEMO_12_MEMCHECK_OFFBYONE_HPP
#define VERNIER_DEMO_12_MEMCHECK_OFFBYONE_HPP
/**
 * @file 12_MemcheckProfiler_OffByOne.hpp
 * @brief Demo 12's deliberately wrong join: an off-by-one for memcheck to
 *        find.
 *
 * Private to BenchDemo_12_MemcheckProfiler. It is not a version of the shared
 * join example: the example's versions are held to the same answers and are
 * safe to run anywhere, and this one is not. Only Memcheck.JoinOffByOne calls
 * it, and that case runs only under valgrind. Do not copy it.
 */

#include <cstddef>

#include <string>
#include <vector>

namespace vernier {
namespace bench {
namespace demo {
namespace memcheck_demo {

/* ----------------------------- API ----------------------------- */

/**
 * @brief Join parts, appending @p sep after each part, through a raw buffer
 *        sized for the characters alone: the terminator is written one byte
 *        past its end, and read back from there.
 * @return The joined string, identical to joinV1's for the same arguments.
 * @note NOT RT-safe: allocates. Not memory-safe either: one invalid write and
 *       one invalid read per call, which memcheck reports.
 */
[[nodiscard, gnu::noinline]] std::string joinOffByOne(const std::vector<std::string>& parts,
                                                      char sep);

} // namespace memcheck_demo
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_12_MEMCHECK_OFFBYONE_HPP
