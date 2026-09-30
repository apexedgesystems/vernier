#ifndef VERNIER_DEMO_14_HELGRIND_RACY_HPP
#define VERNIER_DEMO_14_HELGRIND_RACY_HPP
/**
 * @file 14_HelgrindProfiler_Racy.hpp
 * @brief Demo 14's deliberately racy addition: a shared total updated
 *        without a lock, for helgrind to find.
 *
 * Private to BenchDemo_14_HelgrindProfiler. It is not a version of the shared
 * join example: it is the unsafe way to share one of its results between
 * threads. Only Helgrind.RacyTotal calls it, and that case runs only under
 * valgrind. Do not copy it.
 */

#include <cstddef>

#include <string>
#include <vector>

namespace vernier {
namespace bench {
namespace demo {
namespace helgrind_demo {

/* ----------------------------- API ----------------------------- */

/**
 * @brief Add the length of joinV1(@p parts, @p sep) to @p total, without a
 *        lock.
 * @note NOT RT-safe: allocates. Not thread-safe either: called from several
 *       threads on one total, the reads and writes of @p total race, which
 *       helgrind reports.
 */
[[gnu::noinline]] void addJoinedLength(std::size_t& total, const std::vector<std::string>& parts,
                                       char sep);

} // namespace helgrind_demo
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_14_HELGRIND_RACY_HPP
