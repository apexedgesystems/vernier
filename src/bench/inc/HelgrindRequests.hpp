#ifndef VERNIER_HELGRINDREQUESTS_HPP
#define VERNIER_HELGRINDREQUESTS_HPP
/**
 * @file HelgrindRequests.hpp
 * @brief Helgrind client requests made by libbench on the harness's behalf.
 *
 * StartGate synchronises its threads through atomic flags, which helgrind
 * cannot see as synchronisation, so it asks helgrind to leave those flags
 * unchecked while the gate exists. libbench makes the requests: whether they
 * are made is decided once, when libbench is built (valgrind's helgrind.h
 * found, NVALGRIND not defined), and is the same for every benchmark that
 * loads it, whatever that benchmark's own build has. No valgrind macro reaches
 * a benchmark through this header.
 */

#include <cstddef>

namespace vernier {
namespace bench {
namespace helgrind {

/* ----------------------------- API ----------------------------- */

/**
 * @brief True when this libbench was built with valgrind's helgrind.h and
 *        without NVALGRIND, so disableChecking() and enableChecking() reach
 *        helgrind when the program runs under it.
 * @note RT-safe: returns a constant.
 */
[[nodiscard]] bool requestsBuiltIn() noexcept;

/**
 * @brief Asks helgrind not to check accesses to the @p len bytes at @p addr.
 *
 * Without helgrind the request does nothing; in a libbench built without the
 * requests (see requestsBuiltIn()), neither does the call.
 * @note RT-safe: a few instructions, no allocation.
 */
void disableChecking(void* addr, std::size_t len) noexcept;

/**
 * @brief Returns the @p len bytes at @p addr to helgrind's ordinary checking,
 *        as memory newly owned by the calling thread.
 * @note RT-safe: a few instructions, no allocation.
 */
void enableChecking(void* addr, std::size_t len) noexcept;

} // namespace helgrind
} // namespace bench
} // namespace vernier

#endif // VERNIER_HELGRINDREQUESTS_HPP
