#ifndef VERNIER_CALLGRINDREQUESTS_HPP
#define VERNIER_CALLGRINDREQUESTS_HPP
/**
 * @file CallgrindRequests.hpp
 * @brief libbench's callgrind client requests (CallgrindRequests.cpp, the one
 *        place it includes valgrind's callgrind.h): the callgrind backend
 *        switches the recording of a wrap started by hand from inside the
 *        process with them. Outside callgrind, each request does nothing.
 *
 * Internal to libbench; not installed.
 */

namespace vernier {
namespace bench {
namespace callgrind {

/// Whether this build compiled the requests in: valgrind's callgrind.h was
/// found when it was built, and NVALGRIND is not defined.
[[nodiscard]] bool requestsBuiltIn() noexcept;

/// The first call in a process drops what callgrind has counted so far
/// (CALLGRIND_ZERO_STATS), so that a profile switched on and off around the
/// measured windows holds the windows alone, whether or not the wrap started
/// with --instr-atstart=no; later calls do nothing.
void zeroStatsOnce() noexcept;

/// Switch callgrind's instrumentation on (CALLGRIND_START_INSTRUMENTATION).
void startInstrumentation() noexcept;

/// Switch callgrind's instrumentation off (CALLGRIND_STOP_INSTRUMENTATION).
void stopInstrumentation() noexcept;

} // namespace callgrind
} // namespace bench
} // namespace vernier

#endif // VERNIER_CALLGRINDREQUESTS_HPP
