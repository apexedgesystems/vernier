/**
 * @file HelgrindRequests.cpp
 * @brief libbench's helgrind client requests: the one place it includes
 *        valgrind's helgrind.h.
 */

#include "src/bench/inc/HelgrindRequests.hpp"

// Compiled in where this build finds valgrind's helgrind.h. NVALGRIND, defined
// by the build or by valgrind.h itself on a platform valgrind does not
// support, reduces each request to nothing.
#if defined(__has_include)
#if __has_include(<valgrind/helgrind.h>)
#include <valgrind/helgrind.h>
#define VERNIER_HELGRIND_HEADER_FOUND 1
#endif
#endif

namespace vernier {
namespace bench {
namespace helgrind {

/* ----------------------------- API ----------------------------- */

bool requestsBuiltIn() noexcept {
#if defined(VERNIER_HELGRIND_HEADER_FOUND) && !defined(NVALGRIND)
  return true;
#else
  return false;
#endif
}

void disableChecking(void* addr, std::size_t len) noexcept {
#if defined(VERNIER_HELGRIND_HEADER_FOUND)
  VALGRIND_HG_DISABLE_CHECKING(addr, len);
#else
  (void)addr;
  (void)len;
#endif
}

void enableChecking(void* addr, std::size_t len) noexcept {
#if defined(VERNIER_HELGRIND_HEADER_FOUND)
  VALGRIND_HG_ENABLE_CHECKING(addr, len);
#else
  (void)addr;
  (void)len;
#endif
}

} // namespace helgrind
} // namespace bench
} // namespace vernier
