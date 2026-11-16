/**
 * @file HelgrindRequests.cpp
 * @brief libbench's helgrind client requests: the one place it includes
 *        valgrind's helgrind.h, the copy in src/bench/third_party/valgrind.
 */

#include "src/bench/inc/HelgrindRequests.hpp"

// The copy libbench carries, so every build has the requests whatever its
// machine has installed. NVALGRIND, defined by the build or by valgrind.h
// itself on a platform valgrind does not support, compiles them out.
#include "src/bench/third_party/valgrind/helgrind.h"

namespace vernier {
namespace bench {
namespace helgrind {

/* ----------------------------- API ----------------------------- */

bool requestsBuiltIn() noexcept {
#if defined(NVALGRIND)
  return false;
#else
  return true;
#endif
}

// Under NVALGRIND the macros would still copy their arguments into locals they
// then leave unused, so the calls are left out instead.

void disableChecking(void* addr, std::size_t len) noexcept {
#if defined(NVALGRIND)
  (void)addr;
  (void)len;
#else
  VALGRIND_HG_DISABLE_CHECKING(addr, len);
#endif
}

void enableChecking(void* addr, std::size_t len) noexcept {
#if defined(NVALGRIND)
  (void)addr;
  (void)len;
#else
  VALGRIND_HG_ENABLE_CHECKING(addr, len);
#endif
}

} // namespace helgrind
} // namespace bench
} // namespace vernier
