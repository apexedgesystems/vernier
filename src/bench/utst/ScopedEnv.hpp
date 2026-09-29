#ifndef VERNIER_SCOPEDENV_HPP
#define VERNIER_SCOPEDENV_HPP
/**
 * @file ScopedEnv.hpp
 * @brief Test helper: set or clear one process environment variable for a scope.
 *
 * Shared by the bench tests: the readiness tests reach it through
 * ReadinessFixtures.hpp; the CUPTI, GPU harness and Nsight tests include it.
 */

#include <cstdlib>

#include <string>

namespace vernier {
namespace bench {
namespace test {

/* ----------------------------- ScopedEnv ----------------------------- */

/**
 * @brief Sets one process environment variable for a scope and restores it.
 *
 * For tests of what reads the live environment: a launch, or a decision taken
 * on a snapshot of this process; readiness decisions take an explicit context
 * instead. The variable's previous state, a value or unset, is restored on
 * destruction.
 */
class ScopedEnv {
public:
  /** @brief Sets @p name to @p value. */
  ScopedEnv(const char* name, const std::string& value) : ScopedEnv(name, value.c_str()) {}

  /** @brief Sets @p name to @p value, or clears it when @p value is null. */
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      old_ = old;
      hadOld_ = true;
    }
    if (value != nullptr) {
      ::setenv(name, value, 1);
    } else {
      ::unsetenv(name);
    }
  }
  ~ScopedEnv() {
    if (hadOld_) {
      ::setenv(name_.c_str(), old_.c_str(), 1);
    } else {
      ::unsetenv(name_.c_str());
    }
  }
  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
  std::string name_;
  std::string old_;
  bool hadOld_ = false;
};

} // namespace test
} // namespace bench
} // namespace vernier

#endif // VERNIER_SCOPEDENV_HPP
