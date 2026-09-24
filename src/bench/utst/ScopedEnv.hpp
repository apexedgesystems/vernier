#ifndef VERNIER_SCOPEDENV_HPP
#define VERNIER_SCOPEDENV_HPP
/**
 * @file ScopedEnv.hpp
 * @brief Test helper: set or clear one environment variable for a scope.
 *
 * The previous state (set to a value, or absent) is restored on destruction,
 * so a test that needs a variable does not leak it into the tests after it.
 */

#include <cstdlib>

#include <optional>
#include <string>

namespace vernier {
namespace bench {
namespace test {

/* ----------------------------- ScopedEnv ----------------------------- */

/** @brief Sets @p name to @p value (or clears it, for nullptr) until destroyed. */
class ScopedEnv {
public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) {
      old_ = old;
    }
    if (value != nullptr) {
      ::setenv(name, value, 1);
    } else {
      ::unsetenv(name);
    }
  }

  ~ScopedEnv() {
    if (old_) {
      ::setenv(name_.c_str(), old_->c_str(), 1);
    } else {
      ::unsetenv(name_.c_str());
    }
  }

  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
  std::string name_;
  std::optional<std::string> old_;
};

} // namespace test
} // namespace bench
} // namespace vernier

#endif // VERNIER_SCOPEDENV_HPP
