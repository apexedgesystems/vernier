#ifndef VERNIER_STDERRCAPTURE_HPP
#define VERNIER_STDERRCAPTURE_HPP
/**
 * @file StderrCapture.hpp
 * @brief Test helper: capture what the code under test writes to stderr.
 *
 * Works at the file-descriptor level, so it sees std::fprintf(stderr, ...)
 * from any translation unit, including the shared library under test.
 */

#include <unistd.h>

#include <cstdio>

#include <string>

namespace vernier {
namespace bench {
namespace test {

/* ----------------------------- StderrCapture ----------------------------- */

/** @brief Redirects the process's stderr to a temp file until text() is read. */
class StderrCapture {
public:
  StderrCapture() : file_(std::tmpfile()) {
    if (file_ == nullptr) {
      return;
    }
    std::fflush(stderr);
    saved_ = ::dup(STDERR_FILENO);
    ::dup2(::fileno(file_), STDERR_FILENO);
  }

  ~StderrCapture() {
    restore();
    if (file_ != nullptr) {
      std::fclose(file_);
    }
  }

  StderrCapture(const StderrCapture&) = delete;
  StderrCapture& operator=(const StderrCapture&) = delete;

  /** @brief Stop capturing and return everything written so far. */
  std::string text() {
    restore();
    std::string out;
    if (file_ == nullptr) {
      return out;
    }
    std::rewind(file_);
    char buf[256];
    std::size_t got = 0;
    while ((got = std::fread(buf, 1, sizeof(buf), file_)) > 0) {
      out.append(buf, got);
    }
    return out;
  }

private:
  void restore() {
    if (saved_ >= 0) {
      std::fflush(stderr);
      ::dup2(saved_, STDERR_FILENO);
      ::close(saved_);
      saved_ = -1;
    }
  }

  std::FILE* file_;
  int saved_ = -1;
};

} // namespace test
} // namespace bench
} // namespace vernier

#endif // VERNIER_STDERRCAPTURE_HPP
