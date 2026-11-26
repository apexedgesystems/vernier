/**
 * @file RunStatusRecord.cpp
 * @brief The record of a run's exit status that libbench keeps for bench run
 *        (RunStatusRecord.hpp).
 */

#include "src/bench/src/RunStatusRecord.hpp"

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace vernier {
namespace bench {
namespace run_status {

namespace {

/// The longest token taken: bench run's are 32 hex digits.
constexpr std::size_t MAX_TOKEN = 64;

/// What bench run asked this process for, taken once from its environment.
struct Request {
  std::string token; ///< The run's token; empty when nothing was asked for.
  std::string path;  ///< The record's absolute path.
  pid_t pid = 0;     ///< The process the token was given to.
};

/// Whether @p text is a token as bench run gives them: 1 to MAX_TOKEN hex digits.
bool isToken(const std::string& text) {
  if (text.empty() || text.size() > MAX_TOKEN) {
    return false;
  }
  for (const char C : text) {
    if (!((C >= '0' && C <= '9') || (C >= 'a' && C <= 'f'))) {
      return false;
    }
  }
  return true;
}

/// The record's path in @p dir, made absolute from the working directory; empty when that
/// directory cannot be read.
std::string recordPath(const char* dir) {
  std::string path;
  if (dir[0] != '/') {
    char* cwd = ::getcwd(nullptr, 0);
    if (cwd == nullptr) {
      return {};
    }
    path = cwd;
    std::free(cwd);
    path += '/';
  }
  path += dir;
  path += '/';
  path += RECORD_NAME;
  return path;
}

/**
 * @brief Take bench run's request from the environment: the token is removed
 * from it, so a program this one starts does not record for this run, and
 * the folder is made absolute, so a later change of the working directory
 * does not move the record.
 */
Request take() noexcept {
  Request request;
  try {
    const char* TOKEN = std::getenv(TOKEN_VARIABLE);
    if (TOKEN == nullptr) {
      return request;
    }
    const std::string VALUE = TOKEN;
    ::unsetenv(TOKEN_VARIABLE);
    const char* DIR = std::getenv("VERNIER_EXTERNAL_WRAP_DIR");
    if (!isToken(VALUE)) {
      std::fprintf(stderr,
                   "[bench] %s='%s' is not a run's token (hex digits), so this run's exit status "
                   "is not recorded for bench run\n",
                   TOKEN_VARIABLE, VALUE.c_str());
      return request;
    }
    if (DIR == nullptr || *DIR == '\0') {
      std::fprintf(stderr,
                   "[bench] %s is set without VERNIER_EXTERNAL_WRAP_DIR, so this run's exit "
                   "status is not recorded for bench run\n",
                   TOKEN_VARIABLE);
      return request;
    }
    request.path = recordPath(DIR);
    if (request.path.empty()) {
      std::fprintf(stderr,
                   "[bench] the working directory cannot be read, so this run's exit status is "
                   "not recorded for bench run in %s\n",
                   DIR);
      return request;
    }
    request.token = VALUE;
    request.pid = ::getpid();
  } catch (...) {
    request = Request{};
  }
  return request;
}

/// The request, taken on first use.
const Request& requested() noexcept {
  static const Request REQUEST = take();
  return REQUEST;
}

/// Taken as libbench loads, before the benchmark can start a program.
[[maybe_unused]] const bool TAKEN_AT_LOAD = (static_cast<void>(requested()), true);

} // namespace

void record(int status) noexcept {
  const Request& REQUEST = requested();
  if (REQUEST.token.empty() || ::getpid() != REQUEST.pid) {
    return;
  }
  const unsigned CODE = static_cast<unsigned>(status) & 0xFFU;
  char line[MAX_TOKEN + 8];
  const int LENGTH = std::snprintf(line, sizeof(line), "%s %u\n", REQUEST.token.c_str(), CODE);
  int err = 0;
  const int FD = ::open(REQUEST.path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (FD < 0) {
    err = errno;
  } else {
    std::size_t done = 0;
    while (done < static_cast<std::size_t>(LENGTH)) {
      const ssize_t WROTE = ::write(FD, line + done, static_cast<std::size_t>(LENGTH) - done);
      if (WROTE < 0) {
        if (errno == EINTR) {
          continue;
        }
        err = errno;
        break;
      }
      done += static_cast<std::size_t>(WROTE);
    }
    if (::close(FD) != 0 && err == 0) {
      err = errno;
    }
  }
  if (err != 0) {
    std::fprintf(stderr,
                 "[bench] the exit status %u could not be recorded for bench run in %s: %s\n", CODE,
                 REQUEST.path.c_str(), std::strerror(err));
  }
}

} // namespace run_status
} // namespace bench
} // namespace vernier
