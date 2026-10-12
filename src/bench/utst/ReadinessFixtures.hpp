#ifndef VERNIER_READINESSFIXTURES_HPP
#define VERNIER_READINESSFIXTURES_HPP
/**
 * @file ReadinessFixtures.hpp
 * @brief Test helper: a private directory of fake tools for readiness tests.
 *
 * The fakes are POSIX sh templates under src/bench/utst/fixtures/readiness/
 * (VERNIER_READINESS_FIXTURE_DIR). A FakeToolDir copies the ones a test needs
 * under the tool's own name, and builds a ReadinessContext whose PATH is only
 * that directory, so a check resolves the fakes and nothing else. The fakes
 * set their own PATH for the utilities they use, and append each invocation
 * to the directory's log (FAKE_LOG), which the test reads back. ScopedEnv
 * (ScopedEnv.hpp), for what a test must set in the live environment, comes
 * with it, and ChildReaping, for a test of how this process reaps children.
 */

#include "src/bench/inc/ProfilerReadiness.hpp"
#include "src/bench/utst/ScopedEnv.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifndef VERNIER_READINESS_FIXTURE_DIR
#error "VERNIER_READINESS_FIXTURE_DIR must name src/bench/utst/fixtures/readiness"
#endif

namespace vernier {
namespace bench {
namespace test {

/* ----------------------------- FakeToolDir ----------------------------- */

/** @brief A private temporary directory of fake tools, removed on destruction. */
class FakeToolDir {
public:
  FakeToolDir() {
    const std::filesystem::path BASE = std::filesystem::temp_directory_path();
    std::string pattern = (BASE / "vernier_readiness_XXXXXX").string();
    if (::mkdtemp(pattern.data()) != nullptr) {
      dir_ = pattern;
    }
    log_ = dir_ + "/fake.log";
  }

  ~FakeToolDir() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }

  FakeToolDir(const FakeToolDir&) = delete;
  FakeToolDir& operator=(const FakeToolDir&) = delete;

  /** @brief True when the directory was created. */
  [[nodiscard]] bool ok() const { return !dir_.empty(); }

  /** @brief The directory. */
  [[nodiscard]] const std::string& path() const { return dir_; }

  /** @brief The log every fake appends its invocation to. */
  [[nodiscard]] const std::string& logPath() const { return log_; }

  /**
   * @brief Copy template @p templateName (e.g. "fake_sudo.sh") into the
   * directory as @p toolName with @p mode; returns the installed path.
   */
  std::string install(const std::string& templateName, const std::string& toolName,
                      mode_t mode = 0755) const {
    const std::string SOURCE = std::string{VERNIER_READINESS_FIXTURE_DIR} + "/" + templateName;
    const std::string TARGET = dir_ + "/" + toolName;
    std::error_code ec;
    std::filesystem::copy_file(SOURCE, TARGET, std::filesystem::copy_options::overwrite_existing,
                               ec);
    ::chmod(TARGET.c_str(), mode);
    return TARGET;
  }

  /** @brief Create a directory named @p name (a directory named like a tool). */
  std::string makeDirectory(const std::string& name) const {
    const std::string TARGET = dir_ + "/" + name;
    std::error_code ec;
    std::filesystem::create_directories(TARGET, ec);
    return TARGET;
  }

  /** @brief Write @p text to a file in the directory; returns its path. */
  std::string writeFile(const std::string& name, const std::string& text) const {
    const std::string TARGET = dir_ + "/" + name;
    std::ofstream out(TARGET);
    out << text;
    return TARGET;
  }

  /** @brief Everything the fakes have logged so far. */
  [[nodiscard]] std::string log() const {
    std::ifstream in(log_);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
  }

  /** @brief The log's lines that start with @p prefix. */
  [[nodiscard]] std::vector<std::string> logLines(const std::string& prefix) const {
    std::vector<std::string> out;
    std::istringstream in(log());
    std::string line;
    while (std::getline(in, line)) {
      if (line.compare(0, prefix.size(), prefix) == 0) {
        out.push_back(line);
      }
    }
    return out;
  }

  /**
   * @brief A context whose PATH is this directory and whose FAKE_LOG is its
   * log, with @p extra added; effective uid @p euid.
   */
  [[nodiscard]] ReadinessContext context(std::map<std::string, std::string> extra = {},
                                         uid_t euid = ::geteuid()) const {
    std::map<std::string, std::string> env{{"PATH", dir_}, {"FAKE_LOG", log_}};
    for (auto& [key, value] : extra) {
      env[key] = std::move(value);
    }
    return ReadinessContext(euid, ::getpid(), std::move(env));
  }

private:
  std::string dir_;
  std::string log_;
};

/* ----------------------------- ChildReaping ----------------------------- */

/**
 * @brief While it lives, this process reaps its children as an application
 * may: SIGCHLD at its default action, ignored, or with SA_NOCLDWAIT; with
 * either of the last two the system discards each child's exit status. The
 * previous action is restored on destruction.
 */
class ChildReaping {
public:
  enum class Policy { DEFAULT, IGNORED, NO_CHILD_WAIT };

  explicit ChildReaping(Policy policy) {
    struct sigaction action{};
    action.sa_handler = policy == Policy::IGNORED ? SIG_IGN : SIG_DFL;
    action.sa_flags = policy == Policy::NO_CHILD_WAIT ? SA_NOCLDWAIT : 0;
    sigemptyset(&action.sa_mask);
    ok_ = ::sigaction(SIGCHLD, &action, &saved_) == 0;
  }
  ~ChildReaping() {
    if (ok_) {
      ::sigaction(SIGCHLD, &saved_, nullptr);
    }
  }
  ChildReaping(const ChildReaping&) = delete;
  ChildReaping& operator=(const ChildReaping&) = delete;

  /** @brief True when the policy is in force. */
  [[nodiscard]] bool ok() const noexcept { return ok_; }

private:
  struct sigaction saved_{};
  bool ok_ = false;
};

} // namespace test
} // namespace bench
} // namespace vernier

#endif // VERNIER_READINESSFIXTURES_HPP
