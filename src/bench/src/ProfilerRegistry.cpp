/**
 * @file ProfilerRegistry.cpp
 * @brief ProfilerRegistry singleton + dispatch implementation.
 */

#include "src/bench/inc/ProfilerRegistry.hpp"

#include <cstdio>
#include <utility>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {

/* ------------------------------- API ------------------------------- */

ProfilerRegistry& ProfilerRegistry::instance() {
  static ProfilerRegistry s_instance;
  return s_instance;
}

void ProfilerRegistry::registerBackend(std::string name, Factory factory, EnvCheck check,
                                       std::string unavailableHint) {
  if (!check) {
    check = []() { return EnvReport{EnvReport::Status::Ok, "no check defined", ""}; };
  }
  backends_[std::move(name)] =
      Entry{std::move(factory), std::move(check), std::move(unavailableHint)};
}

std::unique_ptr<Profiler> ProfilerRegistry::make(const std::string& name, const PerfConfig& cfg,
                                                 const std::string& testName) const {
  const auto it = backends_.find(name);
  if (it == backends_.end()) {
    std::string available;
    for (const auto& [n, _] : backends_) {
      if (!available.empty())
        available += ", ";
      available += n;
    }
    std::fprintf(stderr, "\n[WARN] Unknown profiler '%s'. Available: %s.\n\n", name.c_str(),
                 available.c_str());
    return std::make_unique<detail::NoOpProfiler>(name, "");
  }

  if (auto p = it->second.factory(cfg, testName)) {
    return p;
  }

  std::fprintf(stderr,
               "\n[WARN] Profiler '%s' requested but unavailable on this platform.\n"
               "   %s\n"
               "   Falling back to no-op (measurements will proceed without profiling).\n\n",
               name.c_str(), it->second.unavailableHint.c_str());
  return std::make_unique<detail::NoOpProfiler>(name, "");
}

bool ProfilerRegistry::hasBackend(const std::string& name) const noexcept {
  return backends_.find(name) != backends_.end();
}

std::vector<std::string> ProfilerRegistry::backendNames() const {
  std::vector<std::string> names;
  names.reserve(backends_.size());
  for (const auto& [n, _] : backends_) {
    names.push_back(n);
  }
  return names;
}

EnvReport ProfilerRegistry::runCheck(const std::string& name) const {
  const auto it = backends_.find(name);
  if (it == backends_.end()) {
    return EnvReport{EnvReport::Status::Error, "unknown profiler '" + name + "'", ""};
  }
  return it->second.check();
}

std::vector<std::pair<std::string, EnvReport>> ProfilerRegistry::runAllChecks() const {
  std::vector<std::pair<std::string, EnvReport>> out;
  out.reserve(backends_.size());
  for (const auto& [n, e] : backends_) {
    out.emplace_back(n, e.check());
  }
  return out;
}

int ProfilerRegistry::printDoctor() const {
  const auto reports = runAllChecks();
  std::fprintf(stdout, "\n=== Profiler Backend Doctor ===\n\n");
  int fails = 0;
  for (const auto& [name, rep] : reports) {
    const char* tag = "[OK]  ";
    if (rep.status == EnvReport::Status::Warning)
      tag = "[WARN]";
    if (rep.status == EnvReport::Status::Error) {
      tag = "[FAIL]";
      ++fails;
    }
    std::fprintf(stdout, "  %s %-10s %s\n", tag, name.c_str(), rep.message.c_str());
    if (!rep.hint.empty()) {
      std::fprintf(stdout, "             %s\n", rep.hint.c_str());
    }
  }
  std::fprintf(stdout, "\n  %zu backend(s), %d fail.\n\n", reports.size(), fails);
  return fails;
}

} // namespace bench
} // namespace vernier
