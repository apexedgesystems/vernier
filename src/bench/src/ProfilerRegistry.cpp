/**
 * @file ProfilerRegistry.cpp
 * @brief ProfilerRegistry singleton + dispatch implementation.
 */

#include "src/bench/inc/ProfilerRegistry.hpp"

#include <cstdio>
#include <utility>

#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {

/* ------------------------------- API ------------------------------- */

ProfilerRegistry& ProfilerRegistry::instance() {
  static ProfilerRegistry s_instance;
  return s_instance;
}

void ProfilerRegistry::registerBackend(std::string name, Factory factory, std::string unavailableHint) {
  backends_[std::move(name)] = Entry{std::move(factory), std::move(unavailableHint)};
}

std::unique_ptr<Profiler> ProfilerRegistry::make(const std::string& name,
                                                 const PerfConfig& cfg,
                                                 const std::string& testName) const {
  const auto it = backends_.find(name);
  if (it == backends_.end()) {
    std::string available;
    for (const auto& [n, _] : backends_) {
      if (!available.empty()) available += ", ";
      available += n;
    }
    std::fprintf(stderr,
                 "\n[WARN] Unknown profiler '%s'. Available: %s.\n\n",
                 name.c_str(), available.c_str());
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

} // namespace bench
} // namespace vernier
