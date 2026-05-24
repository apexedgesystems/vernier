#ifndef VERNIER_PROFILERREGISTRY_HPP
#define VERNIER_PROFILERREGISTRY_HPP
/**
 * @file ProfilerRegistry.hpp
 * @brief Self-registration registry for profiler backends.
 *
 * Each backend (perf, gperf, callgrind, bpftrace, rapl, nsight, ...)
 * registers a factory + an availability hint at static-init time via
 * VERNIER_REGISTER_PROFILER_BACKEND. Profiler::make() dispatches by
 * looking up the registered factory rather than a hard-coded if-chain,
 * so new backends slot in by adding a file + one registration line.
 *
 * Threading:
 *  - Registration runs during static init (single-threaded).
 *  - make() / hasBackend() / backendNames() are const lookups, safe
 *    after init completes.
 *
 * @note NOT RT-safe (std::map, std::string, std::function).
 */

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "src/bench/inc/PerfConfig.hpp"

namespace vernier {
namespace bench {

class Profiler;

/* ------------------------------ Registry ------------------------------ */

class ProfilerRegistry {
public:
  using Factory = std::function<std::unique_ptr<Profiler>(const PerfConfig&, const std::string&)>;

  /** @brief Access the process-wide registry singleton. */
  static ProfilerRegistry& instance();

  /**
   * @brief Register a backend factory.
   * @param name           Stable backend name (e.g. "perf", "gperf"); also the `--profile` value.
   * @param factory        Callable returning a Profiler or nullptr if unavailable at runtime.
   * @param unavailableHint Actionable hint printed when the factory returns nullptr
   *                       (e.g. "Install linux-tools-$(uname -r) or run outside Docker.").
   *
   * Registration is idempotent: re-registering the same name replaces the prior entry.
   */
  void registerBackend(std::string name, Factory factory, std::string unavailableHint);

  /**
   * @brief Construct a profiler for the requested backend, or a named no-op.
   *
   * Returns:
   *  - factory result if backend is registered and available
   *  - named no-op (with warning to stderr + hint) if registered but unavailable
   *  - named no-op (with "unknown profiler" warning) if not registered
   *
   * Never returns nullptr.
   */
  std::unique_ptr<Profiler> make(const std::string& name,
                                 const PerfConfig& cfg,
                                 const std::string& testName) const;

  /** @brief True if a backend with this name has been registered. */
  bool hasBackend(const std::string& name) const noexcept;

  /** @brief Sorted list of registered backend names (for help text, bench doctor, etc.). */
  std::vector<std::string> backendNames() const;

private:
  ProfilerRegistry() = default;

  struct Entry {
    Factory factory;
    std::string unavailableHint;
  };
  std::map<std::string, Entry> backends_;
};

/* ------------------------- Registration helper ------------------------- */

namespace detail {

/**
 * @brief RAII registrar; one instance per backend at file scope triggers registration.
 */
struct ProfilerRegistrar {
  ProfilerRegistrar(std::string name, ProfilerRegistry::Factory factory, std::string hint) {
    ProfilerRegistry::instance().registerBackend(std::move(name), std::move(factory), std::move(hint));
  }
};

} // namespace detail

/**
 * @brief Convenience macro for backend self-registration at file scope.
 *
 * Usage at the bottom of a profiler's translation unit:
 * @code
 * VERNIER_REGISTER_PROFILER_BACKEND(
 *     "perf",
 *     makePerfProfiler,
 *     "Install linux-tools-$(uname -r) or run outside Docker.");
 * @endcode
 */
#define VERNIER_REGISTER_PROFILER_BACKEND(NAME, FACTORY, HINT)                                         \
  namespace {                                                                                          \
  const ::vernier::bench::detail::ProfilerRegistrar UB_REGISTRAR_##__LINE__{                           \
      (NAME), (FACTORY), (HINT)};                                                                      \
  }

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERREGISTRY_HPP
