#ifndef VERNIER_PROFILERENV_HPP
#define VERNIER_PROFILERENV_HPP
/**
 * @file ProfilerEnv.hpp
 * @brief Tiny shared utilities used by multiple profiler backends.
 *
 * These checks live here (rather than duplicated in each TU) so the
 * Docker / valgrind / binary-on-PATH detection logic stays consistent
 * across backends. All functions are header-only and side-effect-free.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace vernier {
namespace bench {
namespace profiler_env {

/* ----------------------------- isOnPath ----------------------------- */

/**
 * @brief Return true if @p binaryName resolves to an executable on PATH.
 *
 * Uses `command -v` rather than spawning the binary itself.
 */
inline bool isOnPath(const char* binaryName) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "command -v %s >/dev/null 2>&1", binaryName);
  return std::system(buf) == 0;
}

/* ----------------------------- isInContainer ----------------------------- */

/**
 * @brief Heuristic detection that this process runs inside a container.
 *
 * Checks /proc/1/cgroup for substrings that the major container runtimes
 * leave behind ("docker", "containerd", "kubepods", "podman"). Cheap,
 * portable, and stable across Docker/Podman/k8s. False negatives are
 * possible on exotic runtimes; false positives are unlikely.
 */
inline bool isInContainer() {
  std::FILE* fp = std::fopen("/proc/1/cgroup", "r");
  if (!fp) return false;
  char line[512];
  bool found = false;
  while (std::fgets(line, sizeof(line), fp)) {
    if (std::strstr(line, "docker") || std::strstr(line, "containerd") ||
        std::strstr(line, "kubepods") || std::strstr(line, "podman")) {
      found = true;
      break;
    }
  }
  std::fclose(fp);
  return found;
}

/* ----------------------------- isRunningUnderValgrind ----------------------------- */

/**
 * @brief True when the process is being executed by valgrind.
 *
 * RUNNING_ON_VALGRIND is set by `valgrind --tool=*` for child processes;
 * the libvgpreload_* mapping is the secondary signal for cases where the
 * env var is filtered (e.g. systemd / nspawn).
 */
inline bool isRunningUnderValgrind() {
  if (std::getenv("RUNNING_ON_VALGRIND") != nullptr) return true;
  std::FILE* fp = std::fopen("/proc/self/maps", "r");
  if (!fp) return false;
  char line[512];
  bool found = false;
  while (std::fgets(line, sizeof(line), fp)) {
    if (std::strstr(line, "vgpreload_")) {
      found = true;
      break;
    }
  }
  std::fclose(fp);
  return found;
}

} // namespace profiler_env
} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERENV_HPP
