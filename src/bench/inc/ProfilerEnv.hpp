#ifndef VERNIER_PROFILERENV_HPP
#define VERNIER_PROFILERENV_HPP
/**
 * @file ProfilerEnv.hpp
 * @brief Tiny shared utilities used by multiple profiler backends.
 *
 * These checks live here (rather than duplicated in each TU) so the
 * Docker / valgrind / binary-on-PATH detection logic stays consistent
 * across backends. All functions are header-only; resolveArtifactDir() is the
 * only one that touches the filesystem.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <cerrno>
#include <csignal>
#include <sys/types.h>
#include <unistd.h>

#include "src/bench/inc/ProfilerReadiness.hpp" // policy and probes the helpers delegate to

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
 * Probes in order:
 *  1. `/.dockerenv` (Docker sentinel; survives cgroups v2 unified hierarchy)
 *  2. `/run/.containerenv` (Podman sentinel)
 *  3. CONTAINER env var (set by some runtimes / vernier dev images)
 *  4. /proc/1/cgroup substring match (cgroups v1 hosts only)
 *
 * Cheap, portable, stable across Docker / Podman / k8s. Modern Docker
 * uses cgroups v2 and `/proc/1/cgroup` reduces to "0::/" with no hint,
 * which is why the file sentinels run first.
 */
inline bool isInContainer() {
  // 1. Docker sentinel file (most reliable; cgroups-version-independent)
  if (std::FILE* f = std::fopen("/.dockerenv", "r")) {
    std::fclose(f);
    return true;
  }
  // 2. Podman sentinel
  if (std::FILE* f = std::fopen("/run/.containerenv", "r")) {
    std::fclose(f);
    return true;
  }
  // 3. Runtime-set env var
  if (const char* v = std::getenv("CONTAINER")) {
    // vernier dev images set CONTAINER=yes; some runtimes set it differently.
    if (v[0] != '\0' && v[0] != '0' && std::strcmp(v, "false") != 0)
      return true;
  }
  // 4. cgroups v1 substring match (fallback)
  std::FILE* fp = std::fopen("/proc/1/cgroup", "r");
  if (!fp)
    return false;
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
  if (std::getenv("RUNNING_ON_VALGRIND") != nullptr)
    return true;
  std::FILE* fp = std::fopen("/proc/self/maps", "r");
  if (!fp)
    return false;
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

/* ----------------------------- externalWrapTool ----------------------------- */

/**
 * @brief Name of the tool the runner wrapped this process with, or "".
 *
 * `bench run --profile <tool>` sets VERNIER_EXTERNAL_WRAP=<tool> on the
 * child when it invokes the wrap command itself (valgrind tools, nsys,
 * ncu, ...). Backends use this to stay passive instead of re-attaching
 * or printing manual-wrap hints for a wrap that already happened.
 */
inline std::string externalWrapTool() {
  const char* v = std::getenv("VERNIER_EXTERNAL_WRAP");
  return (v != nullptr) ? std::string{v} : std::string{};
}

/* ----------------------------- Artifact Directories ----------------------------- */

/**
 * @brief Directory the runner's wrap writes into, or "".
 *
 * Set by `bench run` (VERNIER_EXTERNAL_WRAP_DIR) next to VERNIER_EXTERNAL_WRAP.
 * A wrapping tool records the whole process into one place, so that directory,
 * not a per-test one, is where a wrapped run's artifacts are.
 */
inline std::string externalWrapDir() {
  const char* v = std::getenv("VERNIER_EXTERNAL_WRAP_DIR");
  return (v != nullptr) ? std::string{v} : std::string{};
}

/**
 * @brief Folder name for one test's artifacts: the encoded test name, a dot,
 * and @p suffix.
 *
 * GoogleTest puts '/' into the names of parameterized and typed tests
 * (`Sizes/Copy.Run/3`); left in, it would nest the folder inside directories
 * named after fragments of the test name. '/' is written as "+2F" instead, and
 * '+' itself as "+2B"; every other character stays, so a name with neither is
 * unchanged. Because the escape character is encoded too, the folder name
 * decodes back to exactly one test name ("+2F" -> '/', "+2B" -> '+', left to
 * right), so different test names get different folders.
 *
 * The escape is '+' because the folder is handed on, in hints and in commands
 * the backends run, to tools with substitution characters of their own:
 * valgrind and nsys expand '%' in output paths, jemalloc's MALLOC_CONF splits
 * on ',' and ':', and an unquoted shell word must not gain a metacharacter.
 * '+' means nothing to any of them, also at the start of a word or where a
 * word with '=' would read as an assignment.
 */
inline std::string artifactDirName(std::string_view testName, std::string_view suffix) {
  std::string name;
  name.reserve(testName.size() + suffix.size() + 1);
  for (const char CH : testName) {
    if (CH == '/') {
      name += "+2F";
    } else if (CH == '+') {
      name += "+2B";
    } else {
      name += CH;
    }
  }
  name += '.';
  name += suffix;
  return name;
}

/**
 * @brief The one rule for where a backend's artifacts go; creates the folder
 * when this process owns it.
 *
 * A folder is named after what its data covers:
 *  - the runner wrapped this process with @p profileTool: the data covers the
 *    whole process and lives in the runner's folder (externalWrapDir()). No
 *    per-test folder is created. Returns "" when the runner did not say where.
 *  - otherwise: `<artifactRoot or .>/<artifactDirName(testName, suffix)>`,
 *    created here.
 *
 * @param profileTool  The `--profile` value this backend was selected by.
 * @param suffix       Folder suffix without the dot, e.g. "gperf".
 * @note NOT RT-safe (filesystem, heap allocation).
 */
inline std::string resolveArtifactDir(const std::string& profileTool,
                                      const std::string& artifactRoot, std::string_view testName,
                                      std::string_view suffix) {
  if (!profileTool.empty() && externalWrapTool() == profileTool) {
    return externalWrapDir();
  }
  const std::string ROOT = artifactRoot.empty() ? std::string{"."} : artifactRoot;
  std::string dir = ROOT + "/" + artifactDirName(testName, suffix);
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  return dir;
}

/* ----------------------------- cuptiMustYield ----------------------------- */

/**
 * @brief True when in-process CUPTI collection must stay off for this run.
 *
 * CUPTI is single-client per process: if an external Nsight session
 * (nsys/ncu) owns the interface, an in-process subscriber wins the race
 * and the external tool records zero kernels. Yield when:
 *  1. VERNIER_DISABLE_CUPTI is set truthy (explicit operator override),
 *  2. the active --profile tool is nsight or ncu (an external session is
 *     the point of the run, attach-mode or wrapped), or
 *  3. the runner wrapped this process with nsys/ncu
 *     (VERNIER_EXTERNAL_WRAP, see externalWrapTool()).
 */
inline bool cuptiMustYield(const std::string& profileTool) {
  if (const char* v = std::getenv("VERNIER_DISABLE_CUPTI")) {
    if (v[0] != '\0' && v[0] != '0' && std::strcmp(v, "false") != 0)
      return true;
  }
  if (profileTool == "nsight" || profileTool == "ncu")
    return true;
  const std::string wrap = externalWrapTool();
  return wrap == "nsight" || wrap == "ncu";
}

/* ----------------------------- benchSudoActive ----------------------------- */

/**
 * @brief True when privilege-needing backends should elevate via `sudo -n`.
 *
 * Opt-in through BENCH_SUDO (1, true, yes or on, any case) for processes not
 * already running as root; the answer of decidePrivilege() for this process.
 * Pairs with a scoped sudoers grant (bpftrace + kill) so kernel-probe
 * backends work from unprivileged test runs -- the tests and their artifacts
 * stay owned by the user; only the probe tooling elevates.
 */
inline bool benchSudoActive() {
  return decidePrivilege(ReadinessContext::capture()).route == PrivilegeRoute::SCOPED_SUDO;
}

/* ----------------------------- sudoBpftraceUsable ----------------------------- */

/**
 * @brief True when `sudo -n <bpftrace> --version` works for this user.
 *
 * Proves that one invocation only: a grant restricted to other arguments can
 * refuse an attach this allows. The bpftrace and offcpu backends decide with
 * their readiness checks, which attach the selected script instead.
 */
inline bool sudoBpftraceUsable() {
  const ReadinessContext CTX = ReadinessContext::capture();
  const auto SUDO = resolveExecutable("sudo", CTX);
  const auto TOOL = resolveExecutable("bpftrace", CTX);
  if (!SUDO || !SUDO->executable || !TOOL || !TOOL->executable) {
    return false;
  }
  return runBoundedProbe({SUDO->path, "-n", "--", TOOL->path, "--version"}, 5000, CTX).succeeded();
}

/* ----------------------------- bpftraceAttachViable ----------------------------- */

/**
 * @brief Live viability probe: can bpftrace actually attach here?
 *
 * Presence on PATH is not health -- stripped builds break BEGIN/END,
 * missing tracefs breaks attachment, and both fail this real probe in
 * well under its 3s bound where a lookup-based check reports a false OK.
 * The bound is runBoundedProbe()'s own; no timeout(1) is needed.
 *
 * The probe attaches a sched-family tracepoint -- the same surface the
 * bpftrace-backed profilers use. A kprobe would be the wrong probe: some
 * vendor kernels (e.g. NVIDIA L4T) compile kprobes out entirely while
 * still shipping every sched tracepoint, and a kprobe-based check
 * reports a false negative on exactly the targets where the backends
 * work fine.
 */
inline bool bpftraceAttachViable(bool viaSudo) {
  const ReadinessContext CTX = ReadinessContext::capture();
  const auto TOOL = resolveExecutable("bpftrace", CTX);
  if (!TOOL || !TOOL->executable) {
    return false;
  }
  std::vector<std::string> argv;
  if (viaSudo) {
    const auto SUDO = resolveExecutable("sudo", CTX);
    if (!SUDO || !SUDO->executable) {
      return false;
    }
    argv = {SUDO->path, "-n", "--"};
  }
  argv.push_back(TOOL->path);
  argv.push_back("-e");
  argv.push_back("tracepoint:sched:sched_switch { } interval:ms:200 { exit(); }");
  return runBoundedProbe(argv, 3000, CTX).succeeded();
}

/* ----------------------------- processAlive ----------------------------- */

/**
 * @brief True when @p pid exists -- including root children an unprivileged
 * caller cannot signal (kill(pid, 0) failing with EPERM still means alive).
 */
inline bool processAlive(pid_t pid) {
  if (::kill(pid, 0) == 0)
    return true;
  return errno == EPERM;
}

/* ----------------------------- tracerPid ----------------------------- */

/**
 * @brief Resolve the process to signal for a spawned tracer child.
 *
 * sudo sometimes runs its command under a monitor process rather than
 * exec'ing in place, and the monitor declines to relay signals whose
 * sender shares the command's process group -- exactly our shape when a
 * test signals its own fork. Signaling the monitor's child directly
 * sidesteps the relay: if @p child has exactly one living child of its
 * own, that grandchild is the tracer.
 */
inline pid_t tracerPid(pid_t child) {
  char path[96];
  std::snprintf(path, sizeof(path), "/proc/%d/task/%d/children", static_cast<int>(child),
                static_cast<int>(child));
  std::FILE* f = std::fopen(path, "r");
  if (f == nullptr)
    return child;
  long grandchild = 0;
  const int GOT = std::fscanf(f, "%ld", &grandchild);
  std::fclose(f);
  return (GOT == 1 && grandchild > 0) ? static_cast<pid_t>(grandchild) : child;
}

/* ----------------------------- sudoKill ----------------------------- */

/**
 * @brief Deliver @p sig to @p pid, elevating via `sudo -n kill` when the
 * caller is not root.
 *
 * A child spawned through sudo runs as root, and a plain ::kill from its
 * unprivileged parent fails with EPERM -- silently losing e.g. bpftrace's
 * SIGINT-triggered END-block flush. @return true when delivery succeeded.
 */
inline bool sudoKill(pid_t pid, int sig) {
  if (::geteuid() == 0)
    return ::kill(pid, sig) == 0;
  const ReadinessContext CTX = ReadinessContext::capture();
  const auto SUDO = resolveExecutable("sudo", CTX);
  const auto KILL = resolveExecutable("kill", CTX);
  if (!SUDO || !SUDO->executable || !KILL || !KILL->executable) {
    return false;
  }
  return runBoundedProbe({SUDO->path, "-n", "--", KILL->path, "-" + std::to_string(sig),
                          std::to_string(static_cast<int>(pid))},
                         5000, CTX)
      .succeeded();
}

} // namespace profiler_env
} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERENV_HPP
