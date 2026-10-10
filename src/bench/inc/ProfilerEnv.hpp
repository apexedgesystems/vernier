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
#include <optional>
#include <stdexcept>
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

/* ----------------------------- computeSanitizerSession ----------------------------- */

/**
 * @brief True when Compute Sanitizer started the process @p ctx describes.
 *
 * compute-sanitizer cannot attach to a running process, so a session exists
 * only when the tool started this one. `bench run --profile
 * compute-sanitizer` says so through VERNIER_EXTERNAL_WRAP. A wrap typed by
 * hand is recognised from what the tool exports to the process it starts:
 * NV_SANITIZER_INJECTION_PORT_BASE, as exported by compute-sanitizer 2025.3
 * and 2025.4, or CUDA_INJECTION64_PATH naming its collection library, the
 * injection variable a toolkit may use instead. A name in a path decides
 * nothing: the process's own binary or a directory may be called after the
 * tool. Reads only the snapshot.
 */
inline bool computeSanitizerSession(const ReadinessContext& ctx) {
  if (ctx.get("VERNIER_EXTERNAL_WRAP").value_or("") == "compute-sanitizer") {
    return true;
  }
  if (ctx.get("NV_SANITIZER_INJECTION_PORT_BASE")) {
    return true;
  }
  const std::string INJECTION = ctx.get("CUDA_INJECTION64_PATH").value_or("");
  const std::size_t SLASH = INJECTION.rfind('/');
  const std::string_view FILE = SLASH == std::string::npos
                                    ? std::string_view{INJECTION}
                                    : std::string_view{INJECTION}.substr(SLASH + 1);
  return FILE == "libsanitizer-collection.so";
}

/** @brief computeSanitizerSession() on a snapshot of this process, taken now. */
inline bool computeSanitizerSession() {
  return computeSanitizerSession(ReadinessContext::capture());
}

/**
 * @brief True when @p mapsText, the text of a /proc/<pid>/maps, shows one of
 * compute-sanitizer's own libraries mapped into the process.
 *
 * Those are libsanitizer-collection.so and libsanitizer-public.so, matched as
 * the file name at the end of a line (compute-sanitizer 2025.3 and 2025.4
 * map both). The bare word "sanitizer" would also match a binary or a
 * directory named after the tool, and the launcher libraries the tool maps
 * beside its own (libTreeLauncher*, libInterceptorInjectionTarget) ship with
 * Nsight too, so neither decides.
 */
inline bool mapsShowComputeSanitizer(std::string_view mapsText) {
  std::size_t from = 0;
  while (from < mapsText.size()) {
    std::size_t end = mapsText.find('\n', from);
    if (end == std::string_view::npos) {
      end = mapsText.size();
    }
    const std::string_view LINE = mapsText.substr(from, end - from);
    from = end + 1;
    const std::size_t SLASH = LINE.rfind('/');
    if (SLASH == std::string_view::npos) {
      continue;
    }
    const std::string_view FILE = LINE.substr(SLASH + 1);
    if (FILE == "libsanitizer-collection.so" || FILE == "libsanitizer-public.so") {
      return true;
    }
  }
  return false;
}

/**
 * @brief True when this process runs under compute-sanitizer: the session
 * a snapshot of it shows (computeSanitizerSession()), or, for a tool
 * version that exports nothing, the tool's library mapped into the process
 * (mapsShowComputeSanitizer() on /proc/self/maps).
 * @note NOT RT-safe: reads /proc/self/maps.
 */
inline bool isRunningUnderComputeSanitizer() {
  if (computeSanitizerSession()) {
    return true;
  }
  std::FILE* fp = std::fopen("/proc/self/maps", "r");
  if (!fp) {
    return false;
  }
  std::string maps;
  char buf[4096];
  while (const std::size_t GOT = std::fread(buf, 1, sizeof(buf), fp)) {
    maps.append(buf, GOT);
  }
  std::fclose(fp);
  return mapsShowComputeSanitizer(maps);
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

/* ----------------------------- nsightSessionTool ----------------------------- */

/**
 * @brief The Nsight tool running the process @p ctx describes: "nsys", "ncu",
 * or "" for none.
 *
 * Neither nsys nor ncu can attach to a process that is already running, so a
 * session exists only when the tool started this process. `bench run --profile
 * nsight|ncu` says so through VERNIER_EXTERNAL_WRAP; its nsight wrap is nsys,
 * or ncu for nsight's compute mode. A wrap typed by hand is recognised from
 * the variables each tool exports to the process it starts:
 * NSYS_PROFILING_SESSION_ID (nsys) and NV_NSIGHT_INJECTION_PORT_BASE (ncu),
 * as exported by nsys 2025.3 and ncu 2025.3. Reads only the snapshot.
 */
inline std::string nsightSessionTool(const ReadinessContext& ctx) {
  const std::string WRAP = ctx.get("VERNIER_EXTERNAL_WRAP").value_or("");
  if (WRAP == "nsight" || WRAP == "nsys") {
    return ctx.get("NV_NSIGHT_INJECTION_PORT_BASE") ? "ncu" : "nsys";
  }
  if (WRAP == "ncu") {
    return "ncu";
  }
  if (ctx.get("NSYS_PROFILING_SESSION_ID")) {
    return "nsys";
  }
  if (ctx.get("NV_NSIGHT_INJECTION_PORT_BASE")) {
    return "ncu";
  }
  return {};
}

/** @brief nsightSessionTool() on a snapshot of this process, taken now. */
inline std::string nsightSessionTool() { return nsightSessionTool(ReadinessContext::capture()); }

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

/* ----------------------------- cuptiDecision ----------------------------- */

/** @brief What the in-process CUPTI collector does in this process, and why. */
struct CuptiDecision {
  bool yields = false; ///< Stand down: an Nsight session owns the process, or the override asks
  std::string error;   ///< Not empty when VERNIER_DISABLE_CUPTI is not a boolean
  std::string remedy;  ///< How to fix @ref error; empty when there is none
};

/**
 * @brief Whether the in-process CUPTI collector stays off, decided from the
 * snapshot @p ctx alone.
 *
 * With the collector registered, an nsys session records no kernels (nsys
 * 2025.3.2), and under ncu (2025.3.1) the collector records nothing while ncu
 * profiles every launch. So the collector stands down when:
 *  1. VERNIER_DISABLE_CUPTI is true (the explicit override), or
 *  2. an nsys or ncu session owns this process (nsightSessionTool()): one that
 *     `bench run` started, or one typed by hand, recognised from the variables
 *     the tool exports to its target.
 *
 * VERNIER_DISABLE_CUPTI is read with parseEnvBool(), the grammar of every
 * boolean setting. True (1, true, yes or on, in any case) is the override;
 * false (0, false, no, off or empty, in any case) leaves the collector on and
 * never keeps it on inside a session; unset is no override. Any other value is
 * a configuration error: @ref CuptiDecision::error names it and
 * @ref CuptiDecision::remedy lists the accepted values, and the caller must not
 * register with CUPTI. As a readiness result it is ReadinessCause::CONFIGURATION
 * with those two texts, the report whose text cuptiMustYield() throws.
 * `--profile nsight|nsys|ncu` alone starts no session. The session variables
 * are what those tool versions export, not a promised interface; with a version
 * that does not export them, set VERNIER_DISABLE_CUPTI=1 when wrapping. The
 * collector and the GPU harness decide with this function, and so should a
 * readiness check, so that they agree.
 */
inline CuptiDecision cuptiDecision(const ReadinessContext& ctx) {
  CuptiDecision decision;
  const std::optional<std::string> RAW = ctx.get("VERNIER_DISABLE_CUPTI");
  const EnvBool SETTING = parseEnvBool(RAW);
  if (SETTING == EnvBool::INVALID) {
    decision.error = "VERNIER_DISABLE_CUPTI='" + *RAW + "' is not a boolean";
    decision.remedy = "Use 1, true, yes or on to turn the in-process CUPTI collector off; 0, "
                      "false, no, off or an empty value to leave it on (it stands down inside "
                      "an Nsight session either way).";
    return decision;
  }
  decision.yields = SETTING == EnvBool::TRUE_VALUE || !nsightSessionTool(ctx).empty();
  return decision;
}

/** @brief cuptiDecision() on a snapshot of this process, taken now. */
inline CuptiDecision cuptiDecision() { return cuptiDecision(ReadinessContext::capture()); }

/**
 * @brief True when the in-process CUPTI collector must stay off for this run
 * (cuptiDecision() on a snapshot of this process).
 * @throws std::invalid_argument when VERNIER_DISABLE_CUPTI is not a boolean,
 *         with the text of the CONFIGURATION report readinessResult() gives for
 *         the decision's error and remedy: its message, ". ", its hint.
 */
inline bool cuptiMustYield() {
  const CuptiDecision DECISION = cuptiDecision();
  if (!DECISION.error.empty()) {
    const ReadinessResult RESULT =
        readinessResult(ReadinessCause::CONFIGURATION, DECISION.error, DECISION.remedy);
    throw std::invalid_argument(RESULT.report.message + ". " + RESULT.report.hint);
  }
  return DECISION.yields;
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
  const std::vector<pid_t> CHILDREN = childProcesses(child);
  return CHILDREN.size() == 1 ? CHILDREN.front() : child;
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
