#ifndef VERNIER_VALGRINDTOOL_HPP
#define VERNIER_VALGRINDTOOL_HPP
/**
 * @file ValgrindTool.hpp
 * @brief What the valgrind backends (callgrind, massif, memcheck, helgrind
 * and its drd mode) share: the words of a request's mode, the arguments of
 * the wrap, the start probe the doctor runs, the tool a process's memory map
 * shows running it, and the result of a request whose process runs under
 * none.
 *
 * Each backend parses its own mode (the words it takes and the options they
 * add), decides its own analysis and builds its check from these pieces.
 * Nothing here decides a backend's policy. The words of a mode, and the
 * refusal of one, are those `bench run` reads for every wrapped tool, so the
 * heaptrack and rocprof backends read theirs with the same two functions.
 *
 * @note NOT RT-safe (heap allocation, fork/exec probes, file reads).
 */

#include <sys/types.h>

#include <optional>
#include <string>
#include <vector>

#include "src/bench/inc/ProfilerReadiness.hpp"

namespace vernier {
namespace bench {
namespace valgrind_tool {

/* ----------------------------- Mode ----------------------------- */

/** @brief The valgrind tool and options a request runs under. */
struct ValgrindMode {
  std::string tool; ///< valgrind's --tool: callgrind, massif, memcheck, helgrind or drd.
  std::vector<std::string> options; ///< The mode's options, in the order the wrap passes them.
  std::vector<std::string>
      unseen; ///< Those of them the request's words add, which a process cannot see.
};

/**
 * @brief The words of a --profile-args value, split on whitespace and commas,
 * as `bench run` splits them.
 */
[[nodiscard]] std::vector<std::string> modeWords(const std::string& profileArgs);

/**
 * @brief The refusal of a word @p backend does not take: a CONFIGURATION error
 * "'<word>' is not a mode of <backend>; its modes are <accepted>", or "...,
 * which takes none" when @p accepted is empty, as `bench run` refuses it.
 */
[[nodiscard]] ReadinessResult refusedWord(const std::string& backend, const std::string& word,
                                          const std::vector<std::string>& accepted);

/** @brief The output option of @p backend's wrap and the file it writes. */
struct OutputFile {
  std::string option; ///< "--callgrind-out-file", "--massif-out-file" or "--log-file".
  std::string name;   ///< "callgrind.out", "massif.out", "memcheck.log" or "helgrind.log".
};

/** @brief @p backend's output file; empty for a name that is not a valgrind backend. */
[[nodiscard]] OutputFile outputFile(const std::string& backend);

/**
 * @brief valgrind's arguments for @p mode writing into @p dir, before the
 * benchmark: --tool, the mode's options and the output option.
 *
 * The same as the route `bench run` starts (tools/rust/src/bench/runner.rs);
 * the shared route table pins the two together.
 */
[[nodiscard]] std::vector<std::string>
wrapArguments(const std::string& backend, const ValgrindMode& mode, const std::string& dir);

/**
 * @brief The remedy of a request that is not wrapped as it needs: the valgrind
 * command that wraps the benchmark by hand (with @p extraOptions after the
 * tool, writing into the working directory), then `bench run` for the same
 * request.
 *
 * "Wrap it: valgrind --tool=massif --massif-out-file=./massif.out
 * <this-binary> --profile massif [...]; or run it with bench run --profile
 * massif, which wraps it." The wrap command comes first and ends at
 * `<this-binary>`, so a script can take it from the report.
 */
[[nodiscard]] std::string wrapRemedy(const std::string& backend, const ValgrindMode& mode,
                                     const std::string& profileArgs,
                                     const std::vector<std::string>& extraOptions = {});

/* ----------------------------- Start Probe ----------------------------- */

/** @brief How long the doctor's start probe of one tool may run. */
inline constexpr int PROBE_TIMEOUT_MS = 30000;

/**
 * @brief Whether valgrind starts @p mode's tool here, for the doctor.
 *
 * Resolves valgrind and /bin/true in @p ctx, then runs `valgrind --tool=<tool>
 * <options> <output option>=/dev/null /bin/true`, bounded, so the probe
 * writes nothing. Sets @p valgrindPath to the resolved valgrind.
 * @return READY naming the probe; MISSING without valgrind; MISSING_HELPER
 *         without /bin/true; UNUSABLE, with valgrind's words, when the tool
 *         does not start.
 */
[[nodiscard]] ReadinessResult probeTool(const std::string& backend, const ValgrindMode& mode,
                                        const ReadinessContext& ctx, std::string& valgrindPath);

/** @brief The MISSING result of a request whose valgrind is not on PATH. */
[[nodiscard]] ReadinessResult valgrindMissing();

/* ----------------------------- Identity ----------------------------- */

/**
 * @brief What a process's memory map shows of the valgrind tool running it.
 *
 * valgrind maps its core preload (vgpreload_core-<platform>.so) into every
 * process it runs, and the executable of the tool running it
 * (<tool>-<platform>, in the same directory); memcheck, massif, helgrind and
 * drd also map a preload of their own (vgpreload_<tool>-<platform>.so),
 * callgrind none. Only files beside the core preload count, so a benchmark
 * named like a tool is not taken for one.
 */
struct ValgrindIdentity {
  bool underValgrind = false;           ///< The core preload is mapped.
  std::vector<std::string> executables; ///< Tools whose executable is mapped, sorted.
  std::vector<std::string> preloads;    ///< Tools whose own preload is mapped, sorted.

  /**
   * @brief The one tool this establishes: exactly one tool executable, and no
   * preload of another tool. Empty otherwise, the core preload alone included.
   */
  [[nodiscard]] std::string tool() const;

  /** @brief What the map shows, for reports. */
  [[nodiscard]] std::string describe() const;
};

/** @brief The identity @p mapsText (a /proc/<pid>/maps) shows. */
[[nodiscard]] ValgrindIdentity identityFromMaps(const std::string& mapsText);

/** @brief The identity of process @p pid: not under valgrind when its map cannot be read. */
[[nodiscard]] ValgrindIdentity identityOf(pid_t pid);

/* ----------------------------- Decisions ----------------------------- */

/** @brief What a valgrind backend's check verified, for the launch to use. */
struct ValgrindPlan final : ReadinessPlan {
  std::string backend;                              ///< The registered name.
  ValgrindMode mode;                                ///< The tool and options of the request.
  LaunchContext launch = LaunchContext::IN_PROCESS; ///< At the runtime scope: how the wrap began.
  std::string valgrind;                             ///< The resolved valgrind (doctor scopes).
  bool canToggle = false; ///< callgrind: instrumentation can be switched for the window.
};

/**
 * @brief The runtime collection decision for @p mode in a process whose map
 * shows @p identity.
 *
 * READY when the requested tool runs the process, from `bench run`'s wrap
 * (@p launch RUNNER_WRAPPED) or one started by hand; UNVERIFIED for a wrap
 * started by hand whose mode adds options the process cannot see, and when
 * valgrind runs the process but its map does not establish the tool;
 * UNSUPPORTED when another tool runs it (naming the request with
 * @p profileArgs' words); when no valgrind runs it,
 * valgrindMissing() without valgrind on @p ctx's PATH and otherwise MISSING:
 * "<backend> collects only when valgrind's <tool> runs the process, and
 * valgrind does not run this one". Every Error carries @p remedy. The
 * plan names the launch context: RUNNER_WRAPPED, MANUALLY_WRAPPED or
 * NOT_WRAPPED.
 */
[[nodiscard]] ReadinessResult decideRuntime(const std::string& backend, const ValgrindMode& mode,
                                            const std::string& profileArgs,
                                            const ValgrindIdentity& identity, LaunchContext launch,
                                            const std::string& remedy, const ReadinessContext& ctx);

/**
 * @brief The collection decision for @p request: the start probe at the
 * doctor's scopes (whose process is never wrapped), decideRuntime() at the
 * runtime scope on @p identity, or, when it is null, on the map of @p ctx's
 * process.
 */
[[nodiscard]] ReadinessResult decideCollection(const std::string& backend, const ValgrindMode& mode,
                                               const ReadinessRequest& request,
                                               const ReadinessContext& ctx,
                                               const std::string& remedy,
                                               const ValgrindIdentity* identity = nullptr);

/**
 * @brief @p collection with the request's analysis decided as @p analysis:
 * an analysis-stage Error replaces the result and keeps its plan, so the
 * capture still runs and the run fails at its end.
 */
[[nodiscard]] ReadinessResult withAnalysis(ReadinessResult collection, ReadinessResult analysis);

} // namespace valgrind_tool
} // namespace bench
} // namespace vernier

#endif // VERNIER_VALGRINDTOOL_HPP
