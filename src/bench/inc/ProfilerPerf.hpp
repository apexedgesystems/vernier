#ifndef VERNIER_PROFILERPERF_HPP
#define VERNIER_PROFILERPERF_HPP
/**
 * @file ProfilerPerf.hpp
 * @brief Linux perf backend for the benchmarking profiler facade.
 *
 * Behavior:
 *  - Default: `perf stat -e cpu-cycles,instructions,branches,branch-misses,cache-misses -p <PID>`
 *             Output is written to `<artifactDir>/stat.txt` (perf writes to stderr; we redirect).
 *  - If `cfg.profileArgs` begins with "record", we run `perf record <args> -p <PID>` instead and
 *    write `<artifactDir>/perf.data` (+ `record.err.txt`); "mem" and "c2c" select
 *    `perf mem record` and `perf c2c record`.
 *  - Only the measured window is profiled: perf starts in beforeMeasure() and is
 *    stopped in afterMeasure(), which returns once perf has finished writing.
 *  - The measured phase starts once perf answers a `ping` on its `--control`
 *    fifo, which it does from its main loop after its counters are on
 *    (bounded at PERF_ACK_WAIT_MS; no answer fails the request). `perf mem`,
 *    and a perf the check found not answering, start after a fixed
 *    PERF_START_GRACE_MS instead.
 *  - perf runs as this process's own child (OwnedHelper): a perf that ends
 *    before or during the measured phase, needs SIGTERM or SIGKILL to stop, or
 *    leaves no counts (stat) or no data (record, mem, c2c) is reported through
 *    ProfilerRegistry::reportFailure(), which fails the run.
 *
 * Readiness (checkPerfRequest): perf is resolved on PATH, `perf --version`
 * must run, and a bounded `perf stat` on this process must open the counters
 * as this user; that access, not the perf_event_paranoid value, decides. The
 * same probe sends a `ping` on a `--control` fifo, and a perf that does not
 * answer it is a caveat. The profiler launches the absolute path that check
 * verified (PerfPlan).
 *
 * Notes:
 *  - Linux-only. Safe no-op on other platforms (compile-time guard).
 *  - vernier never elevates perf: counter access comes from root, CAP_PERFMON or
 *    kernel.perf_event_paranoid.
 */

#include <cstdint>
#include <memory>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- Readiness ----------------------------- */

/** @brief What `--profile-args` selects for perf. */
enum class PerfMode : std::uint8_t { STAT, RECORD, MEM, C2C };

/** @brief The perf mode @p profileArgs selects (one parser for check and launch). */
PerfMode parsePerfMode(const std::string& profileArgs);

/** @brief The events the stat mode and the access probe count. */
inline constexpr const char* PERF_STAT_EVENTS =
    "cpu-cycles,instructions,branches,branch-misses,cache-misses";

/** @brief What the perf check verified, for the launch to use. */
struct PerfPlan final : ReadinessPlan {
  std::string perf; ///< Absolute path of the perf that ran the probes.
  PerfMode mode = PerfMode::STAT;
  bool answersPing = false; ///< The probe's perf answered a `ping` on its `--control` fifo.
};

/**
 * @brief The perf backend's readiness decision for @p request in @p ctx.
 *
 * On success the result's plan is a PerfPlan. Modes other than stat get the
 * same access probe and are reported unverified beyond it.
 */
ReadinessResult checkPerfRequest(const ReadinessRequest& request, const ReadinessContext& ctx);

/* ----------------------------- PerfStatProfiler ----------------------------- */

/**
 * @brief Linux perf profiler implementation.
 *
 * Supports both `perf stat` (default) and `perf record` modes.
 * Attaches to running process via `-p <PID>`. perf is started as
 * `/bin/sh -c "exec <perf command>"`, so `--profile-args` stays shell text and
 * perf itself is the owned child that afterMeasure() stops: SIGINT, with
 * PERF_WRITE_WAIT_MS for perf to write, then SIGTERM, then SIGKILL.
 */
class PerfStatProfiler final : public Profiler {
public:
  /**
   * @brief Construct perf profiler, deciding the request itself; when it
   * cannot run, prints why and does nothing in the hooks.
   * @param cfg Configuration with profileArgs and artifactRoot
   * @param testName Test identifier (e.g., "Suite.Case")
   */
  PerfStatProfiler(const PerfConfig& cfg, std::string testName);

  /** @brief Construct from a decision already made (the registry's path). */
  PerfStatProfiler(const PerfConfig& cfg, std::string testName,
                   std::shared_ptr<const PerfPlan> plan);
  ~PerfStatProfiler() override = default;

  std::string toolName() const noexcept override { return "perf"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

  /** @brief How long perf may take to write its output after SIGINT. */
  static constexpr int PERF_WRITE_WAIT_MS = 5000;

  /** @brief How long the measured phase waits for perf to answer that it is counting. */
  static constexpr int PERF_ACK_WAIT_MS = 5000;

  /** @brief The fixed start for a perf that cannot answer on `--control`, and for `perf mem`. */
  static constexpr int PERF_START_GRACE_MS = 200;

private:
  /** @brief Record a failure of this test's capture at @p stage. */
  void fail(ReadinessCause cause, const std::string& detail, ReadinessStage stage) const;

  /** @brief Check the output perf left after a clean stop. */
  void checkOutput() const;

  // State
  PerfConfig cfg_;
  std::string testName_;
  std::string artifactDir_;
  std::shared_ptr<const PerfPlan> plan_;

#ifdef __linux__
  OwnedHelper helper_;
  bool started_ = false;
  std::string statPath_;
  std::string dataPath_;
  std::string errPath_;
#endif
};

/**
 * @brief Factory function for perf profiler.
 *
 * Decides the request in a snapshot of this process first.
 * @return Profiler instance, or nullptr if the request cannot run here.
 */
std::unique_ptr<Profiler> makePerfProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERPERF_HPP
