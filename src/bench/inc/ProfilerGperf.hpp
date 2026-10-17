#ifndef VERNIER_PROFILERGPERF_HPP
#define VERNIER_PROFILERGPERF_HPP
/**
 * @file ProfilerGperf.hpp
 * @brief gperftools backend for the benchmarking profiler facade.
 *
 * Modes (the words of --profile-args):
 *  - cpu, or no word (the default): generates "<artifactDir>/cpu.prof"
 *  - heap: starts gperftools' heap profiler
 *  - both, or cpu and heap together: both captures
 *
 * Readiness (checkGperfRequest): any other word is refused as a configuration
 * error; every requested mode must be compiled in, and a request that is not
 * is a collection error. With --profile-analyze the analyzer is the first of
 * google-pprof and pprof found on PATH, and it must run; a missing or broken
 * analyzer is an analysis error, and the capture still runs and keeps
 * cpu.prof. The analysis runs exactly the analyzer the check found.
 *
 * Sampling rate (decideGperfSampling): gperftools reads CPUPROFILE_FREQUENCY
 * once, as the process starts, and samples 100 times a second without it. A
 * rate asked for (--profile-frequency, PerfConfig::profileFrequency) is
 * therefore applied before the start: `bench run` sets the variable on the
 * benchmark it starts, and a benchmark started directly needs it in its own
 * environment. The check refuses a request this process's environment does
 * not carry, with the command that does, and a value in the environment that
 * is not a whole number from 1 to 4000. Every finished cpu.prof is read
 * (readGperfProfile): one that is not a complete profile, or that records
 * another sampling period than the decision expects, fails its case and is
 * kept.
 *
 * A run fails a case whose capture gperftools does not start (another
 * profile of the same kind already runs in the process, or the file cannot
 * be written), one whose previous run's files cannot all be removed before
 * it starts (it then does not start, since what it wrote could not be told
 * from them), and one whose stopped capture leaves no file of this run, or
 * an empty one, whether or not an analysis follows; only a capture that left
 * its file is analyzed. A profiler stops only the captures it started: when
 * its measured window ends (afterMeasure()), at once when the window ends by
 * an exception (abortMeasure(), unanalyzed), and before it starts again if a
 * window ended through neither.
 *
 * Notes:
 *  - Requires gperftools headers/libraries to be available at build/link time.
 *  - Heap profiling additionally requires a build with
 *    -DVERNIER_LINK_TCMALLOC=ON: the heap profiler is part of tcmalloc, and
 *    tcmalloc replaces the allocator for the whole process, so it is never
 *    linked implicitly. Without it, a heap request is a readiness error that
 *    says how to enable it.
 *  - If unavailable, makeGperfProfiler(...) returns nullptr, reporting why
 *    as the registry does.
 */

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp" // base

// Detect availability at compile time
#if __has_include(<gperftools/profiler.h>)
#define UB_HAS_GPERF_CPU 1
#else
#define UB_HAS_GPERF_CPU 0
#endif

#if __has_include(<gperftools/heap-profiler.h>) && defined(VERNIER_HAS_TCMALLOC)
#define UB_HAS_GPERF_HEAP 1
#else
#define UB_HAS_GPERF_HEAP 0
#endif

namespace vernier {
namespace bench {

/* ----------------------------- Readiness ----------------------------- */

/** @brief The modes `--profile-args` selects for gperf. */
struct GperfModes {
  bool cpu = false;
  bool heap = false;
};

/**
 * @brief The modes the words of @p profileArgs select, or the refusal of a
 * word that is not one.
 *
 * The words are split on whitespace and commas, as `bench run` splits them.
 * No word selects CPU profiling; "cpu" selects it, "heap" heap profiling and
 * "both" the two, in any combination. Any other word is refused with the
 * CONFIGURATION error every backend gives a word it does not take, naming
 * the modes; @p modes is then left with no mode.
 */
std::optional<ReadinessResult> parseGperfModes(const std::string& profileArgs, GperfModes& modes);

/** @brief Where the sampling rate of a gperf CPU capture comes from. */
enum class GperfRateSource : std::uint8_t {
  DEFAULT,   ///< None asked for and none in the environment: gperftools' own 100 Hz.
  INHERITED, ///< CPUPROFILE_FREQUENCY in the environment, with no rate asked for.
  REQUESTED  ///< A rate asked for, which the environment carries.
};

/** @brief What the gperf check verified, for the profiler to use. */
struct GperfPlan final : ReadinessPlan {
  GperfModes modes;
  bool analyze = false;        ///< --profile-analyze was requested.
  std::string analyzer;        ///< Absolute path of the selected analyzer; empty when none.
  bool analysisReady = false;  ///< The promised analysis can run with that analyzer.
  std::string analysisSkipped; ///< Why it cannot, for the run's note; empty when it can.
  /// The sampling period every finished cpu.prof must record, in microseconds;
  /// unset without a CPU capture, and in a plan made by hand, whose profiles
  /// are checked for their format only.
  std::optional<int> expectedPeriodUs;
  int samplingHz = 0; ///< The rate asked for or inherited; 0 at gperftools' default.
  GperfRateSource samplingSource = GperfRateSource::DEFAULT;
};

/** @brief The analysis half of a gperf decision. */
struct GperfAnalysis {
  std::string analyzer;                 ///< The selected analyzer's path; empty when none.
  std::optional<ReadinessResult> error; ///< The analysis-stage Error, when there is one.
};

/**
 * @brief Which analyzer a promised analysis runs, and whether it can.
 *
 * The analyzer is the first of google-pprof and pprof on the snapshot's PATH.
 * It matters only to a request that promises an analysis of a CPU capture:
 * then a missing analyzer, or one that does not answer `--help`, is an
 * ANALYSIS-stage Error naming it and the remedy, while collection still runs.
 * Otherwise the analyzer is only named, never run. Needs no gperftools, so
 * the rule holds, and is tested, on every build.
 */
GperfAnalysis decideGperfAnalysis(const GperfModes& modes, bool analyze,
                                  const ReadinessContext& ctx);

/** @brief The sampling half of a gperf decision. */
struct GperfSampling {
  /// The sampling period every finished cpu.prof must record, in
  /// microseconds; unset without a CPU capture, and with an error.
  std::optional<int> expectedPeriodUs;
  int hz = 0; ///< The rate asked for or inherited; 0 at gperftools' default.
  GperfRateSource source = GperfRateSource::DEFAULT;
  std::string clause;                   ///< What the READY message adds; empty at the default.
  std::optional<ReadinessResult> error; ///< The CONFIGURATION error, when there is one.
};

/**
 * @brief The sampling rate a capture of @p modes takes, given the rate asked
 * for (@p requestedHz; 0 for none) and the snapshot's CPUPROFILE_FREQUENCY.
 *
 * gperftools reads the variable once, as the process starts, so a rate asked
 * for is taken only when this process's environment carries it. In order: a
 * rate outside 1 to 4000 (only a source value can be one) is refused; a rate
 * with heap profiling only is refused, and without a rate heap profiling
 * alone reads nothing. For a CPU capture: no rate and no variable, the
 * default (period 10000 us, nothing added to the message); no rate and a
 * valid variable, that rate; no rate and an invalid variable, refused; a
 * rate the variable equals as a number, that rate; a rate with no variable
 * or another value, refused, the remedy naming the command that starts the
 * process with it. Needs no gperftools, so the rule holds, and is tested, on
 * every build.
 */
GperfSampling decideGperfSampling(const GperfModes& modes, int requestedHz,
                                  const ReadinessContext& ctx);

/** @brief What a finished gperftools CPU profile records. */
struct GperfProfileSummary {
  bool valid = false;             ///< Its header, every record and its trailer were read.
  long long periodUs = 0;         ///< The sampling period its header records, in microseconds.
  unsigned long long samples = 0; ///< The samples its records count.
  long long binaryBytes = 0;      ///< Where its binary part (header to trailer) ends.
  std::string problem;            ///< Why it is not valid; empty when it is.
};

/**
 * @brief Read the gperftools CPU profile at @p path, in the format gperftools
 * documents: slots the size of a pointer, in the machine's byte order; a
 * header of 0, the number n of header slots that follow (at least 3), the
 * format version 0, the sampling period in microseconds, and padding to n;
 * records of a sample count, a depth and that many program counters; the
 * trailer 0, 1, 0. The list of mapped objects after the trailer is not read.
 *
 * A valid profile's period lies in 250 to 1000000 microseconds, the periods
 * of 4000 Hz to 1 Hz. The profile is this process's own, so its slot size and
 * byte order are this build's. Needs no gperftools.
 */
GperfProfileSummary readGperfProfile(const std::string& path);

/**
 * @brief The gperf backend's readiness decision for @p request in @p ctx.
 *
 * On success, or on an analysis-stage error, the result's plan is a GperfPlan.
 */
ReadinessResult checkGperfRequest(const ReadinessRequest& request, const ReadinessContext& ctx);

/* ----------------------------- GperfProfiler ----------------------------- */

class GperfProfiler final : public Profiler {
public:
  /**
   * @brief Construct, deciding the request itself and reporting it as the
   * registry does: one that cannot run fails the run, creates no folder and
   * does nothing in the hooks.
   */
  GperfProfiler(const PerfConfig& cfg, std::string testName);

  /** @brief Construct from a decision already made (the registry's path). */
  GperfProfiler(const PerfConfig& cfg, std::string testName, std::shared_ptr<const GperfPlan> plan);

  /** @brief Stops a capture beforeMeasure() started that is still on. */
  ~GperfProfiler() override;

  GperfProfiler(const GperfProfiler&) = delete;
  GperfProfiler& operator=(const GperfProfiler&) = delete;

  std::string toolName() const noexcept override { return "gperf"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;
  /** @brief Stops the captures beforeMeasure() started, unanalyzed. */
  void abortMeasure() noexcept override;

private:
  PerfConfig cfg_{};
  std::string testName_;
  std::string artifactDir_;
  std::shared_ptr<const GperfPlan> plan_;

  void applyPlan();
  /// Read the finished cpu.prof and compare its period with the plan's; false
  /// when it is not a complete profile, which is then not analyzed.
  bool checkCapture() const;
  void runPprofAnalysis() const;
  /// Stop the captures this profiler started, without analysis.
  void stopCapture() noexcept;
  /// Record this case's failure, for the run's exit status.
  void fail(ReadinessCause cause, const std::string& detail, const std::string& remedy,
            ReadinessStage stage) const;

  bool wantCpu_{false};
  bool wantHeap_{false};

  // True while a capture beforeMeasure() started runs, so it is stopped
  // exactly once: by afterMeasure(); with no analysis, by abortMeasure() when
  // the measured window ended by an exception, or, when a window ended
  // through neither, by the next beforeMeasure() or the destructor.
  bool cpuActive_{false};
  bool heapActive_{false};

#if UB_HAS_GPERF_CPU
  std::string cpuPath_;
#endif
#if UB_HAS_GPERF_HEAP
  std::string heapPrefix_;
#endif
};

/* --------------------------------- API --------------------------------- */

/**
 * @brief Factory function for gperftools profiler.
 *
 * Decides the request in a snapshot of this process first and reports it
 * as the registry does.
 * @return Profiler instance, or nullptr (the run then fails) if collection
 *         cannot run here.
 */
std::unique_ptr<Profiler> makeGperfProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERGPERF_HPP
