#ifndef VERNIER_PROFILERJEMALLOC_HPP
#define VERNIER_PROFILERJEMALLOC_HPP
/**
 * @file ProfilerJemalloc.hpp
 * @brief jemalloc heap profiler backend: jemalloc's own sampling profiler.
 *
 * jemalloc's heap profiler, inside the allocator, records a sample of the
 * allocations (about one per 512 KiB allocated, by default) with their call
 * stacks. It exists only in a jemalloc built with profiling (configure
 * --enable-prof). Debian's packaging, which Ubuntu's follows, enables it on
 * amd64 and i386 only; the arm64 libjemalloc2 of both rejects prof:true. The
 * benchmark is not rebuilt: the library is preloaded and MALLOC_CONF turns
 * profiling on, with a dump of the whole process written when it exits
 * (prof_final) holding the bytes allocated over the run (prof_accum):
 *
 *   bench run ./MyTest --profile jemalloc --gtest_filter='Foo.Bar'
 *
 *   # or by hand:
 *   LD_PRELOAD=libjemalloc.so.2 \
 *   MALLOC_CONF=prof:true,prof_final:true,prof_accum:true,prof_prefix:./jeprof \
 *       ./MyTest --profile jemalloc --gtest_filter='Foo.Bar'
 *
 *   jeprof --text --alloc_space ./MyTest ./jeprof.<pid>.0.f.heap
 *
 * The loader resolves libjemalloc.so.2 through LD_LIBRARY_PATH first, so a
 * jemalloc built with profiling outside the system's library folders is
 * selected by putting its folder first there.
 *
 * Who decides what: this backend decides whether this process's jemalloc
 * collects (the doctor's probe and the run's own allocator answer the same
 * question); `bench run` prepares the wrap and, after the process exits,
 * checks the dump it wrote. jeprof only reads dumps: it never decides
 * whether anything is collected.
 *
 * When to reach for which:
 *   - massif       full timeline; every instruction runs under valgrind
 *   - heaptrack    every allocation with its stack; cost grows with the
 *                  allocation rate
 *   - jemalloc     a sample of the allocations, inside the allocator; needs a
 *                  jemalloc built with profiling
 */

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"
#include "src/bench/inc/ProfilerReadiness.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- Constants ----------------------------- */

/** @brief What `bench run` preloads: jemalloc's soname, which the loader resolves. */
inline constexpr const char* JEMALLOC_PRELOAD = "libjemalloc.so.2";

/** @brief How long the doctor's jemalloc probe may run. */
inline constexpr int JEMALLOC_PROBE_TIMEOUT_MS = 30000;

/* ----------------------------- Facts ----------------------------- */

/**
 * @brief What a process's allocator says of itself, through jemalloc's
 * mallctl, and where its library is mapped.
 *
 * Every field but @ref jemalloc, @ref sanitizer, @ref ubsanLinkedIn and
 * @ref library is read only when jemalloc is the allocator, and the profiling
 * ones only when it is built with profiling.
 */
struct JemallocRuntimeFacts {
  bool sanitizer = false; ///< The address or thread sanitizer's runtime is in the process.
  /// The undefined-behaviour sanitizer's runtime is linked into the program
  /// itself, as clang links it: a preloaded jemalloc does not take
  /// MALLOC_CONF there (GCC's runtime, a shared library, leaves it alone).
  bool ubsanLinkedIn = false;
  bool jemalloc = false;     ///< mallctl resolves in the process: jemalloc is its allocator.
  std::string library;       ///< The mapped libjemalloc's path; "" when none is mapped.
  std::string version;       ///< jemalloc's "version".
  bool configProf = false;   ///< "config.prof": built with profiling.
  bool optProf = false;      ///< "opt.prof": MALLOC_CONF turned profiling on.
  bool profActive = false;   ///< "prof.active": sampling now.
  bool optProfFinal = false; ///< "opt.prof_final": a dump when the process exits.
  bool optProfAccum = false; ///< "opt.prof_accum": dumps hold the bytes allocated over the run.
  bool optProfGdump = false; ///< "opt.prof_gdump": a dump each time the heap's peak grows.
  long lgProfInterval = -1;  ///< "opt.lg_prof_interval": -1, no interval dumps.
  std::string prefix;        ///< "opt.prof_prefix": the dumps' path prefix.
};

/** @brief This process's facts: mallctl looked up at run time, /proc/self/maps. */
[[nodiscard]] JemallocRuntimeFacts jemallocRuntimeFacts();

/** @brief A file the doctor's probe left in its private directory. */
struct JemallocDump {
  std::string name;      ///< Its name, e.g. "jeprof.4711.0.f.heap".
  std::string firstLine; ///< Its first line, e.g. "heap_v2/524288".
};

/**
 * @brief What the doctor's probe observed: the probe program run once with
 * jemalloc preloaded and the options `bench run` sets.
 */
struct JemallocProbeFacts {
  ProbeResult run;                 ///< output: the probe program's memory map; errorOutput:
                                   ///< what the loader and jemalloc printed.
  std::vector<JemallocDump> dumps; ///< What it left in its private directory.
};

/** @brief Facts given to the check instead of reading them, for tests. */
struct JemallocFacts {
  const JemallocRuntimeFacts* runtime = nullptr; ///< Runtime scope: this process's when null.
  const JemallocProbeFacts* probe = nullptr;     ///< Doctor scopes: the probe runs when null.
};

/* ----------------------------- JemallocPlan ----------------------------- */

/** @brief What the jemalloc check verified, for the profiler to use. */
struct JemallocPlan final : ReadinessPlan {
  std::string library;                              ///< The library the decision named.
  std::string prefix;                               ///< Runtime scope: where the dumps go.
  bool finalDump = true;                            ///< Runtime scope: a dump at exit.
  LaunchContext launch = LaunchContext::IN_PROCESS; ///< Runtime scope: how the wrap began.
};

/* ----------------------------- JemallocProfiler ----------------------------- */

class JemallocProfiler final : public Profiler {
public:
  /**
   * @brief Decide the request now and report it as the registry does: one
   * that cannot collect fails the run and creates nothing.
   */
  JemallocProfiler(const PerfConfig& cfg, std::string testName);

  /** @brief Build from a decision that lets the request run. */
  JemallocProfiler(const PerfConfig& cfg, std::string testName,
                   std::shared_ptr<const JemallocPlan> plan);

  ~JemallocProfiler() override = default;

  std::string toolName() const noexcept override { return "jemalloc"; }
  std::string artifactDir() const noexcept override { return artifactDir_; }

  /** @brief Say where jemalloc writes its dump, and when; nothing is written yet. */
  void beforeMeasure() override;
  void afterMeasure(const Stats& s) override;

private:
  PerfConfig cfg_;
  std::string testName_;
  std::string artifactDir_;
  std::shared_ptr<const JemallocPlan> plan_;
};

/* --------------------------------- API --------------------------------- */

/**
 * @brief The variables `bench run` sets to wrap a benchmark writing into
 * @p dir: the preload and jemalloc's options. The route table both languages
 * test against pins them (tools/rust/tests/fixtures/profile_routes.tsv).
 */
[[nodiscard]] std::vector<std::pair<std::string, std::string>>
jemallocWrapEnvironment(const std::string& dir);

/**
 * @brief The jemalloc backend's readiness decision for @p request in @p ctx.
 *
 * Any --profile-args word is a configuration error (jemalloc takes no mode),
 * and a program built with the address or thread sanitizer (whose allocator
 * serves it), or with clang's undefined-behaviour sanitizer linked in (a
 * preloaded jemalloc does not take MALLOC_CONF there), is unsupported. The
 * doctor's scopes run one bounded
 * probe: `/usr/bin/env LD_PRELOAD=libjemalloc.so.2 MALLOC_CONF=<bench run's
 * options, the prefix in a private directory> /bin/cat /proc/self/maps`. The
 * loader's refusal, and a jemalloc that rejects prof:true (built without
 * profiling), are MISSING; a final dump in the private directory is READY,
 * naming the library the loader resolved. A run asks its own allocator
 * through mallctl, starting no process: not jemalloc, or jemalloc built
 * without profiling, is MISSING; profiling off although MALLOC_CONF asks for
 * it is UNUSABLE (jemalloc did not take MALLOC_CONF); profiling off, or no
 * dump configured, is a configuration error; sampling off at start, no dump
 * at exit beside interval
 * or growth dumps, or bytes allocated not accumulated, is a caveat. jeprof is
 * named beside a READY verdict and never changes one. --profile-analyze is an
 * analysis-stage error: jemalloc has no automatic analysis. On success the
 * result's plan is a JemallocPlan.
 */
ReadinessResult checkJemallocRequest(const ReadinessRequest& request, const ReadinessContext& ctx);

/** @brief checkJemallocRequest() with @p facts in place of the ones it reads or probes. */
ReadinessResult checkJemallocRequestWithFacts(const ReadinessRequest& request,
                                              const ReadinessContext& ctx,
                                              const JemallocFacts& facts);

std::unique_ptr<Profiler> makeJemallocProfiler(const PerfConfig& cfg, const std::string& testName);

} // namespace bench
} // namespace vernier

#endif // VERNIER_PROFILERJEMALLOC_HPP
