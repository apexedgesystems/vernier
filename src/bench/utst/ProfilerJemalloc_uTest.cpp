/**
 * @file ProfilerJemalloc_uTest.cpp
 * @brief The jemalloc backend's decisions: the doctor's probe, a run's own
 * allocator, the requests it refuses, and what a profiler built from a
 * decision says and creates.
 *
 * Every verdict is checked on every machine from injected facts: what an
 * allocator's mallctl answers, and what the doctor's probe printed, captured
 * on real systems (the Raspberry Pi 4 rig's stock jemalloc, which rejects
 * profiling; a jemalloc 5.3.0 built there with profiling; this laptop's
 * loader, which has no jemalloc to preload). The doctor's probe also runs for
 * real against a file that is not a library, first on LD_LIBRARY_PATH, which
 * every loader refuses. This test program has no jemalloc: a run's decision
 * in it is that of a run without a wrap.
 */

#include "src/bench/inc/ProfilerJemalloc.hpp"

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/ProfilerReadiness.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/utst/ReadinessFixtures.hpp"
#include "src/bench/utst/StderrCapture.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef VERNIER_SHARED_FIXTURE_DIR
#error "VERNIER_SHARED_FIXTURE_DIR must name tools/rust/tests/fixtures"
#endif

using ::testing::HasSubstr;
using ::testing::Not;
using vernier::bench::checkJemallocRequestWithFacts;
using vernier::bench::EnvReport;
using vernier::bench::JemallocDump;
using vernier::bench::JemallocFacts;
using vernier::bench::JemallocPlan;
using vernier::bench::JemallocProbeFacts;
using vernier::bench::JemallocProfiler;
using vernier::bench::JemallocRuntimeFacts;
using vernier::bench::LaunchContext;
using vernier::bench::PerfConfig;
using vernier::bench::ProbeResult;
using vernier::bench::ProfileFailure;
using vernier::bench::ProfilerRegistry;
using vernier::bench::ReadinessCause;
using vernier::bench::ReadinessContext;
using vernier::bench::ReadinessRequest;
using vernier::bench::ReadinessResult;
using vernier::bench::ReadinessScope;
using vernier::bench::ReadinessStage;
using vernier::bench::test::FakeToolDir;
using vernier::bench::test::StderrCapture;

namespace {

/* ----------------------------- Captured Outputs ----------------------------- */

/// The loader with no jemalloc to preload (this laptop, Ubuntu 22.04 x86-64).
constexpr const char* LOADER_REFUSAL =
    "ERROR: ld.so: object 'libjemalloc.so.2' from LD_PRELOAD cannot be preloaded (cannot open "
    "shared object file): ignored.\n";

/// The Pi rig's stock jemalloc (Debian 13 libjemalloc2 5.3.0-3 arm64), preloaded
/// with bench run's options: built without profiling, it rejects each of them.
constexpr const char* STOCK_REJECTION =
    "<jemalloc>: Invalid conf pair: prof:true\n"
    "<jemalloc>: Invalid conf pair: prof_final:true\n"
    "<jemalloc>: Invalid conf pair: prof_accum:true\n"
    "<jemalloc>: Invalid conf pair: prof_prefix:/tmp/vernier_probe_Ab12Cd/jeprof\n";

/// The probe program's map lines for the stock library on the Pi.
constexpr const char* STOCK_MAPS =
    "557e6f0000-557e6fb000 r-xp 00000000 b3:02 133570                         /usr/bin/cat\n"
    "7fab4e0000-7fab569000 r-xp 00000000 b3:02 135370                         "
    "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2\n"
    "7fab569000-7fab57a000 ---p 00089000 b3:02 135370                         "
    "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2\n";

/// The same for jemalloc 5.3.0 built with profiling on the Pi (its folder
/// renamed), selected through LD_LIBRARY_PATH.
constexpr const char* BUILD_MAPS =
    "557e6f0000-557e6fb000 r-xp 00000000 b3:02 133570                         /usr/bin/cat\n"
    "7f90ae0000-7f90b84000 r-xp 00000000 b3:02 1054481                        "
    "/opt/jemalloc-prof/lib/libjemalloc.so.2\n";

/// The probe program's map without jemalloc in it.
constexpr const char* PLAIN_MAPS =
    "557e6f0000-557e6fb000 r-xp 00000000 b3:02 133570                         /usr/bin/cat\n";

/// What that build's exit dump starts with.
constexpr const char* HEAP_HEADER = "heap_v2/524288";

/// The options bench run sets, which the remedies give.
constexpr const char* RUN_OPTIONS = "prof:true,prof_final:true,prof_accum:true";

/* ----------------------------- Helpers ----------------------------- */

/// A request for jemalloc at @p scope.
ReadinessRequest requestFor(ReadinessScope scope, const std::string& profileArgs = "",
                            bool analyze = false,
                            LaunchContext launch = LaunchContext::IN_PROCESS) {
  ReadinessRequest request;
  request.backend = "jemalloc";
  request.scope = scope;
  request.profileArgs = profileArgs;
  request.analyze = analyze;
  request.launch = launch;
  return request;
}

/// A context holding @p env alone.
ReadinessContext contextWith(std::map<std::string, std::string> env = {}) {
  return ReadinessContext(::geteuid(), ::getpid(), std::move(env));
}

/// A jemalloc built with profiling, every profiling option bench run sets on.
JemallocRuntimeFacts profiling() {
  JemallocRuntimeFacts facts;
  facts.jemalloc = true;
  facts.library = "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2";
  facts.version = "5.3.0-0-g54eaed1d8b56b1aa528be3bdd1877e59c56fa90c";
  facts.configProf = true;
  facts.optProf = true;
  facts.profActive = true;
  facts.optProfFinal = true;
  facts.optProfAccum = true;
  facts.prefix = "bench-out/MyTest.jemalloc/jeprof";
  return facts;
}

/// A run's decision on @p facts.
ReadinessResult
runtimeDecision(const JemallocRuntimeFacts& facts, const ReadinessContext& ctx = contextWith(),
                const ReadinessRequest& request = requestFor(ReadinessScope::RUNTIME)) {
  JemallocFacts given;
  given.runtime = &facts;
  return checkJemallocRequestWithFacts(request, ctx, given);
}

/// A probe that exited 0 with @p maps on stdout and @p errors on stderr.
JemallocProbeFacts probeOf(const std::string& maps, const std::string& errors,
                           std::vector<JemallocDump> dumps = {}) {
  JemallocProbeFacts probe;
  probe.run.started = true;
  probe.run.statusObtained = true;
  probe.run.exited = true;
  probe.run.exitCode = 0;
  probe.run.output = maps;
  probe.run.errorOutput = errors;
  probe.dumps = std::move(dumps);
  return probe;
}

/// The doctor's decision on @p probe, in a process without a sanitizer.
ReadinessResult doctorDecision(const JemallocProbeFacts& probe,
                               const ReadinessContext& ctx = contextWith(),
                               ReadinessScope scope = ReadinessScope::PREFLIGHT) {
  static const JemallocRuntimeFacts NO_SANITIZER;
  JemallocFacts given;
  given.runtime = &NO_SANITIZER;
  given.probe = &probe;
  return checkJemallocRequestWithFacts(requestFor(scope), ctx, given);
}

/// The plan a decision carries, or null.
std::shared_ptr<const JemallocPlan> planOf(const ReadinessResult& result) {
  return std::dynamic_pointer_cast<const JemallocPlan>(result.plan);
}

/// A fake jeprof in @p dir: a script, executable.
std::string installJeprof(const FakeToolDir& dir) {
  const std::string PATH = dir.writeFile("jeprof", "#!/bin/sh\nexit 0\n");
  ::chmod(PATH.c_str(), 0755);
  return PATH;
}

/// The shared table's rows of @p kind; '-' is an empty field.
std::vector<std::vector<std::string>> sharedTableRows(const std::string& kind) {
  std::ifstream in(std::string{VERNIER_SHARED_FIXTURE_DIR} + "/profile_routes.tsv");
  std::vector<std::vector<std::string>> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::vector<std::string> fields;
    std::size_t start = 0;
    for (std::size_t tab = line.find('\t'); tab != std::string::npos;
         start = tab + 1, tab = line.find('\t', start)) {
      fields.push_back(line.substr(start, tab - start));
    }
    fields.push_back(line.substr(start));
    if (fields[0] == kind) {
      rows.push_back(fields);
    }
  }
  return rows;
}

} // namespace

/* ----------------------------- The Wrap ----------------------------- */

/** @test The preload and options bench run sets are the shared table's, in order. */
TEST(JemallocWrap, EnvironmentMatchesTheSharedTable) {
  std::vector<std::pair<std::string, std::string>> rows;
  for (const auto& row : sharedTableRows("env")) {
    ASSERT_EQ(row.size(), 4U);
    if (row[1] == "jemalloc") {
      rows.emplace_back(row[2], row[3]);
    }
  }
  EXPECT_EQ(rows, vernier::bench::jemallocWrapEnvironment("<dir>"));
  ASSERT_EQ(rows.size(), 2U) << "the table lost jemalloc's environment rows";
  EXPECT_EQ(rows[1].second, std::string{RUN_OPTIONS} + ",prof_prefix:<dir>/jeprof");
}

/* ----------------------------- A Run ----------------------------- */

/**
 * @test Without jemalloc as its allocator a run collects nothing: MISSING,
 * with the wrap command; when LD_PRELOAD names jemalloc, the loader refused
 * it, and the remedy is a loadable jemalloc with profiling.
 */
TEST(JemallocRun, NotTheAllocatorIsMissing) {
  const JemallocRuntimeFacts NONE;
  const ReadinessResult PLAIN = runtimeDecision(NONE);
  EXPECT_EQ(PLAIN.cause, ReadinessCause::MISSING);
  EXPECT_FALSE(PLAIN.collectionReady());
  EXPECT_EQ(PLAIN.report.message, "missing: jemalloc collects only as the process's allocator, "
                                  "and this process does not run on jemalloc");
  EXPECT_EQ(PLAIN.report.hint,
            std::string{"Wrap it: LD_PRELOAD=libjemalloc.so.2 MALLOC_CONF="} + RUN_OPTIONS +
                ",prof_prefix:./jeprof <this-binary> --profile jemalloc [...]; or run it with "
                "bench run --profile jemalloc, which wraps it.");
  ASSERT_NE(planOf(PLAIN), nullptr);
  EXPECT_EQ(planOf(PLAIN)->launch, LaunchContext::NOT_WRAPPED);

  const ReadinessResult REFUSED =
      runtimeDecision(NONE, contextWith({{"LD_PRELOAD", "libjemalloc.so.2"}}));
  EXPECT_EQ(REFUSED.cause, ReadinessCause::MISSING);
  EXPECT_EQ(REFUSED.report.message,
            "missing: LD_PRELOAD names libjemalloc.so.2, but jemalloc is not this process's "
            "allocator: the loader did not preload it (it said why when this program started)");
  EXPECT_THAT(REFUSED.report.hint, HasSubstr("configure --enable-prof"));
  EXPECT_THAT(REFUSED.report.hint, HasSubstr("LD_LIBRARY_PATH"));
}

/**
 * @test A jemalloc built without profiling is MISSING a build feature,
 * named by its path and version, and the remedy is a build with profiling,
 * never the analyzer.
 */
TEST(JemallocRun, BuiltWithoutProfilingIsMissing) {
  JemallocRuntimeFacts facts = profiling();
  facts.library = "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2";
  facts.configProf = false;
  const ReadinessResult RESULT = runtimeDecision(facts);
  EXPECT_EQ(RESULT.cause, ReadinessCause::MISSING);
  EXPECT_EQ(RESULT.report.message,
            "missing: the jemalloc this process runs on, "
            "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2 "
            "(5.3.0-0-g54eaed1d8b56b1aa528be3bdd1877e59c56fa90c), is built without profiling");
  EXPECT_EQ(RESULT.report.hint,
            "Use a jemalloc built with profiling (configure --enable-prof; Debian's and Ubuntu's "
            "libjemalloc2 have it on amd64 and i386, not on arm64), with its library folder first "
            "on LD_LIBRARY_PATH when it is not the system's; or profile the heap with --profile "
            "heaptrack.");
  EXPECT_THAT(RESULT.report.hint, Not(HasSubstr("jeprof")));

  facts.library.clear();
  EXPECT_THAT(runtimeDecision(facts).report.message,
              HasSubstr("jemalloc 5.3.0-0-g54eaed1d8b56b1aa528be3bdd1877e59c56fa90c, linked into "
                        "this program"));
}

/** @test Profiling built in but not turned on, or turned on with no dump, is a configuration error.
 */
TEST(JemallocRun, OptionsThatWriteNothingAreConfigurationErrors) {
  JemallocRuntimeFacts off = profiling();
  off.optProf = false;
  const ReadinessResult OFF = runtimeDecision(off);
  EXPECT_EQ(OFF.cause, ReadinessCause::CONFIGURATION);
  EXPECT_THAT(OFF.report.message, HasSubstr("its profiling is off: MALLOC_CONF has no prof:true"));
  EXPECT_EQ(OFF.report.hint, std::string{"Set MALLOC_CONF="} + RUN_OPTIONS +
                                 ",prof_prefix:./jeprof, as bench run --profile jemalloc does.");

  JemallocRuntimeFacts silent = profiling();
  silent.optProfFinal = false;
  const ReadinessResult SILENT = runtimeDecision(silent);
  EXPECT_EQ(SILENT.cause, ReadinessCause::CONFIGURATION);
  EXPECT_THAT(SILENT.report.message,
              HasSubstr("but writes no dump: MALLOC_CONF has no prof_final:true, and no interval "
                        "or growth dumps"));
}

/** @test Each limit of a run that still collects is a caveat, and only that one is named. */
TEST(JemallocRun, LimitsAreCaveats) {
  struct Limit {
    const char* what;
    void (*apply)(JemallocRuntimeFacts&);
    const char* words;
  };
  const Limit LIMITS[] = {
      {"interval dumps only",
       [](JemallocRuntimeFacts& f) {
         f.optProfFinal = false;
         f.lgProfInterval = 30;
       },
       "no dump at exit (prof_final is off): only its interval or growth dumps are written"},
      {"growth dumps only",
       [](JemallocRuntimeFacts& f) {
         f.optProfFinal = false;
         f.optProfGdump = true;
       },
       "no dump at exit (prof_final is off)"},
      {"sampling off", [](JemallocRuntimeFacts& f) { f.profActive = false; },
       "nothing is sampled (prof.active is off) until the program turns sampling on"},
      {"bytes not accumulated", [](JemallocRuntimeFacts& f) { f.optProfAccum = false; },
       "its dumps hold the bytes in use when they are written, not those allocated over the run "
       "(prof_accum is off), which jeprof --alloc_space reads"},
  };
  for (const Limit& LIMIT : LIMITS) {
    JemallocRuntimeFacts facts = profiling();
    LIMIT.apply(facts);
    const ReadinessResult RESULT = runtimeDecision(facts);
    EXPECT_EQ(RESULT.cause, ReadinessCause::CAVEAT) << LIMIT.what;
    EXPECT_EQ(RESULT.report.status, EnvReport::Status::Warning) << LIMIT.what;
    EXPECT_TRUE(RESULT.collectionReady()) << LIMIT.what;
    EXPECT_THAT(RESULT.report.message, HasSubstr(LIMIT.words)) << LIMIT.what;
    EXPECT_EQ(std::string(RESULT.report.message).find("; "), std::string::npos)
        << LIMIT.what << ": one limit, one clause: " << RESULT.report.message;
    ASSERT_NE(planOf(RESULT), nullptr) << LIMIT.what;
  }
}

/**
 * @test Every option on is READY: under bench run's wrap when the wrap names
 * jemalloc and the dumps go into its folder, by hand otherwise.
 */
TEST(JemallocRun, ReadyNamesWhoWrapped) {
  const JemallocRuntimeFacts FACTS = profiling();
  const ReadinessContext WRAPPED =
      contextWith({{"VERNIER_EXTERNAL_WRAP", "jemalloc"},
                   {"VERNIER_EXTERNAL_WRAP_DIR", "bench-out/MyTest.jemalloc"}});
  const ReadinessResult BY_RUNNER = runtimeDecision(
      FACTS, WRAPPED,
      requestFor(ReadinessScope::RUNTIME, "", false, LaunchContext::RUNNER_WRAPPED));
  EXPECT_EQ(BY_RUNNER.cause, ReadinessCause::READY);
  EXPECT_EQ(BY_RUNNER.report.message,
            "jemalloc profiles this process (/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 "
            "(5.3.0-0-g54eaed1d8b56b1aa528be3bdd1877e59c56fa90c)), under bench run's wrap");
  ASSERT_NE(planOf(BY_RUNNER), nullptr);
  EXPECT_EQ(planOf(BY_RUNNER)->launch, LaunchContext::RUNNER_WRAPPED);
  EXPECT_EQ(planOf(BY_RUNNER)->prefix, "bench-out/MyTest.jemalloc/jeprof");
  EXPECT_TRUE(planOf(BY_RUNNER)->finalDump);

  const ReadinessResult BY_HAND = runtimeDecision(FACTS);
  EXPECT_EQ(BY_HAND.cause, ReadinessCause::READY);
  EXPECT_THAT(BY_HAND.report.message, HasSubstr("under a wrap started by hand"));
  EXPECT_EQ(planOf(BY_HAND)->launch, LaunchContext::MANUALLY_WRAPPED);

  // The wrap's variables, but jemalloc writing elsewhere: not the runner's wrap.
  JemallocRuntimeFacts elsewhere = FACTS;
  elsewhere.prefix = "/tmp/mine/jeprof";
  const ReadinessResult MIXED = runtimeDecision(
      elsewhere, WRAPPED,
      requestFor(ReadinessScope::RUNTIME, "", false, LaunchContext::RUNNER_WRAPPED));
  EXPECT_EQ(planOf(MIXED)->launch, LaunchContext::MANUALLY_WRAPPED);
}

/**
 * @test A run decides from its own allocator and starts no process: a
 * context in which no probe could start or keep its files does not change
 * the verdict, which is this test program's own (no jemalloc).
 */
TEST(JemallocRun, DecidesWithoutStartingAProcess) {
  const FakeToolDir EMPTY;
  ASSERT_TRUE(EMPTY.ok());
  const ReadinessContext NOWHERE =
      contextWith({{"PATH", EMPTY.path()}, {"TMPDIR", EMPTY.path() + "/missing/dir"}});
  const ReadinessResult RESULT =
      vernier::bench::checkJemallocRequest(requestFor(ReadinessScope::RUNTIME), NOWHERE);
  const JemallocRuntimeFacts LIVE = vernier::bench::jemallocRuntimeFacts();
  ASSERT_FALSE(LIVE.jemalloc) << "this test program runs on jemalloc";
  EXPECT_EQ(RESULT.cause, LIVE.sanitizer ? ReadinessCause::UNSUPPORTED : ReadinessCause::MISSING)
      << RESULT.report.message;
  EXPECT_THAT(RESULT.report.message, Not(HasSubstr("probe")));
}

/* ----------------------------- The Doctor ----------------------------- */

/** @test The loader's refusal is MISSING, in its own words, with the profiling remedy. */
TEST(JemallocDoctor, LoaderRefusalIsMissing) {
  const ReadinessResult RESULT = doctorDecision(probeOf(PLAIN_MAPS, LOADER_REFUSAL));
  EXPECT_EQ(RESULT.cause, ReadinessCause::MISSING);
  EXPECT_EQ(RESULT.report.message,
            "missing: the loader cannot preload libjemalloc.so.2 (cannot open shared object file)");
  EXPECT_THAT(RESULT.report.hint, HasSubstr("configure --enable-prof"));
  EXPECT_THAT(RESULT.report.hint, Not(HasSubstr("jeprof")));
}

/**
 * @test The Pi's stock jemalloc, which rejects bench run's options, is
 * MISSING profiling, named by the path the loader resolved and in jemalloc's
 * own words; an analyzer on PATH changes nothing.
 */
TEST(JemallocDoctor, BuiltWithoutProfilingIsMissing) {
  const FakeToolDir DIR;
  ASSERT_TRUE(DIR.ok());
  installJeprof(DIR);
  const ReadinessResult RESULT =
      doctorDecision(probeOf(STOCK_MAPS, STOCK_REJECTION), contextWith({{"PATH", DIR.path()}}));
  EXPECT_EQ(RESULT.cause, ReadinessCause::MISSING);
  EXPECT_EQ(RESULT.report.message,
            "missing: /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 is built without profiling: "
            "preloaded with bench run's options, it rejects them (\"<jemalloc>: Invalid conf "
            "pair: prof:true\")");
  EXPECT_THAT(RESULT.report.hint, Not(HasSubstr("jeprof")));
  EXPECT_THAT(RESULT.report.hint, HasSubstr("not on arm64"));
}

/**
 * @test A final heap dump from the probe is READY, naming the library the
 * loader resolved; the analyzer is named beside it, or said to be missing,
 * and never changes the verdict.
 */
TEST(JemallocDoctor, FinalDumpIsReady) {
  const JemallocProbeFacts PROBE =
      probeOf(BUILD_MAPS, "", {{"jeprof.79492.0.f.heap", HEAP_HEADER}});
  const FakeToolDir WITH;
  ASSERT_TRUE(WITH.ok());
  const std::string JEPROF = installJeprof(WITH);
  const ReadinessResult FOUND = doctorDecision(PROBE, contextWith({{"PATH", WITH.path()}}));
  EXPECT_EQ(FOUND.cause, ReadinessCause::READY);
  EXPECT_EQ(FOUND.report.message,
            "/opt/jemalloc-prof/lib/libjemalloc.so.2 profiles: preloaded with bench run's "
            "options, it wrote its final heap dump at exit (probe: /bin/cat /proc/self/maps); "
            "analyzer " +
                JEPROF);
  ASSERT_NE(planOf(FOUND), nullptr);
  EXPECT_EQ(planOf(FOUND)->library, "/opt/jemalloc-prof/lib/libjemalloc.so.2");

  const FakeToolDir WITHOUT;
  ASSERT_TRUE(WITHOUT.ok());
  for (const ReadinessScope SCOPE :
       {ReadinessScope::DEFAULT_INVENTORY, ReadinessScope::PREFLIGHT}) {
    const ReadinessResult MISSING =
        doctorDecision(PROBE, contextWith({{"PATH", WITHOUT.path()}}), SCOPE);
    EXPECT_EQ(MISSING.cause, ReadinessCause::READY);
    EXPECT_EQ(MISSING.report.status, EnvReport::Status::Ok);
    EXPECT_THAT(MISSING.report.message,
                HasSubstr("; jeprof is not on PATH (only reading the dumps needs it)"));
  }
}

/**
 * @test A probe that ends otherwise is UNUSABLE, saying how: no final dump,
 * a dump that is not a heap profile, jemalloc's other complaints, a failing,
 * killed or unobserved probe, or no jemalloc mapped although the loader
 * said nothing.
 */
TEST(JemallocDoctor, OtherEndingsAreUnusable) {
  struct Ending {
    const char* what;
    JemallocProbeFacts probe;
    const char* words;
  };
  JemallocProbeFacts failing = probeOf(BUILD_MAPS, "cat: write error\n");
  failing.run.exitCode = 1;
  JemallocProbeFacts killed = probeOf(BUILD_MAPS, "");
  killed.run.exited = false;
  killed.run.timedOut = true;
  JemallocProbeFacts unobserved = probeOf(BUILD_MAPS, "");
  unobserved.run.exited = false;
  unobserved.run.statusObtained = false;
  unobserved.run.statusLoss = vernier::bench::StatusLoss::SIGCHLD_IGNORED;
  const std::vector<Ending> ENDINGS = {
      {"no dump", probeOf(BUILD_MAPS, ""),
       "/opt/jemalloc-prof/lib/libjemalloc.so.2 takes bench run's options but wrote no final heap "
       "dump when its probe exited"},
      {"an interval dump only", probeOf(BUILD_MAPS, "", {{"jeprof.79492.0.i0.heap", HEAP_HEADER}}),
       "wrote no final heap dump"},
      {"not a heap profile", probeOf(BUILD_MAPS, "", {{"jeprof.79492.0.f.heap", "garbage"}}),
       "wrote jeprof.79492.0.f.heap at its probe's exit, which is not a heap profile (it starts "
       "'garbage')"},
      {"another complaint", probeOf(BUILD_MAPS, "<jemalloc>: Malformed conf string\n"),
       "does not take bench run's options: \"<jemalloc>: Malformed conf string\""},
      {"a failing probe", failing, ": exit status 1: cat: write error"},
      {"a killed probe", killed, ": did not finish within its bound and was killed"},
      {"an unobserved probe", unobserved, "ended without an exit status this process could read"},
      {"nothing mapped", probeOf(PLAIN_MAPS, "", {{"jeprof.79492.0.f.heap", HEAP_HEADER}}),
       "the loader preloaded libjemalloc.so.2 without a word, but no libjemalloc is mapped"},
  };
  for (const Ending& ENDING : ENDINGS) {
    const ReadinessResult RESULT = doctorDecision(ENDING.probe);
    EXPECT_EQ(RESULT.cause, ReadinessCause::UNUSABLE) << ENDING.what;
    EXPECT_THAT(RESULT.report.message, HasSubstr(ENDING.words)) << ENDING.what;
    EXPECT_THAT(RESULT.report.hint, Not(HasSubstr("jeprof --"))) << ENDING.what;
  }
  EXPECT_EQ(doctorDecision(unobserved).report.hint, vernier::bench::CHILD_STATUS_REMEDY);
}

/**
 * @test The real probe, through the real loader: a file that is not a
 * library first on LD_LIBRARY_PATH is refused by the loader, whatever
 * jemalloc the system has, and the doctor says so.
 */
TEST(JemallocDoctor, RealLoaderRefusesAFileThatIsNotALibrary) {
  const FakeToolDir DIR;
  ASSERT_TRUE(DIR.ok());
  DIR.writeFile("libjemalloc.so.2", "this is not a shared library, only a file with its name\n");
  static const JemallocRuntimeFacts NO_SANITIZER;
  JemallocFacts given;
  given.runtime = &NO_SANITIZER;
  const ReadinessResult RESULT = checkJemallocRequestWithFacts(
      requestFor(ReadinessScope::PREFLIGHT),
      contextWith({{"PATH", DIR.path()}, {"LD_LIBRARY_PATH", DIR.path()}}), given);
  EXPECT_EQ(RESULT.cause, ReadinessCause::MISSING) << RESULT.report.message;
  EXPECT_THAT(RESULT.report.message,
              HasSubstr("missing: the loader cannot preload libjemalloc.so.2 ("));
}

/* ----------------------------- The Request ----------------------------- */

/** @test jemalloc takes no mode: any word is refused, in every scope, before anything runs. */
TEST(JemallocRequest, ModeWordsAreRefused) {
  const JemallocRuntimeFacts FACTS = profiling();
  const JemallocProbeFacts PROBE =
      probeOf(BUILD_MAPS, "", {{"jeprof.79492.0.f.heap", HEAP_HEADER}});
  JemallocFacts given;
  given.runtime = &FACTS;
  given.probe = &PROBE;
  for (const ReadinessScope SCOPE :
       {ReadinessScope::DEFAULT_INVENTORY, ReadinessScope::PREFLIGHT, ReadinessScope::RUNTIME}) {
    const ReadinessResult RESULT =
        checkJemallocRequestWithFacts(requestFor(SCOPE, "x"), contextWith(), given);
    EXPECT_EQ(RESULT.cause, ReadinessCause::CONFIGURATION);
    EXPECT_EQ(RESULT.report.message,
              "configuration: 'x' is not a mode of jemalloc, which takes none");
    EXPECT_EQ(RESULT.report.hint, "Drop --profile-args: jemalloc takes no mode.");
  }
}

/**
 * @test --profile-analyze is an analysis-stage error that keeps the capture
 * and its plan; a request that cannot collect stays a collection error.
 */
TEST(JemallocRequest, AnalyzeKeepsTheCapture) {
  const ReadinessResult READY = runtimeDecision(
      profiling(), contextWith(), requestFor(ReadinessScope::RUNTIME, "", /*analyze=*/true));
  EXPECT_EQ(READY.stage, ReadinessStage::ANALYSIS);
  EXPECT_EQ(READY.report.status, EnvReport::Status::Error);
  EXPECT_TRUE(READY.collectionReady());
  EXPECT_EQ(READY.report.message,
            "analysis: unsupported: --profile-analyze: jemalloc has no "
            "automatic analysis; the capture still runs and its dump is kept");
  EXPECT_THAT(READY.report.hint, HasSubstr("jeprof --text --alloc_space"));
  ASSERT_NE(planOf(READY), nullptr);

  const ReadinessResult MISSING = runtimeDecision(JemallocRuntimeFacts{}, contextWith(),
                                                  requestFor(ReadinessScope::RUNTIME, "", true));
  EXPECT_EQ(MISSING.stage, ReadinessStage::COLLECTION);
  EXPECT_EQ(MISSING.cause, ReadinessCause::MISSING);
}

/** @test A program built with the address or thread sanitizer is unsupported, doctor and run alike.
 */
TEST(JemallocRequest, SanitizerIsUnsupported) {
  JemallocRuntimeFacts facts = profiling();
  facts.sanitizer = true;
  const JemallocProbeFacts PROBE =
      probeOf(BUILD_MAPS, "", {{"jeprof.79492.0.f.heap", HEAP_HEADER}});
  JemallocFacts given;
  given.runtime = &facts;
  given.probe = &PROBE;
  for (const ReadinessScope SCOPE :
       {ReadinessScope::DEFAULT_INVENTORY, ReadinessScope::PREFLIGHT, ReadinessScope::RUNTIME}) {
    const ReadinessResult RESULT =
        checkJemallocRequestWithFacts(requestFor(SCOPE), contextWith(), given);
    EXPECT_EQ(RESULT.cause, ReadinessCause::UNSUPPORTED);
    EXPECT_EQ(RESULT.report.message,
              "unsupported: jemalloc does not profile a program built with the address or thread "
              "sanitizer: the sanitizer's allocator serves its allocations");
  }
}

/* ----------------------------- The Profiler ----------------------------- */

/**
 * @test Built directly in this program, which does not run on jemalloc, the
 * profiler refuses as the registry would: no folder, the failure recorded,
 * the run's status 4.
 */
TEST(JemallocProfilerTest, DirectConstructionHereRefusesAndCreatesNothing) {
  const FakeToolDir ROOT;
  ASSERT_TRUE(ROOT.ok());
  PerfConfig cfg;
  cfg.profileTool = "jemalloc";
  cfg.artifactRoot = ROOT.path() + "/root";
  const StderrCapture QUIET;
  ProfilerRegistry& registry = ProfilerRegistry::instance();
  for (int i = 0; i < 2; ++i) {
    registry.resetFailures();
    registry.resetReadiness();
    const std::unique_ptr<vernier::bench::Profiler> PROFILER =
        i == 0 ? std::make_unique<JemallocProfiler>(cfg, "Suite.Case")
               : vernier::bench::makeJemallocProfiler(cfg, "Suite.Case");
    PROFILER->beforeMeasure();
    PROFILER->afterMeasure(vernier::bench::Stats{});
    const std::vector<ProfileFailure> FAILED = registry.failures();
    const int STATUS = registry.finishRun(cfg, 0, 1);
    registry.resetFailures();
    registry.resetReadiness();
    EXPECT_EQ(PROFILER->artifactDir(), "") << i;
    EXPECT_FALSE(std::filesystem::exists(cfg.artifactRoot)) << i;
    ASSERT_EQ(FAILED.size(), 1U) << i;
    EXPECT_EQ(FAILED[0].backend, "jemalloc");
    EXPECT_EQ(FAILED[0].result.stage, ReadinessStage::COLLECTION);
    EXPECT_EQ(STATUS, 4) << i;
  }
}

/**
 * @test What the profiler prints before the measured window says where and
 * when jemalloc will write its dump, and who checks it; it never says a dump
 * was written.
 */
TEST(JemallocProfilerTest, AnnouncesTheDumpWithoutClaimingIt) {
  const FakeToolDir ROOT;
  ASSERT_TRUE(ROOT.ok());
  PerfConfig cfg;
  cfg.artifactRoot = ROOT.path();
  const std::string PID = std::to_string(static_cast<long>(::getpid()));
  struct Case {
    LaunchContext launch;
    bool finalDump;
    const char* words;
  };
  const Case CASES[] = {
      {LaunchContext::RUNNER_WRAPPED, true, "bench run checks that dump after the exit."},
      {LaunchContext::MANUALLY_WRAPPED, true,
       "This run does not check that dump; bench run --profile jemalloc does."},
      {LaunchContext::MANUALLY_WRAPPED, false, "none when it exits."},
  };
  for (const Case& CASE : CASES) {
    auto plan = std::make_shared<JemallocPlan>();
    plan->prefix = "bench-out/MyTest.jemalloc/jeprof";
    plan->launch = CASE.launch;
    plan->finalDump = CASE.finalDump;
    JemallocProfiler profiler(cfg, "Suite.Case", plan);
    StderrCapture capture;
    profiler.beforeMeasure();
    const std::string TEXT = capture.text();
    EXPECT_THAT(TEXT, HasSubstr(CASE.words));
    EXPECT_THAT(TEXT, HasSubstr("bench-out/MyTest.jemalloc/jeprof." + PID + ".<n>."));
    EXPECT_THAT(TEXT, Not(HasSubstr("written")));
    if (CASE.finalDump) {
      EXPECT_THAT(TEXT, HasSubstr("writes its heap dump when the\n[jemalloc] process exits"));
    }
  }
}
