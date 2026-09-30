/**
 * @file ProfilerRegistry_uTest.cpp
 * @brief Unit tests for the self-registering profiler backend registry.
 *
 * Covers the registry surface that `--profile`, `bench doctor`, and
 * `Profiler::make()` rely on: every CPU backend compiled into libbench
 * self-registers on load, name lookups are stable, make() never returns
 * nullptr, and each backend's environment check is callable. The readiness
 * tests register their own backends under unique names, decide in explicit
 * contexts, and remove the backends again: one decision serves the doctor
 * rows, the selected row and construction, zero-argument checks never vouch
 * for more than they verified, and decisions are memoized per request and
 * context until resetReadiness(). The outcome tests pin the run's result: a
 * failed request is recorded once, finishRun() reports it and returns 4 when
 * the tests passed, and a run that created no profiler says so.
 */

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/Profiler.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/utst/StderrCapture.hpp"

#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using vernier::bench::EnvReport;
using vernier::bench::LaunchContext;
using vernier::bench::PerfConfig;
using vernier::bench::ProfileFailure;
using vernier::bench::Profiler;
using vernier::bench::ProfilerRegistry;
using vernier::bench::ReadinessCause;
using vernier::bench::ReadinessContext;
using vernier::bench::ReadinessPlan;
using vernier::bench::ReadinessRequest;
using vernier::bench::ReadinessResult;
using vernier::bench::readinessResult;
using vernier::bench::ReadinessScope;
using vernier::bench::ReadinessStage;
using vernier::bench::test::StderrCapture;

// Backends compiled into libbench that self-register on load. compute-sanitizer
// and nsight self-register too, but they live in the separate CUDA library and
// so are absent from this CPU-only test binary.
const std::vector<std::string> CPU_BACKENDS = {"perf",   "gperf",     "callgrind", "bpftrace",
                                               "rapl",   "massif",    "memcheck",  "helgrind",
                                               "offcpu", "heaptrack", "jemalloc",  "rocprof"};

/** @test Every CPU profiler backend self-registers. */
TEST(ProfilerRegistryTest, CpuBackendsRegistered) {
  const ProfilerRegistry& reg = ProfilerRegistry::instance();
  for (const std::string& NAME : CPU_BACKENDS) {
    EXPECT_TRUE(reg.hasBackend(NAME)) << "backend not registered: " << NAME;
  }
}

/** @test backendNames() is non-empty, sorted, and lists the v1.0.2 additions. */
TEST(ProfilerRegistryTest, BackendNamesSortedAndComplete) {
  const std::vector<std::string> names = ProfilerRegistry::instance().backendNames();
  EXPECT_FALSE(names.empty());
  EXPECT_TRUE(std::is_sorted(names.begin(), names.end()));
  for (const char* NAME : {"helgrind", "heaptrack", "massif", "memcheck", "offcpu"}) {
    EXPECT_NE(std::find(names.begin(), names.end(), NAME), names.end())
        << "backendNames() missing: " << NAME;
  }
}

/** @test An unregistered name reports as absent. */
TEST(ProfilerRegistryTest, UnknownBackendNotRegistered) {
  EXPECT_FALSE(ProfilerRegistry::instance().hasBackend("not-a-real-backend"));
}

/** @test make() never returns nullptr: real backend or named no-op. */
TEST(ProfilerRegistryTest, MakeNeverReturnsNull) {
  const ProfilerRegistry& reg = ProfilerRegistry::instance();
  const PerfConfig cfg{};
  EXPECT_NE(reg.make("perf", cfg, "RegistryTest"), nullptr);
  EXPECT_NE(reg.make("not-a-real-backend", cfg, "RegistryTest"), nullptr);
}

/** @test runCheck() yields a valid status per backend, Error for unknown. */
TEST(ProfilerRegistryTest, RunCheckReturnsStatus) {
  const ProfilerRegistry& reg = ProfilerRegistry::instance();
  for (const std::string& NAME : CPU_BACKENDS) {
    const EnvReport report = reg.runCheck(NAME);
    EXPECT_TRUE(report.status == EnvReport::Status::Ok ||
                report.status == EnvReport::Status::Warning ||
                report.status == EnvReport::Status::Error);
  }
  EXPECT_EQ(reg.runCheck("not-a-real-backend").status, EnvReport::Status::Error);
}

/** @test runAllChecks() reports one entry per registered backend, name-aligned. */
TEST(ProfilerRegistryTest, RunAllChecksMatchesBackendNames) {
  const ProfilerRegistry& reg = ProfilerRegistry::instance();
  const std::vector<std::string> names = reg.backendNames();
  const std::vector<std::pair<std::string, EnvReport>> checks = reg.runAllChecks();
  ASSERT_EQ(checks.size(), names.size());
  for (std::size_t i = 0; i < names.size(); ++i) {
    EXPECT_EQ(checks[i].first, names[i]);
  }
}

/* ----------------------------- Readiness Routing ----------------------------- */

/** @brief A profiler the test factories return; remembers the plan it was built from. */
class FakeProfiler final : public Profiler {
public:
  FakeProfiler(std::string tool, std::shared_ptr<const ReadinessPlan> plan)
      : tool_(std::move(tool)), plan_(std::move(plan)) {}
  std::string toolName() const noexcept override { return tool_; }
  std::string artifactDir() const noexcept override { return "fake-artifacts"; }
  const std::shared_ptr<const ReadinessPlan>& plan() const { return plan_; }

private:
  std::string tool_;
  std::shared_ptr<const ReadinessPlan> plan_;
};

/** @brief A plan type only these tests know. */
struct TestPlan final : ReadinessPlan {
  explicit TestPlan(std::string text) : tag(std::move(text)) {}
  std::string tag;
};

/** @brief Counts calls to a test backend's check and factory. */
struct Calls {
  std::atomic<int> checks{0};
  std::atomic<int> factories{0};
};

/** @brief A test backend under a name no other test uses; removed on destruction. */
class ScopedBackend {
public:
  explicit ScopedBackend(const std::string& stem)
      : name_(stem + "-" + std::to_string(::getpid()) + "-" + std::to_string(serial()++)) {}
  ~ScopedBackend() { ProfilerRegistry::instance().unregisterBackend(name_); }
  ScopedBackend(const ScopedBackend&) = delete;
  ScopedBackend& operator=(const ScopedBackend&) = delete;

  const std::string& name() const { return name_; }
  const std::shared_ptr<Calls>& calls() const { return calls_; }

  /** @brief Register a readiness backend whose check is @p decide. */
  void readiness(std::function<ReadinessResult(const ReadinessRequest&)> decide,
                 std::vector<std::string> contextKeys = {}) {
    auto calls = calls_;
    ProfilerRegistry::instance().registerReadinessBackend(
        name_,
        [calls, decide](const ReadinessRequest& request, const ReadinessContext&) {
          calls->checks.fetch_add(1);
          return decide(request);
        },
        [calls, name = name_](const PerfConfig&, const std::string&, const ReadinessResult& r) {
          calls->factories.fetch_add(1);
          return std::unique_ptr<Profiler>(std::make_unique<FakeProfiler>(name, r.plan));
        },
        "install the test tool", std::move(contextKeys));
  }

  /** @brief Register a legacy backend; an empty @p check registers none. */
  void legacy(std::function<EnvReport()> check) {
    auto calls = calls_;
    ProfilerRegistry::EnvCheck counted;
    if (check) {
      counted = [calls, check] {
        calls->checks.fetch_add(1);
        return check();
      };
    }
    ProfilerRegistry::instance().registerBackend(
        name_,
        [calls, name = name_](const PerfConfig&, const std::string&) {
          calls->factories.fetch_add(1);
          return std::unique_ptr<Profiler>(std::make_unique<FakeProfiler>(name, nullptr));
        },
        counted, "install the test tool");
  }

private:
  static int& serial() {
    static int value = 0;
    return value;
  }
  std::string name_;
  std::shared_ptr<Calls> calls_ = std::make_shared<Calls>();
};

/** @brief A context with @p env only (no process environment). */
ReadinessContext contextWith(std::map<std::string, std::string> env = {}) {
  return ReadinessContext(::geteuid(), ::getpid(), std::move(env));
}

/** @brief A request for @p name in @p scope. */
ReadinessRequest requestOf(const std::string& name, ReadinessScope scope,
                           const std::string& args = "") {
  ReadinessRequest request;
  request.backend = name;
  request.scope = scope;
  request.profileArgs = args;
  return request;
}

PerfConfig configFor(const std::string& name, const std::string& args = "") {
  PerfConfig cfg;
  cfg.profileTool = name;
  cfg.profileArgs = args;
  return cfg;
}

/** @test A zero-argument check answers the inventory as it always did. */
TEST(ProfilerReadinessRouting, LegacyCheckAnswersInventoryAsIs) {
  ScopedBackend backend("legacy-ok");
  backend.legacy([] { return EnvReport{EnvReport::Status::Ok, "legacy ok", ""}; });
  const EnvReport REPORT = ProfilerRegistry::instance().runCheck(backend.name());
  EXPECT_EQ(REPORT.status, EnvReport::Status::Ok);
  EXPECT_EQ(REPORT.message, "legacy ok");
}

/** @test A zero-argument check that passed does not vouch for a requested mode. */
TEST(ProfilerReadinessRouting, LegacyCheckDoesNotVouchForAMode) {
  ScopedBackend backend("legacy-mode");
  backend.legacy([] { return EnvReport{EnvReport::Status::Ok, "legacy ok", ""}; });
  for (const ReadinessScope SCOPE : {ReadinessScope::PREFLIGHT, ReadinessScope::RUNTIME}) {
    const ReadinessResult R = ProfilerRegistry::instance().checkRequest(
        requestOf(backend.name(), SCOPE, "record -g"), contextWith());
    EXPECT_EQ(R.report.status, EnvReport::Status::Warning);
    EXPECT_EQ(R.report.message, "unverified: " + backend.name() +
                                    " checks its default mode only; 'record -g' was not checked");
  }
  // A failing check is reported as it is, mode or not.
  ScopedBackend failing("legacy-fail");
  failing.legacy([] { return EnvReport{EnvReport::Status::Error, "tool missing", "install"}; });
  const ReadinessResult F = ProfilerRegistry::instance().checkRequest(
      requestOf(failing.name(), ReadinessScope::PREFLIGHT, "record"), contextWith());
  EXPECT_EQ(F.report.status, EnvReport::Status::Error);
  EXPECT_EQ(F.report.message, "tool missing");
  EXPECT_EQ(F.report.hint, "install");
}

/** @test A backend registered without a check is unverified, never Ok. */
TEST(ProfilerReadinessRouting, NoCheckIsUnverified) {
  ScopedBackend backend("no-check");
  backend.legacy(nullptr);
  const EnvReport REPORT = ProfilerRegistry::instance().runCheck(backend.name());
  EXPECT_EQ(REPORT.status, EnvReport::Status::Warning);
  EXPECT_EQ(REPORT.message, "unverified: no environment check is registered for " + backend.name());
}

/** @test Under the runner's wrap the zero-argument check is not run at all. */
TEST(ProfilerReadinessRouting, WrappedRunDoesNotRunLegacyCheck) {
  ScopedBackend backend("legacy-wrapped");
  backend.legacy([] { return EnvReport{EnvReport::Status::Ok, "legacy ok", ""}; });
  const ReadinessContext WRAPPED = contextWith({{"VERNIER_EXTERNAL_WRAP", backend.name()}});
  ReadinessRequest request = requestOf(backend.name(), ReadinessScope::RUNTIME);
  request.launch = LaunchContext::RUNNER_WRAPPED;
  const ReadinessResult R = ProfilerRegistry::instance().checkRequest(request, WRAPPED);
  EXPECT_EQ(R.report.status, EnvReport::Status::Warning);
  EXPECT_EQ(R.report.message, "unverified: collection is owned by the " + backend.name() +
                                  " wrap; completion is checked at exit");

  StderrCapture quiet;
  const auto PROFILER =
      ProfilerRegistry::instance().make(backend.name(), configFor(backend.name()), "T.W", WRAPPED);
  EXPECT_EQ(PROFILER->artifactDir(), "fake-artifacts") << "the wrapped run still constructs";
  EXPECT_EQ(backend.calls()->checks.load(), 0) << "the legacy check ran under the wrap";
}

/** @test A collection error is printed once with its remedy; the factory is never called. */
TEST(ProfilerReadinessRouting, CollectionErrorSkipsTheFactory) {
  ScopedBackend backend("denied");
  backend.readiness([](const ReadinessRequest&) {
    return readinessResult(ReadinessCause::DENIED, "the tool refused", "grant it");
  });
  const PerfConfig CFG = configFor(backend.name());
  const ReadinessContext CTX = contextWith();
  StderrCapture capture;
  const auto FIRST = ProfilerRegistry::instance().make(backend.name(), CFG, "T.A", CTX);
  const auto SECOND = ProfilerRegistry::instance().make(backend.name(), CFG, "T.B", CTX);
  const std::string ERR = capture.text();
  EXPECT_EQ(FIRST->toolName(), "") << "a no-op for a failed request names no profiler";
  EXPECT_EQ(FIRST->artifactDir(), "") << "a no-op, not the backend's profiler";
  EXPECT_EQ(backend.calls()->factories.load(), 0);
  EXPECT_EQ(backend.calls()->checks.load(), 1);
  const std::string EXPECTED = "[FAIL] Profiler '" + backend.name() +
                               "': denied: the tool refused\n   grant it\n   Nothing is collected "
                               "for this request; the run will fail (exit status 4 if the tests "
                               "pass).\n";
  EXPECT_NE(ERR.find(EXPECTED), std::string::npos) << ERR;
  EXPECT_EQ(ERR.find(EXPECTED), ERR.rfind(EXPECTED)) << "printed more than once:\n" << ERR;
}

/** @test An analysis-stage error is printed once, recorded, and collection still runs. */
TEST(ProfilerReadinessRouting, AnalysisErrorStillConstructs) {
  ScopedBackend backend("analysis");
  backend.readiness([](const ReadinessRequest&) {
    return readinessResult(ReadinessCause::MISSING, "no analyzer", "install one",
                           ReadinessStage::ANALYSIS);
  });
  ProfilerRegistry::instance().resetFailures();
  StderrCapture capture;
  const auto PROFILER = ProfilerRegistry::instance().make(backend.name(), configFor(backend.name()),
                                                          "T.A", contextWith());
  const std::string ERR = capture.text();
  const std::vector<ProfileFailure> FAILED = ProfilerRegistry::instance().failures();
  ProfilerRegistry::instance().resetFailures();
  ASSERT_EQ(FAILED.size(), 1U) << "the analysis error is the run's failure";
  EXPECT_EQ(FAILED[0].result.stage, ReadinessStage::ANALYSIS);
  EXPECT_EQ(PROFILER->artifactDir(), "fake-artifacts");
  EXPECT_EQ(backend.calls()->factories.load(), 1);
  EXPECT_NE(ERR.find("[FAIL] Profiler '" + backend.name() + "': analysis: missing: no analyzer"),
            std::string::npos)
      << ERR;
  EXPECT_NE(ERR.find("   Collection proceeds and keeps the raw capture; the analysis is skipped "
                     "and the run will fail (exit status 4 if the tests pass).\n"),
            std::string::npos)
      << ERR;
  EXPECT_EQ(ERR.find("Nothing is collected"), std::string::npos) << ERR;
}

/** @test A warning is printed once per decision and every case is still profiled. */
TEST(ProfilerReadinessRouting, WarningPrintedOnceAndConstructs) {
  ScopedBackend backend("caveat");
  backend.readiness([](const ReadinessRequest&) {
    return readinessResult(ReadinessCause::CAVEAT, "kernel samples excluded", "lower paranoid");
  });
  StderrCapture capture;
  for (int i = 0; i < 3; ++i) {
    (void)ProfilerRegistry::instance().make(backend.name(), configFor(backend.name()), "T.C",
                                            contextWith());
  }
  const std::string ERR = capture.text();
  const std::string LINE = "[WARN] Profiler '" + backend.name() + "': kernel samples excluded";
  EXPECT_NE(ERR.find(LINE), std::string::npos) << ERR;
  EXPECT_EQ(ERR.find(LINE), ERR.rfind(LINE)) << ERR;
  EXPECT_EQ(backend.calls()->factories.load(), 3);
  EXPECT_EQ(backend.calls()->checks.load(), 1);
}

/** @test An Ok decision constructs silently. */
TEST(ProfilerReadinessRouting, OkIsSilent) {
  ScopedBackend backend("ready");
  backend.readiness(
      [](const ReadinessRequest&) { return readinessResult(ReadinessCause::READY, "ok", ""); });
  StderrCapture capture;
  const auto PROFILER = ProfilerRegistry::instance().make(backend.name(), configFor(backend.name()),
                                                          "T.O", contextWith());
  EXPECT_EQ(capture.text(), "");
  EXPECT_EQ(PROFILER->artifactDir(), "fake-artifacts");
}

/** @test The factory receives the decision itself, plan included. */
TEST(ProfilerReadinessRouting, PlannedFactoryReceivesTheDecision) {
  ScopedBackend backend("planned");
  auto plan = std::make_shared<const TestPlan>("resolved /abs/tool");
  backend.readiness([plan](const ReadinessRequest&) {
    ReadinessResult r = readinessResult(ReadinessCause::READY, "ok", "");
    r.plan = plan;
    return r;
  });
  const auto PROFILER = ProfilerRegistry::instance().make(backend.name(), configFor(backend.name()),
                                                          "T.P", contextWith());
  const auto* fake = dynamic_cast<const FakeProfiler*>(PROFILER.get());
  ASSERT_NE(fake, nullptr);
  EXPECT_EQ(fake->plan().get(), plan.get());
}

/** @test One decision per request and context; a changed input decides again. */
TEST(ProfilerReadinessRouting, MemoizedPerRequestAndContext) {
  ScopedBackend backend("memo");
  backend.readiness(
      [](const ReadinessRequest&) { return readinessResult(ReadinessCause::READY, "ok", ""); },
      {"MEMO_TEST_TOOL_DIR"});
  ProfilerRegistry& reg = ProfilerRegistry::instance();
  const auto MAKE = [&](const PerfConfig& cfg, const ReadinessContext& ctx) {
    (void)reg.make(backend.name(), cfg, "T.M", ctx);
  };
  const PerfConfig CFG = configFor(backend.name());
  MAKE(CFG, contextWith({{"PATH", "/a"}}));
  MAKE(CFG, contextWith({{"PATH", "/a"}, {"UNRELATED", "x"}}));
  EXPECT_EQ(backend.calls()->checks.load(), 1) << "same request and decision inputs";
  MAKE(configFor(backend.name(), "mode"), contextWith({{"PATH", "/a"}}));
  EXPECT_EQ(backend.calls()->checks.load(), 2) << "another mode is another request";
  MAKE(CFG, contextWith({{"PATH", "/b"}}));
  EXPECT_EQ(backend.calls()->checks.load(), 3) << "PATH changed";
  MAKE(CFG, contextWith({{"PATH", "/a"}, {"MEMO_TEST_TOOL_DIR", "/t"}}));
  EXPECT_EQ(backend.calls()->checks.load(), 4) << "a declared input changed";
  MAKE(CFG, contextWith({{"PATH", "/a"}, {"BENCH_SUDO", "1"}}));
  EXPECT_EQ(backend.calls()->checks.load(), 5) << "the privilege opt-in changed";
  PerfConfig analyzed = CFG;
  analyzed.profileAnalyze = true;
  MAKE(analyzed, contextWith({{"PATH", "/a"}}));
  EXPECT_EQ(backend.calls()->checks.load(), 6) << "a promised analysis is another request";
  PerfConfig scripted = CFG;
  scripted.bpfScripts = {"one"};
  MAKE(scripted, contextWith({{"PATH", "/a"}}));
  EXPECT_EQ(backend.calls()->checks.load(), 7) << "a script selection is another request";
  MAKE(analyzed, contextWith({{"PATH", "/a"}}));
  MAKE(scripted, contextWith({{"PATH", "/a"}}));
  EXPECT_EQ(backend.calls()->checks.load(), 7) << "each is kept once decided";
}

/** @test A repair is seen only after resetReadiness(), the one reset boundary. */
TEST(ProfilerReadinessRouting, ResetThenRetryAfterRepair) {
  ScopedBackend backend("repair");
  auto repaired = std::make_shared<std::atomic<bool>>(false);
  backend.readiness([repaired](const ReadinessRequest&) {
    return repaired->load() ? readinessResult(ReadinessCause::READY, "ok", "")
                            : readinessResult(ReadinessCause::MISSING, "tool", "install it");
  });
  ProfilerRegistry& reg = ProfilerRegistry::instance();
  const PerfConfig CFG = configFor(backend.name());
  const ReadinessContext CTX = contextWith();
  StderrCapture quiet;
  EXPECT_EQ(reg.make(backend.name(), CFG, "T.R", CTX)->artifactDir(), "");
  repaired->store(true);
  EXPECT_EQ(reg.make(backend.name(), CFG, "T.R", CTX)->artifactDir(), "")
      << "the error is kept until the reset";
  EXPECT_EQ(backend.calls()->checks.load(), 1);
  reg.resetReadiness();
  EXPECT_EQ(reg.make(backend.name(), CFG, "T.R", CTX)->artifactDir(), "fake-artifacts");
  EXPECT_EQ(backend.calls()->checks.load(), 2);
}

/** @test Concurrent constructions of one request decide once. */
TEST(ProfilerReadinessRouting, ConcurrentMakeDecidesOnce) {
  ScopedBackend backend("concurrent");
  backend.readiness([](const ReadinessRequest&) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return readinessResult(ReadinessCause::READY, "ok", "");
  });
  const PerfConfig CFG = configFor(backend.name());
  const ReadinessContext CTX = contextWith();
  std::vector<std::thread> threads;
  std::atomic<int> built{0};
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&] {
      if (ProfilerRegistry::instance().make(backend.name(), CFG, "T.X", CTX)->artifactDir() ==
          "fake-artifacts") {
        built.fetch_add(1);
      }
    });
  }
  for (std::thread& t : threads) {
    t.join();
  }
  EXPECT_EQ(backend.calls()->checks.load(), 1);
  EXPECT_EQ(built.load(), 8);
}

/** @test Re-registering a backend is a new decision. */
TEST(ProfilerReadinessRouting, ReRegistrationDecidesAgain) {
  ScopedBackend backend("rereg");
  backend.readiness(
      [](const ReadinessRequest&) { return readinessResult(ReadinessCause::READY, "ok", ""); });
  (void)ProfilerRegistry::instance().make(backend.name(), configFor(backend.name()), "T.G",
                                          contextWith());
  backend.readiness(
      [](const ReadinessRequest&) { return readinessResult(ReadinessCause::READY, "ok", ""); });
  (void)ProfilerRegistry::instance().make(backend.name(), configFor(backend.name()), "T.G",
                                          contextWith());
  EXPECT_EQ(backend.calls()->checks.load(), 2);
}

/** @test A check that throws is an internal error naming the backend. */
TEST(ProfilerReadinessRouting, ThrowingCheckIsInternalError) {
  ScopedBackend backend("throws");
  backend.readiness(
      [](const ReadinessRequest&) -> ReadinessResult { throw std::runtime_error("boom"); });
  const ReadinessResult R = ProfilerRegistry::instance().checkRequest(
      requestOf(backend.name(), ReadinessScope::PREFLIGHT), contextWith());
  EXPECT_EQ(R.report.status, EnvReport::Status::Error);
  EXPECT_EQ(R.report.message, "internal: the " + backend.name() + " check failed: boom");
  StderrCapture quiet;
  EXPECT_EQ(ProfilerRegistry::instance()
                .make(backend.name(), configFor(backend.name()), "T.T", contextWith())
                ->artifactDir(),
            "");
}

/** @test An unknown backend is an error in checkRequest(), as in runCheck(), naming the others. */
TEST(ProfilerReadinessRouting, UnknownBackendIsError) {
  const ReadinessResult R = ProfilerRegistry::instance().checkRequest(
      requestOf("not-a-real-backend", ReadinessScope::PREFLIGHT), contextWith());
  EXPECT_EQ(R.report.status, EnvReport::Status::Error);
  EXPECT_EQ(R.cause, ReadinessCause::MISSING);
  EXPECT_EQ(R.report.message, "unknown profiler 'not-a-real-backend'");
  EXPECT_EQ(R.report.hint.rfind("Available: ", 0), 0U) << R.report.hint;
  EXPECT_NE(R.report.hint.find("gperf, "), std::string::npos) << R.report.hint;
}

/** @test The run prints exactly the selected doctor row's report for the same request. */
TEST(ProfilerReadinessRouting, RunPrintsTheSelectedDoctorReport) {
  ScopedBackend backend("agree");
  backend.readiness([](const ReadinessRequest& request) {
    return readinessResult(ReadinessCause::UNSUPPORTED, "mode '" + request.profileArgs + "'",
                           "use a supported mode");
  });
  const PerfConfig CFG = configFor(backend.name(), "odd");
  const ReadinessContext CTX = contextWith();
  const ReadinessResult DOCTOR = ProfilerRegistry::instance().checkRequest(
      vernier::bench::readinessRequestFor(CFG, ReadinessScope::PREFLIGHT, CTX), CTX);
  StderrCapture capture;
  (void)ProfilerRegistry::instance().make(backend.name(), CFG, "T.S", CTX);
  const std::string ERR = capture.text();
  EXPECT_EQ(DOCTOR.report.message, "unsupported: mode 'odd'");
  EXPECT_NE(ERR.find("'" + backend.name() + "': " + DOCTOR.report.message + "\n   " +
                     DOCTOR.report.hint + "\n"),
            std::string::npos)
      << ERR;
}

/** @test The doctor labels its rows as default-mode checks and never memoizes. */
TEST(ProfilerReadinessRouting, DoctorLabelsRowsAndDecidesAfresh) {
  ScopedBackend backend("doctor");
  backend.readiness([](const ReadinessRequest& request) {
    return readinessResult(ReadinessCause::CAVEAT, "asked about '" + request.profileArgs + "'", "");
  });
  testing::internal::CaptureStdout();
  (void)ProfilerRegistry::instance().printDoctor();
  (void)ProfilerRegistry::instance().printDoctor();
  const std::string OUT = testing::internal::GetCapturedStdout();
  EXPECT_NE(OUT.find("=== Profiler Backend Doctor (default mode of each backend) ==="),
            std::string::npos)
      << OUT;
  EXPECT_NE(OUT.find("Each row checks one backend's default mode for this user and environment."),
            std::string::npos);
  EXPECT_NE(OUT.find("only cases built with the profiler guard create one"), std::string::npos);
  EXPECT_NE(OUT.find("[WARN] " + backend.name()), std::string::npos) << OUT;
  EXPECT_EQ(OUT.find("Selected request"), std::string::npos);
  EXPECT_EQ(backend.calls()->checks.load(), 2) << "the doctor decides afresh every time";
}

/** @test With a request the doctor adds one selected row for exactly that request. */
TEST(ProfilerReadinessRouting, DoctorSelectedRow) {
  ScopedBackend backend("selected");
  backend.readiness([](const ReadinessRequest& request) {
    return readinessResult(ReadinessCause::CAVEAT, "asked about '" + request.profileArgs + "'",
                           "a caveat");
  });
  PerfConfig cfg = configFor(backend.name(), "mode-x");
  cfg.profileAnalyze = true;
  testing::internal::CaptureStdout();
  (void)ProfilerRegistry::instance().printDoctor(cfg);
  const std::string OUT = testing::internal::GetCapturedStdout();
  EXPECT_NE(OUT.find("Selected request: --profile " + backend.name() +
                     " --profile-args 'mode-x' --profile-analyze\n  [WARN] " + backend.name()),
            std::string::npos)
      << OUT;
  EXPECT_NE(OUT.find("asked about 'mode-x'\n             a caveat"), std::string::npos) << OUT;
}

/** @test The JSON document labels its rows and carries the selected decision. */
TEST(ProfilerReadinessRouting, DoctorJsonKeys) {
  ScopedBackend backend("json");
  backend.readiness([](const ReadinessRequest& request) {
    return readinessResult(ReadinessCause::CAVEAT, "asked \"" + request.profileArgs + "\"", "hint");
  });
  testing::internal::CaptureStdout();
  vernier::bench::runProfileCheckJson(PerfConfig{});
  const std::string PLAIN = testing::internal::GetCapturedStdout();
  EXPECT_NE(PLAIN.find("\"backendScope\": \"default-mode\""), std::string::npos) << PLAIN;
  EXPECT_EQ(PLAIN.find("\"selected\""), std::string::npos) << PLAIN;
  EXPECT_NE(PLAIN.find("\"backendFailures\": "), std::string::npos);

  testing::internal::CaptureStdout();
  vernier::bench::runProfileCheckJson(configFor(backend.name(), "m"));
  const std::string SELECTED = testing::internal::GetCapturedStdout();
  EXPECT_NE(SELECTED.find("\"selected\": {\"name\": \"" + backend.name() +
                          "\", \"profileArgs\": \"m\", \"status\": \"warn\", "
                          "\"message\": \"asked \\\"m\\\"\", \"hint\": \"hint\"}"),
            std::string::npos)
      << SELECTED;
}

/**
 * @test An analysis-stage Error is a failure in every view of the request, and its analysis-off
 * twin is Ok
 *
 * The run prints it as [FAIL] and still builds the profiler; the doctor's
 * selected row is [FAIL]; the JSON's selected.status is "fail". The same
 * backend asked without the analysis is Ok everywhere.
 */
TEST(ProfilerReadinessRouting, AnalysisStageErrorFailsEveryViewAndItsTwinIsOk) {
  ScopedBackend backend("analysis-views");
  backend.readiness([](const ReadinessRequest& request) {
    return request.analyze ? readinessResult(ReadinessCause::MISSING, "no analyzer", "install one",
                                             ReadinessStage::ANALYSIS)
                           : readinessResult(ReadinessCause::READY, "collects", "");
  });
  PerfConfig promised = configFor(backend.name());
  promised.profileAnalyze = true;
  const PerfConfig PLAIN = configFor(backend.name());

  testing::internal::CaptureStdout();
  vernier::bench::runProfileCheckJson(promised);
  const std::string JSON_ON = testing::internal::GetCapturedStdout();
  EXPECT_NE(JSON_ON.find("\"selected\": {\"name\": \"" + backend.name() +
                         "\", \"profileArgs\": \"\", \"status\": \"fail\""),
            std::string::npos)
      << JSON_ON;
  testing::internal::CaptureStdout();
  vernier::bench::runProfileCheckJson(PLAIN);
  const std::string JSON_OFF = testing::internal::GetCapturedStdout();
  EXPECT_NE(JSON_OFF.find("\"status\": \"ok\", \"message\": \"collects\""), std::string::npos)
      << JSON_OFF;

  testing::internal::CaptureStdout();
  (void)ProfilerRegistry::instance().printDoctor(promised);
  const std::string TEXT = testing::internal::GetCapturedStdout();
  EXPECT_NE(TEXT.find("--profile-analyze\n  [FAIL] " + backend.name()), std::string::npos) << TEXT;

  StderrCapture capture;
  const auto PROFILER =
      ProfilerRegistry::instance().make(backend.name(), promised, "T.V", contextWith());
  const auto QUIET = ProfilerRegistry::instance().make(backend.name(), PLAIN, "T.V", contextWith());
  const std::string ERR = capture.text();
  EXPECT_NE(ERR.find("[FAIL] Profiler '" + backend.name() +
                     "': analysis: missing: no analyzer\n"
                     "   install one\n   Collection proceeds and keeps the raw capture;"),
            std::string::npos)
      << ERR;
  EXPECT_EQ(PROFILER->artifactDir(), "fake-artifacts") << "collection still runs";
  EXPECT_EQ(QUIET->artifactDir(), "fake-artifacts");
  EXPECT_EQ(ERR.find("collects"), std::string::npos) << "the Ok twin prints nothing";
}

/* ----------------------------- Run Outcome ----------------------------- */

/** @brief Empties the process's run outcome before and after each test. */
class ProfilerOutcomeTest : public ::testing::Test {
protected:
  void SetUp() override { ProfilerRegistry::instance().resetFailures(); }
  void TearDown() override { ProfilerRegistry::instance().resetFailures(); }
};

/** @test A failure while the tests pass gives status 4; the tests' own failure wins. */
TEST_F(ProfilerOutcomeTest, TestFailureWins) {
  ScopedBackend backend("outcome-status");
  backend.readiness(
      [](const ReadinessRequest&) { return readinessResult(ReadinessCause::READY, "ok", ""); });
  ProfilerRegistry& reg = ProfilerRegistry::instance();
  const PerfConfig CFG = configFor(backend.name());
  StderrCapture capture;
  reg.reportFailure(
      backend.name(), "T.F",
      readinessResult(ReadinessCause::MISSING, "no output", "rerun", ReadinessStage::COMPLETION));
  EXPECT_EQ(reg.finishRun(CFG, 0, 1), 4);
  EXPECT_EQ(reg.finishRun(CFG, 1, 1), 1);
  const std::string ERR = capture.text();
  EXPECT_NE(ERR.find("\n[FAIL] Profiler '" + backend.name() +
                     "' (T.F): completion: missing: no output\n   rerun\n   The run will fail "
                     "(exit status 4 if the tests pass).\n"),
            std::string::npos)
      << ERR;
  EXPECT_NE(ERR.find("[profile] --profile " + backend.name() +
                     " failed; the run exits with status 4:\n[profile]   " + backend.name() +
                     " (T.F): completion: missing: no output\n"),
            std::string::npos)
      << ERR;
  EXPECT_NE(ERR.find("[profile] --profile " + backend.name() +
                     " failed; the run exits with the tests' status 1:\n"),
            std::string::npos)
      << ERR;
}

/** @test Nothing recorded: the tests' status, and nothing printed without a request. */
TEST_F(ProfilerOutcomeTest, NothingFailedKeepsTheTestsStatus) {
  StderrCapture capture;
  EXPECT_EQ(ProfilerRegistry::instance().finishRun(PerfConfig{}, 0, 3), 0);
  EXPECT_EQ(ProfilerRegistry::instance().finishRun(PerfConfig{}, 1, 3), 1);
  EXPECT_EQ(capture.text(), "");
}

/** @test A decision's error is recorded once for every case it serves; a warning is not. */
TEST_F(ProfilerOutcomeTest, RecordedOncePerDecision) {
  ScopedBackend denied("outcome-denied");
  denied.readiness([](const ReadinessRequest&) {
    return readinessResult(ReadinessCause::DENIED, "refused", "grant it");
  });
  ScopedBackend caveat("outcome-caveat");
  caveat.readiness(
      [](const ReadinessRequest&) { return readinessResult(ReadinessCause::CAVEAT, "short", ""); });
  ProfilerRegistry& reg = ProfilerRegistry::instance();
  StderrCapture quiet;
  for (const char* test : {"T.A", "T.B", "T.C"}) {
    (void)reg.make(denied.name(), configFor(denied.name()), test, contextWith());
    (void)reg.make(caveat.name(), configFor(caveat.name()), test, contextWith());
    (void)reg.make("no-such-profiler", configFor("no-such-profiler"), test, contextWith());
  }
  const std::vector<ProfileFailure> FAILED = reg.failures();
  ASSERT_EQ(FAILED.size(), 2U);
  EXPECT_EQ(FAILED[0].backend, denied.name());
  EXPECT_EQ(FAILED[0].test, "") << "a decision concerns the whole request";
  EXPECT_EQ(FAILED[0].result.stage, ReadinessStage::COLLECTION);
  EXPECT_EQ(FAILED[0].result.cause, ReadinessCause::DENIED);
  EXPECT_EQ(FAILED[0].result.report.message, "denied: refused");
  EXPECT_EQ(FAILED[1].backend, "no-such-profiler") << "recorded when asked, not only at the end";
  EXPECT_EQ(FAILED[1].result.cause, ReadinessCause::MISSING);
}

/** @test A factory that still builds nothing is printed once and recorded as unusable. */
TEST_F(ProfilerOutcomeTest, UnavailableFactoryIsRecorded) {
  ScopedBackend backend("outcome-null");
  ProfilerRegistry& reg = ProfilerRegistry::instance();
  reg.registerReadinessBackend(
      backend.name(),
      [](const ReadinessRequest&, const ReadinessContext&) {
        return readinessResult(ReadinessCause::READY, "ok", "");
      },
      [](const PerfConfig&, const std::string&, const ReadinessResult&) {
        return std::unique_ptr<Profiler>{};
      },
      "build it with the tool");
  StderrCapture capture;
  const auto FIRST = reg.make(backend.name(), configFor(backend.name()), "T.A", contextWith());
  const auto SECOND = reg.make(backend.name(), configFor(backend.name()), "T.B", contextWith());
  const std::string ERR = capture.text();
  EXPECT_EQ(FIRST->toolName(), "");
  const std::string NOTICE = "[FAIL] Profiler '" + backend.name() +
                             "': unusable: " + backend.name() +
                             " is unavailable on this platform\n   build it with the tool\n"
                             "   Nothing is collected for this request;";
  EXPECT_NE(ERR.find(NOTICE), std::string::npos) << ERR;
  EXPECT_EQ(ERR.find(NOTICE), ERR.rfind(NOTICE)) << "printed more than once:\n" << ERR;
  const std::vector<ProfileFailure> FAILED = reg.failures();
  ASSERT_EQ(FAILED.size(), 1U);
  EXPECT_EQ(FAILED[0].result.cause, ReadinessCause::UNUSABLE);
}

/** @test A hook's failure is recorded and printed once per case and message; a non-error is not. */
TEST_F(ProfilerOutcomeTest, HookFailuresOncePerCaseAndMessage) {
  ProfilerRegistry& reg = ProfilerRegistry::instance();
  const ReadinessResult LOST =
      readinessResult(ReadinessCause::MISSING, "no output", "", ReadinessStage::COMPLETION);
  StderrCapture capture;
  reg.reportFailure("b", "T.A", LOST);
  reg.reportFailure("b", "T.A", LOST);
  reg.reportFailure("b", "T.B", LOST);
  reg.reportFailure(
      "b", "T.A",
      readinessResult(ReadinessCause::UNUSABLE, "truncated", "", ReadinessStage::COMPLETION));
  reg.reportFailure("b", "T.A", readinessResult(ReadinessCause::CAVEAT, "short", ""));
  const std::string ERR = capture.text();
  EXPECT_EQ(reg.failures().size(), 3U);
  const std::string LINE = "[FAIL] Profiler 'b' (T.A): completion: missing: no output\n";
  EXPECT_NE(ERR.find(LINE), std::string::npos) << ERR;
  EXPECT_EQ(ERR.find(LINE), ERR.rfind(LINE)) << "printed more than once:\n" << ERR;
  EXPECT_EQ(ERR.find("short"), std::string::npos) << "a warning is not a failure:\n" << ERR;
}

/** @test Reports from many threads are all recorded, each once. */
TEST_F(ProfilerOutcomeTest, ConcurrentReportsAllRecorded) {
  ProfilerRegistry& reg = ProfilerRegistry::instance();
  StderrCapture quiet;
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&reg, i] {
      for (int j = 0; j < 50; ++j) {
        reg.reportFailure(
            "b", "T." + std::to_string(i),
            readinessResult(ReadinessCause::MISSING, "no output", "", ReadinessStage::COMPLETION));
        reg.reportFailure("b", "", readinessResult(ReadinessCause::MISSING, "shared", ""));
      }
    });
  }
  for (std::thread& t : threads) {
    t.join();
  }
  EXPECT_EQ(reg.failures().size(), 9U);
}

/** @test An unknown name fails the run at its end, even when no case asked for a profiler. */
TEST_F(ProfilerOutcomeTest, UnknownNameFailsAtTheEnd) {
  ProfilerRegistry& reg = ProfilerRegistry::instance();
  reg.resetReadiness(); // the notice is printed once per process until this reset
  StderrCapture capture;
  EXPECT_EQ(reg.finishRun(configFor("no-such-profiler"), 0, 1), 4);
  const std::string ERR = capture.text();
  EXPECT_NE(ERR.find("[FAIL] Profiler 'no-such-profiler': unknown profiler 'no-such-profiler'\n"
                     "   Available: "),
            std::string::npos)
      << ERR;
  EXPECT_NE(ERR.find("[profile] --profile no-such-profiler failed; the run exits with status 4:\n"
                     "[profile]   no-such-profiler: unknown profiler 'no-such-profiler'\n"),
            std::string::npos)
      << ERR;
  ASSERT_EQ(reg.failures().size(), 1U);
  reg.resetFailures();
  StderrCapture quiet;
  EXPECT_EQ(reg.finishRun(configFor("cupti"), 0, 1), 0) << "cupti needs no --profile";
  EXPECT_TRUE(reg.failures().empty());
}

/** @test A run that ran tests and created no profiler says so; one that created one is silent. */
TEST_F(ProfilerOutcomeTest, NoProfilerCreatedNotice) {
  ScopedBackend backend("outcome-created");
  backend.readiness(
      [](const ReadinessRequest&) { return readinessResult(ReadinessCause::READY, "ok", ""); });
  ProfilerRegistry& reg = ProfilerRegistry::instance();
  const PerfConfig CFG = configFor(backend.name());
  const std::string NOTICE = "[profile] --profile " + backend.name() +
                             ": no case that ran was built with the profiler guard, so nothing "
                             "was profiled.\n";
  {
    StderrCapture capture;
    EXPECT_EQ(reg.finishRun(CFG, 0, 2), 0) << "a notice, not a failure";
    EXPECT_EQ(reg.finishRun(CFG, 0, 0), 0);
    const std::string ERR = capture.text();
    EXPECT_NE(ERR.find(NOTICE), std::string::npos) << ERR;
    EXPECT_EQ(ERR.find(NOTICE), ERR.rfind(NOTICE)) << "no notice when no test ran:\n" << ERR;
  }
  (void)reg.make(backend.name(), CFG, "T.N", contextWith());
  StderrCapture capture;
  EXPECT_EQ(reg.finishRun(CFG, 0, 2), 0);
  EXPECT_EQ(capture.text(), "") << "a profiler was created";
}

/* ----------------------------- Shared Route Table ----------------------------- */

/** @brief The rows of @p kind in the table the Rust tools are tested against too. */
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

/** @test The aliases and the profile-failed status are the ones `bench run` uses. */
TEST(ProfilerRoutesTable, AliasesAndExitStatus) {
  const auto ALIASES = sharedTableRows("alias");
  ASSERT_GE(ALIASES.size(), 16U) << "the shared table was not read";
  for (const auto& row : ALIASES) {
    ASSERT_EQ(row.size(), 3U);
    EXPECT_EQ(ProfilerRegistry::canonicalName(row[1]), row[2]) << "alias " << row[1];
  }
  const auto EXITS = sharedTableRows("exit");
  ASSERT_EQ(EXITS.size(), 1U);
  EXPECT_EQ(EXITS[0][1], "profile-failed");
  EXPECT_EQ(std::to_string(vernier::bench::BENCH_PROFILE_FAILED_EXIT_CODE), EXITS[0][2]);
}

} // namespace
