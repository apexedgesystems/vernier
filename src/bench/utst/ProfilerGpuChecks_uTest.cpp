/**
 * @file ProfilerGpuChecks_uTest.cpp
 * @brief Unit tests for the Nsight (nsight, ncu) and Compute Sanitizer
 * readiness checks, which are CPU code in libbench, and for libbench's
 * default registration of those names with a passive profiler.
 *
 * The doctor's scopes probe `<tool> --version` through a fake tool
 * (fixtures/readiness/fake_nvidia_tool.sh) in a private PATH; the runtime
 * scope decides from a snapshot's environment (the session variables nsys,
 * ncu and compute-sanitizer export, and the runner's wrap variables) and, for
 * compute-sanitizer, a memory map given as text. The modes and refusals are
 * pinned to the route table `bench run` reads. This binary links libbench
 * alone, so the registry holds libbench's registrations, as in a build without
 * CUDA.
 */

#include "src/bench/inc/Profiler.hpp"
#include "src/bench/inc/ProfilerComputeSanitizerChecks.hpp"
#include "src/bench/inc/ProfilerNsightChecks.hpp"
#include "src/bench/inc/ProfilerReadiness.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/utst/ReadinessFixtures.hpp"
#include "src/bench/utst/StderrCapture.hpp"

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#ifndef VERNIER_SHARED_FIXTURE_DIR
#error "VERNIER_SHARED_FIXTURE_DIR must name tools/rust/tests/fixtures"
#endif

using vernier::bench::checkComputeSanitizerRequest;
using vernier::bench::checkComputeSanitizerRequestWithMaps;
using vernier::bench::checkNsightRequest;
using vernier::bench::checkNsightRequestWith;
using vernier::bench::NsightMode;
using vernier::bench::nsightModeTool;
using vernier::bench::parseNsightMode;
using vernier::bench::parseSanitizerTool;
using vernier::bench::PerfConfig;
using vernier::bench::Profiler;
using vernier::bench::ProfilerRegistry;
using vernier::bench::ReadinessCause;
using vernier::bench::ReadinessContext;
using vernier::bench::ReadinessRequest;
using vernier::bench::ReadinessResult;
using vernier::bench::ReadinessScope;
using vernier::bench::ReadinessStage;
using vernier::bench::test::FakeToolDir;
using vernier::bench::test::ScopedEnv;
using vernier::bench::test::StderrCapture;

namespace {

/** @brief The rows of @p kind in the route table `bench run` reads; '-' is an empty field. */
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
    for (auto& field : fields) {
      if (field == "-") {
        field.clear();
      }
    }
    if (fields[0] == kind) {
      rows.push_back(fields);
    }
  }
  return rows;
}

/** @brief A request for @p backend with @p profileArgs at @p scope. */
ReadinessRequest request(const std::string& backend, const std::string& profileArgs,
                         ReadinessScope scope, bool analyze = false) {
  ReadinessRequest out;
  out.backend = backend;
  out.profileArgs = profileArgs;
  out.scope = scope;
  out.analyze = analyze;
  return out;
}

/** @brief A snapshot of this process with exactly @p env, effective uid @p euid. */
ReadinessContext snapshot(std::map<std::string, std::string> env, uid_t euid = 0) {
  return ReadinessContext(euid, ::getpid(), std::move(env));
}

/** @brief The mode @p backend's parser reads from @p profileArgs; it must be accepted. */
NsightMode modeOf(const std::string& backend, const std::string& profileArgs) {
  NsightMode mode = NsightMode::Systems;
  const auto REFUSED = parseNsightMode(backend, profileArgs, mode);
  EXPECT_FALSE(REFUSED.has_value())
      << backend << " '" << profileArgs << "': " << (REFUSED ? REFUSED->report.message : "");
  return mode;
}

} // namespace

/* ----------------------------- Modes ----------------------------- */

/** @test Each Nsight mode needs one tool: nsys for Systems, ncu for the compute modes. */
TEST(NsightModes, ParserSelectsTheModeAndItsTool) {
  EXPECT_EQ(modeOf("nsight", ""), NsightMode::Systems);
  EXPECT_EQ(modeOf("nsight", "compute"), NsightMode::Compute);
  EXPECT_EQ(modeOf("nsight", "ncu"), NsightMode::Compute);
  EXPECT_EQ(modeOf("nsight", "replay"), NsightMode::ComputeReplay);
  EXPECT_EQ(modeOf("nsight", "compute,replay"), NsightMode::ComputeReplay);
  EXPECT_EQ(modeOf("ncu", ""), NsightMode::Compute);
  EXPECT_EQ(modeOf("ncu", "replay"), NsightMode::ComputeReplay);
  EXPECT_STREQ(nsightModeTool(NsightMode::Systems), "nsys");
  EXPECT_STREQ(nsightModeTool(NsightMode::Compute), "ncu");
  EXPECT_STREQ(nsightModeTool(NsightMode::ComputeReplay), "ncu");
}

/** @test A word a mode does not take is refused, naming the words it takes. */
TEST(NsightModes, UnknownWordsAreRefused) {
  NsightMode mode = NsightMode::Compute;
  const auto HEAP = parseNsightMode("nsight", "heap", mode);
  ASSERT_TRUE(HEAP.has_value());
  EXPECT_EQ(HEAP->cause, ReadinessCause::CONFIGURATION);
  EXPECT_EQ(HEAP->report.message,
            "configuration: 'heap' is not a mode of nsight; its modes are compute, ncu, replay");
  EXPECT_EQ(mode, NsightMode::Systems);
  const auto COMPUTE = parseNsightMode("ncu", "compute", mode);
  ASSERT_TRUE(COMPUTE.has_value());
  EXPECT_EQ(COMPUTE->report.message,
            "configuration: 'compute' is not a mode of ncu; its modes are replay");
  // The words it takes still select the mode, for a backend built directly.
  const auto QUOTED = parseNsightMode("ncu", "replay it's", mode);
  ASSERT_TRUE(QUOTED.has_value());
  EXPECT_EQ(QUOTED->report.message,
            "configuration: 'it's' is not a mode of ncu; its modes are replay");
  EXPECT_EQ(mode, NsightMode::ComputeReplay);
}

/**
 * @test The routes `bench run` starts for nsight, nsys and ncu are the tools
 * these modes need, and every word it refuses is refused here too, but for a
 * kernel replay, which the benchmark itself prints the ncu command for.
 */
TEST(NsightModes, RoutesMatchTheSharedTable) {
  int routes = 0;
  for (const auto& ROW : sharedTableRows("route")) {
    if (ROW[1] != "nsight" && ROW[1] != "nsys" && ROW[1] != "ncu") {
      continue;
    }
    const std::string BACKEND = ROW[1] == "ncu" ? "ncu" : "nsight";
    EXPECT_STREQ(nsightModeTool(modeOf(BACKEND, ROW[2])), ROW[3].c_str())
        << ROW[1] << " '" << ROW[2] << "'";
    ++routes;
  }
  EXPECT_EQ(routes, 5) << "the table lost its Nsight routes";
  int refusals = 0;
  for (const auto& ROW : sharedTableRows("refuse")) {
    if ((ROW[1] != "nsight" && ROW[1] != "nsys" && ROW[1] != "ncu") ||
        ROW[2].find("replay") != std::string::npos) {
      continue;
    }
    NsightMode mode = NsightMode::Systems;
    EXPECT_TRUE(parseNsightMode(ROW[1] == "ncu" ? "ncu" : "nsight", ROW[2], mode).has_value())
        << ROW[1] << " '" << ROW[2] << "' was accepted";
    ++refusals;
  }
  EXPECT_GE(refusals, 1);
}

/** @test compute-sanitizer runs one tool, memcheck by default. */
TEST(SanitizerTools, ParserSelectsOneTool) {
  for (const auto& [ARGS, TOOL] :
       std::vector<std::pair<std::string, std::string>>{{"", "memcheck"},
                                                        {"memcheck", "memcheck"},
                                                        {"racecheck", "racecheck"},
                                                        {"synccheck", "synccheck"},
                                                        {"initcheck", "initcheck"}}) {
    std::string tool;
    EXPECT_FALSE(parseSanitizerTool(ARGS, tool).has_value()) << ARGS;
    EXPECT_EQ(tool, TOOL) << ARGS;
  }
  std::string tool;
  const auto TWO = parseSanitizerTool("memcheck,racecheck", tool);
  ASSERT_TRUE(TWO.has_value());
  EXPECT_EQ(TWO->report.message,
            "configuration: compute-sanitizer runs one tool at a time; choose one of memcheck, "
            "racecheck, synccheck, initcheck");
  const auto LEAKS = parseSanitizerTool("leaks", tool);
  ASSERT_TRUE(LEAKS.has_value());
  EXPECT_EQ(LEAKS->report.message,
            "configuration: 'leaks' is not a mode of compute-sanitizer; its modes are memcheck, "
            "racecheck, synccheck, initcheck");
  EXPECT_EQ(tool, "memcheck");
  EXPECT_TRUE(parseSanitizerTool("racecheck leaks", tool).has_value());
  EXPECT_EQ(tool, "racecheck") << "the tool named, for a backend built directly";
}

/** @test The tool `bench run` passes for each mode is the one parsed here; its refusals too. */
TEST(SanitizerTools, RoutesMatchTheSharedTable) {
  int routes = 0;
  for (const auto& ROW : sharedTableRows("route")) {
    if (ROW[1] != "compute-sanitizer") {
      continue;
    }
    std::string tool;
    ASSERT_FALSE(parseSanitizerTool(ROW[2], tool).has_value()) << ROW[2];
    EXPECT_EQ(ROW[4].rfind("--tool=" + tool + " ", 0), 0U) << ROW[4];
    ++routes;
  }
  EXPECT_EQ(routes, 5) << "the table lost its compute-sanitizer routes";
  for (const auto& ROW : sharedTableRows("refuse")) {
    if (ROW[1] != "compute-sanitizer") {
      continue;
    }
    std::string tool;
    const auto REFUSED = parseSanitizerTool(ROW[2], tool);
    ASSERT_TRUE(REFUSED.has_value()) << ROW[2];
    EXPECT_NE(REFUSED->report.message.find(ROW[3]), std::string::npos) << REFUSED->report.message;
  }
}

/* ----------------------------- Doctor ----------------------------- */

/** @test The doctor probes only the selected mode's tool, and a tool that runs is unverified. */
TEST(GpuDoctorRows, EachModeProbesOnlyItsTool) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string NSYS = dir.install("fake_nvidia_tool.sh", "nsys");
  const auto CTX = dir.context({}, 0);
  const ReadinessResult SYSTEMS =
      checkNsightRequest(request("nsight", "", ReadinessScope::PREFLIGHT), CTX);
  EXPECT_EQ(SYSTEMS.cause, ReadinessCause::UNVERIFIED);
  EXPECT_EQ(SYSTEMS.report.message,
            "unverified: " + NSYS +
                " runs here (NVIDIA Nsight Systems version 2026.3.1.157-263138048394v0); whether "
                "it captures the benchmark's GPU work is not checked before the run");
  const ReadinessResult COMPUTE =
      checkNsightRequest(request("nsight", "compute", ReadinessScope::PREFLIGHT), CTX);
  EXPECT_EQ(COMPUTE.cause, ReadinessCause::MISSING) << "nsys does not serve the compute mode";
  EXPECT_EQ(COMPUTE.report.message, "missing: ncu not found on PATH");
  EXPECT_EQ(dir.logLines("nsys --version").size(), 1U) << dir.log();

  const std::string NCU = dir.install("fake_nvidia_tool.sh", "ncu");
  const ReadinessResult NCU_ROW =
      checkNsightRequest(request("ncu", "", ReadinessScope::DEFAULT_INVENTORY), CTX);
  EXPECT_EQ(NCU_ROW.report.message,
            "unverified: " + NCU +
                " runs here (Version 2026.2.0.0 (build 37790515) (public-release)); whether it "
                "captures the benchmark's GPU work is not checked before the run");
  EXPECT_EQ(dir.logLines("ncu --version").size(), 1U) << dir.log();
  EXPECT_EQ(dir.logLines("nsys --version").size(), 1U) << "ncu's row ran nsys:\n" << dir.log();
}

/** @test A tool that does not run is unusable, with its own words; a missing one is missing. */
TEST(GpuDoctorRows, BrokenOrMissingTools) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string NSYS = dir.install("fake_nvidia_tool.sh", "nsys");
  const ReadinessResult BROKEN =
      checkNsightRequest(request("nsight", "", ReadinessScope::PREFLIGHT),
                         dir.context({{"FAKE_NVIDIA_TOOL_MODE", "broken"}}, 0));
  EXPECT_EQ(BROKEN.cause, ReadinessCause::UNUSABLE);
  EXPECT_EQ(BROKEN.report.message,
            "unusable: " + NSYS +
                " --version: exit status 127: nsys: error while loading shared libraries: "
                "libcuda.so.1: cannot open shared object file");
  FakeToolDir empty;
  ASSERT_TRUE(empty.ok());
  const ReadinessResult NO_SANITIZER = checkComputeSanitizerRequest(
      request("compute-sanitizer", "", ReadinessScope::PREFLIGHT), empty.context({}, 0));
  EXPECT_EQ(NO_SANITIZER.report.message, "missing: compute-sanitizer not found on PATH");
  const std::string SANITIZER = empty.install("fake_nvidia_tool.sh", "compute-sanitizer");
  const ReadinessResult SANITIZER_ROW = checkComputeSanitizerRequest(
      request("compute-sanitizer", "racecheck", ReadinessScope::PREFLIGHT), empty.context({}, 0));
  EXPECT_EQ(SANITIZER_ROW.report.message,
            "unverified: " + SANITIZER +
                " runs here (Version 2025.4.0.0 (build 36782660) (public-release)); whether its "
                "racecheck checks the benchmark's kernels is not checked before the run");
}

/**
 * @test An ncu mode names a driver that gives the GPU's counters to root
 * only, for a process that is not root; Nsight Systems and root never get it.
 */
TEST(GpuDoctorRows, CountersForRootOnlyAreNamed) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_nvidia_tool.sh", "ncu");
  dir.install("fake_nvidia_tool.sh", "nsys");
  const std::string PARAMS = dir.path() + "/params";
  std::ofstream(PARAMS) << "ResmanDebugLevel: 4294967295\nRmProfilingAdminOnly: 1\n";
  const std::string NOTE = "; the driver gives GPU performance counters to root only "
                           "(RmProfilingAdminOnly: 1 in " +
                           PARAMS + "), and this process is not root";
  const auto ROW = [&](const std::string& backend, uid_t euid) {
    return checkNsightRequestWith(request(backend, "", ReadinessScope::PREFLIGHT),
                                  dir.context({}, euid), PARAMS)
        .report.message;
  };
  EXPECT_NE(ROW("ncu", 1000).find(NOTE), std::string::npos) << ROW("ncu", 1000);
  EXPECT_EQ(ROW("ncu", 0).find("root only"), std::string::npos);
  EXPECT_EQ(ROW("nsight", 1000).find("root only"), std::string::npos);
  std::ofstream(PARAMS) << "RmProfilingAdminOnly: 0\n";
  EXPECT_EQ(ROW("ncu", 1000).find("root only"), std::string::npos);
}

/* ----------------------------- Runtime ----------------------------- */

/**
 * @test A run no Nsight tool started fails with the command that captures it,
 * starting nothing; without the tool the request is missing it.
 */
TEST(GpuRuntime, UnwrappedRequestFailsWithTheCommand) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_nvidia_tool.sh", "nsys");
  dir.install("fake_nvidia_tool.sh", "ncu");
  const auto CTX = dir.context({}, 0);
  const ReadinessResult SYSTEMS =
      checkNsightRequest(request("nsight", "", ReadinessScope::RUNTIME), CTX);
  EXPECT_EQ(SYSTEMS.cause, ReadinessCause::MISSING);
  EXPECT_FALSE(SYSTEMS.collectionReady());
  EXPECT_EQ(SYSTEMS.report.message,
            "missing: nsight collects only when nsys starts the process, and nsys did not start "
            "this one");
  EXPECT_EQ(SYSTEMS.report.hint,
            "Wrap it: nsys profile -o ./profile -t cuda,nvtx --force-overwrite true "
            "<this-binary> --profile nsight [...]; or run it with bench run --profile nsight, "
            "which wraps it and writes the summary reports.");
  const ReadinessResult COMPUTE =
      checkNsightRequest(request("ncu", "", ReadinessScope::RUNTIME), CTX);
  EXPECT_EQ(COMPUTE.report.hint,
            "Wrap it, with few launches (ncu replays each one): ncu -o ./kernel_profile -f "
            "--target-processes all <this-binary> --profile ncu --cycles 3 --repeats 1 [...]; or "
            "run it with bench run --profile ncu --cycles 3 --repeats 1, which wraps it.");
  const ReadinessResult REPLAY =
      checkNsightRequest(request("nsight", "replay", ReadinessScope::RUNTIME), CTX);
  EXPECT_NE(
      REPLAY.report.hint.find("ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,"),
      std::string::npos)
      << REPLAY.report.hint;
  EXPECT_NE(REPLAY.report.hint.find("<this-binary> --profile nsight --profile-args replay "
                                    "--cycles 3 --repeats 1 [...]; bench run does not wrap a "
                                    "kernel replay."),
            std::string::npos)
      << REPLAY.report.hint;
  EXPECT_EQ(dir.log(), "") << "a run's check started a tool";

  FakeToolDir empty;
  ASSERT_TRUE(empty.ok());
  EXPECT_EQ(checkNsightRequest(request("nsight", "", ReadinessScope::RUNTIME), empty.context({}, 0))
                .report.message,
            "missing: nsys not found on PATH");
}

/**
 * @test A run nsys or ncu started may proceed, unverified, whether bench run
 * or a wrap typed by hand started it; under nsight's compute route the tool
 * is ncu, though the runner names the wrap nsight.
 */
TEST(GpuRuntime, SessionIsUnverified) {
  const auto CHECK = [](const std::string& backend, const std::string& args,
                        std::map<std::string, std::string> env) {
    return checkNsightRequest(request(backend, args, ReadinessScope::RUNTIME),
                              snapshot(std::move(env)));
  };
  const std::string BY_NSYS =
      "unverified: nsys started this process and writes its report when the process exits; "
      "whether it captured the benchmark's GPU work is not checked from inside the process";
  for (const auto& ENV : std::vector<std::map<std::string, std::string>>{
           {{"VERNIER_EXTERNAL_WRAP", "nsight"}}, {{"NSYS_PROFILING_SESSION_ID", "10121"}}}) {
    const ReadinessResult R = CHECK("nsight", "", ENV);
    EXPECT_EQ(R.report.message, BY_NSYS) << ENV.begin()->first;
    EXPECT_TRUE(R.collectionReady());
  }
  const ReadinessResult ROUTE =
      CHECK("nsight", "compute",
            {{"VERNIER_EXTERNAL_WRAP", "nsight"}, {"NV_NSIGHT_INJECTION_PORT_BASE", "49152"}});
  EXPECT_EQ(ROUTE.cause, ReadinessCause::UNVERIFIED) << ROUTE.report.message;
  EXPECT_EQ(ROUTE.report.message.rfind("unverified: ncu started this process", 0), 0U)
      << ROUTE.report.message;
  EXPECT_EQ(CHECK("ncu", "", {{"VERNIER_EXTERNAL_WRAP", "ncu"}}).cause, ReadinessCause::UNVERIFIED);
}

/** @test A process the other Nsight tool runs is not what the mode needs. */
TEST(GpuRuntime, WrongToolIsUnsupported) {
  const ReadinessResult R =
      checkNsightRequest(request("nsight", "compute", ReadinessScope::RUNTIME),
                         snapshot({{"NSYS_PROFILING_SESSION_ID", "10121"}}));
  EXPECT_EQ(R.cause, ReadinessCause::UNSUPPORTED);
  EXPECT_EQ(R.report.message,
            "unsupported: --profile nsight --profile-args compute needs ncu, and this process "
            "runs under nsys");
}

/**
 * @test compute-sanitizer's session is what it exports, or its libraries in
 * the process's map; nsys's injection library and a name in a path are not.
 */
TEST(GpuRuntime, SanitizerSessionIsUnverified) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_nvidia_tool.sh", "compute-sanitizer");
  const auto CHECK = [&](std::map<std::string, std::string> env, const std::string& maps) {
    return checkComputeSanitizerRequestWithMaps(
        request("compute-sanitizer", "racecheck", ReadinessScope::RUNTIME),
        dir.context(std::move(env), 0), maps);
  };
  for (const auto& ENV : std::vector<std::map<std::string, std::string>>{
           {{"VERNIER_EXTERNAL_WRAP", "compute-sanitizer"}},
           {{"NV_SANITIZER_INJECTION_PORT_BASE", "49152"}},
           {{"CUDA_INJECTION64_PATH", "/opt/cuda/compute-sanitizer/libsanitizer-collection.so"}}}) {
    const ReadinessResult R = CHECK(ENV, "");
    EXPECT_EQ(R.report.message,
              "unverified: compute-sanitizer started this process and reports when it exits; "
              "which tool it runs, and what it finds, is not seen from inside the process")
        << ENV.begin()->first;
  }
  const std::string MAPPED =
      "7f00-7f10 r-xp 00000000 08:01 42 /opt/cuda/compute-sanitizer/libsanitizer-public.so\n";
  EXPECT_EQ(CHECK({}, MAPPED).cause, ReadinessCause::UNVERIFIED);
  const ReadinessResult NONE =
      CHECK({{"CUDA_INJECTION64_PATH",
              "/opt/nvidia/nsight-systems-cli/target-linux-x64/libToolsInjection64.so"}},
            "5600-5610 r-xp 00000000 08:01 7 /home/me/sanitizer-demo/MyBench\n");
  EXPECT_EQ(NONE.cause, ReadinessCause::MISSING);
  EXPECT_EQ(NONE.report.message,
            "missing: compute-sanitizer checks a process only when it starts it, and it did not "
            "start this one");
  // The by-hand log is sanitizer.log in the working directory, named whole.
  const std::string LOG = vernier::bench::detail::shellQuote(vernier::bench::detail::escapePercent(
      (std::filesystem::current_path() / "sanitizer.log").string()));
  EXPECT_EQ(NONE.report.hint,
            "Wrap it: compute-sanitizer --tool=racecheck --error-exitcode 5 --log-file=" + LOG +
                " <this-binary> --profile compute-sanitizer --profile-args racecheck [...]; or "
                "run it with bench run --profile compute-sanitizer --profile-args racecheck, "
                "which wraps it and reads the report.");
  EXPECT_EQ(dir.log(), "") << "a run's check started the tool";
}

/** @test The remedy's error exit status is the one bench run reserves for the tool's findings. */
TEST(GpuRuntime, SanitizerRemedyUsesTheReservedStatus) {
  const auto EXITS = sharedTableRows("exit");
  ASSERT_EQ(EXITS.size(), 2U);
  ASSERT_EQ(EXITS[1][1], "tool-findings");
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_nvidia_tool.sh", "compute-sanitizer");
  const ReadinessResult R = checkComputeSanitizerRequestWithMaps(
      request("compute-sanitizer", "", ReadinessScope::RUNTIME), dir.context({}, 0), "");
  EXPECT_NE(R.report.hint.find(" --error-exitcode " + EXITS[1][2] + " "), std::string::npos)
      << R.report.hint;
}

/**
 * @test The by-hand wrap names its log whole, in the working directory, with
 *       each '%' doubled for the tool and the name quoted for the shell.
 *
 * compute-sanitizer joins a relative log name to its own working directory
 * and reads a '%' in the result as a macro, so from a directory whose path
 * holds one, `--log-file=./sanitizer.log` fails before the program starts.
 */
TEST(GpuRuntime, SanitizerRemedyNamesItsLogWhole) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_nvidia_tool.sh", "compute-sanitizer");
  std::string base = (std::filesystem::temp_directory_path() / "vernier-remedy-XXXXXX").string();
  ASSERT_NE(::mkdtemp(base.data()), nullptr);
  const std::filesystem::path HERE = std::filesystem::path(base) / "a%b c'd";
  std::error_code ec;
  std::filesystem::create_directories(HERE, ec);
  ASSERT_FALSE(ec) << ec.message();
  const std::filesystem::path BEFORE = std::filesystem::current_path();
  std::filesystem::current_path(HERE);
  const ReadinessResult R = checkComputeSanitizerRequestWithMaps(
      request("compute-sanitizer", "", ReadinessScope::RUNTIME), dir.context({}, 0), "");
  std::filesystem::current_path(BEFORE);
  std::filesystem::remove_all(base, ec);
  EXPECT_NE(R.report.hint.find(" --log-file='" + base +
                               "/a%%b c'\\''d/sanitizer.log' <this-binary> --profile "
                               "compute-sanitizer [...]; "),
            std::string::npos)
      << R.report.hint;
}

/**
 * @test --profile-analyze promises an analysis neither tool has: an
 * analysis-stage error naming the reader, the capture kept; a run that cannot
 * collect keeps its collection error.
 */
TEST(GpuRuntime, AnalysisIsNotTheBackends) {
  const ReadinessResult NSIGHT =
      checkNsightRequest(request("nsight", "", ReadinessScope::RUNTIME, /*analyze=*/true),
                         snapshot({{"VERNIER_EXTERNAL_WRAP", "nsight"}}));
  EXPECT_EQ(NSIGHT.stage, ReadinessStage::ANALYSIS);
  EXPECT_TRUE(NSIGHT.collectionReady());
  EXPECT_EQ(NSIGHT.report.message,
            "analysis: unsupported: --profile-analyze: nsight has no analysis of its own; the "
            "capture still runs and its report is kept");
  EXPECT_EQ(NSIGHT.report.hint, "Read the report with nsys stats (an ncu report with ncu --import) "
                                "after the process exits, and drop --profile-analyze.");
  const ReadinessResult SANITIZER = checkComputeSanitizerRequestWithMaps(
      request("compute-sanitizer", "", ReadinessScope::RUNTIME, /*analyze=*/true),
      snapshot({{"VERNIER_EXTERNAL_WRAP", "compute-sanitizer"}}), "");
  EXPECT_EQ(SANITIZER.stage, ReadinessStage::ANALYSIS);
  EXPECT_TRUE(SANITIZER.collectionReady());
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_nvidia_tool.sh", "nsys");
  const ReadinessResult UNWRAPPED = checkNsightRequest(
      request("nsight", "", ReadinessScope::RUNTIME, /*analyze=*/true), dir.context({}, 0));
  EXPECT_EQ(UNWRAPPED.stage, ReadinessStage::COLLECTION);
  EXPECT_EQ(UNWRAPPED.cause, ReadinessCause::MISSING);
}

/* ----------------------------- Registration ----------------------------- */

namespace {

/** @brief A factory whose profiler names @p tool, for a request the check let collect. */
vernier::bench::PlannedFactory namingFactory(const std::string& tool) {
  return [tool](const PerfConfig&, const std::string&,
                const ReadinessResult&) -> std::unique_ptr<Profiler> {
    return std::make_unique<vernier::bench::detail::NoOpProfiler>(tool, "");
  };
}

ReadinessResult readyCheck(const ReadinessRequest&, const ReadinessContext&) {
  return vernier::bench::readinessResult(ReadinessCause::READY, "ready", "");
}

/** @brief The tool the profiler registered as @p name names. */
std::string madeBy(const std::string& name) {
  PerfConfig cfg;
  cfg.profileTool = name;
  StderrCapture quiet;
  const auto PROFILER =
      ProfilerRegistry::instance().make(name, cfg, "Gpu.Registration", snapshot({}));
  return PROFILER ? PROFILER->toolName() : std::string{"(none)"};
}

} // namespace

/**
 * @test A default registered if absent loses to a full registration in
 * either order: before it, the full one replaces it; after it, it is not
 * registered.
 */
TEST(GpuRegistration, FallbackLosesInEitherOrder) {
  auto& registry = ProfilerRegistry::instance();
  EXPECT_TRUE(registry.registerReadinessBackendIfAbsent("gpu-test-first", readyCheck,
                                                        namingFactory("fallback"), ""));
  registry.registerReadinessBackend("gpu-test-first", readyCheck, namingFactory("full"), "");
  EXPECT_EQ(madeBy("gpu-test-first"), "full");

  registry.registerReadinessBackend("gpu-test-second", readyCheck, namingFactory("full"), "");
  EXPECT_FALSE(registry.registerReadinessBackendIfAbsent("gpu-test-second", readyCheck,
                                                         namingFactory("fallback"), ""));
  EXPECT_EQ(madeBy("gpu-test-second"), "full");

  EXPECT_TRUE(registry.registerReadinessBackendIfAbsent("gpu-test-alone", readyCheck,
                                                        namingFactory("fallback"), ""));
  EXPECT_EQ(madeBy("gpu-test-alone"), "fallback");
  for (const char* NAME : {"gpu-test-first", "gpu-test-second", "gpu-test-alone"}) {
    EXPECT_TRUE(registry.unregisterBackend(NAME));
  }
}

/**
 * @test In a build without CUDA (this binary links libbench alone) the three
 * names are registered: a run the tool started gets a passive profiler naming
 * the tool and the runner's folder, and a run it did not start gets none and
 * fails, naming the wrap.
 */
TEST(GpuRegistration, WithoutCudaTheNamesAreChecked) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_nvidia_tool.sh", "nsys");
  dir.install("fake_nvidia_tool.sh", "compute-sanitizer");
  auto& registry = ProfilerRegistry::instance();
  registry.resetFailures();
  for (const std::string NAME : {"nsight", "ncu", "compute-sanitizer"}) {
    EXPECT_TRUE(registry.hasBackend(NAME)) << NAME;
  }
  for (const auto& [NAME, TOOL] : std::vector<std::pair<std::string, std::string>>{
           {"nsight", "nsight"}, {"compute-sanitizer", "compute-sanitizer"}}) {
    const std::string WRAP_DIR = dir.path() + "/bench-out/bench." + NAME;
    ScopedEnv wrap("VERNIER_EXTERNAL_WRAP", NAME);
    ScopedEnv wrapDir("VERNIER_EXTERNAL_WRAP_DIR", WRAP_DIR);
    PerfConfig cfg;
    cfg.profileTool = NAME;
    StderrCapture err;
    const auto PROFILER =
        registry.make(NAME, cfg, "Gpu.Wrapped", dir.context({{"VERNIER_EXTERNAL_WRAP", NAME}}, 0));
    ASSERT_NE(PROFILER, nullptr) << NAME;
    EXPECT_EQ(PROFILER->toolName(), TOOL);
    EXPECT_EQ(PROFILER->artifactDir(), WRAP_DIR);
    EXPECT_NE(err.text().find("[WARN] Profiler '" + NAME + "': unverified: "), std::string::npos)
        << err.text();
  }
  EXPECT_TRUE(registry.failures().empty());

  PerfConfig cfg;
  cfg.profileTool = "nsight";
  cfg.artifactRoot = dir.path() + "/unwrapped";
  StderrCapture err;
  const auto NONE = registry.make("nsight", cfg, "Gpu.Unwrapped", dir.context({}, 0));
  ASSERT_NE(NONE, nullptr);
  EXPECT_EQ(NONE->toolName(), "") << "the no-op names no tool";
  const auto FAILURES = registry.failures();
  ASSERT_EQ(FAILURES.size(), 1U);
  EXPECT_EQ(FAILURES[0].result.report.message,
            "missing: nsight collects only when nsys starts the process, and nsys did not start "
            "this one");
  EXPECT_FALSE(std::filesystem::exists(cfg.artifactRoot)) << "a refused request made a folder";
  registry.resetFailures();
}
