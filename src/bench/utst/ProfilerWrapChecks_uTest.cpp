/**
 * @file ProfilerWrapChecks_uTest.cpp
 * @brief Unit tests for the readiness checks of the backends a tool wraps:
 * valgrind's (callgrind, massif, memcheck, helgrind and drd), heaptrack and
 * rocprof.
 *
 * The doctor's scopes run a start probe through a fake tool in a private PATH
 * (ReadinessFixtures.hpp); the runtime scope decides from memory maps given
 * as text, in the shapes valgrind 3.18 to 3.24 and heaptrack 1.3 to 1.5 map on
 * amd64 and arm64, from this test process's own map (never under a tool), or,
 * for rocprof, from the snapshot's environment. The modes and wrap arguments
 * are pinned to the route table `bench run` reads. Nothing here changes the
 * process environment or needs privileges.
 */

#include "src/bench/inc/ProfilerCallgrind.hpp"
#include "src/bench/inc/ProfilerHeaptrack.hpp"
#include "src/bench/inc/ProfilerHelgrind.hpp"
#include "src/bench/inc/ProfilerMassif.hpp"
#include "src/bench/inc/ProfilerMemcheck.hpp"
#include "src/bench/inc/ProfilerReadiness.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/inc/ProfilerRocprof.hpp"
#include "src/bench/inc/ValgrindTool.hpp"
#include "src/bench/utst/ReadinessFixtures.hpp"
#include "src/bench/utst/StderrCapture.hpp"

#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#ifndef VERNIER_SHARED_FIXTURE_DIR
#error "VERNIER_SHARED_FIXTURE_DIR must name tools/rust/tests/fixtures"
#endif

using vernier::bench::EnvReport;
using vernier::bench::LaunchContext;
using vernier::bench::ProfilerRegistry;
using vernier::bench::ReadinessCause;
using vernier::bench::ReadinessContext;
using vernier::bench::ReadinessRequest;
using vernier::bench::ReadinessResult;
using vernier::bench::ReadinessScope;
using vernier::bench::ReadinessStage;
using vernier::bench::test::FakeToolDir;
using vernier::bench::valgrind_tool::identityFromMaps;
using vernier::bench::valgrind_tool::ValgrindIdentity;
using vernier::bench::valgrind_tool::ValgrindMode;
using vernier::bench::valgrind_tool::ValgrindPlan;

namespace {

/** @brief The mode @p backend's own parser reads from @p profileArgs; the refusal otherwise. */
std::optional<ReadinessResult> parseFor(const std::string& backend, const std::string& profileArgs,
                                        ValgrindMode& mode) {
  if (backend == "callgrind") {
    return vernier::bench::parseCallgrindMode(profileArgs, mode);
  }
  if (backend == "massif") {
    return vernier::bench::parseMassifMode(profileArgs, mode);
  }
  if (backend == "memcheck") {
    return vernier::bench::parseMemcheckMode(profileArgs, mode);
  }
  return vernier::bench::parseHelgrindMode(profileArgs, mode);
}

/** @brief The mode @p backend reads from @p profileArgs, which must be accepted. */
ValgrindMode modeOf(const std::string& backend, const std::string& profileArgs = "") {
  ValgrindMode mode;
  const auto REFUSED = parseFor(backend, profileArgs, mode);
  EXPECT_FALSE(REFUSED.has_value())
      << backend << " '" << profileArgs << "': " << (REFUSED ? REFUSED->report.message : "");
  return mode;
}

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

bool isValgrindBackend(const std::string& name) {
  return name == "callgrind" || name == "massif" || name == "memcheck" || name == "helgrind";
}

std::string joined(const std::vector<std::string>& words) {
  std::string out;
  for (const auto& word : words) {
    out += (out.empty() ? "" : " ") + word;
  }
  return out;
}

/** @brief A /proc/<pid>/maps as valgrind leaves it: its files under @p dir on @p platform. */
std::string valgrindMaps(const std::string& dir, const std::string& platform,
                         const std::vector<std::string>& executables,
                         const std::vector<std::string>& preloads, bool core = true) {
  std::string maps = "55d0c1a00000-55d0c1a2f000 r--p 00000000 fd:01 1048601 "
                     "/home/user/build/bin/ReadinessFixtureTarget\n";
  std::size_t inode = 700000;
  const auto LINE = [&](const std::string& file) {
    maps += "58000000-58200000 r-xp 00000000 fd:01 " + std::to_string(++inode) + "          " +
            dir + "/" + file + "\n";
  };
  if (core) {
    LINE("vgpreload_core-" + platform + ".so");
  }
  for (const auto& tool : preloads) {
    LINE("vgpreload_" + tool + "-" + platform + ".so");
  }
  for (const auto& tool : executables) {
    LINE(tool + "-" + platform);
  }
  maps += "7ffd1c9e0000-7ffd1ca01000 rw-p 00000000 00:00 0                          [stack]\n";
  return maps;
}

/** @brief The identity of a process @p tool runs, as valgrind maps it on amd64. */
ValgrindIdentity running(const std::string& tool) {
  const std::vector<std::string> PRELOAD =
      tool == "callgrind" ? std::vector<std::string>{} : std::vector<std::string>{tool};
  return identityFromMaps(valgrindMaps("/usr/libexec/valgrind", "amd64-linux", {tool}, PRELOAD));
}

/** @brief A request for @p backend at @p scope. */
ReadinessRequest requestFor(const std::string& backend, ReadinessScope scope,
                            const std::string& profileArgs = "", bool analyze = false,
                            LaunchContext launch = LaunchContext::IN_PROCESS) {
  ReadinessRequest request;
  request.backend = backend;
  request.profileArgs = profileArgs;
  request.analyze = analyze;
  request.scope = scope;
  request.launch = launch;
  return request;
}

std::shared_ptr<const ValgrindPlan> planOf(const ReadinessResult& result) {
  return std::dynamic_pointer_cast<const ValgrindPlan>(result.plan);
}

/** @brief Entries of the working directory whose names start with a tool's default output. */
std::set<std::string> defaultOutputsHere() {
  std::set<std::string> names;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(".", ec)) {
    const std::string NAME = entry.path().filename().string();
    if (NAME.rfind("massif.out.", 0) == 0 || NAME.rfind("callgrind.out.", 0) == 0) {
      names.insert(NAME);
    }
  }
  return names;
}

} // namespace

/* ----------------------------- Modes and the Route Table ----------------------------- */

/** @test Every valgrind route of the shared table is what the backend's own parser builds. */
TEST(ValgrindModes, RoutesMatchTheSharedTable) {
  int checked = 0;
  for (const auto& row : sharedTableRows("route")) {
    ASSERT_EQ(row.size(), 5U);
    const std::string BACKEND = ProfilerRegistry::canonicalName(row[1]);
    if (row[3] != "valgrind") {
      continue;
    }
    ASSERT_TRUE(isValgrindBackend(BACKEND)) << row[1];
    const ValgrindMode MODE = modeOf(BACKEND, row[2]);
    std::vector<std::string> args =
        vernier::bench::valgrind_tool::wrapArguments(BACKEND, MODE, "<dir>");
    args.emplace_back("<bin>");
    EXPECT_EQ(joined(args), row[4]) << row[1] << " '" << row[2] << "'";
    ++checked;
  }
  EXPECT_GE(checked, 10) << "the table lost its valgrind routes";
}

/** @test Every refusal the shared table records for a valgrind backend is the parser's too. */
TEST(ValgrindModes, RefusalsMatchTheSharedTable) {
  int checked = 0;
  for (const auto& row : sharedTableRows("refuse")) {
    const std::string BACKEND = ProfilerRegistry::canonicalName(row[1]);
    if (!isValgrindBackend(BACKEND)) {
      continue;
    }
    ValgrindMode mode;
    const auto REFUSED = parseFor(BACKEND, row[2], mode);
    ASSERT_TRUE(REFUSED.has_value()) << row[1] << " '" << row[2] << "' was accepted";
    EXPECT_EQ(REFUSED->report.status, EnvReport::Status::Error);
    EXPECT_EQ(REFUSED->cause, ReadinessCause::CONFIGURATION);
    EXPECT_NE(REFUSED->report.message.find(row[3]), std::string::npos) << REFUSED->report.message;
    ++checked;
  }
  EXPECT_GE(checked, 5) << "the table lost its valgrind refusals";
}

/** @test The words select the tool and its options; only options the process cannot see are unseen.
 */
TEST(ValgrindModes, WordsSelectToolAndOptions) {
  EXPECT_EQ(modeOf("helgrind", "drd").tool, "drd");
  EXPECT_EQ(modeOf("helgrind").tool, "helgrind");
  EXPECT_TRUE(modeOf("helgrind", "drd").unseen.empty()) << "the tool itself is seen";
  EXPECT_EQ(modeOf("massif", "pages").unseen, std::vector<std::string>{"--pages-as-heap=yes"});
  EXPECT_EQ(modeOf("memcheck", "leak-full, track-origins").unseen,
            std::vector<std::string>{"--track-origins=yes"});
  EXPECT_TRUE(modeOf("memcheck", "leak-full").unseen.empty()) << "the default leak check";
  EXPECT_TRUE(modeOf("callgrind").options.empty());
}

/* ----------------------------- Identity ----------------------------- */

/** @test Each tool is known by its executable, beside valgrind's core preload, on amd64 and arm64.
 */
TEST(ValgrindIdentityTest, EachToolIsKnownByItsExecutable) {
  for (const char* PLATFORM : {"amd64-linux", "arm64-linux"}) {
    for (const char* TOOL : {"memcheck", "massif", "helgrind", "drd"}) {
      const ValgrindIdentity ID =
          identityFromMaps(valgrindMaps("/usr/libexec/valgrind", PLATFORM, {TOOL}, {TOOL}));
      EXPECT_TRUE(ID.underValgrind);
      EXPECT_EQ(ID.tool(), TOOL) << PLATFORM << ": " << ID.describe();
    }
    // callgrind maps no preload of its own: its executable is the evidence.
    const ValgrindIdentity CALLGRIND =
        identityFromMaps(valgrindMaps("/usr/libexec/valgrind", PLATFORM, {"callgrind"}, {}));
    EXPECT_EQ(CALLGRIND.tool(), "callgrind") << PLATFORM << ": " << CALLGRIND.describe();
  }
  // Another installation directory, as valgrind was configured.
  EXPECT_EQ(
      identityFromMaps(valgrindMaps("/opt/vg/lib/valgrind", "amd64-linux", {"massif"}, {"massif"}))
          .tool(),
      "massif");
}

/** @test valgrind's core preload alone does not establish callgrind, or any tool. */
TEST(ValgrindIdentityTest, CorePreloadAloneEstablishesNoTool) {
  const ValgrindIdentity ID =
      identityFromMaps(valgrindMaps("/usr/libexec/valgrind", "amd64-linux", {}, {}));
  EXPECT_TRUE(ID.underValgrind);
  EXPECT_EQ(ID.tool(), "");
}

/** @test Evidence that names two tools, or disagrees, establishes none. */
TEST(ValgrindIdentityTest, AmbiguousEvidenceEstablishesNoTool) {
  EXPECT_EQ(identityFromMaps(valgrindMaps("/usr/libexec/valgrind", "amd64-linux",
                                          {"memcheck", "massif"}, {"memcheck", "massif"}))
                .tool(),
            "");
  EXPECT_EQ(identityFromMaps(
                valgrindMaps("/usr/libexec/valgrind", "amd64-linux", {"callgrind"}, {"memcheck"}))
                .tool(),
            "")
      << "an executable and another tool's preload disagree";
}

/** @test Without the core preload there is no valgrind, whatever else is named like a tool. */
TEST(ValgrindIdentityTest, NothingCountsWithoutTheCorePreload) {
  const ValgrindIdentity NONE =
      identityFromMaps(valgrindMaps("/usr/libexec/valgrind", "amd64-linux", {"massif"}, {"massif"},
                                    /*core=*/false));
  EXPECT_FALSE(NONE.underValgrind);
  EXPECT_EQ(NONE.tool(), "");
  // A file named like a tool, but not beside the core preload, is not one.
  std::string maps = valgrindMaps("/usr/libexec/valgrind", "amd64-linux", {}, {});
  maps += "5a000000-5a100000 r-xp 00000000 fd:01 1 /home/user/bin/massif-amd64-linux\n";
  EXPECT_EQ(identityFromMaps(maps).tool(), "");
  EXPECT_FALSE(identityFromMaps("").underValgrind);
}

/** @test This test process, never run under valgrind, reads as not under valgrind. */
TEST(ValgrindIdentityTest, ThisProcessIsNotUnderValgrind) {
  EXPECT_FALSE(vernier::bench::valgrind_tool::identityOf(::getpid()).underValgrind);
}

/* ----------------------------- Runtime Decisions ----------------------------- */

/** @test The requested tool running the process is READY, under bench run's wrap or one started by
 * hand. */
TEST(ValgrindRuntime, RequestedToolRunningIsReady) {
  const FakeToolDir DIR;
  ASSERT_TRUE(DIR.ok());
  const ReadinessContext CTX = DIR.context();
  const ValgrindMode MODE = modeOf("massif");
  for (const LaunchContext LAUNCH : {LaunchContext::RUNNER_WRAPPED, LaunchContext::IN_PROCESS}) {
    const ReadinessResult RESULT = vernier::bench::valgrind_tool::decideRuntime(
        "massif", MODE, "", running("massif"), LAUNCH, "remedy", CTX);
    EXPECT_EQ(RESULT.cause, ReadinessCause::READY) << RESULT.report.message;
    ASSERT_NE(planOf(RESULT), nullptr);
    EXPECT_EQ(planOf(RESULT)->launch, LAUNCH == LaunchContext::RUNNER_WRAPPED
                                          ? LaunchContext::RUNNER_WRAPPED
                                          : LaunchContext::MANUALLY_WRAPPED);
  }
}

/** @test Another valgrind tool running the process is UNSUPPORTED, naming both, with the remedy. */
TEST(ValgrindRuntime, WrongToolIsUnsupported) {
  const FakeToolDir DIR;
  ASSERT_TRUE(DIR.ok());
  const ReadinessResult RESULT = vernier::bench::valgrind_tool::decideRuntime(
      "massif", modeOf("massif"), "", running("memcheck"), LaunchContext::IN_PROCESS, "the remedy",
      DIR.context());
  EXPECT_EQ(RESULT.cause, ReadinessCause::UNSUPPORTED);
  EXPECT_FALSE(RESULT.collectionReady());
  EXPECT_EQ(RESULT.report.message, "unsupported: --profile massif needs valgrind's massif, and "
                                   "this process runs under valgrind's memcheck");
  EXPECT_EQ(RESULT.report.hint, "the remedy");
}

/** @test drd is helgrind's mode and valgrind's tool: helgrind running a drd request is the wrong
 * tool. */
TEST(ValgrindRuntime, DrdRequestNeedsDrd) {
  const FakeToolDir DIR;
  ASSERT_TRUE(DIR.ok());
  const ValgrindMode DRD = modeOf("helgrind", "drd");
  const ReadinessResult WRONG = vernier::bench::valgrind_tool::decideRuntime(
      "helgrind", DRD, "drd", running("helgrind"), LaunchContext::IN_PROCESS, "", DIR.context());
  EXPECT_EQ(WRONG.cause, ReadinessCause::UNSUPPORTED);
  EXPECT_EQ(WRONG.report.message,
            "unsupported: --profile helgrind --profile-args drd needs valgrind's drd, and this "
            "process runs under valgrind's helgrind");
  EXPECT_EQ(vernier::bench::valgrind_tool::decideRuntime("helgrind", DRD, "drd", running("drd"),
                                                         LaunchContext::IN_PROCESS, "",
                                                         DIR.context())
                .cause,
            ReadinessCause::READY);
}

/** @test valgrind running the process without a map that shows the tool is not READY. */
TEST(ValgrindRuntime, AmbiguousEvidenceIsUnverified) {
  const FakeToolDir DIR;
  ASSERT_TRUE(DIR.ok());
  const ValgrindIdentity CORE_ONLY =
      identityFromMaps(valgrindMaps("/usr/libexec/valgrind", "arm64-linux", {}, {}));
  const ReadinessResult RESULT = vernier::bench::valgrind_tool::decideRuntime(
      "callgrind", modeOf("callgrind"), "", CORE_ONLY, LaunchContext::RUNNER_WRAPPED, "the remedy",
      DIR.context());
  EXPECT_EQ(RESULT.cause, ReadinessCause::UNVERIFIED);
  EXPECT_EQ(RESULT.report.status, EnvReport::Status::Warning);
  EXPECT_NE(RESULT.report.message.find("does not show which tool"), std::string::npos)
      << RESULT.report.message;
}

/** @test A wrap started by hand cannot show the options a mode adds; bench run's wrap passed them.
 */
TEST(ValgrindRuntime, ManualWrapCannotShowTheModesOptions) {
  const FakeToolDir DIR;
  ASSERT_TRUE(DIR.ok());
  const ValgrindMode PAGES = modeOf("massif", "pages");
  const ReadinessResult BY_HAND = vernier::bench::valgrind_tool::decideRuntime(
      "massif", PAGES, "pages", running("massif"), LaunchContext::IN_PROCESS, "the remedy",
      DIR.context());
  EXPECT_EQ(BY_HAND.cause, ReadinessCause::UNVERIFIED);
  EXPECT_NE(BY_HAND.report.message.find("--pages-as-heap=yes"), std::string::npos);
  EXPECT_TRUE(BY_HAND.collectionReady());
  EXPECT_EQ(
      vernier::bench::valgrind_tool::decideRuntime("massif", PAGES, "pages", running("massif"),
                                                   LaunchContext::RUNNER_WRAPPED, "", DIR.context())
          .cause,
      ReadinessCause::READY);
}

/** @test No valgrind around the process is an Error naming the wrap; the run collects nothing. */
TEST(ValgrindRuntime, UnwrappedRequestFailsWithTheWrapCommand) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_valgrind.sh", "valgrind");
  const ValgrindMode MODE = modeOf("massif", "pages");
  const std::string REMEDY = vernier::bench::valgrind_tool::wrapRemedy("massif", MODE, "pages");
  const ReadinessResult RESULT = vernier::bench::valgrind_tool::decideRuntime(
      "massif", MODE, "pages", ValgrindIdentity{}, LaunchContext::IN_PROCESS, REMEDY,
      dir.context());
  EXPECT_EQ(RESULT.cause, ReadinessCause::MISSING);
  EXPECT_FALSE(RESULT.collectionReady());
  EXPECT_EQ(RESULT.report.message, "missing: massif collects only when valgrind's massif runs "
                                   "the process, and valgrind does not run this one");
  EXPECT_EQ(RESULT.report.hint,
            "Wrap it: valgrind --tool=massif --pages-as-heap=yes --massif-out-file=./massif.out "
            "<this-binary> --profile massif --profile-args pages [...]; or run it with bench run "
            "--profile massif --profile-args pages, which wraps it.");
  ASSERT_NE(planOf(RESULT), nullptr);
  EXPECT_EQ(planOf(RESULT)->launch, LaunchContext::NOT_WRAPPED);
}

/** @test Without valgrind on PATH, the unwrapped request says so, as the doctor does. */
TEST(ValgrindRuntime, UnwrappedWithoutValgrindSaysValgrindIsMissing) {
  const FakeToolDir DIR;
  ASSERT_TRUE(DIR.ok());
  const ReadinessResult RESULT = vernier::bench::valgrind_tool::decideRuntime(
      "memcheck", modeOf("memcheck"), "", ValgrindIdentity{}, LaunchContext::IN_PROCESS, "remedy",
      DIR.context());
  EXPECT_EQ(RESULT.report.message, "missing: valgrind not found on PATH");
  EXPECT_EQ(RESULT.report.hint, vernier::bench::valgrind_tool::valgrindMissing().report.hint);
}

/** @test A run of this test process (not under valgrind) fails each valgrind request through the
 * registry. */
TEST(ValgrindRuntime, RegistryFailsAnUnwrappedRun) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_valgrind.sh", "valgrind");
  for (const char* BACKEND : {"callgrind", "massif", "memcheck", "helgrind"}) {
    const ReadinessResult RESULT = ProfilerRegistry::instance().checkRequest(
        requestFor(BACKEND, ReadinessScope::RUNTIME), dir.context());
    EXPECT_EQ(RESULT.cause, ReadinessCause::MISSING) << BACKEND << ": " << RESULT.report.message;
    EXPECT_FALSE(RESULT.collectionReady()) << BACKEND;
    EXPECT_NE(RESULT.report.hint.find("Wrap it: valgrind --tool="), std::string::npos) << BACKEND;
  }
  EXPECT_EQ(dir.log(), "") << "a run's check starts no valgrind";
}

/* ----------------------------- Doctor Scopes ----------------------------- */

/** @test The doctor starts the requested tool, with the mode's options, and names the probe. */
TEST(ValgrindDoctor, ProbeStartsTheRequestedTool) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const std::string VALGRIND = dir.install("fake_valgrind.sh", "valgrind");
  const ReadinessResult RESULT = vernier::bench::checkMassifRequest(
      requestFor("massif", ReadinessScope::PREFLIGHT, "pages"), dir.context());
  EXPECT_EQ(RESULT.cause, ReadinessCause::READY) << RESULT.report.message;
  const std::string PROBE =
      VALGRIND + " --tool=massif --pages-as-heap=yes --massif-out-file=/dev/null /bin/true";
  EXPECT_EQ(RESULT.report.message,
            "valgrind starts massif with --pages-as-heap=yes (probe: " + PROBE + ")");
  EXPECT_EQ(dir.logLines("valgrind "), std::vector<std::string>{"valgrind " + PROBE});
  ASSERT_NE(planOf(RESULT), nullptr);
  EXPECT_EQ(planOf(RESULT)->valgrind, VALGRIND);
}

/** @test drd is probed as drd, and each backend's output option points at /dev/null. */
TEST(ValgrindDoctor, EachToolIsProbedAsRequested) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_valgrind.sh", "valgrind");
  const ReadinessContext CTX = dir.context();
  EXPECT_EQ(vernier::bench::checkHelgrindRequest(
                requestFor("helgrind", ReadinessScope::PREFLIGHT, "drd"), CTX)
                .cause,
            ReadinessCause::READY);
  EXPECT_EQ(vernier::bench::checkCallgrindRequest(
                requestFor("callgrind", ReadinessScope::DEFAULT_INVENTORY), CTX)
                .cause,
            ReadinessCause::READY);
  EXPECT_EQ(vernier::bench::checkMemcheckRequest(
                requestFor("memcheck", ReadinessScope::PREFLIGHT, "track-origins"), CTX)
                .cause,
            ReadinessCause::READY);
  const std::string LOG = dir.log();
  EXPECT_NE(LOG.find(" --tool=drd --log-file=/dev/null /bin/true"), std::string::npos) << LOG;
  EXPECT_NE(LOG.find(" --tool=callgrind --callgrind-out-file=/dev/null /bin/true"),
            std::string::npos)
      << LOG;
  EXPECT_NE(LOG.find(" --tool=memcheck --leak-check=full --error-exitcode=0 --track-origins=yes "
                     "--log-file=/dev/null /bin/true"),
            std::string::npos)
      << LOG;
}

/** @test The start probe leaves no output file in the working directory. */
TEST(ValgrindDoctor, ProbeWritesNothingInTheWorkingDirectory) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_valgrind.sh", "valgrind");
  const std::set<std::string> BEFORE = defaultOutputsHere();
  for (const char* BACKEND : {"massif", "callgrind"}) {
    (void)ProfilerRegistry::instance().checkRequest(
        requestFor(BACKEND, ReadinessScope::DEFAULT_INVENTORY), dir.context());
  }
  EXPECT_EQ(defaultOutputsHere(), BEFORE);
}

/** @test A tool that does not start is UNUSABLE with valgrind's words; `--version` is not a probe.
 */
TEST(ValgrindDoctor, ToolThatDoesNotStartIsUnusable) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_valgrind.sh", "valgrind");
  const ReadinessResult RESULT =
      vernier::bench::checkMassifRequest(requestFor("massif", ReadinessScope::DEFAULT_INVENTORY),
                                         dir.context({{"FAKE_VALGRIND_MODE", "broken"}}));
  EXPECT_EQ(RESULT.cause, ReadinessCause::UNUSABLE);
  EXPECT_NE(RESULT.report.message.find("failed to start tool 'massif'"), std::string::npos)
      << RESULT.report.message;
}

/** @test A word a tool does not take is refused before anything starts. */
TEST(ValgrindDoctor, UnknownModeIsRefusedBeforeAnyProbe) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_valgrind.sh", "valgrind");
  const ReadinessResult RESULT = vernier::bench::checkMassifRequest(
      requestFor("massif", ReadinessScope::PREFLIGHT, "heap"), dir.context());
  EXPECT_EQ(RESULT.cause, ReadinessCause::CONFIGURATION);
  EXPECT_EQ(RESULT.report.message,
            "configuration: 'heap' is not a mode of massif; its modes are pages, stacks");
  EXPECT_EQ(dir.log(), "");
}

/** @test Without valgrind the doctor says it is missing; a valgrind that is not executable is
 * unusable. */
TEST(ValgrindDoctor, MissingOrBrokenValgrind) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const ReadinessResult MISSING = vernier::bench::checkMemcheckRequest(
      requestFor("memcheck", ReadinessScope::DEFAULT_INVENTORY), dir.context());
  EXPECT_EQ(MISSING.report.message, "missing: valgrind not found on PATH");
  dir.install("fake_valgrind.sh", "valgrind", 0644);
  EXPECT_EQ(vernier::bench::checkMemcheckRequest(
                requestFor("memcheck", ReadinessScope::DEFAULT_INVENTORY), dir.context())
                .cause,
            ReadinessCause::UNUSABLE);
}

/** @test The registry decides each valgrind backend's whole request, not its default mode only. */
TEST(ValgrindDoctor, RegistryDecidesTheRequestedMode) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_valgrind.sh", "valgrind");
  const ReadinessResult RESULT = ProfilerRegistry::instance().checkRequest(
      requestFor("helgrind", ReadinessScope::PREFLIGHT, "drd"), dir.context());
  EXPECT_EQ(RESULT.cause, ReadinessCause::READY) << RESULT.report.message;
  EXPECT_NE(RESULT.report.message.find("valgrind starts drd"), std::string::npos)
      << RESULT.report.message;
}

/* ----------------------------- Analysis ----------------------------- */

/**
 * @test --profile-analyze with massif, memcheck or helgrind is an analysis-stage
 * error naming the reader; the capture still runs.
 */
TEST(ValgrindAnalysis, NoAutomaticAnalysisFailsTheAnalysisStage) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_valgrind.sh", "valgrind");
  const std::map<std::string, std::string> READERS = {
      {"massif", "ms_print"}, {"memcheck", "memcheck's log"}, {"helgrind", "helgrind's log"}};
  for (const auto& [BACKEND, READER] : READERS) {
    const ReadinessResult RESULT = ProfilerRegistry::instance().checkRequest(
        requestFor(BACKEND, ReadinessScope::PREFLIGHT, "", /*analyze=*/true), dir.context());
    EXPECT_EQ(RESULT.report.status, EnvReport::Status::Error) << BACKEND;
    EXPECT_EQ(RESULT.stage, ReadinessStage::ANALYSIS) << BACKEND;
    EXPECT_TRUE(RESULT.collectionReady()) << BACKEND;
    EXPECT_EQ(RESULT.report.message.rfind("analysis: unsupported: --profile-analyze: ", 0), 0U)
        << RESULT.report.message;
    EXPECT_NE(RESULT.report.hint.find(READER), std::string::npos) << RESULT.report.hint;
    EXPECT_NE(planOf(RESULT), nullptr) << BACKEND << ": the plan that collects is kept";
  }
}

/** @test In the doctor, callgrind's --profile-analyze needs callgrind_annotate. */
TEST(ValgrindAnalysis, CallgrindAnalyzeInTheDoctorNeedsTheAnnotator) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_valgrind.sh", "valgrind");
  const ReadinessRequest REQUEST =
      requestFor("callgrind", ReadinessScope::PREFLIGHT, "", /*analyze=*/true);
  const ReadinessResult WITHOUT = vernier::bench::checkCallgrindRequest(REQUEST, dir.context());
  EXPECT_EQ(WITHOUT.cause, ReadinessCause::MISSING);
  EXPECT_EQ(WITHOUT.stage, ReadinessStage::ANALYSIS);
  EXPECT_TRUE(WITHOUT.collectionReady());
  const std::string ANNOTATE = dir.install("fake_pprof.sh", "callgrind_annotate");
  const ReadinessResult WITH = vernier::bench::checkCallgrindRequest(REQUEST, dir.context());
  EXPECT_EQ(WITH.cause, ReadinessCause::READY);
  EXPECT_NE(
      WITH.report.message.find("annotates the profile after the process exits with " + ANNOTATE),
      std::string::npos)
      << WITH.report.message;
}

/** @test Under bench run's wrap, callgrind's annotation is bench run's, after the exit. */
TEST(ValgrindAnalysis, CallgrindUnderBenchRunAnnotatesAfterExit) {
  const FakeToolDir DIR;
  ASSERT_TRUE(DIR.ok());
  const ReadinessResult RESULT = vernier::bench::checkCallgrindRequestWithIdentity(
      requestFor("callgrind", ReadinessScope::RUNTIME, "", /*analyze=*/true,
                 LaunchContext::RUNNER_WRAPPED),
      DIR.context(), running("callgrind"));
  EXPECT_EQ(RESULT.cause, ReadinessCause::READY);
  EXPECT_NE(RESULT.report.message.find("bench run annotates the profile after the process exits"),
            std::string::npos)
      << RESULT.report.message;
}

/** @test Under a wrap started by hand, callgrind's --profile-analyze is an analysis-stage error. */
TEST(ValgrindAnalysis, CallgrindManualWrapCannotAnnotate) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_helper.sh", "callgrind_control");
  const ReadinessResult RESULT = vernier::bench::checkCallgrindRequestWithIdentity(
      requestFor("callgrind", ReadinessScope::RUNTIME, "", /*analyze=*/true), dir.context(),
      running("callgrind"));
  EXPECT_EQ(RESULT.cause, ReadinessCause::UNSUPPORTED);
  EXPECT_EQ(RESULT.stage, ReadinessStage::ANALYSIS);
  EXPECT_TRUE(RESULT.collectionReady());
  ASSERT_NE(planOf(RESULT), nullptr);
  EXPECT_TRUE(planOf(RESULT)->canToggle);
}

/* ----------------------------- Callgrind's Window ----------------------------- */

/** @test A wrap started by hand switches the window when callgrind_control is there. */
TEST(CallgrindWindowCheck, ManualWrapWithControlToggles) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_helper.sh", "callgrind_control");
  const ReadinessResult RESULT = vernier::bench::checkCallgrindRequestWithIdentity(
      requestFor("callgrind", ReadinessScope::RUNTIME), dir.context(), running("callgrind"));
  EXPECT_EQ(RESULT.cause, ReadinessCause::READY) << RESULT.report.message;
  ASSERT_NE(planOf(RESULT), nullptr);
  EXPECT_TRUE(planOf(RESULT)->canToggle);
  EXPECT_EQ(planOf(RESULT)->launch, LaunchContext::MANUALLY_WRAPPED);
}

/** @test Without callgrind_control the window cannot be switched: a caveat, and no toggling. */
TEST(CallgrindWindowCheck, ManualWrapWithoutControlIsACaveat) {
  const FakeToolDir DIR;
  ASSERT_TRUE(DIR.ok());
  const ReadinessResult RESULT = vernier::bench::checkCallgrindRequestWithIdentity(
      requestFor("callgrind", ReadinessScope::RUNTIME), DIR.context(), running("callgrind"));
  EXPECT_EQ(RESULT.cause, ReadinessCause::CAVEAT);
  EXPECT_NE(RESULT.report.message.find("callgrind_control is not on PATH"), std::string::npos);
  ASSERT_NE(planOf(RESULT), nullptr);
  EXPECT_FALSE(planOf(RESULT)->canToggle);
}

/** @test bench run's wrap records the whole process: nothing is switched. */
TEST(CallgrindWindowCheck, BenchRunsWrapIsLeftAlone) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_helper.sh", "callgrind_control");
  const ReadinessResult RESULT = vernier::bench::checkCallgrindRequestWithIdentity(
      requestFor("callgrind", ReadinessScope::RUNTIME, "", false, LaunchContext::RUNNER_WRAPPED),
      dir.context(), running("callgrind"));
  EXPECT_EQ(RESULT.cause, ReadinessCause::READY);
  ASSERT_NE(planOf(RESULT), nullptr);
  EXPECT_FALSE(planOf(RESULT)->canToggle);
}

/** @test The printed wrap opens the window with --instr-atstart=no only when it can be switched. */
TEST(CallgrindWindowCheck, WrapCommandOpensTheWindowWhenItCan) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_valgrind.sh", "valgrind");
  const ReadinessRequest REQUEST = requestFor("callgrind", ReadinessScope::RUNTIME);
  const ValgrindIdentity NONE;
  const ReadinessResult WITHOUT =
      vernier::bench::checkCallgrindRequestWithIdentity(REQUEST, dir.context(), NONE);
  EXPECT_EQ(WITHOUT.report.hint.rfind("Wrap it: valgrind --tool=callgrind "
                                      "--callgrind-out-file=./callgrind.out <this-binary> "
                                      "--profile callgrind [...]",
                                      0),
            0U)
      << WITHOUT.report.hint;
  dir.install("fake_helper.sh", "callgrind_control");
  const ReadinessResult WITH =
      vernier::bench::checkCallgrindRequestWithIdentity(REQUEST, dir.context(), NONE);
  EXPECT_EQ(WITH.report.hint.rfind("Wrap it: valgrind --tool=callgrind --instr-atstart=no "
                                   "--callgrind-out-file=./callgrind.out <this-binary>",
                                   0),
            0U)
      << WITH.report.hint;
}

/** @test valgrind's core preload alone is not proof of callgrind. */
TEST(CallgrindWindowCheck, CorePreloadAloneIsNotCallgrind) {
  const FakeToolDir DIR;
  ASSERT_TRUE(DIR.ok());
  const ReadinessResult RESULT = vernier::bench::checkCallgrindRequestWithIdentity(
      requestFor("callgrind", ReadinessScope::RUNTIME), DIR.context(),
      identityFromMaps(valgrindMaps("/usr/libexec/valgrind", "amd64-linux", {}, {})));
  EXPECT_EQ(RESULT.cause, ReadinessCause::UNVERIFIED);
  EXPECT_NE(RESULT.cause, ReadinessCause::READY);
}

/* ----------------------------- heaptrack ----------------------------- */

namespace {

/** @brief A /proc/<pid>/maps of a process heaptrack started (preload) or attached to (inject). */
std::string heaptrackMaps(const std::string& library) {
  return "55d0c1a00000-55d0c1a2f000 r--p 00000000 fd:01 1048601 "
         "/home/user/build/bin/ReadinessFixtureTarget\n"
         "7f1c3a000000-7f1c3a020000 r-xp 00000000 fd:01 900001 /usr/lib/heaptrack/" +
         library + "\n";
}

} // namespace

/** @test heaptrack's route and refusal in the shared table are the backend's. */
TEST(HeaptrackCheck, RouteAndRefusalMatchTheSharedTable) {
  int routes = 0;
  for (const auto& row : sharedTableRows("route")) {
    if (row[3] != "heaptrack") {
      continue;
    }
    std::vector<std::string> args = vernier::bench::heaptrackWrapArguments("<dir>");
    args.emplace_back("<bin>");
    EXPECT_EQ(joined(args), row[4]);
    ++routes;
  }
  EXPECT_EQ(routes, 1);
  const FakeToolDir DIR;
  ASSERT_TRUE(DIR.ok());
  int refusals = 0;
  for (const auto& row : sharedTableRows("refuse")) {
    if (row[1] != "heaptrack") {
      continue;
    }
    const ReadinessResult RESULT = vernier::bench::checkHeaptrackRequest(
        requestFor("heaptrack", ReadinessScope::PREFLIGHT, row[2]), DIR.context());
    EXPECT_EQ(RESULT.cause, ReadinessCause::CONFIGURATION);
    EXPECT_NE(RESULT.report.message.find(row[3]), std::string::npos) << RESULT.report.message;
    ++refusals;
  }
  EXPECT_EQ(refusals, 1);
}

/** @test The doctor records /bin/true with heaptrack into a private directory, which it removes. */
TEST(HeaptrackCheck, DoctorRecordsTheProbe) {
  for (const std::string MODE : {"ok", "gz"}) {
    FakeToolDir dir;
    ASSERT_TRUE(dir.ok());
    dir.install("fake_heaptrack.sh", "heaptrack");
    const ReadinessResult RESULT = vernier::bench::checkHeaptrackRequestWithMaps(
        requestFor("heaptrack", ReadinessScope::DEFAULT_INVENTORY),
        dir.context({{"FAKE_HEAPTRACK_MODE", MODE}}), "");
    EXPECT_EQ(RESULT.cause, ReadinessCause::READY) << MODE << ": " << RESULT.report.message;
    EXPECT_NE(RESULT.report.message.find(std::string{"which wrote probe."} +
                                         (MODE == "gz" ? "gz" : "zst")),
              std::string::npos)
        << RESULT.report.message;
    const std::vector<std::string> LINES = dir.logLines("heaptrack ");
    ASSERT_EQ(LINES.size(), 1U) << dir.log();
    const std::size_t AT = LINES.front().find(" -o ");
    ASSERT_NE(AT, std::string::npos) << LINES.front();
    const std::string OUTPUT =
        LINES.front().substr(AT + 4, LINES.front().find(' ', AT + 4) - AT - 4);
    EXPECT_FALSE(std::filesystem::exists(std::filesystem::path(OUTPUT).parent_path()))
        << "the probe's directory was left behind: " << OUTPUT;
    EXPECT_EQ(LINES.front().substr(LINES.front().size() - 10), " /bin/true") << LINES.front();
  }
}

/** @test A heaptrack that fails, or exits 0 without a trace, is UNUSABLE; none is MISSING. */
TEST(HeaptrackCheck, BrokenSilentOrMissingHeaptrack) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const ReadinessRequest REQUEST = requestFor("heaptrack", ReadinessScope::DEFAULT_INVENTORY);
  EXPECT_EQ(
      vernier::bench::checkHeaptrackRequestWithMaps(REQUEST, dir.context(), "").report.message,
      "missing: heaptrack not found on PATH");
  dir.install("fake_heaptrack.sh", "heaptrack");
  const ReadinessResult BROKEN = vernier::bench::checkHeaptrackRequestWithMaps(
      REQUEST, dir.context({{"FAKE_HEAPTRACK_MODE", "broken"}}), "");
  EXPECT_EQ(BROKEN.cause, ReadinessCause::UNUSABLE);
  EXPECT_NE(BROKEN.report.message.find("cannot find libheaptrack_preload.so"), std::string::npos)
      << BROKEN.report.message;
  const ReadinessResult SILENT = vernier::bench::checkHeaptrackRequestWithMaps(
      REQUEST, dir.context({{"FAKE_HEAPTRACK_MODE", "empty"}}), "");
  EXPECT_EQ(SILENT.cause, ReadinessCause::UNUSABLE);
  EXPECT_NE(SILENT.report.message.find("wrote no trace"), std::string::npos)
      << SILENT.report.message;
}

/** @test With libtcmalloc in the process, the doctor's heaptrack row is a caveat naming it. */
TEST(HeaptrackCheck, TcmallocInTheProcessIsACaveat) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_heaptrack.sh", "heaptrack");
  const ReadinessResult RESULT = vernier::bench::checkHeaptrackRequestWithMaps(
      requestFor("heaptrack", ReadinessScope::PREFLIGHT), dir.context(),
      "7f00-7f10 r-xp 0 fd:01 1 /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4\n");
  EXPECT_EQ(RESULT.cause, ReadinessCause::CAVEAT);
  EXPECT_NE(RESULT.report.message.find("libtcmalloc"), std::string::npos);
  EXPECT_NE(RESULT.report.hint.find("VERNIER_LINK_TCMALLOC"), std::string::npos);
}

/** @test heaptrack's library in the map (preloaded or injected) establishes heaptrack. */
TEST(HeaptrackCheck, MappedLibraryIsTheEvidence) {
  EXPECT_TRUE(vernier::bench::heaptrackMapped(heaptrackMaps("libheaptrack_preload.so")));
  EXPECT_TRUE(vernier::bench::heaptrackMapped(heaptrackMaps("libheaptrack_inject.so")));
  EXPECT_FALSE(vernier::bench::heaptrackMapped(heaptrackMaps("libstdc++.so.6")));
  EXPECT_FALSE(vernier::bench::heaptrackMapped(""));
}

/** @test A run under heaptrack is READY; one without it fails with the wrap command. */
TEST(HeaptrackCheck, RunDecidesFromTheMap) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const ReadinessResult NONE = vernier::bench::checkHeaptrackRequestWithMaps(
      requestFor("heaptrack", ReadinessScope::RUNTIME), dir.context(), "");
  EXPECT_EQ(NONE.report.message, "missing: heaptrack not found on PATH")
      << "without heaptrack, the doctor's own words";
  dir.install("fake_heaptrack.sh", "heaptrack");
  const ReadinessResult UNWRAPPED = vernier::bench::checkHeaptrackRequestWithMaps(
      requestFor("heaptrack", ReadinessScope::RUNTIME), dir.context(), "");
  EXPECT_EQ(UNWRAPPED.cause, ReadinessCause::MISSING);
  EXPECT_FALSE(UNWRAPPED.collectionReady());
  EXPECT_EQ(UNWRAPPED.report.hint,
            "Wrap it: heaptrack -o ./run <this-binary> --profile heaptrack [...]; or run it with "
            "bench run --profile heaptrack, which wraps it.");
  for (const LaunchContext LAUNCH : {LaunchContext::RUNNER_WRAPPED, LaunchContext::IN_PROCESS}) {
    const ReadinessResult WRAPPED = vernier::bench::checkHeaptrackRequestWithMaps(
        requestFor("heaptrack", ReadinessScope::RUNTIME, "", false, LAUNCH), dir.context(),
        heaptrackMaps("libheaptrack_preload.so"));
    EXPECT_EQ(WRAPPED.cause, ReadinessCause::READY) << WRAPPED.report.message;
    const auto PLAN = std::dynamic_pointer_cast<const vernier::bench::HeaptrackPlan>(WRAPPED.plan);
    ASSERT_NE(PLAN, nullptr);
    EXPECT_EQ(PLAN->launch, LAUNCH == LaunchContext::RUNNER_WRAPPED
                                ? LaunchContext::RUNNER_WRAPPED
                                : LaunchContext::MANUALLY_WRAPPED);
  }
  EXPECT_EQ(dir.log(), "") << "a run's check starts no heaptrack";
}

/** @test --profile-analyze with heaptrack fails the analysis stage, naming heaptrack_print. */
TEST(HeaptrackCheck, AnalyzeFailsTheAnalysisStage) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const ReadinessResult RESULT = vernier::bench::checkHeaptrackRequestWithMaps(
      requestFor("heaptrack", ReadinessScope::RUNTIME, "", /*analyze=*/true), dir.context(),
      heaptrackMaps("libheaptrack_preload.so"));
  EXPECT_EQ(RESULT.cause, ReadinessCause::UNSUPPORTED);
  EXPECT_EQ(RESULT.stage, ReadinessStage::ANALYSIS);
  EXPECT_TRUE(RESULT.collectionReady());
  EXPECT_NE(RESULT.report.hint.find("heaptrack_print"), std::string::npos) << RESULT.report.hint;
}

/* ----------------------------- rocprof ----------------------------- */

/** @test rocprof is never Ok: the doctor reports it unverified, or missing. */
TEST(RocprofCheck, NeverOk) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const ReadinessRequest REQUEST = requestFor("rocprof", ReadinessScope::DEFAULT_INVENTORY);
  EXPECT_EQ(vernier::bench::checkRocprofRequest(REQUEST, dir.context()).report.message,
            "missing: rocprof not found on PATH");
  const std::string ROCPROF = dir.install("fake_rocprof.sh", "rocprof");
  const ReadinessResult RESULT = vernier::bench::checkRocprofRequest(REQUEST, dir.context());
  EXPECT_EQ(RESULT.cause, ReadinessCause::UNVERIFIED);
  EXPECT_EQ(RESULT.report.message,
            "unverified: AMD collection is not validated (legacy rocprof): rocprof is " + ROCPROF +
                "; no AMD device access or capture is checked");
  const ReadinessResult INJECTED = vernier::bench::checkRocprofRequest(
      requestFor("rocprof", ReadinessScope::RUNTIME), dir.context({{"ROCP_TOOL_LIB", "x.so"}}));
  EXPECT_EQ(INJECTED.cause, ReadinessCause::UNVERIFIED);
  EXPECT_TRUE(INJECTED.collectionReady());
  EXPECT_EQ(dir.log(), "") << "the rocprof check starts nothing";
}

/** @test Each of rocprof's injection markers is its evidence at run time. */
TEST(RocprofCheck, InjectionMarkersAreTheEvidence) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_rocprof.sh", "rocprof");
  for (const auto& [NAME, VALUE] : std::map<std::string, std::string>{
           {"ROCP_TOOL_LIB", "/opt/rocm/lib/librocprof-tool.so"},
           {"ROCPROFILER_LIBRARY", "/opt/rocm/lib/librocprofiler64.so"},
           {"LD_PRELOAD", "/opt/rocm/lib/librocprofiler-sdk-tool.so:librocprof.so"}}) {
    const ReadinessResult RESULT = vernier::bench::checkRocprofRequest(
        requestFor("rocprof", ReadinessScope::RUNTIME), dir.context({{NAME, VALUE}}));
    EXPECT_EQ(RESULT.cause, ReadinessCause::UNVERIFIED) << NAME << ": " << RESULT.report.message;
    EXPECT_NE(RESULT.report.message.find("injection is present"), std::string::npos)
        << RESULT.report.message;
  }
}

/** @test Without rocprof's injection a run fails with the wrap command; bench run does not wrap it.
 */
TEST(RocprofCheck, UnwrappedRunFailsWithTheWrapCommand) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  dir.install("fake_rocprof.sh", "rocprof");
  const ReadinessResult RESULT = vernier::bench::checkRocprofRequest(
      requestFor("rocprof", ReadinessScope::RUNTIME, "stats"), dir.context());
  EXPECT_EQ(RESULT.cause, ReadinessCause::MISSING);
  EXPECT_FALSE(RESULT.collectionReady());
  EXPECT_EQ(RESULT.report.hint,
            "Wrap it: rocprof --stats -o ./results.csv <this-binary> --profile rocprof "
            "--profile-args stats [...]; bench run does not wrap rocprof.");
}

/** @test rocprof's words select its flags; any other word is refused. */
TEST(RocprofCheck, ModesSelectFlags) {
  std::vector<std::string> flags;
  EXPECT_FALSE(vernier::bench::parseRocprofMode("stats,hip-trace", flags).has_value());
  EXPECT_EQ(flags, (std::vector<std::string>{"--stats", "--hip-trace"}));
  EXPECT_FALSE(vernier::bench::parseRocprofMode("", flags).has_value());
  EXPECT_TRUE(flags.empty());
  const auto REFUSED = vernier::bench::parseRocprofMode("hsa", flags);
  ASSERT_TRUE(REFUSED.has_value());
  EXPECT_EQ(REFUSED->report.message,
            "configuration: 'hsa' is not a mode of rocprof; its modes are stats, hsa-trace, "
            "hip-trace");
}

/** @test --profile-analyze with rocprof fails the analysis stage; the capture still runs. */
TEST(RocprofCheck, AnalyzeFailsTheAnalysisStage) {
  FakeToolDir dir;
  ASSERT_TRUE(dir.ok());
  const ReadinessResult RESULT = vernier::bench::checkRocprofRequest(
      requestFor("rocprof", ReadinessScope::RUNTIME, "", /*analyze=*/true),
      dir.context({{"ROCPROFILER_LIBRARY", "x.so"}}));
  EXPECT_EQ(RESULT.cause, ReadinessCause::UNSUPPORTED);
  EXPECT_EQ(RESULT.stage, ReadinessStage::ANALYSIS);
  EXPECT_TRUE(RESULT.collectionReady());
}

/** @test A rocprof profiler built directly for a request that cannot run creates no folder. */
TEST(RocprofCheck, RefusedDirectConstructionCreatesNoFolder) {
  for (const char* NAME : {"ROCP_TOOL_LIB", "ROCPROFILER_LIBRARY"}) {
    if (std::getenv(NAME) != nullptr) {
      GTEST_SKIP() << NAME << " is set in this process: rocprof's injection is present";
    }
  }
  const FakeToolDir ROOT;
  ASSERT_TRUE(ROOT.ok());
  vernier::bench::PerfConfig cfg;
  cfg.profileTool = "rocprof";
  cfg.artifactRoot = ROOT.path();
  const vernier::bench::test::StderrCapture QUIET;
  const vernier::bench::RocprofProfiler PROFILER(cfg, "Suite.Case");
  EXPECT_EQ(PROFILER.artifactDir(), "");
  EXPECT_FALSE(std::filesystem::exists(ROOT.path() + "/Suite.Case.rocprof"));
}
