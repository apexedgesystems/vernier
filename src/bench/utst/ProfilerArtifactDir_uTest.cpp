/**
 * @file ProfilerArtifactDir_uTest.cpp
 * @brief Unit tests for where profiler backends put their artifacts.
 *
 * Notes:
 *  - Backends are constructed directly, or given a decision that lets them
 *    run, or registered as a fixture whose request always runs, so the tests
 *    do not depend on which profiling tools are installed.
 *  - Every test works in its own temporary directory and restores the wrap
 *    environment variables it sets.
 */

#include "src/bench/inc/ProfilerEnv.hpp"

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfRegistry.hpp"
#include "src/bench/inc/Profiler.hpp"
#include "src/bench/inc/ProfilerCallgrind.hpp"
#include "src/bench/inc/ProfilerHeaptrack.hpp"
#include "src/bench/inc/ProfilerHelgrind.hpp"
#include "src/bench/inc/ProfilerJemalloc.hpp"
#include "src/bench/inc/ProfilerMassif.hpp"
#include "src/bench/inc/ProfilerMemcheck.hpp"
#include "src/bench/inc/ProfilerBpftrace.hpp"
#include "src/bench/inc/ProfilerGperf.hpp"
#include "src/bench/inc/ProfilerOffCpu.hpp"
#include "src/bench/inc/ProfilerPerf.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/utst/ReadinessFixtures.hpp"
#include "src/bench/utst/StderrCapture.hpp"

#include <gtest/gtest.h>

#include <cstdlib>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

using vernier::bench::PerfConfig;
using vernier::bench::PerfPlan;
using vernier::bench::PerfRegistry;
using vernier::bench::PerfRow;
using vernier::bench::Profiler;
using vernier::bench::ProfilerRegistry;
using vernier::bench::profiler_env::artifactDirName;
using vernier::bench::profiler_env::resolveArtifactDir;

/* ----------------------------- Fixture ----------------------------- */

class ProfilerArtifactDirTest : public ::testing::Test {
protected:
  fs::path root_;
  std::optional<std::string> savedWrap_;
  std::optional<std::string> savedWrapDir_;

  static std::optional<std::string> readEnv(const char* name) {
    const char* v = std::getenv(name);
    return (v != nullptr) ? std::optional<std::string>{v} : std::nullopt;
  }

  static void writeEnv(const char* name, const std::optional<std::string>& value) {
    if (value) {
      ::setenv(name, value->c_str(), 1);
    } else {
      ::unsetenv(name);
    }
  }

  void SetUp() override {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    root_ = fs::temp_directory_path() / ("vernier_artifact_dir_" + std::string(info->name()) + "_" +
                                         std::to_string(::getpid()));
    fs::remove_all(root_);
    fs::create_directories(root_);
    savedWrap_ = readEnv("VERNIER_EXTERNAL_WRAP");
    savedWrapDir_ = readEnv("VERNIER_EXTERNAL_WRAP_DIR");
    setWrap(std::nullopt, std::nullopt);
  }

  void TearDown() override {
    writeEnv("VERNIER_EXTERNAL_WRAP", savedWrap_);
    writeEnv("VERNIER_EXTERNAL_WRAP_DIR", savedWrapDir_);
    std::error_code ec;
    fs::remove_all(root_, ec);
  }

  void setWrap(const std::optional<std::string>& tool, const std::optional<std::string>& dir) {
    writeEnv("VERNIER_EXTERNAL_WRAP", tool);
    writeEnv("VERNIER_EXTERNAL_WRAP_DIR", dir);
  }

  PerfConfig configFor(const std::string& tool) const {
    PerfConfig cfg;
    cfg.profileTool = tool;
    cfg.artifactRoot = root_.string();
    return cfg;
  }

  /// Names of the entries directly under the test's root, sorted.
  std::vector<std::string> entries() const {
    std::vector<std::string> names;
    for (const auto& entry : fs::directory_iterator(root_)) {
      names.push_back(entry.path().filename().string());
    }
    std::sort(names.begin(), names.end());
    return names;
  }
};

/// Backends `bench run` wraps, with the suffix of their per-test folder. The
/// GPU ones live in libbench_cuda and are covered by the same shared function.
struct WrappedBackend {
  const char* tool;
  const char* suffix;
  std::function<std::unique_ptr<Profiler>(const PerfConfig&, const std::string&)> construct;
};

template <typename T> WrappedBackend wrapped(const char* tool, const char* suffix) {
  return {tool, suffix, [](const PerfConfig& cfg, const std::string& name) {
            return std::unique_ptr<Profiler>(new T(cfg, name));
          }};
}

static std::vector<WrappedBackend> wrappedBackends() {
  return {wrapped<vernier::bench::CallgrindProfiler>("callgrind", "callgrind"),
          wrapped<vernier::bench::MassifProfiler>("massif", "massif"),
          wrapped<vernier::bench::MemcheckProfiler>("memcheck", "memcheck"),
          wrapped<vernier::bench::HelgrindProfiler>("helgrind", "helgrind"),
          wrapped<vernier::bench::HeaptrackProfiler>("heaptrack", "heaptrack"),
          wrapped<vernier::bench::JemallocProfiler>("jemalloc", "jemalloc")};
}

/// A perf decision that lets the profiler run; its tool is never launched here.
static std::shared_ptr<const PerfPlan> perfThatRuns() {
  auto plan = std::make_shared<PerfPlan>();
  plan->perf = "/nonexistent/perf";
  return plan;
}

/* ----------------------------- API Tests ----------------------------- */

/// Inverse of the encoding artifactDirName() applies to a test name: the
/// folder name without its ".<suffix>", decoded left to right.
static std::string decodeTestName(const std::string& folder, const std::string& suffix) {
  const std::string ENCODED = folder.substr(0, folder.size() - suffix.size() - 1);
  std::string name;
  for (std::size_t i = 0; i < ENCODED.size(); ++i) {
    if (ENCODED.compare(i, 3, "+2F") == 0) {
      name += '/';
      i += 2;
    } else if (ENCODED.compare(i, 3, "+2B") == 0) {
      name += '+';
      i += 2;
    } else {
      name += ENCODED[i];
    }
  }
  return name;
}

/** @test A name without '/' or '+' is used as it is; '/' never reaches the folder name */
TEST(ArtifactDirNameTest, OrdinaryNamesUnchangedAndSlashesEncoded) {
  EXPECT_EQ(artifactDirName("Suite.Case", "perf"), "Suite.Case.perf");
  EXPECT_EQ(artifactDirName("My_Suite.Case_1-a b", "gperf"), "My_Suite.Case_1-a b.gperf");
  EXPECT_EQ(artifactDirName("Load.At50%", "perf"), "Load.At50%.perf");
  EXPECT_EQ(artifactDirName("Copy.n=1000", "perf"), "Copy.n=1000.perf");
  EXPECT_EQ(artifactDirName("Parts/Join.V0/n1000", "gperf"), "Parts+2FJoin.V0+2Fn1000.gperf");
  EXPECT_EQ(artifactDirName("Lang.C++/0", "perf"), "Lang.C+2B+2B+2F0.perf");
}

/**
 * @test Encoding a GoogleTest name adds no character that a consumer of the path interprets
 *
 * The folder is passed, unquoted, in output-path options to valgrind and nsys
 * ('%' starts a substitution), inside jemalloc's MALLOC_CONF (',' and ':'
 * separate), to heaptrack -o and perf -o, and in shell command lines.
 */
TEST(ArtifactDirNameTest, EncodingAddsNoCharacterItsConsumersInterpret) {
  const std::string INTERPRETED = "/%,: \t\n\"'$`\\*?[];&|<>()#!=@{}^";
  // GoogleTest names are letters, digits and '_', with '/' and '.' as separators.
  const std::vector<std::string> NAMES = {
      "A/B.Run/0",   "Sizes/Copy.Run/3",     "P/S/0.T", "/", "//",
      "A_B/C.Run/0", "Deep/er/Suite.Case/12"};
  for (const auto& name : NAMES) {
    for (const char* suffix : {"callgrind", "nsight", "jemalloc", "compute-sanitizer"}) {
      const std::string FOLDER = artifactDirName(name, suffix);
      EXPECT_EQ(FOLDER.find_first_of(INTERPRETED), std::string::npos)
          << "'" << name << "' -> '" << FOLDER << "'";
      EXPECT_NE(FOLDER.front(), '~') << FOLDER;
      EXPECT_NE(FOLDER.front(), '-') << FOLDER;
    }
  }
}

/** @test Names built to collide under a plain substitution map to distinct, decodable folders */
TEST(ArtifactDirNameTest, AdversarialNamesMapToDistinctFolders) {
  const std::vector<std::string> NAMES = {
      "A_B/C.Run/0", "A/B_C.Run/0", "A_B_C.Run_0", "A/B/C.Run/0", "A//B",      "A/_B",
      "A_/B",        "A__B",        "A+2FB",       "A/B",         "A+2B2FB",   "A+/B",
      "A+2B/B",      "A+2F/B",      "+",           "++",          "+2F",       "/",
      "//",          "+/",          "/+",          "A+",          "A+2",       "A+2f/B",
      "S.T/0",       "S.T+2F0",     "S.T_0",       "S.T.0",       ".",         "..",
      "a/b.perf/c",  "a/b",         "a+2Fb.perf",  "trailing/",   "trailing_", "trailing+2F",
      "A%2FB",       "A%/B",        "A=2FB",       "A=/B",
  };
  ASSERT_EQ(std::set<std::string>(NAMES.begin(), NAMES.end()).size(), NAMES.size())
      << "the input names themselves must be distinct";

  std::set<std::string> folders;
  for (const auto& name : NAMES) {
    const std::string FOLDER = artifactDirName(name, "gperf");
    EXPECT_EQ(FOLDER.find('/'), std::string::npos) << name << " -> " << FOLDER;
    EXPECT_EQ(decodeTestName(FOLDER, "gperf"), name) << FOLDER << " does not decode to its name";
    EXPECT_TRUE(folders.insert(FOLDER).second)
        << "'" << name << "' shares the folder name '" << FOLDER << "' with another test";
  }
  EXPECT_EQ(folders.size(), NAMES.size());
}

/** @test A parameterized test name yields one flat folder under the root, not a nested path */
TEST_F(ProfilerArtifactDirTest, ParameterizedNameGetsOneFlatFolder) {
  const vernier::bench::test::StderrCapture QUIET;
  const vernier::bench::PerfStatProfiler PERF(configFor("perf"), "Parts/Join.V0/n1000",
                                              perfThatRuns());
  const vernier::bench::MassifProfiler MASSIF(configFor("massif"), "Parts/Join.V0/n1000");

  EXPECT_EQ(entries(), (std::vector<std::string>{"Parts+2FJoin.V0+2Fn1000.massif",
                                                 "Parts+2FJoin.V0+2Fn1000.perf"}));
  EXPECT_EQ(PERF.artifactDir(), (root_ / "Parts+2FJoin.V0+2Fn1000.perf").string());
  EXPECT_EQ(MASSIF.artifactDir(), (root_ / "Parts+2FJoin.V0+2Fn1000.massif").string());
}

/** @test Outside a wrap every wrapped backend keeps its per-test folder and reports it */
TEST_F(ProfilerArtifactDirTest, UnwrappedBackendOwnsPerTestFolder) {
  const vernier::bench::test::StderrCapture QUIET;
  for (const auto& backend : wrappedBackends()) {
    const auto PROF = backend.construct(configFor(backend.tool), "Suite.Case");
    const fs::path EXPECTED = root_ / (std::string("Suite.Case.") + backend.suffix);
    EXPECT_EQ(PROF->artifactDir(), EXPECTED.string()) << backend.tool;
    EXPECT_TRUE(fs::is_directory(EXPECTED)) << backend.tool;
  }
}

/** @test Under the runner's wrap no per-test folder appears and the wrap folder is reported */
TEST_F(ProfilerArtifactDirTest, WrappedBackendCreatesNoFolderAndReportsWrapFolder) {
  const vernier::bench::test::StderrCapture QUIET;
  for (const auto& backend : wrappedBackends()) {
    const std::string WRAP_DIR = "bench-out/Binary_PTEST." + std::string(backend.tool);
    setWrap(backend.tool, WRAP_DIR);
    const auto PROF = backend.construct(configFor(backend.tool), "Suite.Case");
    EXPECT_EQ(PROF->artifactDir(), WRAP_DIR) << backend.tool;
    EXPECT_TRUE(entries().empty()) << backend.tool << " left '" << entries().front()
                                   << "' behind for a run whose data is in the wrap folder";
  }
}

/** @test The CSV row's profileDir carries the wrap folder for a wrapped run */
TEST_F(ProfilerArtifactDirTest, CsvProfileDirPointsAtWrapFolder) {
  const vernier::bench::test::StderrCapture QUIET;
  const std::string WRAP_DIR = "bench-out/Binary_PTEST.jemalloc";
  setWrap("jemalloc", WRAP_DIR);

  // jemalloc is the one wrapped backend whose factory needs no installed tool.
  PerfConfig cfg = configFor("jemalloc");
  cfg.cycles = 1;
  cfg.repeats = 2;
  cfg.warmup = 0;
  auto perf = vernier::bench::makePerfCaseWithProfiler("Suite.Case", cfg);
  static_cast<void>(PerfRegistry::instance().take());
  perf.throughputLoop([] {});

  const std::optional<PerfRow> ROW = PerfRegistry::instance().take();
  ASSERT_TRUE(ROW.has_value());
  ASSERT_TRUE(ROW->profileDir.has_value());
  EXPECT_EQ(*ROW->profileDir, WRAP_DIR);
  EXPECT_TRUE(entries().empty());
}

/** @test A wrap by another tool does not redirect this backend */
TEST_F(ProfilerArtifactDirTest, WrapByAnotherToolIsIgnored) {
  const vernier::bench::test::StderrCapture QUIET;
  setWrap("massif", "bench-out/Binary_PTEST.massif");
  const vernier::bench::PerfStatProfiler PERF(configFor("perf"), "Suite.Case", perfThatRuns());
  EXPECT_EQ(PERF.artifactDir(), (root_ / "Suite.Case.perf").string());
  EXPECT_EQ(entries(), std::vector<std::string>{"Suite.Case.perf"});
}

/**
 * @test A backend built for a request that cannot run creates no folder
 *
 * The four backends that decide their own request when constructed directly
 * leave nothing behind when the decision rejects it, like every backend the
 * registry does not build. The tools are made absent with an empty PATH.
 */
TEST_F(ProfilerArtifactDirTest, RejectedRequestCreatesNoFolder) {
  const vernier::bench::test::FakeToolDir empty;
  ASSERT_TRUE(empty.ok());
  const vernier::bench::test::ScopedEnv path("PATH", empty.path());
  const vernier::bench::test::StderrCapture QUIET;
  const vernier::bench::PerfStatProfiler PERF(configFor("perf"), "Suite.Perf");
  const vernier::bench::BpftraceProfiler BPF(configFor("bpftrace"), "Suite.Bpf");
  const vernier::bench::OffCpuProfiler OFFCPU(configFor("offcpu"), "Suite.OffCpu");
  EXPECT_EQ(PERF.artifactDir(), "");
  EXPECT_EQ(BPF.artifactDir(), "");
  EXPECT_EQ(OFFCPU.artifactDir(), "");
  if (UB_HAS_GPERF_HEAP == 0) {
    // Heap mode is not compiled in (or gperftools is absent): rejected either way.
    PerfConfig heap = configFor("gperf");
    heap.profileArgs = "heap";
    const vernier::bench::GperfProfiler GPERF(heap, "Suite.Gperf");
    EXPECT_EQ(GPERF.artifactDir(), "");
  }
  EXPECT_TRUE(entries().empty()) << "a rejected request left '" << entries().front() << "'";
}

/** @test A wrap that does not say where it writes yields no folder and no claim about one */
TEST_F(ProfilerArtifactDirTest, WrapWithoutFolderReportsNothing) {
  setWrap("massif", std::nullopt);
  EXPECT_EQ(resolveArtifactDir("massif", root_.string(), "Suite.Case", "massif"), "");
  EXPECT_TRUE(entries().empty());
}

/* ----------------------------- Parameterized Case Tests ----------------------------- */

// Two real parameterized suites whose full names, "A_B/C.Run/0" and
// "A/B_C.Run/0", differ only in where '/' and '_' sit. Each case profiles
// itself, leaves a file in the folder its CSV row reports, and checks that the
// other case's folder is a different one whose file it did not touch. The
// checks are symmetric, so they hold in either execution order.
class C : public ::testing::TestWithParam<int> {};
class B_C : public ::testing::TestWithParam<int> {};

static fs::path sharedCaptureRoot() {
  return fs::temp_directory_path() /
         ("vernier_artifact_dir_param_cases_" + std::to_string(::getpid()));
}

// The two cases need each other's folder to outlive them, so the shared root
// is removed when the test program exits rather than after each case.
struct SharedCaptureRootCleanup {
  ~SharedCaptureRootCleanup() {
    std::error_code ec;
    fs::remove_all(sharedCaptureRoot(), ec);
  }
};
static const SharedCaptureRootCleanup SHARED_CAPTURE_ROOT_CLEANUP;

/// A profiler that only takes its per-test folder, through the same shared
/// rule every backend uses.
class FolderOnlyProfiler final : public Profiler {
public:
  FolderOnlyProfiler(const PerfConfig& cfg, const std::string& testName)
      : dir_(resolveArtifactDir(cfg.profileTool, cfg.artifactRoot, testName, "fixture")) {}
  std::string toolName() const noexcept override { return "folder-fixture"; }
  std::string artifactDir() const noexcept override { return dir_; }

private:
  std::string dir_;
};

/// Registers, for one case, a backend whose request always runs: its check is
/// always ready and it needs no installed tool.
class FolderFixtureBackend {
public:
  static constexpr const char* NAME = "folder-fixture";
  FolderFixtureBackend() {
    ProfilerRegistry::instance().registerReadinessBackend(
        NAME,
        [](const vernier::bench::ReadinessRequest&, const vernier::bench::ReadinessContext&) {
          return vernier::bench::readinessResult(vernier::bench::ReadinessCause::READY,
                                                 "always ready", "");
        },
        [](const PerfConfig& cfg, const std::string& testName,
           const vernier::bench::ReadinessResult&) {
          return std::unique_ptr<Profiler>(std::make_unique<FolderOnlyProfiler>(cfg, testName));
        },
        "");
  }
  ~FolderFixtureBackend() { ProfilerRegistry::instance().unregisterBackend(NAME); }
  FolderFixtureBackend(const FolderFixtureBackend&) = delete;
  FolderFixtureBackend& operator=(const FolderFixtureBackend&) = delete;
};

static std::string readFile(const fs::path& path) {
  std::ifstream in(path);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

static void captureAndCheckSibling(const std::string& siblingName) {
  const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
  const std::string OWN_NAME = std::string(info->test_suite_name()) + "." + info->name();
  ASSERT_NE(OWN_NAME, siblingName);

  const vernier::bench::test::StderrCapture QUIET;
  ::unsetenv("VERNIER_EXTERNAL_WRAP");
  ::unsetenv("VERNIER_EXTERNAL_WRAP_DIR");
  const fs::path ROOT = sharedCaptureRoot();
  fs::create_directories(ROOT);

  // A registered fixture whose request always runs: the case builds its
  // profiler through the registry like any guarded case, with no installed
  // tool involved, and the profiler takes its folder like every backend.
  const FolderFixtureBackend BACKEND;
  PerfConfig cfg;
  cfg.profileTool = FolderFixtureBackend::NAME;
  cfg.artifactRoot = ROOT.string();
  cfg.cycles = 1;
  cfg.repeats = 2;
  cfg.warmup = 0;
  auto perf = vernier::bench::makePerfCaseWithProfiler(OWN_NAME, cfg);
  static_cast<void>(PerfRegistry::instance().take());
  perf.throughputLoop([] {});
  const std::optional<PerfRow> ROW = PerfRegistry::instance().take();
  ASSERT_TRUE(ROW.has_value());
  ASSERT_TRUE(ROW->profileDir.has_value());

  const fs::path OWN_DIR = *ROW->profileDir;
  const fs::path SIBLING_DIR = ROOT / artifactDirName(siblingName, "fixture");
  EXPECT_EQ(OWN_DIR.parent_path(), ROOT) << "the folder is not directly under the root";
  EXPECT_NE(OWN_DIR, SIBLING_DIR) << "'" << OWN_NAME << "' and '" << siblingName
                                  << "' report the same artifact folder";
  ASSERT_TRUE(fs::is_directory(OWN_DIR));

  // Stand-in for a backend's fixed file name (gperf's cpu.prof).
  std::ofstream(OWN_DIR / "capture.txt") << OWN_NAME;
  EXPECT_EQ(readFile(OWN_DIR / "capture.txt"), OWN_NAME);
  if (fs::exists(SIBLING_DIR / "capture.txt")) {
    EXPECT_EQ(readFile(SIBLING_DIR / "capture.txt"), siblingName)
        << "this case overwrote the capture of '" << siblingName << "'";
  }
}

/** @test Case "A_B/C.Run/0" gets a folder of its own, distinct from "A/B_C.Run/0" */
TEST_P(C, Run) { captureAndCheckSibling("A/B_C.Run/0"); }

/** @test Case "A/B_C.Run/0" gets a folder of its own, distinct from "A_B/C.Run/0" */
TEST_P(B_C, Run) { captureAndCheckSibling("A_B/C.Run/0"); }

INSTANTIATE_TEST_SUITE_P(A_B, C, ::testing::Values(1));
INSTANTIATE_TEST_SUITE_P(A, B_C, ::testing::Values(1));
