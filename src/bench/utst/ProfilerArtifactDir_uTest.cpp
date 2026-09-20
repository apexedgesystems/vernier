/**
 * @file ProfilerArtifactDir_uTest.cpp
 * @brief Unit tests for where profiler backends put their artifacts.
 *
 * Notes:
 *  - Backends are constructed directly, so the tests do not depend on which
 *    profiling tools are installed.
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
#include "src/bench/inc/ProfilerPerf.hpp"
#include "src/bench/utst/StderrCapture.hpp"

#include <gtest/gtest.h>

#include <cstdlib>

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

using vernier::bench::PerfConfig;
using vernier::bench::PerfRegistry;
using vernier::bench::PerfRow;
using vernier::bench::Profiler;
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

/* ----------------------------- API Tests ----------------------------- */

/** @test Replaces every '/' of a parameterized test name and appends the suffix */
TEST(ArtifactDirNameTest, FlattensSlashesAndAppendsSuffix) {
  EXPECT_EQ(artifactDirName("Parts/Join.V0/n1000", "gperf"), "Parts_Join.V0_n1000.gperf");
  EXPECT_EQ(artifactDirName("Suite.Case", "perf"), "Suite.Case.perf");
  EXPECT_EQ(artifactDirName("A/B/C/D", "x").find('/'), std::string::npos);
}

/** @test A parameterized test name yields one flat folder under the root, not a nested path */
TEST_F(ProfilerArtifactDirTest, ParameterizedNameGetsOneFlatFolder) {
  const vernier::bench::test::StderrCapture QUIET;
  const vernier::bench::PerfStatProfiler PERF(configFor("perf"), "Parts/Join.V0/n1000");
  const vernier::bench::MassifProfiler MASSIF(configFor("massif"), "Parts/Join.V0/n1000");

  EXPECT_EQ(entries(),
            (std::vector<std::string>{"Parts_Join.V0_n1000.massif", "Parts_Join.V0_n1000.perf"}));
  EXPECT_EQ(PERF.artifactDir(), (root_ / "Parts_Join.V0_n1000.perf").string());
  EXPECT_EQ(MASSIF.artifactDir(), (root_ / "Parts_Join.V0_n1000.massif").string());
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
  const vernier::bench::PerfStatProfiler PERF(configFor("perf"), "Suite.Case");
  EXPECT_EQ(PERF.artifactDir(), (root_ / "Suite.Case.perf").string());
  EXPECT_EQ(entries(), std::vector<std::string>{"Suite.Case.perf"});
}

/** @test A wrap that does not say where it writes yields no folder and no claim about one */
TEST_F(ProfilerArtifactDirTest, WrapWithoutFolderReportsNothing) {
  setWrap("massif", std::nullopt);
  EXPECT_EQ(resolveArtifactDir("massif", root_.string(), "Suite.Case", "massif"), "");
  EXPECT_TRUE(entries().empty());
}
