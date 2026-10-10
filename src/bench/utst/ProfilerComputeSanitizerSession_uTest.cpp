/**
 * @file ProfilerComputeSanitizerSession_uTest.cpp
 * @brief Unit tests for the Compute Sanitizer detection in profiler_env:
 *        computeSanitizerSession() on snapshots and mapsShowComputeSanitizer()
 *        on map texts.
 *
 * Notes:
 *  - Snapshots are built by hand, so no test touches the process environment
 *    and the tests are independent of each other and of their order.
 *  - The map lines are the ones an instrumented process showed on a Jetson
 *    AGX Thor (compute-sanitizer 2025.3.1, CUDA 13.0), with the paths that
 *    must not count: the demo binary's own, a directory named after the
 *    tool, the launcher libraries Nsight ships too, another sanitizer.
 */

#include "src/bench/inc/ProfilerEnv.hpp"

#include "src/bench/inc/ProfilerReadiness.hpp"

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <utility>

using vernier::bench::ReadinessContext;
using vernier::bench::profiler_env::computeSanitizerSession;
using vernier::bench::profiler_env::mapsShowComputeSanitizer;

/* ----------------------------- Constants ----------------------------- */

namespace {

/// The tool's own libraries, as an instrumented process maps them.
constexpr const char* COLLECTION_LINE =
    "ffff8f260000-ffff8fd97000 r-xp 00000000 103:01 18484268                  "
    "/usr/local/cuda-13.0/compute-sanitizer/libsanitizer-collection.so\n";
constexpr const char* PUBLIC_LINE =
    "ffff873a0000-ffff87584000 r-xp 00000000 103:01 18484269                  "
    "/usr/local/cuda-13.0/compute-sanitizer/libsanitizer-public.so\n";

/// The launcher libraries the tool maps beside its own; Nsight ships them too.
constexpr const char* LAUNCHER_LINES =
    "ffff8ece0000-ffff8f023000 r-xp 00000000 103:01 18484266                  "
    "/usr/local/cuda-13.0/compute-sanitizer/libTreeLauncherTargetInjection.so\n"
    "ffff8f060000-ffff8f119000 r-xp 00000000 103:01 18484264                  "
    "/usr/local/cuda-13.0/compute-sanitizer/libInterceptorInjectionTarget.so\n"
    "ffff8f150000-ffff8f23f000 r-xp 00000000 103:01 18484267                  "
    "/usr/local/cuda-13.0/compute-sanitizer/libTreeLauncherTargetUpdatePreloadInjection.so\n";

/// The demo binary's own mappings: its path names the tool.
constexpr const char* DEMO_BINARY_LINES =
    "aaaacd600000-aaaacd6a3000 r-xp 00000000 103:01 41571166                  "
    "/b/bin/ptests/BenchDemo_Gpu_04_ComputeSanitizerProfiler\n"
    "aaaacd6ba000-aaaacd6c0000 r--p 000aa000 103:01 41571166                  "
    "/b/bin/ptests/BenchDemo_Gpu_04_ComputeSanitizerProfiler\n"
    "aaaacd6c0000-aaaacd6c1000 rw-p 000b0000 103:01 41571166                  "
    "/b/bin/ptests/BenchDemo_Gpu_04_ComputeSanitizerProfiler\n";

/// Mappings every process has, without a path or with an ordinary one.
constexpr const char* ORDINARY_LINES =
    "ffff90a00000-ffff90a30000 rw-p 00000000 00:00 0 \n"
    "ffff90a30000-ffff90a31000 r--p 00000000 00:00 0                          [vvar]\n"
    "ffff90c00000-ffff90d80000 r-xp 00000000 103:01 3959                      "
    "/usr/lib/aarch64-linux-gnu/libc.so.6\n";

/// A snapshot of an unprivileged process with @p environment.
ReadinessContext snapshot(std::map<std::string, std::string> environment) {
  return ReadinessContext(1000, 4242, std::move(environment));
}

} // namespace

/* ----------------------------- Session Tests ----------------------------- */

/** @test An empty snapshot shows no session */
TEST(ComputeSanitizerSessionTest, EmptySnapshotIsNoSession) {
  EXPECT_FALSE(computeSanitizerSession(snapshot({})));
}

/** @test The runner's wrap counts when it names this tool, not another */
TEST(ComputeSanitizerSessionTest, RunnerWrapCountsForThisToolOnly) {
  EXPECT_TRUE(computeSanitizerSession(snapshot({{"VERNIER_EXTERNAL_WRAP", "compute-sanitizer"}})));
  EXPECT_FALSE(computeSanitizerSession(snapshot({{"VERNIER_EXTERNAL_WRAP", "memcheck"}})));
  EXPECT_FALSE(computeSanitizerSession(snapshot({{"VERNIER_EXTERNAL_WRAP", "ncu"}})));
}

/** @test The port variable the tool exports to its target counts */
TEST(ComputeSanitizerSessionTest, ExportedPortBaseCounts) {
  EXPECT_TRUE(computeSanitizerSession(snapshot({{"NV_SANITIZER_INJECTION_PORT_BASE", "49152"}})));
  // The other variables of the same export are not the decision
  EXPECT_FALSE(
      computeSanitizerSession(snapshot({{"NV_SANITIZER_INJECTION_TRANSPORT_TYPE", "uds"},
                                        {"NVTX_INJECTION64_PATH", "libsanitizer-collection.so"}})));
}

/** @test An injection path counts only when it names the collection library */
TEST(ComputeSanitizerSessionTest, InjectionPathCountsByFileName) {
  EXPECT_TRUE(computeSanitizerSession(
      snapshot({{"CUDA_INJECTION64_PATH",
                 "/usr/local/cuda/compute-sanitizer/libsanitizer-collection.so"}})));
  EXPECT_TRUE(
      computeSanitizerSession(snapshot({{"CUDA_INJECTION64_PATH", "libsanitizer-collection.so"}})));
  // A directory named after the tool, or another tool's injection, is not it
  EXPECT_FALSE(computeSanitizerSession(
      snapshot({{"CUDA_INJECTION64_PATH", "/usr/local/cuda/compute-sanitizer/libother.so"}})));
  EXPECT_FALSE(computeSanitizerSession(
      snapshot({{"CUDA_INJECTION64_PATH", "/opt/nsight/libToolsInjection64.so"}})));
  EXPECT_FALSE(computeSanitizerSession(snapshot({{"CUDA_INJECTION64_PATH", ""}})));
}

/** @test Nsight's session variables are not this tool's */
TEST(ComputeSanitizerSessionTest, NsightSessionIsNotThisTool) {
  EXPECT_FALSE(computeSanitizerSession(snapshot({{"NSYS_PROFILING_SESSION_ID", "1"}})));
  EXPECT_FALSE(computeSanitizerSession(snapshot({{"NV_NSIGHT_INJECTION_PORT_BASE", "49152"}})));
}

/* ----------------------------- Maps Tests ----------------------------- */

/** @test The tool's own libraries count, each on its own, among ordinary lines */
TEST(ComputeSanitizerMapsTest, TheToolsLibrariesCount) {
  EXPECT_TRUE(mapsShowComputeSanitizer(std::string(ORDINARY_LINES) + COLLECTION_LINE));
  EXPECT_TRUE(mapsShowComputeSanitizer(std::string(ORDINARY_LINES) + PUBLIC_LINE));
  EXPECT_TRUE(mapsShowComputeSanitizer(std::string(DEMO_BINARY_LINES) + LAUNCHER_LINES +
                                       COLLECTION_LINE + PUBLIC_LINE + ORDINARY_LINES));
}

/** @test A last line without a newline is read too */
TEST(ComputeSanitizerMapsTest, TheLastLineNeedsNoNewline) {
  const std::string TEXT = std::string(ORDINARY_LINES) + COLLECTION_LINE;
  EXPECT_TRUE(mapsShowComputeSanitizer(TEXT.substr(0, TEXT.size() - 1)));
}

/** @test The binary's own path, named after the tool, does not count */
TEST(ComputeSanitizerMapsTest, TheBinarysOwnPathDoesNotCount) {
  EXPECT_FALSE(mapsShowComputeSanitizer(std::string(DEMO_BINARY_LINES) + ORDINARY_LINES));
}

/** @test A directory named after the tool does not count */
TEST(ComputeSanitizerMapsTest, ADirectoryNamedAfterTheToolDoesNotCount) {
  EXPECT_FALSE(mapsShowComputeSanitizer(
      "ffff90c00000-ffff90d80000 r-xp 00000000 103:01 3959                      "
      "/opt/compute-sanitizer/lib/libfoo.so\n"
      "ffff90e00000-ffff90e80000 r-xp 00000000 103:01 3960                      "
      "/home/sanitizer/bin/Sanitizer\n"));
}

/** @test The launcher libraries alone do not count: Nsight maps them too */
TEST(ComputeSanitizerMapsTest, LauncherLibrariesAloneDoNotCount) {
  EXPECT_FALSE(mapsShowComputeSanitizer(std::string(LAUNCHER_LINES) + ORDINARY_LINES));
}

/** @test Another sanitizer's runtime does not count */
TEST(ComputeSanitizerMapsTest, AnotherSanitizerDoesNotCount) {
  EXPECT_FALSE(mapsShowComputeSanitizer(
      "7f0000000000-7f0000100000 r-xp 00000000 08:01 1234                      "
      "/usr/lib/x86_64-linux-gnu/libasan.so.8\n"
      "7f0000200000-7f0000300000 r-xp 00000000 08:01 1235                      "
      "/usr/lib/llvm-21/lib/clang/21/lib/linux/libclang_rt.asan-x86_64.so\n"));
}

/** @test An empty text and lines without a path show nothing */
TEST(ComputeSanitizerMapsTest, EmptyAndPathlessLinesShowNothing) {
  EXPECT_FALSE(mapsShowComputeSanitizer(""));
  EXPECT_FALSE(mapsShowComputeSanitizer("\n\n"));
  EXPECT_FALSE(mapsShowComputeSanitizer("ffff90a00000-ffff90a30000 rw-p 00000000 00:00 0 \n"));
}
