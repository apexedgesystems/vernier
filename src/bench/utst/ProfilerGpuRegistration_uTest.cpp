/**
 * @file ProfilerGpuRegistration_uTest.cpp
 * @brief The Nsight and Compute Sanitizer names in a binary that links the
 * CUDA library: its registrations replace libbench's passive defaults, so a
 * request the readiness check lets collect builds the real backend.
 *
 * Built only where libbench_cuda is; it makes no CUDA call and needs no
 * device. Each case runs under the runner's wrap of its tool, which the
 * snapshot and this process's environment both state.
 */

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/ProfilerComputeSanitizer.hpp"
#include "src/bench/inc/ProfilerNsight.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/utst/ScopedEnv.hpp"
#include "src/bench/utst/StderrCapture.hpp"

#include <gtest/gtest.h>

#include <unistd.h>

#include <map>
#include <memory>
#include <string>

using vernier::bench::ComputeSanitizerProfiler;
using vernier::bench::NsightProfiler;
using vernier::bench::PerfConfig;
using vernier::bench::Profiler;
using vernier::bench::ProfilerRegistry;
using vernier::bench::ReadinessContext;
using vernier::bench::test::ScopedEnv;
using vernier::bench::test::StderrCapture;

namespace {

/** @brief The profiler the registry builds for @p name under the runner's wrap of it. */
std::unique_ptr<Profiler> madeUnderItsWrap(const std::string& name) {
  ScopedEnv wrap("VERNIER_EXTERNAL_WRAP", name);
  ScopedEnv wrapDir("VERNIER_EXTERNAL_WRAP_DIR", "bench-out/registration." + name);
  PerfConfig cfg;
  cfg.profileTool = name;
  StderrCapture quiet;
  return ProfilerRegistry::instance().make(
      name, cfg, "Gpu.Registration",
      ReadinessContext(::geteuid(), ::getpid(),
                       std::map<std::string, std::string>{{"VERNIER_EXTERNAL_WRAP", name}}));
}

} // namespace

/** @test nsight and ncu build the Nsight backend, and compute-sanitizer its own. */
TEST(GpuRegistrationWithCuda, TheCudaBackendsReplaceTheDefaults) {
  for (const char* NAME : {"nsight", "ncu"}) {
    const auto PROFILER = madeUnderItsWrap(NAME);
    EXPECT_NE(dynamic_cast<NsightProfiler*>(PROFILER.get()), nullptr) << NAME;
  }
  const auto SANITIZER = madeUnderItsWrap("compute-sanitizer");
  EXPECT_NE(dynamic_cast<ComputeSanitizerProfiler*>(SANITIZER.get()), nullptr);
  ProfilerRegistry::instance().resetFailures();
}
