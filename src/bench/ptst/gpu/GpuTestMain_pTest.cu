/**
 * @file GpuTestMain_pTest.cu
 * @brief Main entry point for GPU benchmark tests
 *
 * This file provides the main() function for the GPU test binary,
 * ensuring that both CPU and GPU command-line flags are properly parsed.
 */

#include <gtest/gtest.h>
#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfGpuConfig.hpp"
#include "src/bench/inc/PerfRegistry.hpp"
#include "src/bench/inc/PerfListener.hpp"
#include "src/bench/inc/PerfTestMacros.hpp"
#include "src/bench/inc/PerfGpuTestMacros.hpp"

int main(int argc, char** argv) {
  // Parse CPU flags (--cycles, --repeats, --profile, --artifact-root, etc.)
  auto& cfg = vernier::bench::detail::perfConfigSingleton();
  vernier::bench::parsePerfFlags(cfg, &argc, argv);

  // Parse GPU-specific flags (--gpu-memory, --gpu-device, etc.)
  vernier::bench::PerfGpuConfig gpuCfg;
  vernier::bench::parseGpuFlags(gpuCfg, &argc, argv);

  // Register configs globally
  vernier::bench::detail::setGlobalGpuConfig(gpuCfg);
  vernier::bench::setGlobalPerfConfig(&cfg);

  // Install CSV listener if --csv provided
  vernier::bench::installPerfEventListener(cfg);

  // Initialize GoogleTest with remaining args
  ::testing::InitGoogleTest(&argc, argv);

  // Run all tests; a failed --profile request makes the run's status 4 when
  // they pass, as PERF_GPU_MAIN() does.
  const int RC = RUN_ALL_TESTS();
  return vernier::bench::ProfilerRegistry::instance().finishRun(
      cfg, RC, ::testing::UnitTest::GetInstance()->test_to_run_count());
}
