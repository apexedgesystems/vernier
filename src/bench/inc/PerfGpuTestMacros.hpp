#ifndef VERNIER_PERFGPUTESTMACROS_HPP
#define VERNIER_PERFGPUTESTMACROS_HPP
/**
 * @file PerfGpuTestMacros.hpp
 * @brief Test declaration macros for GPU benchmarks with multi-GPU support.
 *
 * Added GPU flag parsing to PERF_MAIN macro
 */

#include <string>

#include "src/bench/inc/PerfGpuHarness.hpp"
#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfGpuConfig.hpp"
#include "src/bench/inc/PerfTestMacros.hpp"

/* ----------------------------- Test Declaration Macros ----------------------------- */

#define PERF_GPU_TEST(Suite, Name) TEST(Suite, Name)
#define PERF_GPU_THROUGHPUT(Suite, Name) TEST(Suite, Name)
#define PERF_GPU_LATENCY(Suite, Name) TEST(Suite, Name)
#define PERF_GPU_COMPARISON(Suite, Name) TEST(Suite, Name)
#define PERF_GPU_BANDWIDTH(Suite, Name) TEST(Suite, Name)

/**
 * @brief Multi-GPU scaling test macro.
 *
 * Use for tests that scale across multiple GPUs or vary launch configurations.
 *
 * Example:
 * @code{.cpp}
 * PERF_GPU_SCALING(MatMul, MultiGpu) {
 *   UB_PERF_GPU_GUARD(perf);
 *
 *   auto results = perf.cudaKernelMultiGpu(4,
 *     [&](int dev, cudaStream_t s) {
 *       kernel<<<grid, block, 0, s>>>(d_data[dev]);
 *     }
 *   ).measure();
 *
 *   EXPECT_GT(results.aggregatedStats.multiGpu->scalingEfficiency, 0.8);
 * }
 * @endcode
 */
#define PERF_GPU_SCALING(Suite, Name) TEST(Suite, Name)

/* ----------------------------- Scoped Guard ----------------------------- */

#define UB_PERF_GPU_GUARD(varName)                                                                 \
  vernier::bench::PerfGpuCase varName {                                                            \
    ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name() +                   \
        std::string(".") + ::testing::UnitTest::GetInstance()->current_test_info()->name(),        \
        vernier::bench::detail::getPerfConfig()                                                    \
  }

/* ----------------------------- GPU Main Macro ----------------------------- */

/**
 * @brief GPU-aware main() that parses both CPU and GPU flags.
 *
 * Use this instead of PERF_MAIN() in GPU test binaries. Calls parseGpuFlags()
 * to enable --gpu-memory, --gpu-device, and other GPU-specific CLI options.
 *
 * Usage in GPU test files:
 *   PERF_GPU_MAIN()  // Replaces entire main() function
 */
#define PERF_GPU_MAIN()                                                                            \
  int main(int argc, char** argv) {                                                                \
    auto& cfg = vernier::bench::detail::perfConfigSingleton();                                     \
    vernier::bench::parsePerfFlags(cfg, &argc, argv);                                              \
    vernier::bench::PerfGpuConfig gpuCfg;                                                          \
    vernier::bench::parseGpuFlags(gpuCfg, &argc, argv);                                            \
    vernier::bench::detail::setGlobalGpuConfig(gpuCfg);                                            \
    vernier::bench::setGlobalPerfConfig(&cfg);                                                     \
    vernier::bench::installPerfEventListener(cfg);                                                 \
    ::testing::InitGoogleTest(&argc, argv);                                                        \
    return RUN_ALL_TESTS();                                                                        \
  }

/* -------------------------------- Detail -------------------------------- */

namespace vernier {
namespace bench {
namespace detail {

// Global GPU config storage (similar to perfConfigSingleton)
inline PerfGpuConfig& gpuConfigSingleton() {
  static PerfGpuConfig cfg{};
  return cfg;
}

inline void setGlobalGpuConfig(const PerfGpuConfig& cfg) { gpuConfigSingleton() = cfg; }

inline const PerfGpuConfig& getGlobalGpuConfig() { return gpuConfigSingleton(); }

} // namespace detail

} // namespace bench
} // namespace vernier

#endif // VERNIER_PERFGPUTESTMACROS_HPP