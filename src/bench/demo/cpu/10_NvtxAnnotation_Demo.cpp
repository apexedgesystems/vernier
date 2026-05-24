/**
 * @file 10_NvtxAnnotation_Demo.cpp
 * @brief Demo 10: NVTX scope annotation for Nsight Systems timelines.
 *
 * BENCH_NVTX_SCOPE pushes a named NVTX range for its enclosing scope; nsys
 * picks the range up automatically when profiling. Annotating a benchmark
 * turns an unlabeled wall of activity into a timeline with named phases.
 *
 * The macro compiles to a no-op when NVTX headers are not present, so the
 * same code builds on CPU-only targets without conditional includes.
 *
 * Usage:
 *   @code{.sh}
 *   # Inside the CUDA dev container, annotated ranges show up in nsys:
 *   nsys profile -o nvtx_demo -t cuda,nvtx --force-overwrite=true \
 *     ./build/native-linux-debug/bin/ptests/BenchDemo_10_NvtxAnnotation \
 *     --gtest_filter='Nvtx.PhasedWorkload'
 *
 *   nsys stats --report nvtx_pushpop_sum nvtx_demo.nsys-rep
 *   # -> per-phase totals: gen, transform, accumulate, finalize
 *   @endcode
 *
 * On a CPU-only build, the macros are no-ops; the test still runs but no
 * ranges are emitted. NsightProfiler also auto-injects a top-level range
 * named after the test, so the timeline groups by test even without
 * explicit annotations.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <vector>

#include "src/bench/inc/Nvtx.hpp"
#include "src/bench/inc/Perf.hpp"
#include "helpers/DemoWorkloads.hpp"

namespace ub = vernier::bench;
namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

static constexpr std::size_t WORK_SIZE = 50000;

/* ----------------------------- Tests ----------------------------- */

/**
 * @test Multi-phase workload with NVTX scopes per stage.
 *
 * Demonstrates the canonical pattern: wrap each conceptual phase in its own
 * BENCH_NVTX_SCOPE so the nsys timeline reads like a flame graph of phases
 * rather than a single opaque hotspot.
 */
PERF_THROUGHPUT(Nvtx, PhasedWorkload) {
  UB_PERF_GUARD(perf);

  std::vector<double> data;

  perf.warmup([&] {
    BENCH_NVTX_SCOPE("warmup");
    data = demo::makeRandomDoubles(WORK_SIZE);
  });

  volatile double sink = 0.0;
  auto result = perf.throughputLoop(
      [&] {
        // Each phase appears as a separately named range in nsys.
        {
          BENCH_NVTX_SCOPE("phase_gen");
          for (std::size_t i = 0; i < data.size(); ++i) {
            data[i] = data[i] * 1.0001 + 1.0e-6;
          }
        }
        {
          BENCH_NVTX_SCOPE("phase_transform");
          for (auto& v : data) {
            v = (v * v) - v + 1.0;
          }
        }
        double sum;
        {
          BENCH_NVTX_SCOPE("phase_accumulate");
          sum = std::accumulate(data.begin(), data.end(), 0.0);
        }
        {
          BENCH_NVTX_SCOPE("phase_finalize");
          sink = sink + sum;
        }
      },
      "phased_workload");

  // Marker for downstream tooling that wants to find this point in the timeline.
  BENCH_NVTX_MARK("phased_workload.completed");

  EXPECT_GT(result.callsPerSecond, 10.0);
  (void)sink;
}

PERF_MAIN()
