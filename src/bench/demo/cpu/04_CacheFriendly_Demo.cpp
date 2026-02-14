/**
 * @file 04_CacheFriendly_Demo.cpp
 * @brief Demo 04: AoS vs SoA data layout transformation
 *
 * Demonstrates the performance impact of data layout on cache utilization.
 * Array-of-Structs (AoS) wastes cache lines loading unused fields, while
 * Struct-of-Arrays (SoA) accesses only the data it needs.
 *
 * Slow: ParticleAoS (128 bytes/particle, only 24 bytes of x/y/z used per access)
 * Fast: ParticleSoA (sequential doubles, 100% of cache line is useful data)
 *
 * Usage:
 *   @code{.sh}
 *   # Run both versions
 *   ./BenchDemo_04_CacheFriendly --csv results.csv
 *
 *   # Summary shows bandwidth difference
 *   bench summary results.csv
 *   @endcode
 *
 * @see docs/04_CACHE_FRIENDLY.md for step-by-step walkthrough
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "src/bench/inc/Perf.hpp"
#include "helpers/DemoWorkloads.hpp"

namespace ub = vernier::bench;
namespace demo = vernier::bench::demo;

/* ----------------------------- Constants ----------------------------- */

// 50K particles: AoS = 6.4 MB, SoA x/y/z = 1.2 MB
static constexpr std::size_t PARTICLE_COUNT = 50000;

/* ----------------------------- Tests ----------------------------- */

/**
 * @test Slow: AoS layout wastes cache lines.
 *
 * Each ParticleAoS is 128 bytes (2 cache lines). When summing x+y+z,
 * only 24 bytes of each 128-byte struct are used -- 81% cache waste.
 * The hardware prefetcher works but loads mostly useless data.
 *
 * MemoryProfile shows low effective bandwidth utilization.
 */
PERF_THROUGHPUT(CacheFriendly, ArrayOfStructs) {
  UB_PERF_GUARD(perf);

  auto particles = demo::makeParticlesAoS(PARTICLE_COUNT);

  perf.warmup([&] {
    volatile double sum = demo::sumPositionsAoS(particles);
    (void)sum;
  });

  // Report actual bytes read: entire struct is loaded into cache per particle
  ub::MemoryProfile memProfile{.bytesRead = PARTICLE_COUNT * sizeof(demo::ParticleAoS),
                               .bytesWritten = 0,
                               .bytesAllocated = 0};

  volatile double sink = 0.0;
  auto result = perf.throughputLoop([&] { sink = sink + demo::sumPositionsAoS(particles); },
                                    "aos_sum_positions", memProfile);

  EXPECT_GT(result.callsPerSecond, 10.0);

  (void)sink;
}

/**
 * @test Fast: SoA layout maximizes cache utilization.
 *
 * ParticleSoA stores x[], y[], z[] as separate contiguous arrays.
 * When summing positions, every byte loaded into the cache is useful data.
 * The hardware prefetcher sees clean sequential access patterns.
 *
 * Expected improvement: 3-5x due to cache utilization.
 * MemoryProfile shows much higher effective bandwidth.
 */
PERF_THROUGHPUT(CacheFriendly, StructOfArrays) {
  UB_PERF_GUARD(perf);

  auto particles = demo::makeParticlesSoA(PARTICLE_COUNT);

  perf.warmup([&] {
    volatile double sum = demo::sumPositionsSoA(particles, PARTICLE_COUNT);
    (void)sum;
  });

  // Only x, y, z arrays are read: 3 * count * sizeof(double)
  ub::MemoryProfile memProfile{
      .bytesRead = PARTICLE_COUNT * 3 * sizeof(double), .bytesWritten = 0, .bytesAllocated = 0};

  volatile double sink = 0.0;
  auto result =
      perf.throughputLoop([&] { sink = sink + demo::sumPositionsSoA(particles, PARTICLE_COUNT); },
                          "soa_sum_positions", memProfile);

  EXPECT_GT(result.callsPerSecond, 10.0);

  (void)sink;
}

PERF_MAIN()
