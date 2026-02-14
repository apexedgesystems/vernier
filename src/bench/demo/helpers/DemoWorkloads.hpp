/**
 * @file DemoWorkloads.hpp
 * @brief Shared slow/fast workload implementations for benchmarking demos
 *
 * Each workload pair provides an intentionally inefficient version (for
 * baseline measurement and profiling) and an optimized version (to
 * demonstrate measurable improvement).
 *
 * Workloads are designed to be:
 *  - Realistic enough to demonstrate real optimization patterns
 *  - Simple enough to understand quickly
 *  - Deterministic (same input = same output)
 *  - Compiler-resistant (optimizations not eliminated by -O2)
 */

#ifndef VERNIER_DEMO_WORKLOADS_HPP
#define VERNIER_DEMO_WORKLOADS_HPP

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <numeric>
#include <random>
#include <vector>

namespace vernier {
namespace bench {
namespace demo {

/* ----------------------------- Data Generators ----------------------------- */

/** @brief Generate deterministic random doubles in [0, 1). */
inline std::vector<double> makeRandomDoubles(std::size_t count, std::uint32_t seed = 12345) {
  std::vector<double> v(count);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (auto& x : v) {
    x = dist(rng);
  }
  return v;
}

/** @brief Generate deterministic random int32 in [-1000, 1000]. */
inline std::vector<std::int32_t> makeRandomInts(std::size_t count, std::uint32_t seed = 42) {
  std::vector<std::int32_t> v(count);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<std::int32_t> dist(-1000, 1000);
  for (auto& x : v) {
    x = dist(rng);
  }
  return v;
}

/** @brief Generate a sorted copy of input data. */
inline std::vector<double> makeSorted(std::vector<double> data) {
  std::sort(data.begin(), data.end());
  return data;
}

/* ----------------------------- AoS vs SoA ----------------------------- */

/** @brief Array-of-Structs particle (cache-unfriendly: 128 bytes per particle). */
struct ParticleAoS {
  double x, y, z;
  double vx, vy, vz;
  double mass;
  double padding[9]; // Inflate to 128 bytes (2 cache lines)
};

static_assert(sizeof(ParticleAoS) == 128, "ParticleAoS must be 128 bytes");

inline std::vector<ParticleAoS> makeParticlesAoS(std::size_t count, std::uint32_t seed = 42) {
  std::vector<ParticleAoS> v(count);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (auto& p : v) {
    p.x = dist(rng);
    p.y = dist(rng);
    p.z = dist(rng);
    p.vx = dist(rng);
    p.vy = dist(rng);
    p.vz = dist(rng);
    p.mass = 1.0;
    std::memset(p.padding, 0, sizeof(p.padding));
  }
  return v;
}

/** @brief Struct-of-Arrays particle data (cache-friendly: sequential access). */
struct ParticleSoA {
  std::vector<double> x, y, z;
  std::vector<double> vx, vy, vz;
  std::vector<double> mass;
};

inline ParticleSoA makeParticlesSoA(std::size_t count, std::uint32_t seed = 42) {
  ParticleSoA soa;
  soa.x.resize(count);
  soa.y.resize(count);
  soa.z.resize(count);
  soa.vx.resize(count);
  soa.vy.resize(count);
  soa.vz.resize(count);
  soa.mass.resize(count, 1.0);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (std::size_t i = 0; i < count; ++i) {
    soa.x[i] = dist(rng);
    soa.y[i] = dist(rng);
    soa.z[i] = dist(rng);
    soa.vx[i] = dist(rng);
    soa.vy[i] = dist(rng);
    soa.vz[i] = dist(rng);
  }
  return soa;
}

/* ----------------------------- Cache Workloads ----------------------------- */

/** @brief Slow: Sum positions from AoS layout (128B stride, poor spatial locality). */
inline double sumPositionsAoS(const std::vector<ParticleAoS>& particles) {
  double sum = 0.0;
  for (const auto& p : particles) {
    sum += p.x + p.y + p.z;
  }
  return sum;
}

/** @brief Fast: Sum positions from SoA layout (sequential doubles, excellent locality). */
inline double sumPositionsSoA(const ParticleSoA& particles, std::size_t count) {
  double sum = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    sum += particles.x[i] + particles.y[i] + particles.z[i];
  }
  return sum;
}

/** @brief Slow: Stride-512 array walk (constant cache misses). */
inline std::uint64_t stridedArrayWalk(const std::uint8_t* data, std::size_t len,
                                      std::size_t stride) {
  std::uint64_t sum = 0;
  for (std::size_t i = 0; i < len; i += stride) {
    sum += data[i];
  }
  return sum;
}

/** @brief Fast: Sequential array walk (hardware prefetching). */
inline std::uint64_t sequentialArrayWalk(const std::uint8_t* data, std::size_t len) {
  std::uint64_t sum = 0;
  for (std::size_t i = 0; i < len; ++i) {
    sum += data[i];
  }
  return sum;
}

/* ----------------------------- Branch Workloads ----------------------------- */

/** @brief Slow: Conditional sum with unpredictable branches (50% miss rate). */
inline std::int64_t conditionalSumBranchy(const double* data, std::size_t len, double threshold) {
  std::int64_t sum = 0;
  for (std::size_t i = 0; i < len; ++i) {
    if (data[i] > threshold) {
      sum += static_cast<std::int64_t>(data[i] * 1000.0);
    }
  }
  return sum;
}

/** @brief Fast: Branchless conditional sum using multiply-by-predicate. */
inline std::int64_t conditionalSumBranchless(const double* data, std::size_t len,
                                             double threshold) {
  std::int64_t sum = 0;
  for (std::size_t i = 0; i < len; ++i) {
    const auto val = static_cast<std::int64_t>(data[i] * 1000.0);
    sum += val * static_cast<std::int64_t>(data[i] > threshold);
  }
  return sum;
}

/* ----------------------------- Sort Workloads ----------------------------- */

/** @brief Slow: Bubble sort O(n^2). */
inline void bubbleSort(double* data, std::size_t len) {
  for (std::size_t i = 0; i < len; ++i) {
    for (std::size_t j = 0; j + 1 < len - i; ++j) {
      if (data[j] > data[j + 1]) {
        const double tmp = data[j];
        data[j] = data[j + 1];
        data[j + 1] = tmp;
      }
    }
  }
}

/** @brief Fast: std::sort O(n log n). */
inline void fastSort(double* data, std::size_t len) { std::sort(data, data + len); }

/* ----------------------------- Search Workloads ----------------------------- */

/** @brief Slow: Linear search O(n). */
inline std::size_t linearSearch(const double* sortedData, std::size_t len, double target) {
  for (std::size_t i = 0; i < len; ++i) {
    if (sortedData[i] >= target) {
      return i;
    }
  }
  return len;
}

/** @brief Fast: Binary search O(log n). */
inline std::size_t binarySearch(const double* sortedData, std::size_t len, double target) {
  const double* it = std::lower_bound(sortedData, sortedData + len, target);
  return static_cast<std::size_t>(it - sortedData);
}

/* ----------------------------- Contention Workloads ----------------------------- */

/** @brief Slow: Mutex-protected counter increment. */
inline void incrementMutex(std::mutex& mtx, std::uint64_t& counter, int iterations) {
  for (int i = 0; i < iterations; ++i) {
    std::lock_guard<std::mutex> lock(mtx);
    ++counter;
  }
}

/** @brief Fast: Atomic counter increment (lock-free). */
inline void incrementAtomic(std::atomic<std::uint64_t>& counter, int iterations) {
  for (int i = 0; i < iterations; ++i) {
    counter.fetch_add(1, std::memory_order_relaxed);
  }
}

/* ----------------------------- Dot Product Workloads ----------------------------- */

/** @brief Slow: Naive element-by-element dot product (not vectorizable by some compilers). */
inline double naiveDotProduct(const double* a, const double* b, std::size_t len) {
  double sum = 0.0;
  for (std::size_t i = 0; i < len; ++i) {
    const double product = a[i] * b[i];
    sum = sum + product;
    // Intentionally prevent auto-vectorization by introducing a dependency
    if (sum > 1e18) {
      sum *= 1.0; // Compiler barrier
    }
  }
  return sum;
}

/** @brief Fast: std::inner_product (compiler can auto-vectorize). */
inline double fastDotProduct(const double* a, const double* b, std::size_t len) {
  return std::inner_product(a, a + len, b, 0.0);
}

/* ----------------------------- I/O Workloads ----------------------------- */

/** @brief Slow: Many small writes (one syscall per byte). */
inline void manySmallWrites(int fd, const std::uint8_t* data, std::size_t len) {
  for (std::size_t i = 0; i < len; ++i) {
    [[maybe_unused]] auto r = ::write(fd, &data[i], 1);
  }
}

/** @brief Fast: Single batched write. */
inline void singleBatchedWrite(int fd, const std::uint8_t* data, std::size_t len) {
  [[maybe_unused]] auto r = ::write(fd, data, len);
}

} // namespace demo

} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_WORKLOADS_HPP
