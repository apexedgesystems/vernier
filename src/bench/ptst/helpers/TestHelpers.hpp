/**
 * @file TestHelpers.hpp
 * @brief Shared utilities for benchmark framework tests
 *
 * Provides common test utilities, data generators, and validation helpers
 * used across the benchmark test suite.
 */

#ifndef VERNIER_TEST_HELPERS_HPP
#define VERNIER_TEST_HELPERS_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <random>
#include <algorithm>

namespace vernier {
namespace bench {
namespace test {

/**
 * @brief Generate deterministic test data with specified pattern
 *
 * @param size Number of bytes to generate
 * @param seed Random seed for reproducibility
 * @return Vector of bytes with deterministic pattern
 */
inline std::vector<std::uint8_t> makeTestData(std::size_t size, std::uint32_t seed = 42) {
  std::vector<std::uint8_t> data(size);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 255);

  for (std::size_t i = 0; i < size; ++i) {
    data[i] = static_cast<std::uint8_t>(dist(rng));
  }

  return data;
}

/**
 * @brief Generate signed integer test data
 *
 * @param count Number of integers to generate
 * @param seed Random seed for reproducibility
 * @return Vector of signed integers with mix of positive and negative values
 */
inline std::vector<std::int32_t> makeSignedData(std::size_t count, std::uint32_t seed = 42) {
  std::vector<std::int32_t> data(count);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<std::int32_t> dist(-1000, 1000);

  for (std::size_t i = 0; i < count; ++i) {
    data[i] = dist(rng);
  }

  return data;
}

/**
 * @brief Simple sum workload for testing
 *
 * @param data Pointer to byte array
 * @param len Length of array
 * @return Sum of all bytes
 */
inline std::uint64_t sumBytes(const std::uint8_t* data, std::size_t len) {
  std::uint64_t sum = 0;
  for (std::size_t i = 0; i < len; ++i) {
    sum += data[i];
  }
  return sum;
}

/**
 * @brief Strided sum workload for cache testing
 *
 * @param data Pointer to byte array
 * @param len Length of array
 * @param stride Access stride in bytes
 * @return Sum of accessed bytes
 */
inline std::uint64_t sumBytesStrided(const std::uint8_t* data, std::size_t len,
                                     std::size_t stride) {
  std::uint64_t sum = 0;
  for (std::size_t i = 0; i < len; i += stride) {
    sum += data[i];
  }
  return sum;
}

/**
 * @brief Count positive values with branches (branch-heavy)
 *
 * @param data Pointer to signed integer array
 * @param len Length of array
 * @return Count of positive values
 */
inline std::size_t countPositiveBranchy(const std::int32_t* data, std::size_t len) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < len; ++i) {
    if (data[i] > 0) {
      count++;
    }
  }
  return count;
}

/**
 * @brief Count positive values without branches (branchless)
 *
 * @param data Pointer to signed integer array
 * @param len Length of array
 * @return Count of positive values
 */
inline std::size_t countPositiveBranchless(const std::int32_t* data, std::size_t len) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < len; ++i) {
    count += (data[i] > 0);
  }
  return count;
}

/**
 * @brief Allocate and fill buffer (simulates allocation overhead)
 *
 * @param size Buffer size in bytes
 */
inline void allocateAndFill(std::size_t size) {
  std::vector<std::uint8_t> buf(size);
  std::fill(buf.begin(), buf.end(), std::uint8_t{0xFF});
  volatile auto val = buf[0];
  (void)val;
}

/**
 * @brief Reuse buffer (simulates zero allocation overhead)
 *
 * @param buf Reusable buffer
 * @param size Size to fill
 */
inline void reuseAndFill(std::vector<std::uint8_t>& buf, std::size_t size) {
  buf.resize(size);
  std::fill(buf.begin(), buf.end(), std::uint8_t{0xFF});
  volatile auto val = buf[0];
  (void)val;
}

} // namespace test

} // namespace bench
} // namespace vernier

#endif // VERNIER_TEST_HELPERS_HPP