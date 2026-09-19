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
 * @brief Make a buffer and the writes to it observable
 *
 * Publishes the pointer through a volatile object and reads one byte back
 * through the reloaded pointer. The compiler cannot tell what the reloaded
 * pointer refers to, so the buffer has to exist and hold its contents at
 * this point. A volatile copy of buf[0] alone is not enough: the value is
 * known at compile time and the allocation and the fill are removed.
 *
 * @param data Start of the buffer (at least one byte)
 */
inline void observeBuffer(const std::uint8_t* data) {
  const std::uint8_t* volatile published = data;
  volatile std::uint8_t val = *published;
  (void)val;
}

/**
 * @brief Allocate, fill and free a buffer (one heap allocation per call)
 *
 * @param size Buffer size in bytes (at least one)
 */
inline void allocateAndFill(std::size_t size) {
  std::vector<std::uint8_t> buf(size);
  std::fill(buf.begin(), buf.end(), std::uint8_t{0xFF});
  observeBuffer(buf.data());
}

/**
 * @brief Fill a caller-owned buffer (no allocation once it has grown to size)
 *
 * @param buf Reusable buffer
 * @param size Size to fill (at least one)
 */
inline void reuseAndFill(std::vector<std::uint8_t>& buf, std::size_t size) {
  buf.resize(size);
  std::fill(buf.begin(), buf.end(), std::uint8_t{0xFF});
  observeBuffer(buf.data());
}

} // namespace test

} // namespace bench
} // namespace vernier

#endif // VERNIER_TEST_HELPERS_HPP