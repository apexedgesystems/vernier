#ifndef VERNIER_PERFUTILS_HPP
#define VERNIER_PERFUTILS_HPP
/**
 * @file PerfUtils.hpp
 * @brief Internal utility functions for benchmarking framework.
 *
 * Consolidates small utility headers:
 * - Clock utilities (steady clock for wall-time measurements)
 * - File utilities (temp files, slurp, line counting)
 * - Synchronization primitives (start gate for multi-threaded tests)
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#endif

namespace vernier {
namespace bench {

/* ----------------------------- Spin Hints ----------------------------- */

/**
 * @brief Hint the CPU to relax/yield in a spin loop.
 *
 * Uses architecture-specific instructions:
 *  - x86/x64: PAUSE instruction
 *  - ARM: YIELD instruction
 *  - Others: No-op (safe fallback)
 *
 * @note RT-safe (single CPU instruction).
 */
inline void cpuRelax() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(__GNUC__) || defined(__clang__)
  __builtin_ia32_pause();
#elif defined(_MSC_VER)
  _mm_pause();
#else
  (void)0;
#endif
#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || defined(_M_ARM64)
#if defined(__GNUC__) || defined(__clang__)
  __asm__ __volatile__("yield" ::: "memory");
#else
  (void)0;
#endif
#else
  (void)0;
#endif
}

/* ---------------------------- Clock Utilities ---------------------------- */

/**
 * @brief Current time in microseconds from a monotonic clock.
 *
 * Uses steady_clock to ensure measurements aren't affected by system
 * clock adjustments (NTP, daylight savings, etc.).
 *
 * @return Microseconds since epoch (arbitrary reference point)
 * @note NOT RT-safe (system call).
 */
inline double nowUs() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::microseconds>(clock::now().time_since_epoch())
      .count();
}

/* ----------------------------- File Utilities ----------------------------- */

/**
 * @brief Create a unique temp file path with a given stem.
 *
 * Generates a random filename in the system temp directory to avoid
 * collisions across concurrent test runs.
 *
 * @param stem Prefix for the filename (e.g., "benchmark")
 * @return Unique path like /tmp/benchmark_12345678901234567890.log
 * @note NOT RT-safe (filesystem access, heap allocation).
 */
inline std::filesystem::path uniqTempFile(const std::string& stem) {
  const auto DIR = std::filesystem::temp_directory_path();
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<unsigned long long> dist;
  return DIR / (stem + "_" + std::to_string(dist(gen)) + ".log");
}

/**
 * @brief Read whole file into memory (binary mode).
 *
 * @param path File to read
 * @return File contents as string
 * @note NOT RT-safe (file I/O, heap allocation).
 */
inline std::string slurp(const std::filesystem::path& path) {
  std::ifstream ifs(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

/**
 * @brief Count newlines in a string.
 *
 * Useful for validating log output or parsing line-oriented data.
 *
 * @param s String to count newlines in
 * @return Number of '\n' characters
 * @note RT-safe (no allocations, bounded iteration).
 */
inline int countLines(const std::string& s) {
  return static_cast<int>(std::count(s.begin(), s.end(), '\n'));
}

/* ------------------------ Synchronization Primitives ------------------------ */

/**
 * @brief Start-line gate for multi-threaded measurements.
 *
 * Ensures all threads begin work simultaneously to measure contention
 * accurately. Without this, early threads would complete before late
 * threads even start, skewing results.
 *
 * Usage:
 * @code{.cpp}
 * StartGate gate(numThreads);
 *
 * // In each thread:
 * gate.start();  // Blocks until all threads ready
 * // ... do work ...
 *
 * // Main thread:
 * gate.releaseWhenAllReady();  // Unblocks all threads
 * @endcode
 *
 * This is a simple portable C++17 implementation using atomics and
 * spin-waiting. For production code with many threads, consider
 * condition variables, but for benchmarking (typically <=16 threads),
 * this is sufficient and has minimal overhead.
 *
 * @note RT-safe (lock-free atomics, spin-wait only).
 */
class StartGate {
public:
  /**
   * @brief Construct gate for specified number of threads.
   * @param total Number of threads that will call start()
   */
  explicit StartGate(int total) noexcept : total_(total) {}

  /**
   * @brief Worker thread calls this to wait at the start line.
   *
   * Increments ready counter and spins until go flag is set.
   * Last thread to call start() triggers releaseWhenAllReady().
   */
  void start() noexcept {
    ready_.fetch_add(1, std::memory_order_acq_rel);
    while (!go_.load(std::memory_order_acquire)) {
      cpuRelax();
    }
  }

  /**
   * @brief Main thread calls this to release all waiting workers.
   *
   * Spins until all threads have called start(), then sets go flag.
   */
  void releaseWhenAllReady() noexcept {
    while (ready_.load(std::memory_order_acquire) < total_) {
      cpuRelax();
    }
    go_.store(true, std::memory_order_release);
  }

private:
  int total_;
  std::atomic<int> ready_{0};
  std::atomic<bool> go_{false};
};

} // namespace bench
} // namespace vernier

#endif // VERNIER_PERFUTILS_HPP