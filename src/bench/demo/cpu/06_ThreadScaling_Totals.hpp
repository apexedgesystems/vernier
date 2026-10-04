#ifndef VERNIER_DEMO_06_THREAD_SCALING_TOTALS_HPP
#define VERNIER_DEMO_06_THREAD_SCALING_TOTALS_HPP
/**
 * @file 06_ThreadScaling_Totals.hpp
 * @brief Two ways for threads to add a joined length to one total: under one
 *        lock held for the whole call, and with nothing shared while they run.
 *
 * One definition for demo 06 and for the test that holds both versions to the
 * same total.
 */

#include "src/bench/demo/examples/join/inc/Join.hpp"

#include <atomic>
#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

namespace vernier {
namespace bench {
namespace demo {
namespace thread_scaling_demo {

/* ----------------------------- Constants ----------------------------- */

/// Parts per join, as in demo 01, so the walkthroughs measure the same call.
inline constexpr std::size_t PART_COUNT = 1000;

/// Fixed seed: every run joins the same words.
inline constexpr unsigned PART_SEED = 42;

inline constexpr char SEPARATOR = ',';

/* ----------------------------- Version 0: Coarse Lock ----------------------------- */

/// One total for every thread, and the lock that guards it.
struct SharedTotal {
  std::mutex lock;
  std::size_t value = 0;
};

/**
 * @brief Join @p parts and add the length to @p total, holding its lock for
 *        the whole call: the join runs under the lock, so threads calling this
 *        take turns.
 * @note NOT RT-safe: allocates, and waits while another thread holds the lock.
 */
inline void addUnderCoarseLock(SharedTotal& total, const std::vector<std::string>& parts) {
  std::lock_guard<std::mutex> guard(total.lock);
  total.value += joinV1(parts, SEPARATOR).size();
}

/* ----------------------------- Version 1: No Sharing ----------------------------- */

/// What the threads that have ended added. Each thread adds to it once.
inline std::atomic<std::size_t> finishedThreadsTotal{0};

/// One thread's own total. Only its thread adds to it, so it needs no lock;
/// the destructor runs when the thread ends and hands the total over.
struct ThreadTotal {
  std::size_t value = 0;

  ThreadTotal() = default;
  ThreadTotal(const ThreadTotal&) = delete;
  ThreadTotal& operator=(const ThreadTotal&) = delete;
  ~ThreadTotal() { finishedThreadsTotal += value; }
};

/// The calling thread's total.
inline thread_local ThreadTotal threadTotal;

/**
 * @brief Join @p parts and add the length to the calling thread's own total.
 *        Nothing is shared until the thread ends.
 * @note NOT RT-safe: allocates.
 */
inline void addToThreadTotal(const std::vector<std::string>& parts) {
  threadTotal.value += joinV1(parts, SEPARATOR).size();
}

} // namespace thread_scaling_demo
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_06_THREAD_SCALING_TOTALS_HPP
