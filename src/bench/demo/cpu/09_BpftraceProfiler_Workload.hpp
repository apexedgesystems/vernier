#ifndef VERNIER_DEMO_09_BPFTRACE_WORKLOAD_HPP
#define VERNIER_DEMO_09_BPFTRACE_WORKLOAD_HPP
/**
 * @file 09_BpftraceProfiler_Workload.hpp
 * @brief What demo 09 measures: a text written one write() per line against
 *        the same text in one write(), and threads adding joined lengths to
 *        one total under a lock against totals of their own.
 *
 * One definition for demo 09 and for the program that checks it, so both
 * describe the same calls.
 */

#include "src/bench/demo/examples/join/inc/Join.hpp"

#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

namespace vernier {
namespace bench {
namespace demo {
namespace bpftrace_demo {

/* ----------------------------- Constants ----------------------------- */

/// Words per text and per join, as in demo 01, so the walkthroughs use the
/// same words.
inline constexpr std::size_t PART_COUNT = 1000;

/// Fixed seed: every run uses the same words.
inline constexpr unsigned PART_SEED = 42;

/// What ends each line of the text.
inline constexpr char LINE_END = '\n';

/// What the threaded cases join the words with.
inline constexpr char SEPARATOR = ',';

/* ----------------------------- The Text ----------------------------- */

/**
 * @brief The text's lines: each word followed by LINE_END. Laid end to end
 *        they are joinV1(words, LINE_END).
 * @note NOT RT-safe: allocates.
 */
inline std::vector<std::string> linesOf(const std::vector<std::string>& words) {
  std::vector<std::string> lines;
  lines.reserve(words.size());
  for (const std::string& word : words) {
    lines.push_back(word + LINE_END);
  }
  return lines;
}

/**
 * @brief The lines both write cases write: linesOf() the join example's
 *        PART_COUNT words from PART_SEED. WritePerLine writes them one write()
 *        each, WriteBatched textOf() them with one.
 * @note NOT RT-safe: allocates.
 */
inline std::vector<std::string> demoLines() { return linesOf(makeParts(PART_COUNT, PART_SEED)); }

/**
 * @brief @p lines end to end, the text WriteBatched writes; for demoLines(),
 *        joinV1(words, LINE_END).
 * @note NOT RT-safe: allocates.
 */
inline std::string textOf(const std::vector<std::string>& lines) {
  std::string text;
  for (const std::string& line : lines) {
    text += line;
  }
  return text;
}

/* ----------------------------- Writes: One per Line ----------------------------- */

/**
 * @brief Write @p lines to @p fd with one write() per line, as a logger that
 *        writes every line out as it comes does.
 * @return The bytes written.
 * @note NOT RT-safe: one system call per line.
 */
inline std::size_t writeEachLine(int fd, const std::vector<std::string>& lines) {
  std::size_t written = 0;
  for (const std::string& line : lines) {
    const ssize_t RESULT = ::write(fd, line.data(), line.size());
    written += RESULT > 0 ? static_cast<std::size_t>(RESULT) : 0;
  }
  return written;
}

/* ----------------------------- Writes: One for the Text ----------------------------- */

/**
 * @brief Write @p text, the lines end to end, to @p fd with one write().
 * @return The bytes written.
 * @note NOT RT-safe: a system call.
 */
inline std::size_t writeBatched(int fd, const std::string& text) {
  const ssize_t RESULT = ::write(fd, text.data(), text.size());
  return RESULT > 0 ? static_cast<std::size_t>(RESULT) : 0;
}

/* ----------------------------- Threads: One Lock ----------------------------- */

/// One total for every thread, and the lock that guards it.
struct SharedTotal {
  std::mutex lock;
  std::size_t value = 0;
};

/**
 * @brief Join @p words and add the length to @p total, holding its lock for
 *        the whole call. The joins take turns: a thread that finds the lock
 *        held sleeps until the thread that holds it wakes it.
 * @note NOT RT-safe: allocates, and waits while another thread holds the lock.
 */
inline void addUnderCoarseLock(SharedTotal& total, const std::vector<std::string>& words) {
  std::lock_guard<std::mutex> guard(total.lock);
  total.value += joinV1(words, SEPARATOR).size();
}

/* ----------------------------- Threads: Nothing Shared ----------------------------- */

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
 * @brief Join @p words and add the length to the calling thread's own total.
 *        Nothing is shared until the thread ends.
 * @note NOT RT-safe: allocates.
 */
inline void addToThreadTotal(const std::vector<std::string>& words) {
  threadTotal.value += joinV1(words, SEPARATOR).size();
}

} // namespace bpftrace_demo
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_09_BPFTRACE_WORKLOAD_HPP
