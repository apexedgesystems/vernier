#ifndef VERNIER_MONITORQUEUE_HPP
#define VERNIER_MONITORQUEUE_HPP
/**
 * @file MonitorQueue.hpp
 * @brief Lock-free MPMC ring buffer for Sample records.
 *
 * Design goals:
 *  - Vyukov's bounded MPMC queue with per-cell sequence counters
 *  - Hot path (tryPush) is lock-free, allocation-free, RT-safe
 *  - Overflow drops the sample and increments a counter
 */

#include "src/monitor/inc/MonitorTag.hpp"

#include <cstddef>
#include <cstdint>

#include <atomic>
#include <new>
#include <vector>

namespace vernier {
namespace monitor {

/* ----------------------------- MonitorQueue ----------------------------- */

/** @brief Lock-free MPMC ring buffer for Sample records. */
class MonitorQueue {
public:
  /**
   * @brief Construct a queue with the given capacity (will be rounded to power of 2).
   * @param capacity Desired queue capacity.
   * @note NOT RT-safe: Heap allocation during construction.
   */
  explicit MonitorQueue(std::size_t capacity)
      : mask_{roundUp(capacity) - 1}, cells_(roundUp(capacity)), dropped_{0} {
    for (std::size_t i = 0; i < cells_.size(); ++i) {
      cells_[i].seq.store(i, std::memory_order_relaxed);
    }
    enqPos_.store(0, std::memory_order_relaxed);
    deqPos_.store(0, std::memory_order_relaxed);
  }

  /**
   * @brief Try to enqueue a sample.
   * @param sample The sample to enqueue.
   * @return True on success, false if full (dropped).
   * @note RT-safe: Lock-free, no allocation, no formatting.
   */
  bool tryPush(const Sample& sample) noexcept {
    std::size_t pos = enqPos_.load(std::memory_order_relaxed);

    for (;;) {
      Cell& cell = cells_[pos & mask_];
      std::size_t seq = cell.seq.load(std::memory_order_acquire);
      auto diff = static_cast<std::ptrdiff_t>(seq) - static_cast<std::ptrdiff_t>(pos);

      if (diff == 0) {
        // Cell is ready for writing
        if (enqPos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
          cell.data = sample;
          cell.seq.store(pos + 1, std::memory_order_release);
          return true;
        }
      } else if (diff < 0) {
        // Queue is full
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return false;
      } else {
        // Another thread advanced enqPos_; reload
        pos = enqPos_.load(std::memory_order_relaxed);
      }
    }
  }

  /**
   * @brief Try to dequeue a sample.
   * @param out Destination for the dequeued sample.
   * @return True on success, false if empty.
   * @note RT-safe: Lock-free, no allocation.
   */
  bool tryPop(Sample& out) noexcept {
    std::size_t pos = deqPos_.load(std::memory_order_relaxed);

    for (;;) {
      Cell& cell = cells_[pos & mask_];
      std::size_t seq = cell.seq.load(std::memory_order_acquire);
      auto diff = static_cast<std::ptrdiff_t>(seq) - static_cast<std::ptrdiff_t>(pos + 1);

      if (diff == 0) {
        // Cell is ready for reading
        if (deqPos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
          out = cell.data;
          cell.seq.store(pos + mask_ + 1, std::memory_order_release);
          return true;
        }
      } else if (diff < 0) {
        // Queue is empty
        return false;
      } else {
        pos = deqPos_.load(std::memory_order_relaxed);
      }
    }
  }

  /**
   * @brief Number of samples dropped due to queue overflow (monotonic).
   * @return Drop count.
   * @note RT-safe: Atomic load.
   */
  [[nodiscard]] std::uint64_t droppedCount() const noexcept {
    return dropped_.load(std::memory_order_relaxed);
  }

  /**
   * @brief Queue capacity.
   * @return Capacity in number of samples.
   * @note RT-safe: Bounded computation, no allocation.
   */
  [[nodiscard]] std::size_t capacity() const noexcept { return mask_ + 1; }

private:
  struct Cell {
    std::atomic<std::size_t> seq;
    Sample data;
  };

  static std::size_t roundUp(std::size_t v) noexcept {
    if (v < 2)
      v = 2;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
  }

  const std::size_t mask_;
  std::vector<Cell> cells_;

  // Separate cache lines for enqueue and dequeue positions
  alignas(64) std::atomic<std::size_t> enqPos_;
  alignas(64) std::atomic<std::size_t> deqPos_;
  alignas(64) std::atomic<std::uint64_t> dropped_;
};

} // namespace monitor
} // namespace vernier

#endif // VERNIER_MONITORQUEUE_HPP
