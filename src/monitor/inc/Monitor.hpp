#ifndef VERNIER_MONITOR_HPP
#define VERNIER_MONITOR_HPP
/**
 * @file Monitor.hpp
 * @brief Main runtime performance monitor API.
 *
 * Usage:
 *   vernier::monitor::Monitor mon(config);
 *   mon.start();
 *   {
 *       VERNIER_MONITOR_SCOPE(mon, "stage", tag);
 *       doWork();
 *   }
 *   mon.stop(); // prints summary
 */

#include "src/monitor/inc/MonitorConfig.hpp"
#include "src/monitor/inc/MonitorConsoleSink.hpp"
#include "src/monitor/inc/MonitorFileSink.hpp"
#include "src/monitor/inc/MonitorQueue.hpp"
#include "src/monitor/inc/MonitorScopeGuard.hpp"
#include "src/monitor/inc/MonitorSummary.hpp"
#include "src/monitor/inc/MonitorTag.hpp"

#include <cstring>

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace vernier {
namespace monitor {

/* ----------------------------- Monitor ----------------------------- */

/** @brief Main runtime performance monitor with async I/O backend. */
class Monitor {
public:
  /**
   * @brief Construct a monitor with the given configuration.
   * @param cfg Monitor configuration.
   * @note NOT RT-safe: Heap allocation for queue and internal structures.
   */
  explicit Monitor(const MonitorConfig& cfg = {})
      : cfg_{cfg}, queue_{roundUpPow2(cfg.queueCapacity)}, enabled_{cfg.enabled}, running_{false},
        totalSamples_{0}, startTimeNs_{0} {}

  ~Monitor() { stop(); }

  Monitor(const Monitor&) = delete;
  Monitor& operator=(const Monitor&) = delete;

  /* ----------------------------- Lifecycle ----------------------------- */

  /**
   * @brief Start the async I/O backend. Idempotent.
   * @note NOT RT-safe: Thread creation, heap allocation.
   */
  void start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true))
      return;

    startTimeNs_ = nowNs();

    // Freeze thresholds into immutable snapshot for RT-safe hot-path reads
    if (!pendingThresholds_.empty()) {
      auto table = std::make_unique<std::vector<ThresholdEntry>>();
      table->reserve(pendingThresholds_.size());
      for (const auto& [KEY, US] : pendingThresholds_) {
        ThresholdEntry entry;
        // key format: "scopeName/tagId"
        const auto SEP = KEY.rfind('/');
        if (SEP != std::string::npos) {
          std::strncpy(entry.scope, KEY.substr(0, SEP).c_str(), sizeof(entry.scope) - 1);
          entry.tagId = static_cast<std::uint16_t>(std::stoul(KEY.substr(SEP + 1)));
        }
        entry.thresholdUs = US;
        table->push_back(entry);
      }
      frozenThresholds_ = std::move(table);
    }

    // Create sinks
    if (cfg_.sinks & SINK_CONSOLE) {
      sinks_.push_back(std::make_unique<MonitorConsoleSink>(cfg_.consoleLevel));
    }
    if ((cfg_.sinks & SINK_FILE) && !cfg_.filePath.empty()) {
      auto fs = std::make_unique<MonitorFileSink>(cfg_.filePath, cfg_.fileLevel);
      if (fs->isOpen()) {
        sinks_.push_back(std::move(fs));
      }
    }

    // Start I/O drain thread
    ioThread_ = std::thread([this] { drainLoop(); });
  }

  /**
   * @brief Stop the I/O backend, flush remaining samples, and print summary.
   * @note NOT RT-safe: Thread join, I/O operations.
   */
  void stop() {
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false))
      return;

    // Wake the I/O thread and wait for it to drain
    if (ioThread_.joinable()) {
      ioThread_.join();
    }

    // Print summary
    const std::uint64_t WALL_NS = nowNs() - startTimeNs_;
    summary_.print(WALL_NS, totalSamples_.load(), queue_.droppedCount());

    sinks_.clear();
  }

  /**
   * @brief True if the monitor is actively collecting samples.
   * @return Running status.
   * @note RT-safe: Atomic load.
   */
  [[nodiscard]] bool isRunning() const noexcept { return running_.load(std::memory_order_relaxed); }

  /* ----------------------------- Instrumentation ----------------------------- */

  /**
   * @brief Record a completed scope measurement.
   * @param scopeName Name of the scope being measured.
   * @param tag Monitor tag identifying the source.
   * @param startNs Start timestamp in nanoseconds.
   * @param endNs End timestamp in nanoseconds.
   * @note RT-safe: Lock-free enqueue, no allocation.
   */
  void recordScope(const char* scopeName, const MonitorTag& tag, std::uint64_t startNs,
                   std::uint64_t endNs) noexcept {
    if (!enabled_.load(std::memory_order_relaxed))
      return;

    Sample s;
    s.timestampNs = startNs;
    s.durationNs = endNs - startNs;
    s.tag = tag;
    s.kind = SampleKind::SCOPE;
    s.value = 0.0;
    if (scopeName) {
      std::strncpy(s.scope, scopeName, sizeof(s.scope) - 1);
    }

    // Check threshold
    const auto THRESHOLD_US = getThreshold(scopeName, tag.id);
    if (THRESHOLD_US > 0) {
      const double DURATION_US = static_cast<double>(s.durationNs) / 1e3;
      if (DURATION_US > static_cast<double>(THRESHOLD_US)) {
        s.kind = SampleKind::THRESHOLD_BREACH;
        s.value = static_cast<double>(THRESHOLD_US);
      }
    }

    totalSamples_.fetch_add(1, std::memory_order_relaxed);
    queue_.tryPush(s);
  }

  /**
   * @brief Record a counter increment.
   * @param scopeName Name of the counter scope.
   * @param tag Monitor tag identifying the source.
   * @param delta Increment value (default 1.0).
   * @note RT-safe: Lock-free enqueue, no allocation.
   */
  void increment(const char* scopeName, const MonitorTag& tag, double delta = 1.0) noexcept {
    if (!enabled_.load(std::memory_order_relaxed))
      return;

    Sample s;
    s.timestampNs = nowNs();
    s.kind = SampleKind::COUNTER;
    s.tag = tag;
    s.value = delta;
    if (scopeName) {
      std::strncpy(s.scope, scopeName, sizeof(s.scope) - 1);
    }

    totalSamples_.fetch_add(1, std::memory_order_relaxed);
    queue_.tryPush(s);
  }

  /**
   * @brief Record a gauge (point-in-time) value.
   * @param scopeName Name of the gauge scope.
   * @param tag Monitor tag identifying the source.
   * @param value Gauge value.
   * @note RT-safe: Lock-free enqueue, no allocation.
   */
  void gauge(const char* scopeName, const MonitorTag& tag, double value) noexcept {
    if (!enabled_.load(std::memory_order_relaxed))
      return;

    Sample s;
    s.timestampNs = nowNs();
    s.kind = SampleKind::GAUGE;
    s.tag = tag;
    s.value = value;
    if (scopeName) {
      std::strncpy(s.scope, scopeName, sizeof(s.scope) - 1);
    }

    totalSamples_.fetch_add(1, std::memory_order_relaxed);
    queue_.tryPush(s);
  }

  /* ----------------------------- Configuration ----------------------------- */

  /**
   * @brief Set a threshold (in microseconds) for a scope+id combo.
   *
   * If a scope duration exceeds this, it is flagged as THRESHOLD_BREACH.
   * MUST be called before start(). Thresholds are frozen into an immutable
   * snapshot at start() time so the hot path is lock-free and allocation-free.
   *
   * @param scopeName Name of the scope.
   * @param tagId Numeric tag identifier.
   * @param thresholdUs Threshold in microseconds.
   * @note NOT RT-safe: Map insertion.
   */
  void setThreshold(const char* scopeName, std::uint16_t tagId, std::uint64_t thresholdUs) {
    const std::string KEY = makeThresholdKey(scopeName, tagId);
    pendingThresholds_[KEY] = thresholdUs;
  }

  /**
   * @brief Enable or disable monitoring at runtime.
   * @param on True to enable, false to disable.
   * @note RT-safe: Atomic store.
   */
  void setEnabled(bool on) noexcept { enabled_.store(on, std::memory_order_relaxed); }

  /**
   * @brief Access the queue (for testing).
   * @return Const reference to the queue.
   * @note RT-safe: No allocation.
   */
  [[nodiscard]] const MonitorQueue& queue() const noexcept { return queue_; }

  /**
   * @brief Access the summary (for testing).
   * @return Const reference to the summary.
   * @note RT-safe: No allocation.
   */
  [[nodiscard]] const MonitorSummary& summary() const noexcept { return summary_; }

private:
  [[nodiscard]] static std::string makeThresholdKey(const char* scopeName, std::uint16_t tagId) {
    return std::string(scopeName ? scopeName : "") + "/" + std::to_string(tagId);
  }

  /// RT-safe: reads from immutable frozen snapshot. No lock, no allocation.
  [[nodiscard]] std::uint64_t getThreshold(const char* scopeName,
                                           std::uint16_t tagId) const noexcept {
    if (!frozenThresholds_)
      return 0;
    // Linear scan of frozen (immutable) table. Typically < 10 entries.
    const auto& TABLE = *frozenThresholds_;
    for (const auto& ENTRY : TABLE) {
      if (ENTRY.tagId == tagId && std::strncmp(ENTRY.scope, scopeName ? scopeName : "", 32) == 0) {
        return ENTRY.thresholdUs;
      }
    }
    return 0;
  }

  void drainLoop() {
    Sample sample;
    while (running_.load(std::memory_order_relaxed) || queue_.tryPop(sample)) {
      // Drain batch
      while (queue_.tryPop(sample)) {
        for (auto& sink : sinks_) {
          sink->write(sample);
        }
        summary_.record(sample);
      }

      if (running_.load(std::memory_order_relaxed)) {
        // Brief sleep to avoid busy-spinning when queue is empty
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }

    // Final drain after stop
    while (queue_.tryPop(sample)) {
      for (auto& sink : sinks_) {
        sink->write(sample);
      }
      summary_.record(sample);
    }

    for (auto& sink : sinks_) {
      sink->flush();
    }
  }

  /// Frozen (immutable) threshold entry for RT-safe reads.
  struct ThresholdEntry {
    char scope[32]{};
    std::uint16_t tagId{0};
    std::uint64_t thresholdUs{0};
  };

  MonitorConfig cfg_;
  MonitorQueue queue_;
  std::atomic<bool> enabled_;
  std::atomic<bool> running_;
  std::atomic<std::uint64_t> totalSamples_;
  std::uint64_t startTimeNs_;

  std::thread ioThread_;
  std::vector<std::unique_ptr<MonitorSink>> sinks_;
  MonitorSummary summary_;

  /// Setup-time thresholds (written before start(), never read on hot path).
  std::map<std::string, std::uint64_t> pendingThresholds_;

  /// Frozen immutable snapshot created at start(). Read by hot path with
  /// zero synchronization -- no mutex, no allocation, no heap.
  std::unique_ptr<std::vector<ThresholdEntry>> frozenThresholds_;
};

/* ----------------------------- ScopeGuard dtor ----------------------------- */

inline ScopeGuard::~ScopeGuard() noexcept {
  const std::uint64_t END_NS = nowNs();
  monitor_.recordScope(scope_, tag_, startNs_, END_NS);
}

/* ----------------------------- Convenience Macros ----------------------------- */

/// Token-pasting helper (two levels for proper expansion of `__LINE__`).
#define VERNIER_MONITOR_CAT_INNER(a, b) a##b
#define VERNIER_MONITOR_CAT(a, b) VERNIER_MONITOR_CAT_INNER(a, b)

/// Scoped timer macro. Creates a ScopeGuard that auto-records on scope exit.
/// Usage: VERNIER_MONITOR_SCOPE(monitor, "stage_name", tag);
#define VERNIER_MONITOR_SCOPE(mon, scopeName, tag)                                                 \
  ::vernier::monitor::ScopeGuard VERNIER_MONITOR_CAT(_vmon_guard_, __LINE__)(mon, scopeName, tag)

/// Point-in-time gauge sample. Useful for "current queue depth", etc.
/// Usage: VERNIER_MONITOR_GAUGE(monitor, "queue_depth", tag, queue.size());
#define VERNIER_MONITOR_GAUGE(mon, scopeName, tag, value)                                          \
  (mon).gauge((scopeName), (tag), static_cast<double>(value))

/// Counter increment. Default delta is 1; pass a value to advance by N.
/// Usage: VERNIER_MONITOR_INCREMENT(monitor, "events", tag);
#define VERNIER_MONITOR_INCREMENT(mon, scopeName, tag, ...)                                        \
  (mon).increment((scopeName), (tag) __VA_OPT__(, static_cast<double>(__VA_ARGS__)))

} // namespace monitor
} // namespace vernier

#endif // VERNIER_MONITOR_HPP
