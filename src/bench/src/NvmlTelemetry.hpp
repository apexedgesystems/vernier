#ifndef VERNIER_BENCH_NVMLTELEMETRY_HPP
#define VERNIER_BENCH_NVMLTELEMETRY_HPP
/**
 * @file NvmlTelemetry.hpp
 * @brief The GPU harness's NVML readings, each one checked: the CSV cells a
 *        measured window's readings can fill, and the statement naming the
 *        cells they cannot.
 *
 * Private to the GPU harness (PerfGpuHarness.cu) and not installed; host code
 * only. COMPAT_NVML_AVAILABLE, which the build sets to 1 when it links NVML
 * and to 0 otherwise (also with VERNIER_USE_NVML off), decides whether NVML is
 * called at all: at 0 this header includes no nvml.h and a session says the
 * build has no NVML.
 *
 * NVML answers each reading with a value or with a result saying why it has
 * none, and a device can report some readings and not others: an integrated
 * GPU may answer "Not Supported" to every one the harness takes. A cell is
 * filled only from readings NVML reported, so no cell carries a zero that
 * nothing measured.
 */

#include "src/bench/inc/PerfGpuStats.hpp"

#include <cstddef>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#if COMPAT_NVML_AVAILABLE
#include <nvml.h>
#endif

namespace vernier {
namespace bench {
namespace nvml_telemetry {

/* ----------------------------- Readings ----------------------------- */

/** @brief One NVML reading: the value NVML reported, or why it reported none. */
template <typename T> struct Reading {
  std::optional<T> value;           ///< Set when NVML reported the reading
  std::string failure = "not read"; ///< NVML's text for why not; empty when reported
};

/** @brief What one end of a measured window read. */
struct Sample {
  Reading<int> smClockMHz;
  Reading<int> memClockMHz;
  Reading<double> powerDrawW;
  Reading<int> temperatureC;
};

/** @brief The readings of one measured window. */
struct WindowReadings {
  Sample start;
  Sample end;
  Reading<int> maxSmClockMHz;  ///< Read with the start sample
  Reading<double> powerLimitW; ///< Read with the start sample
};

/* ----------------------------- Cells ----------------------------- */

/** @brief The CSV cells NVML fills; each is empty unless NVML reported every reading it needs. */
struct Cells {
  std::optional<int> smClockMHz;        ///< The end sample's SM clock
  std::optional<bool> throttling;       ///< The SM clock at both ends and the maximum SM clock
  std::optional<double> powerDrawW;     ///< Mean of the power draw at both ends
  std::optional<double> powerLimitW;    ///< The power limit
  std::optional<int> temperatureC;      ///< The end sample's GPU temperature
  std::optional<int> temperatureDeltaC; ///< The GPU temperature at both ends: end minus start
};

/** @brief The cells @p w can fill, by the rule on Cells. */
inline Cells cellsOf(const WindowReadings& w) {
  Cells cells;
  if (w.end.smClockMHz.value) {
    cells.smClockMHz = *w.end.smClockMHz.value;
  }
  if (w.start.smClockMHz.value && w.end.smClockMHz.value && w.maxSmClockMHz.value) {
    ClockSpeedProfile clocks{};
    clocks.smClockMHzStart = *w.start.smClockMHz.value;
    clocks.smClockMHzEnd = *w.end.smClockMHz.value;
    clocks.boostClockMHz = *w.maxSmClockMHz.value;
    cells.throttling = clocks.isThrottling();
  }
  if (w.start.powerDrawW.value && w.end.powerDrawW.value) {
    cells.powerDrawW = (*w.start.powerDrawW.value + *w.end.powerDrawW.value) * 0.5;
  }
  if (w.powerLimitW.value) {
    cells.powerLimitW = *w.powerLimitW.value;
  }
  if (w.end.temperatureC.value) {
    cells.temperatureC = *w.end.temperatureC.value;
  }
  if (w.start.temperatureC.value && w.end.temperatureC.value) {
    cells.temperatureDeltaC = *w.end.temperatureC.value - *w.start.temperatureC.value;
  }
  return cells;
}

/**
 * @brief The public profiles as the harness fills them: each field from its
 *        reading, 0 where NVML reported none.
 */
inline void fillProfiles(const WindowReadings& w, ClockSpeedProfile& clocks,
                         PowerThermalProfile& powerThermal) {
  clocks.smClockMHzStart = w.start.smClockMHz.value.value_or(0);
  clocks.smClockMHzEnd = w.end.smClockMHz.value.value_or(0);
  clocks.memClockMHzStart = w.start.memClockMHz.value.value_or(0);
  clocks.memClockMHzEnd = w.end.memClockMHz.value.value_or(0);
  clocks.boostClockMHz = w.maxSmClockMHz.value.value_or(0);
  powerThermal.powerDrawWStart = w.start.powerDrawW.value.value_or(0.0);
  powerThermal.powerDrawWEnd = w.end.powerDrawW.value.value_or(0.0);
  powerThermal.powerLimitW = w.powerLimitW.value.value_or(0.0);
  powerThermal.temperatureCStart = w.start.temperatureC.value.value_or(0);
  powerThermal.temperatureCEnd = w.end.temperatureC.value.value_or(0);
}

/**
 * @brief The run's throttling warning for @p w, or "" when its cells do not
 *        show throttling (including when NVML did not report what it needs).
 */
inline std::string throttlingWarning(const WindowReadings& w) {
  const Cells CELLS = cellsOf(w);
  if (!CELLS.throttling.value_or(false)) {
    return {};
  }
  return "Warning: GPU throttling detected (" + std::to_string(*w.start.smClockMHz.value) + " -> " +
         std::to_string(*w.end.smClockMHz.value) + " MHz)";
}

/* ----------------------------- Statements ----------------------------- */

/// The CSV columns NVML fills, in the CSV's order.
constexpr const char* CELL_COLUMNS[] = {"smClockMHz",  "throttling",   "powerDrawW",
                                        "powerLimitW", "temperatureC", "temperatureDeltaC"};

/** @brief @p items as "a", "a and b" or "a, b and c", with @p last in place of "and". */
inline std::string joinList(const std::vector<std::string>& items, const char* last = "and") {
  std::string out;
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (i > 0) {
      out += (i + 1 == items.size()) ? std::string(" ") + last + " " : ", ";
    }
    out += items[i];
  }
  return out;
}

/** @brief "<columns> stays empty." or "<columns> stay empty." */
inline std::string emptyColumns(const std::vector<std::string>& columns) {
  return joinList(columns) + (columns.size() == 1 ? " stays empty." : " stay empty.");
}

/**
 * @brief The statement for a session that samples nothing: every NVML cell
 *        stays empty, for @p reason.
 */
inline std::string absenceStatement(const std::string& reason) {
  return "[gpu] " + reason + ": " +
         emptyColumns(std::vector<std::string>(std::begin(CELL_COLUMNS), std::end(CELL_COLUMNS)));
}

/**
 * @brief The statement naming the readings NVML did not report in @p w and
 *        the cells that leaves empty; "" when every cell is filled.
 *
 * A reading is named once, with NVML's text for the first failure of it; the
 * text is given once at the end when every named reading failed the same way.
 */
inline std::string missingReadingsStatement(const WindowReadings& w) {
  const Cells CELLS = cellsOf(w);
  std::vector<std::string> empty;
  if (!CELLS.smClockMHz) {
    empty.emplace_back("smClockMHz");
  }
  if (!CELLS.throttling) {
    empty.emplace_back("throttling");
  }
  if (!CELLS.powerDrawW) {
    empty.emplace_back("powerDrawW");
  }
  if (!CELLS.powerLimitW) {
    empty.emplace_back("powerLimitW");
  }
  if (!CELLS.temperatureC) {
    empty.emplace_back("temperatureC");
  }
  if (!CELLS.temperatureDeltaC) {
    empty.emplace_back("temperatureDeltaC");
  }
  if (empty.empty()) {
    return {};
  }

  struct Missing {
    std::string reading;
    std::string failure;
  };
  std::vector<Missing> missing;
  const auto NOTE = [&missing](const char* reading, const std::string& first,
                               const std::string& second = std::string()) {
    if (!first.empty()) {
      missing.push_back({reading, first});
    } else if (!second.empty()) {
      missing.push_back({reading, second});
    }
  };
  NOTE("SM clock", w.start.smClockMHz.failure, w.end.smClockMHz.failure);
  NOTE("maximum SM clock", w.maxSmClockMHz.failure);
  NOTE("power draw", w.start.powerDrawW.failure, w.end.powerDrawW.failure);
  NOTE("power limit", w.powerLimitW.failure);
  NOTE("GPU temperature", w.start.temperatureC.failure, w.end.temperatureC.failure);

  bool oneFailure = true;
  for (const Missing& m : missing) {
    oneFailure = oneFailure && m.failure == missing.front().failure;
  }
  std::vector<std::string> named;
  for (const Missing& m : missing) {
    named.push_back(oneFailure ? m.reading : m.reading + " (" + m.failure + ")");
  }
  std::string text = "[gpu] NVML reported no " + joinList(named, "or");
  if (oneFailure && !missing.empty()) {
    text += " (" + missing.front().failure + ")";
  }
  return text + ": " + emptyColumns(empty);
}

/* ----------------------------- Session ----------------------------- */

/**
 * @brief NVML opened for one device, for the life of a GPU test case.
 *
 * Opening checks each step and keeps the reason when it cannot sample; each
 * reading keeps NVML's result.
 */
class Session {
public:
  /**
   * @param capture false when the harness was asked not to sample
   *        (PerfGpuConfig::captureClockSpeeds): NVML is not opened.
   * @param index The device's NVML index.
   */
  Session(bool capture, unsigned index) {
    if (!capture) {
      reason_ = "clock and power capture is off (PerfGpuConfig::captureClockSpeeds)";
      return;
    }
#if COMPAT_NVML_AVAILABLE
    const nvmlReturn_t INIT = nvmlInit();
    if (INIT != NVML_SUCCESS) {
      reason_ = "NVML did not initialize (" + text(INIT) + ")";
      return;
    }
    initialized_ = true;
    const nvmlReturn_t FOUND = nvmlDeviceGetHandleByIndex(index, &device_);
    if (FOUND != NVML_SUCCESS) {
      reason_ = "NVML found no device at index " + std::to_string(index) + " (" + text(FOUND) + ")";
      return;
    }
    ready_ = true;
#else
    (void)index;
    reason_ = "this build has no NVML";
#endif
  }

  ~Session() {
#if COMPAT_NVML_AVAILABLE
    if (initialized_) {
      nvmlShutdown();
    }
#endif
  }

  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /** @return true when NVML is open for the device. */
  [[nodiscard]] bool ready() const noexcept { return ready_; }

  /** @return Why the session samples nothing; empty while ready(). */
  [[nodiscard]] const std::string& unavailableReason() const noexcept { return reason_; }

  /** @brief Reads a window's start: a sample, the maximum SM clock and the power limit. */
  void readStart(WindowReadings& w) const {
    if (!ready_) {
      return;
    }
    w.start = sample();
#if COMPAT_NVML_AVAILABLE
    unsigned maxClock = 0;
    const nvmlReturn_t MAX_CLOCK = nvmlDeviceGetMaxClockInfo(device_, NVML_CLOCK_SM, &maxClock);
    w.maxSmClockMHz = reading(MAX_CLOCK, static_cast<int>(maxClock));
    unsigned limitMw = 0;
    const nvmlReturn_t LIMIT = nvmlDeviceGetPowerManagementLimit(device_, &limitMw);
    w.powerLimitW = reading(LIMIT, static_cast<double>(limitMw) / 1000.0);
#endif
  }

  /** @brief Reads a window's end sample. */
  void readEnd(WindowReadings& w) const {
    if (!ready_) {
      return;
    }
    w.end = sample();
  }

private:
  /** @brief One end's readings: SM and memory clocks, power draw and GPU temperature. */
  [[nodiscard]] Sample sample() const {
    Sample s;
#if COMPAT_NVML_AVAILABLE
    unsigned smClock = 0;
    const nvmlReturn_t SM_CLOCK = nvmlDeviceGetClockInfo(device_, NVML_CLOCK_SM, &smClock);
    s.smClockMHz = reading(SM_CLOCK, static_cast<int>(smClock));
    unsigned memClock = 0;
    const nvmlReturn_t MEM_CLOCK = nvmlDeviceGetClockInfo(device_, NVML_CLOCK_MEM, &memClock);
    s.memClockMHz = reading(MEM_CLOCK, static_cast<int>(memClock));
    // NVML reports milliwatts.
    unsigned powerMw = 0;
    const nvmlReturn_t POWER = nvmlDeviceGetPowerUsage(device_, &powerMw);
    s.powerDrawW = reading(POWER, static_cast<double>(powerMw) / 1000.0);
    // Newer NVML deprecates nvmlDeviceGetTemperature for a versioned variant
    // that not every supported driver has, so the classic call stays, with the
    // deprecation silenced here.
    unsigned tempC = 0;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    const nvmlReturn_t TEMPERATURE =
        nvmlDeviceGetTemperature(device_, NVML_TEMPERATURE_GPU, &tempC);
#pragma GCC diagnostic pop
    s.temperatureC = reading(TEMPERATURE, static_cast<int>(tempC));
#endif
    return s;
  }

#if COMPAT_NVML_AVAILABLE
  /** @brief NVML's text for @p result, or its number when NVML gives none. */
  static std::string text(nvmlReturn_t result) {
    const char* s = nvmlErrorString(result);
    return (s != nullptr) ? std::string(s)
                          : "NVML result " + std::to_string(static_cast<int>(result));
  }

  /**
   * @brief @p value when @p result is success, NVML's text for @p result otherwise.
   *
   * Each caller makes the NVML call in a statement of its own before passing
   * the value it wrote: the order in which a call's arguments are evaluated is
   * unspecified, and GCC reads the value first.
   */
  template <typename T> static Reading<T> reading(nvmlReturn_t result, T value) {
    Reading<T> r;
    if (result == NVML_SUCCESS) {
      r.value = value;
      r.failure.clear();
    } else {
      r.failure = text(result);
    }
    return r;
  }

  nvmlDevice_t device_{};
  bool initialized_ = false; ///< nvmlInit succeeded, so the destructor shuts NVML down
#endif
  bool ready_ = false; ///< The device's handle was found
  std::string reason_; ///< Why the session samples nothing
};

} // namespace nvml_telemetry
} // namespace bench
} // namespace vernier

#endif // VERNIER_BENCH_NVMLTELEMETRY_HPP
