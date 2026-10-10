#ifndef VERNIER_BENCH_UTST_FAKE_NVML_H
#define VERNIER_BENCH_UTST_FAKE_NVML_H
/**
 * @file nvml.h
 * @brief A scripted stand-in for the NVML calls NvmlTelemetry.hpp makes.
 *
 * Only for TestBenchNvmlTelemetry: it lets the real NvmlTelemetry.hpp be
 * compiled and run on a machine without NVML. A test lists the devices NVML
 * numbers, each with its UUID, and, for each reading of each device, what the
 * first call and every later call answer (a window's start and its end), and
 * the stand-in counts initialisations and shutdowns. Result codes and texts
 * are NVML's.
 */

#include <cstring>
#include <string>
#include <vector>

/* ----------------------------- Types ----------------------------- */

typedef enum nvmlReturn_enum {
  NVML_SUCCESS = 0,
  NVML_ERROR_UNINITIALIZED = 1,
  NVML_ERROR_INVALID_ARGUMENT = 2,
  NVML_ERROR_NOT_SUPPORTED = 3,
  NVML_ERROR_NO_PERMISSION = 4,
  NVML_ERROR_NOT_FOUND = 6,
  NVML_ERROR_DRIVER_NOT_LOADED = 9,
  NVML_ERROR_UNKNOWN = 999
} nvmlReturn_t;

typedef enum nvmlClockType_enum {
  NVML_CLOCK_GRAPHICS = 0,
  NVML_CLOCK_SM = 1,
  NVML_CLOCK_MEM = 2
} nvmlClockType_t;

typedef enum nvmlTemperatureSensors_enum { NVML_TEMPERATURE_GPU = 0 } nvmlTemperatureSensors_t;

namespace fake_nvml {

/** @brief One answer to a reading: its result and, on success, its value. */
struct Answer {
  nvmlReturn_t result = NVML_SUCCESS;
  unsigned value = 0;
};

/** @brief What a reading answers: @ref first on its first call, @ref later on every other. */
struct Script {
  Answer first;
  Answer later;
  int calls = 0;

  /** @brief Both calls answer @p answer. */
  static Script always(Answer answer) { return Script{answer, answer, 0}; }

  Answer next() { return (calls++ == 0) ? first : later; }
};

/** @brief One device NVML numbers, with its UUID and its readings. */
struct Device {
  std::string uuid;    ///< As NVML spells it: "GPU-" and 8-4-4-4-12 hex digits
  Script smClock;      ///< MHz
  Script memClock;     ///< MHz
  Script maxSmClock;   ///< MHz
  Script powerMw;      ///< Milliwatts
  Script powerLimitMw; ///< Milliwatts
  Script temperatureC; ///< Celsius
};

/** @brief The calls made so far. */
struct Calls {
  int inits = 0;     ///< nvmlInit
  int shutdowns = 0; ///< nvmlShutdown
};

/** @brief How the stand-in answers; by default nvmlInit succeeds and NVML numbers no device. */
struct Behaviour {
  nvmlReturn_t initResult = NVML_SUCCESS;
  std::vector<Device> devices;
};

inline Calls& calls() {
  static Calls c;
  return c;
}

inline Behaviour& behaviour() {
  static Behaviour b;
  return b;
}

/** @brief No call counted and the default behaviour. */
inline void reset() {
  calls() = {};
  behaviour() = {};
}

} // namespace fake_nvml

/// A handle is the address of one of the behaviour's devices.
typedef fake_nvml::Device* nvmlDevice_t;

/* ----------------------------- API calls ----------------------------- */

inline nvmlReturn_t nvmlInit() {
  ++fake_nvml::calls().inits;
  return fake_nvml::behaviour().initResult;
}

inline nvmlReturn_t nvmlShutdown() {
  ++fake_nvml::calls().shutdowns;
  return NVML_SUCCESS;
}

inline nvmlReturn_t nvmlDeviceGetHandleByUUID(const char* uuid, nvmlDevice_t* device) {
  for (fake_nvml::Device& d : fake_nvml::behaviour().devices) {
    if (std::strcmp(d.uuid.c_str(), uuid) == 0) {
      *device = &d;
      return NVML_SUCCESS;
    }
  }
  return NVML_ERROR_NOT_FOUND;
}

namespace fake_nvml {

/** @brief Answers one reading from @p script, writing the value only on success. */
inline nvmlReturn_t answer(Script& script, unsigned* value) {
  const Answer A = script.next();
  if (A.result == NVML_SUCCESS) {
    *value = A.value;
  }
  return A.result;
}

} // namespace fake_nvml

inline nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type,
                                           unsigned* clock) {
  if (type == NVML_CLOCK_SM) {
    return fake_nvml::answer(device->smClock, clock);
  }
  if (type == NVML_CLOCK_MEM) {
    return fake_nvml::answer(device->memClock, clock);
  }
  return NVML_ERROR_NOT_SUPPORTED;
}

inline nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type,
                                              unsigned* clock) {
  return (type == NVML_CLOCK_SM) ? fake_nvml::answer(device->maxSmClock, clock)
                                 : NVML_ERROR_NOT_SUPPORTED;
}

inline nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned* milliwatts) {
  return fake_nvml::answer(device->powerMw, milliwatts);
}

inline nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned* milliwatts) {
  return fake_nvml::answer(device->powerLimitMw, milliwatts);
}

inline nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, nvmlTemperatureSensors_t,
                                             unsigned* celsius) {
  return fake_nvml::answer(device->temperatureC, celsius);
}

inline const char* nvmlErrorString(nvmlReturn_t result) {
  switch (result) {
  case NVML_SUCCESS:
    return "Success";
  case NVML_ERROR_UNINITIALIZED:
    return "Uninitialized";
  case NVML_ERROR_INVALID_ARGUMENT:
    return "Invalid Argument";
  case NVML_ERROR_NOT_SUPPORTED:
    return "Not Supported";
  case NVML_ERROR_NO_PERMISSION:
    return "Insufficient Permissions";
  case NVML_ERROR_NOT_FOUND:
    return "Not Found";
  case NVML_ERROR_DRIVER_NOT_LOADED:
    return "Driver Not Loaded";
  case NVML_ERROR_UNKNOWN:
    return "Unknown Error";
  }
  return nullptr;
}

#endif // VERNIER_BENCH_UTST_FAKE_NVML_H
