#ifndef VERNIER_BENCH_UTST_FAKE_CUPTI_H
#define VERNIER_BENCH_UTST_FAKE_CUPTI_H
/**
 * @file cupti.h
 * @brief A counting stand-in for the CUPTI Activity API calls CuptiCollector.cu makes.
 *
 * Only for TestBenchCuptiDecision: it lets the real CuptiCollector.cu be
 * compiled and run on a machine without CUPTI. It counts every call that
 * registers with or drives CUPTI, logs them in order, so a test can show what
 * a decision did, and answers each call with the result a test sets. A flush
 * delivers the kernel records, the unified-memory counter records and the
 * dropped-record count a test sets, through the buffer callbacks the
 * collector registered, as CUPTI does; dropped records can also be left to
 * reach no buffer, as when CUPTI had none to hand over. A walk of a buffer's
 * records ends at the buffer's end, or with the error a test sets after the
 * records it lets through.
 */

#include <stddef.h>
#include <stdint.h>

#include <cstring>
#include <string>
#include <vector>

#define CUPTIAPI

/// The API version of CUDA 12.0's CUPTI, the first to declare
/// CUpti_ActivityKernel9, whose shape the stand-in's record imitates.
#define CUPTI_API_VERSION 19

/* ----------------------------- Types ----------------------------- */

using CUptiResult = int;
constexpr CUptiResult CUPTI_SUCCESS = 0;
constexpr CUptiResult CUPTI_ERROR_MAX_LIMIT_REACHED = 1; ///< No further record in a buffer
constexpr CUptiResult CUPTI_ERROR_NOT_COMPATIBLE = 2;
constexpr CUptiResult CUPTI_ERROR_INVALID_KIND = 3; ///< An incomplete or invalid record
constexpr CUptiResult CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED = 4; ///< The Thor's answer
constexpr CUptiResult CUPTI_ERROR_NOT_READY = 5; ///< The kind, after a refused configuration
constexpr CUptiResult CUPTI_ERROR_UNKNOWN = 999;
using CUcontext = void*;

enum CUpti_ActivityKind {
  CUPTI_ACTIVITY_KIND_KERNEL = 3,
  CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10,
  CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER = 25,
};

struct CUpti_Activity {
  CUpti_ActivityKind kind;
};

struct CUpti_ActivityKernel9 {
  CUpti_ActivityKind kind;
  uint16_t registersPerThread;
  int32_t staticSharedMemory;
  int32_t dynamicSharedMemory;
  const char* name;
};

enum CUpti_ActivityUnifiedMemoryCounterScope {
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_UNKNOWN = 0,
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE = 1,
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_ALL_DEVICES = 2,
};

enum CUpti_ActivityUnifiedMemoryCounterKind {
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_UNKNOWN = 0,
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD = 1,
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH = 2,
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT = 3,
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT = 4,
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING = 5,
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING = 6,
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP = 7,
  CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD = 8,
};

struct CUpti_ActivityUnifiedMemoryCounterConfig {
  CUpti_ActivityUnifiedMemoryCounterScope scope;
  CUpti_ActivityUnifiedMemoryCounterKind kind;
  uint32_t deviceId;
  uint32_t enable;
};

/// The layout of CUPTI's own record of this name (64 bytes).
struct CUpti_ActivityUnifiedMemoryCounter2 {
  CUpti_ActivityKind kind;
  CUpti_ActivityUnifiedMemoryCounterKind counterKind;
  uint64_t value;
  uint64_t start;
  uint64_t end;
  uint64_t address;
  uint32_t srcId;
  uint32_t dstId;
  uint32_t streamId;
  uint32_t processId;
  uint32_t flags;
  uint32_t pad;
};

using CUpti_BuffersCallbackRequestFunc = void (*)(uint8_t**, size_t*, size_t*);
using CUpti_BuffersCallbackCompleteFunc = void (*)(CUcontext, uint32_t, uint8_t*, size_t, size_t);

/* ----------------------------- Counting and Behaviour ----------------------------- */

namespace fake_cupti {

/** @brief The calls made so far. */
struct Calls {
  int registrations = 0;                    ///< cuptiActivityRegisterCallbacks
  int enables = 0;                          ///< cuptiActivityEnable
  int disables = 0;                         ///< cuptiActivityDisable
  int flushes = 0;                          ///< cuptiActivityFlushAll
  int configures = 0;                       ///< cuptiActivityConfigureUnifiedMemoryCounter
  std::vector<CUpti_ActivityKind> enabled;  ///< Kinds asked of cuptiActivityEnable, in order
  std::vector<CUpti_ActivityKind> disabled; ///< Kinds asked of cuptiActivityDisable, in order
  std::vector<CUpti_ActivityUnifiedMemoryCounterConfig> configuration; ///< The last one asked for
  std::vector<CUptiResult> walkEnds; ///< cuptiActivityGetNextRecord answers with no record
  std::vector<std::string> log;      ///< "enable 3", "configure", "flush", "disable 25", ...
};

/** @brief How the stand-in answers; every call succeeds and delivers nothing by default. */
struct Behaviour {
  CUptiResult registerResult = CUPTI_SUCCESS;     ///< cuptiActivityRegisterCallbacks
  CUptiResult enableResult = CUPTI_SUCCESS;       ///< cuptiActivityEnable, the kernel kinds
  CUptiResult flushResult = CUPTI_SUCCESS;        ///< cuptiActivityFlushAll
  CUptiResult disableResult = CUPTI_SUCCESS;      ///< cuptiActivityDisable, the kernel kinds
  CUptiResult droppedCountResult = CUPTI_SUCCESS; ///< cuptiActivityGetNumDroppedRecords
  CUptiResult configureResult = CUPTI_SUCCESS;    ///< cuptiActivityConfigureUnifiedMemoryCounter
  CUptiResult unifiedMemoryEnableResult = CUPTI_SUCCESS;  ///< ...Enable, the unified-memory kind
  CUptiResult unifiedMemoryDisableResult = CUPTI_SUCCESS; ///< ...Disable, the unified-memory kind
  size_t recordsPerFlush = 0;                             ///< Kernel records one flush delivers
  uint16_t registersPerThread = 0;                        ///< Every delivered record's registers
  int32_t staticSharedMemory = 0;                         ///< Every delivered record's static bytes
  /// Unified-memory records one flush delivers after the kernel records,
  /// whether or not their kind is on: the collector must read only its own.
  std::vector<CUpti_ActivityUnifiedMemoryCounter2> unifiedMemoryRecords;
  size_t droppedPerFlush = 0;                   ///< Records one flush reports as dropped
  bool dropsReachABuffer = true;                ///< false: dropped records come with no buffer
  CUptiResult nextRecordResult = CUPTI_SUCCESS; ///< A walk's error after readableRecords, if set
  size_t readableRecords = 0;                   ///< Records a walk hands over before that error
};

/** @brief The buffer callbacks the collector registered. */
struct Callbacks {
  CUpti_BuffersCallbackRequestFunc requested = nullptr;
  CUpti_BuffersCallbackCompleteFunc completed = nullptr;
};

/** @brief The one call count of the process. */
inline Calls& calls() {
  static Calls c;
  return c;
}

/** @brief The one behaviour of the process; a test sets it and resets it. */
inline Behaviour& behaviour() {
  static Behaviour b;
  return b;
}

/** @brief The registered callbacks. */
inline Callbacks& callbacks() {
  static Callbacks c;
  return c;
}

/** @brief CUPTI's count of dropped records, which a flush raises and a read resets. */
inline size_t& pendingDropped() {
  static size_t n = 0;
  return n;
}

/** @brief Where each record of the buffer being walked starts, in order. */
inline std::vector<size_t>& recordOffsets() {
  static std::vector<size_t> offsets;
  return offsets;
}

/** @brief No call counted, the default behaviour, no callbacks and nothing dropped. */
inline void reset() {
  calls() = {};
  behaviour() = {};
  callbacks() = {};
  pendingDropped() = 0;
  recordOffsets().clear();
}

/** @brief A unified-memory record of @p kind with @p value, from @p start to @p end ns. */
inline CUpti_ActivityUnifiedMemoryCounter2
unifiedMemoryRecord(CUpti_ActivityUnifiedMemoryCounterKind kind, uint64_t value, uint64_t start = 0,
                    uint64_t end = 0) {
  CUpti_ActivityUnifiedMemoryCounter2 record{};
  record.kind = CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER;
  record.counterKind = kind;
  record.value = value;
  record.start = start;
  record.end = end;
  return record;
}

} // namespace fake_cupti

/* ----------------------------- API calls ----------------------------- */

inline CUptiResult cuptiActivityRegisterCallbacks(CUpti_BuffersCallbackRequestFunc requested,
                                                  CUpti_BuffersCallbackCompleteFunc completed) {
  ++fake_cupti::calls().registrations;
  if (fake_cupti::behaviour().registerResult == CUPTI_SUCCESS) {
    fake_cupti::callbacks() = {requested, completed};
  }
  return fake_cupti::behaviour().registerResult;
}
inline CUptiResult cuptiActivityEnable(CUpti_ActivityKind kind) {
  ++fake_cupti::calls().enables;
  fake_cupti::calls().enabled.push_back(kind);
  fake_cupti::calls().log.push_back("enable " + std::to_string(static_cast<int>(kind)));
  return kind == CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
             ? fake_cupti::behaviour().unifiedMemoryEnableResult
             : fake_cupti::behaviour().enableResult;
}
inline CUptiResult cuptiActivityDisable(CUpti_ActivityKind kind) {
  ++fake_cupti::calls().disables;
  fake_cupti::calls().disabled.push_back(kind);
  fake_cupti::calls().log.push_back("disable " + std::to_string(static_cast<int>(kind)));
  return kind == CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER
             ? fake_cupti::behaviour().unifiedMemoryDisableResult
             : fake_cupti::behaviour().disableResult;
}
inline CUptiResult
cuptiActivityConfigureUnifiedMemoryCounter(CUpti_ActivityUnifiedMemoryCounterConfig* config,
                                           uint32_t count) {
  ++fake_cupti::calls().configures;
  fake_cupti::calls().configuration.assign(config, config + count);
  fake_cupti::calls().log.push_back("configure");
  return fake_cupti::behaviour().configureResult;
}
/**
 * @brief Delivers the behaviour's records in one buffer (the kernel records,
 *        then the unified-memory ones, each at its own size) and counts its
 *        dropped records, then answers with its flush result (a failed flush
 *        has still handed over what it delivered).
 */
inline CUptiResult cuptiActivityFlushAll(uint32_t) {
  ++fake_cupti::calls().flushes;
  fake_cupti::calls().log.push_back("flush");
  const fake_cupti::Behaviour& b = fake_cupti::behaviour();
  if (b.dropsReachABuffer) {
    fake_cupti::pendingDropped() += b.droppedPerFlush;
  }
  const fake_cupti::Callbacks& cb = fake_cupti::callbacks();
  const bool BUFFER = b.recordsPerFlush > 0 || !b.unifiedMemoryRecords.empty() ||
                      (b.droppedPerFlush > 0 && b.dropsReachABuffer);
  if (BUFFER && cb.requested != nullptr && cb.completed != nullptr) {
    uint8_t* buffer = nullptr;
    size_t size = 0;
    size_t maxRecords = 0;
    cb.requested(&buffer, &size, &maxRecords);
    size_t valid = 0;
    std::vector<size_t>& offsets = fake_cupti::recordOffsets();
    offsets.clear();
    const auto PUT = [&](const void* record, size_t bytes) {
      if (buffer == nullptr || valid + bytes > size) {
        return;
      }
      std::memcpy(buffer + valid, record, bytes);
      offsets.push_back(valid);
      valid += bytes;
    };
    for (size_t i = 0; i < b.recordsPerFlush; ++i) {
      CUpti_ActivityKernel9 record{};
      record.kind = CUPTI_ACTIVITY_KIND_KERNEL;
      record.registersPerThread = b.registersPerThread;
      record.staticSharedMemory = b.staticSharedMemory;
      record.name = "fakeKernel";
      PUT(&record, sizeof(record));
    }
    for (const CUpti_ActivityUnifiedMemoryCounter2& record : b.unifiedMemoryRecords) {
      PUT(&record, sizeof(record));
    }
    cb.completed(nullptr, 0, buffer, size, valid);
  }
  if (!b.dropsReachABuffer) {
    fake_cupti::pendingDropped() += b.droppedPerFlush;
  }
  return b.flushResult;
}
/**
 * @brief Walks the records a flush wrote: the first when @p record is null, then
 *        each next, and CUPTI_ERROR_MAX_LIMIT_REACHED past the last. With a
 *        nextRecordResult set, the walk answers it instead, with no record, once
 *        it has handed over readableRecords records. Each answer without a
 *        record is counted.
 */
inline CUptiResult cuptiActivityGetNextRecord(uint8_t* buffer, size_t validSize,
                                              CUpti_Activity** record) {
  const fake_cupti::Behaviour& b = fake_cupti::behaviour();
  const std::vector<size_t>& offsets = fake_cupti::recordOffsets();
  size_t next = 0;
  if (*record != nullptr) {
    const size_t AT = static_cast<size_t>(reinterpret_cast<uint8_t*>(*record) - buffer);
    while (next < offsets.size() && offsets[next] <= AT) {
      ++next;
    }
  }
  CUptiResult answer = CUPTI_SUCCESS;
  if (b.nextRecordResult != CUPTI_SUCCESS && next >= b.readableRecords) {
    answer = b.nextRecordResult;
  } else if (buffer == nullptr || next >= offsets.size() || offsets[next] >= validSize) {
    answer = CUPTI_ERROR_MAX_LIMIT_REACHED;
  }
  if (answer != CUPTI_SUCCESS) {
    *record = nullptr;
    fake_cupti::calls().walkEnds.push_back(answer);
    return answer;
  }
  *record = reinterpret_cast<CUpti_Activity*>(buffer + offsets[next]);
  return CUPTI_SUCCESS;
}
/** @brief Reports the dropped count and resets it, as CUPTI does, unless told to fail. */
inline CUptiResult cuptiActivityGetNumDroppedRecords(CUcontext, uint32_t, size_t* dropped) {
  if (fake_cupti::behaviour().droppedCountResult != CUPTI_SUCCESS) {
    return fake_cupti::behaviour().droppedCountResult;
  }
  *dropped = fake_cupti::pendingDropped();
  fake_cupti::pendingDropped() = 0;
  return CUPTI_SUCCESS;
}
inline CUptiResult cuptiGetResultString(CUptiResult result, const char** text) {
  switch (result) {
  case CUPTI_SUCCESS:
    *text = "CUPTI_SUCCESS";
    return CUPTI_SUCCESS;
  case CUPTI_ERROR_NOT_COMPATIBLE:
    *text = "CUPTI_ERROR_NOT_COMPATIBLE";
    return CUPTI_SUCCESS;
  case CUPTI_ERROR_INVALID_KIND:
    *text = "CUPTI_ERROR_INVALID_KIND";
    return CUPTI_SUCCESS;
  case CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED:
    *text = "CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED";
    return CUPTI_SUCCESS;
  case CUPTI_ERROR_NOT_READY:
    *text = "CUPTI_ERROR_NOT_READY";
    return CUPTI_SUCCESS;
  case CUPTI_ERROR_UNKNOWN:
    *text = "CUPTI_ERROR_UNKNOWN";
    return CUPTI_SUCCESS;
  default:
    *text = nullptr;
    return CUPTI_ERROR_UNKNOWN;
  }
}

#endif // VERNIER_BENCH_UTST_FAKE_CUPTI_H
