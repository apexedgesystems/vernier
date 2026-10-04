#ifndef VERNIER_BENCH_UTST_FAKE_CUPTI_H
#define VERNIER_BENCH_UTST_FAKE_CUPTI_H
/**
 * @file cupti.h
 * @brief A counting stand-in for the CUPTI Activity API calls CuptiCollector.cu makes.
 *
 * Only for TestBenchCuptiDecision: it lets the real CuptiCollector.cu be
 * compiled and run on a machine without CUPTI. It counts every call that
 * registers with or drives CUPTI, so a test can show what a decision did, and
 * answers each call with the result a test sets. A flush delivers the kernel
 * records and the dropped-record count a test sets, through the buffer
 * callbacks the collector registered, as CUPTI does; dropped records can also
 * be left to reach no buffer, as when CUPTI had none to hand over. A walk of a
 * buffer's records ends at the buffer's end, or with the error a test sets
 * after the records it lets through.
 */

#include <stddef.h>
#include <stdint.h>

#include <cstring>
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
constexpr CUptiResult CUPTI_ERROR_UNKNOWN = 999;
using CUcontext = void*;

enum CUpti_ActivityKind {
  CUPTI_ACTIVITY_KIND_KERNEL = 3,
  CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10,
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

using CUpti_BuffersCallbackRequestFunc = void (*)(uint8_t**, size_t*, size_t*);
using CUpti_BuffersCallbackCompleteFunc = void (*)(CUcontext, uint32_t, uint8_t*, size_t, size_t);

/* ----------------------------- Counting and Behaviour ----------------------------- */

namespace fake_cupti {

/** @brief The calls made so far. */
struct Calls {
  int registrations = 0;                   ///< cuptiActivityRegisterCallbacks
  int enables = 0;                         ///< cuptiActivityEnable
  int disables = 0;                        ///< cuptiActivityDisable
  int flushes = 0;                         ///< cuptiActivityFlushAll
  std::vector<CUpti_ActivityKind> enabled; ///< Kinds asked of cuptiActivityEnable, in order
  std::vector<CUptiResult> walkEnds;       ///< cuptiActivityGetNextRecord answers with no record
};

/** @brief How the stand-in answers; every call succeeds and delivers nothing by default. */
struct Behaviour {
  CUptiResult registerResult = CUPTI_SUCCESS;     ///< cuptiActivityRegisterCallbacks
  CUptiResult enableResult = CUPTI_SUCCESS;       ///< cuptiActivityEnable, any kind
  CUptiResult flushResult = CUPTI_SUCCESS;        ///< cuptiActivityFlushAll
  CUptiResult disableResult = CUPTI_SUCCESS;      ///< cuptiActivityDisable, any kind
  CUptiResult droppedCountResult = CUPTI_SUCCESS; ///< cuptiActivityGetNumDroppedRecords
  size_t recordsPerFlush = 0;                     ///< Kernel records one flush delivers
  uint16_t registersPerThread = 0;                ///< Every delivered record's registers
  int32_t staticSharedMemory = 0;                 ///< Every delivered record's static bytes
  size_t droppedPerFlush = 0;                     ///< Records one flush reports as dropped
  bool dropsReachABuffer = true;                  ///< false: dropped records come with no buffer
  CUptiResult nextRecordResult = CUPTI_SUCCESS;   ///< A walk's error after readableRecords, if set
  size_t readableRecords = 0;                     ///< Records a walk hands over before that error
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

/** @brief No call counted, the default behaviour, no callbacks and nothing dropped. */
inline void reset() {
  calls() = {};
  behaviour() = {};
  callbacks() = {};
  pendingDropped() = 0;
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
  return fake_cupti::behaviour().enableResult;
}
inline CUptiResult cuptiActivityDisable(CUpti_ActivityKind) {
  ++fake_cupti::calls().disables;
  return fake_cupti::behaviour().disableResult;
}
/**
 * @brief Delivers the behaviour's records in one buffer and counts its dropped
 *        records, then answers with its flush result (a failed flush has
 *        still handed over what it delivered).
 */
inline CUptiResult cuptiActivityFlushAll(uint32_t) {
  ++fake_cupti::calls().flushes;
  const fake_cupti::Behaviour& b = fake_cupti::behaviour();
  if (b.dropsReachABuffer) {
    fake_cupti::pendingDropped() += b.droppedPerFlush;
  }
  const fake_cupti::Callbacks& cb = fake_cupti::callbacks();
  const bool BUFFER = b.recordsPerFlush > 0 || (b.droppedPerFlush > 0 && b.dropsReachABuffer);
  if (BUFFER && cb.requested != nullptr && cb.completed != nullptr) {
    uint8_t* buffer = nullptr;
    size_t size = 0;
    size_t maxRecords = 0;
    cb.requested(&buffer, &size, &maxRecords);
    size_t valid = 0;
    for (size_t i = 0; i < b.recordsPerFlush && buffer != nullptr; ++i) {
      if (valid + sizeof(CUpti_ActivityKernel9) > size) {
        break;
      }
      CUpti_ActivityKernel9 record{};
      record.kind = CUPTI_ACTIVITY_KIND_KERNEL;
      record.registersPerThread = b.registersPerThread;
      record.staticSharedMemory = b.staticSharedMemory;
      record.name = "fakeKernel";
      std::memcpy(buffer + valid, &record, sizeof(record));
      valid += sizeof(record);
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
  const size_t NEXT = (*record == nullptr)
                          ? 0
                          : static_cast<size_t>(reinterpret_cast<uint8_t*>(*record) - buffer) +
                                sizeof(CUpti_ActivityKernel9);
  CUptiResult answer = CUPTI_SUCCESS;
  if (b.nextRecordResult != CUPTI_SUCCESS &&
      NEXT / sizeof(CUpti_ActivityKernel9) >= b.readableRecords) {
    answer = b.nextRecordResult;
  } else if (buffer == nullptr || NEXT + sizeof(CUpti_ActivityKernel9) > validSize) {
    answer = CUPTI_ERROR_MAX_LIMIT_REACHED;
  }
  if (answer != CUPTI_SUCCESS) {
    *record = nullptr;
    fake_cupti::calls().walkEnds.push_back(answer);
    return answer;
  }
  *record = reinterpret_cast<CUpti_Activity*>(buffer + NEXT);
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
  case CUPTI_ERROR_UNKNOWN:
    *text = "CUPTI_ERROR_UNKNOWN";
    return CUPTI_SUCCESS;
  default:
    *text = nullptr;
    return CUPTI_ERROR_UNKNOWN;
  }
}

#endif // VERNIER_BENCH_UTST_FAKE_CUPTI_H
