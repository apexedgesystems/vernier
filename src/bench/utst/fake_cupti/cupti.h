#ifndef VERNIER_BENCH_UTST_FAKE_CUPTI_H
#define VERNIER_BENCH_UTST_FAKE_CUPTI_H
/**
 * @file cupti.h
 * @brief A counting stand-in for the CUPTI Activity API calls CuptiCollector.cu makes.
 *
 * Only for TestBenchCuptiDecision: it lets the real CuptiCollector.cu be
 * compiled and run on a machine without CUPTI, and counts every call that
 * registers with or drives CUPTI, so a test can show what a decision did
 * before any record exists. No activity record is ever produced.
 */

#include <stddef.h>
#include <stdint.h>

#define CUPTIAPI

/* ----------------------------- Types ----------------------------- */

using CUptiResult = int;
constexpr CUptiResult CUPTI_SUCCESS = 0;
constexpr CUptiResult CUPTI_ERROR_MAX_LIMIT_REACHED = 1; ///< No further record in a buffer
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

/* ----------------------------- Counting ----------------------------- */

namespace fake_cupti {

/** @brief The calls made so far. */
struct Calls {
  int registrations = 0; ///< cuptiActivityRegisterCallbacks
  int enables = 0;       ///< cuptiActivityEnable
  int disables = 0;      ///< cuptiActivityDisable
  int flushes = 0;       ///< cuptiActivityFlushAll
};

/** @brief The one call count of the process. */
inline Calls& calls() {
  static Calls c;
  return c;
}

} // namespace fake_cupti

/* ----------------------------- API calls ----------------------------- */

inline CUptiResult cuptiActivityRegisterCallbacks(CUpti_BuffersCallbackRequestFunc,
                                                  CUpti_BuffersCallbackCompleteFunc) {
  ++fake_cupti::calls().registrations;
  return CUPTI_SUCCESS;
}
inline CUptiResult cuptiActivityEnable(CUpti_ActivityKind) {
  ++fake_cupti::calls().enables;
  return CUPTI_SUCCESS;
}
inline CUptiResult cuptiActivityDisable(CUpti_ActivityKind) {
  ++fake_cupti::calls().disables;
  return CUPTI_SUCCESS;
}
inline CUptiResult cuptiActivityFlushAll(uint32_t) {
  ++fake_cupti::calls().flushes;
  return CUPTI_SUCCESS;
}
inline CUptiResult cuptiActivityGetNextRecord(uint8_t*, size_t, CUpti_Activity** record) {
  *record = nullptr;
  return CUPTI_ERROR_MAX_LIMIT_REACHED;
}

#endif // VERNIER_BENCH_UTST_FAKE_CUPTI_H
