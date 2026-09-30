/**
 * @file CuptiCollector.cu
 * @brief CUPTI Activity API implementation.
 *
 * Buffer-based model: register two callbacks; CUPTI fills our buffers with
 * activity records from kernel-launch threads; we walk the records in
 * stop() to populate the aggregate metrics. Every CUPTI call's result is
 * checked: a refusal to register, or to switch kernel records on or off,
 * leaves the collector unavailable with the reason; a failed flush, dropped
 * records or no record at all leave the window without stats, with the
 * problem named.
 */

#include "src/bench/inc/CuptiCollector.hpp"

#include "src/bench/inc/ProfilerEnv.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// The build decides (src/bench/CMakeLists.txt): COMPAT_CUPTI_AVAILABLE is 1
// only when libcupti is linked. The presence of cupti.h decides nothing; it
// sits in the CUDA toolkit's main include directory with or without the
// library.
#if defined(COMPAT_CUPTI_AVAILABLE) && COMPAT_CUPTI_AVAILABLE
#define VERNIER_HAS_CUPTI 1
#include <cupti.h>
#else
#define VERNIER_HAS_CUPTI 0
#endif

namespace vernier {
namespace bench {

#if VERNIER_HAS_CUPTI

namespace {

// Kernel records are read through CUpti_ActivityKernel9, which CUPTI declares
// from CUDA 12.0 (API version 19) on. CUPTI hands each kernel over as the
// newest kernel record its library knows; the fields read here sit at the same
// offsets in every version from 4 to 11, so this view reads a newer record
// correctly. The asserts hold that for the newest records the header declares.
static_assert(CUPTI_API_VERSION >= 19, "CuptiCollector.cu reads CUpti_ActivityKernel9, which "
                                       "CUPTI declares from CUDA 12.0 (API version 19) on");

/** @brief True when @p Newer keeps the fields read here where Kernel9 has them. */
template <typename Newer> constexpr bool readsThroughKernel9() {
  return offsetof(CUpti_ActivityKernel9, kind) == offsetof(Newer, kind) &&
         offsetof(CUpti_ActivityKernel9, registersPerThread) ==
             offsetof(Newer, registersPerThread) &&
         offsetof(CUpti_ActivityKernel9, staticSharedMemory) ==
             offsetof(Newer, staticSharedMemory) &&
         offsetof(CUpti_ActivityKernel9, dynamicSharedMemory) ==
             offsetof(Newer, dynamicSharedMemory) &&
         offsetof(CUpti_ActivityKernel9, name) == offsetof(Newer, name);
}
#if CUPTI_API_VERSION >= 130000
static_assert(readsThroughKernel9<CUpti_ActivityKernel10>(),
              "CUpti_ActivityKernel10 moves a field read through CUpti_ActivityKernel9");
#endif
#if CUPTI_API_VERSION >= 130100
static_assert(readsThroughKernel9<CUpti_ActivityKernel11>(),
              "CUpti_ActivityKernel11 moves a field read through CUpti_ActivityKernel9");
#endif

constexpr std::size_t BUFFER_SIZE = 32 * 1024; // bytes per CUPTI activity buffer
constexpr std::size_t BUFFER_ALIGN = 8;        // CUPTI requires 8-byte aligned buffers
constexpr std::size_t RECORD_RESERVE = 1024;   // pre-allocate to avoid reallocation under callback

struct KernelRecord {
  std::uint16_t registersPerThread{0};
  std::uint32_t staticSmemBytes{0};
  std::uint32_t dynamicSmemBytes{0};
  std::string name;
};

// Aggregator state is global because CUPTI's buffer-completion callback is a
// free function. Guarded by a mutex; expected contention is low (one drain
// per measured window) and the mutex never appears in the hot path.
struct Aggregator {
  std::mutex mtx;
  std::vector<KernelRecord> records;
  std::size_t dropped{0};                  ///< Records CUPTI dropped in this window
  CUptiResult droppedCount{CUPTI_SUCCESS}; ///< The first failed count of them, if any
  bool enabled{false};

  /** @brief Adds one reading of CUPTI's dropped-record count (call with mtx held). */
  void noteDropped(CUptiResult counted, std::size_t n) {
    if (counted != CUPTI_SUCCESS) {
      if (droppedCount == CUPTI_SUCCESS) {
        droppedCount = counted;
      }
      return;
    }
    dropped += n;
  }
};

Aggregator& aggregator() {
  static Aggregator g;
  return g;
}

/** @brief CUPTI's name for @p result, or its number when CUPTI gives none. */
std::string resultName(CUptiResult result) {
  const char* text = nullptr;
  if (cuptiGetResultString(result, &text) == CUPTI_SUCCESS && text != nullptr) {
    return text;
  }
  return "CUPTI result " + std::to_string(static_cast<long long>(result));
}

extern "C" void CUPTIAPI cuptiBufferRequested(uint8_t** buffer, size_t* size,
                                              size_t* maxNumRecords) {
  void* allocated = nullptr;
  if (posix_memalign(&allocated, BUFFER_ALIGN, BUFFER_SIZE) != 0) {
    *buffer = nullptr;
    *size = 0;
    *maxNumRecords = 0;
    return;
  }
  *buffer = static_cast<uint8_t*>(allocated);
  *size = BUFFER_SIZE;
  *maxNumRecords = 0; // 0 means "as many as fit"
}

extern "C" void CUPTIAPI cuptiBufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer,
                                              size_t /*size*/, size_t validSize) {
  // CUPTI counts the records it had no buffer space for; reading the count
  // resets it. It is read with every buffer so that a window knows whether
  // its records are complete.
  std::size_t dropped = 0;
  const CUptiResult COUNTED = cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped);

  Aggregator& agg = aggregator();
  CUpti_Activity* record = nullptr;
  CUptiResult status = CUPTI_SUCCESS;

  std::lock_guard<std::mutex> guard(agg.mtx);
  if (agg.enabled) {
    agg.noteDropped(COUNTED, dropped);
  }
  if (!agg.enabled || !buffer) {
    std::free(buffer);
    return;
  }

  do {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status != CUPTI_SUCCESS || !record)
      break;

    // The collector enables KERNEL records; CONCURRENT_KERNEL records share
    // their record type, so either kind is read.
    if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL ||
        record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
      auto* k = reinterpret_cast<CUpti_ActivityKernel9*>(record);
      KernelRecord r;
      r.registersPerThread = k->registersPerThread;
      r.staticSmemBytes = k->staticSharedMemory;
      r.dynamicSmemBytes = k->dynamicSharedMemory;
      r.name = k->name ? k->name : "?";
      agg.records.push_back(std::move(r));
    }
  } while (status == CUPTI_SUCCESS);

  std::free(buffer);
}

} // namespace

#endif // VERNIER_HAS_CUPTI

/** @brief What the collector says about itself; kept here, off the class's layout. */
struct CuptiCollector::Impl {
  std::string unavailableReason; ///< Why the collector does not collect; empty while it can
  std::string windowProblem;     ///< What kept the last window's records from being complete
};

CuptiCollector::CuptiCollector(bool forceDisabled) {
  // Decide before acquiring or registering anything: registering the
  // callbacks, before any cuptiActivityEnable, is already enough to keep an
  // nsys session from recording kernels (observed with nsys 2025.3.2), so a
  // later check in start() would come too late. The decision is
  // profiler_env::cuptiMustYield() (the explicit override or an nsys/ncu
  // session), the one the GPU harness passes in as forceDisabled, so a direct
  // caller gets the same rule; an explicit forceDisabled request wins over it.
  // An invalid VERNIER_DISABLE_CUPTI throws here, a configuration error,
  // before any CUPTI call, in a build without CUPTI too. Leaving available_
  // false keeps start()/stop()/stats() as safe no-ops.
  const bool YIELD = forceDisabled || profiler_env::cuptiMustYield();
  impl_ = new Impl();
  if (YIELD) {
    impl_->unavailableReason =
        "the collector stood down (an Nsight session, VERNIER_DISABLE_CUPTI or its caller)";
    return;
  }
#if VERNIER_HAS_CUPTI
  const CUptiResult REGISTERED =
      cuptiActivityRegisterCallbacks(cuptiBufferRequested, cuptiBufferCompleted);
  if (REGISTERED != CUPTI_SUCCESS) {
    impl_->unavailableReason =
        "CUPTI refused the collector's activity callbacks (" + resultName(REGISTERED) + ")";
    return;
  }
  aggregator().records.reserve(RECORD_RESERVE);
  available_ = true;
#else
  impl_->unavailableReason = "this build has no CUPTI";
#endif
}

CuptiCollector::~CuptiCollector() {
  if (running_)
    stop();
  delete impl_;
}

const std::string& CuptiCollector::unavailableReason() const noexcept {
  return impl_->unavailableReason;
}

const std::string& CuptiCollector::windowProblem() const noexcept { return impl_->windowProblem; }

void CuptiCollector::start() {
  if (running_)
    return;
  // A new window: nothing of the last one carries over, also when this one
  // collects nothing.
  stats_ = {};
  impl_->windowProblem.clear();
  // A collector that stood down at construction, has no CUPTI, or was
  // refused by CUPTI is not available.
  if (!available_)
    return;
#if VERNIER_HAS_CUPTI
  {
    std::lock_guard<std::mutex> guard(aggregator().mtx);
    aggregator().records.clear();
    aggregator().dropped = 0;
    aggregator().droppedCount = CUPTI_SUCCESS;
    aggregator().enabled = true;
  }
  // KERNEL records only: with KERNEL on, CUPTI refuses CONCURRENT_KERNEL
  // (CUPTI_ERROR_NOT_COMPATIBLE).
  const CUptiResult ENABLED = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
  if (ENABLED != CUPTI_SUCCESS) {
    {
      std::lock_guard<std::mutex> guard(aggregator().mtx);
      aggregator().enabled = false;
    }
    available_ = false;
    impl_->unavailableReason =
        "CUPTI refused to record kernel activity (" + resultName(ENABLED) + ")";
    return;
  }
#endif
  running_ = true;
}

void CuptiCollector::stop() {
  if (!available_ || !running_)
    return;

#if VERNIER_HAS_CUPTI
  // 1 = force: partly filled buffers are handed over too.
  const CUptiResult FLUSHED = cuptiActivityFlushAll(1);
  const CUptiResult DISABLED = cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
  // Records dropped while CUPTI had no buffer to hand over are counted in the
  // same global queue the completion callback reads, but reach no callback.
  std::size_t droppedUnbuffered = 0;
  const CUptiResult COUNTED = cuptiActivityGetNumDroppedRecords(nullptr, 0, &droppedUnbuffered);

  // Aggregate under the mutex. A buffer CUPTI completes after this (when the
  // flush or the disable failed) finds the aggregator disabled and is freed
  // unread.
  std::vector<KernelRecord> snapshot;
  std::size_t dropped = 0;
  CUptiResult droppedCount = CUPTI_SUCCESS;
  {
    std::lock_guard<std::mutex> guard(aggregator().mtx);
    aggregator().noteDropped(COUNTED, droppedUnbuffered);
    aggregator().enabled = false;
    snapshot = std::move(aggregator().records);
    aggregator().records.clear();
    dropped = aggregator().dropped;
    droppedCount = aggregator().droppedCount;
  }

  // Kernel records still on would arrive between windows and be read into
  // the next one, so the collector collects no further window.
  if (DISABLED != CUPTI_SUCCESS) {
    available_ = false;
    impl_->unavailableReason =
        "CUPTI did not stop recording kernel activity (" + resultName(DISABLED) + ")";
  }

  // A count or a median from part of a window is not published: the window
  // reports no launch and names what went wrong.
  if (FLUSHED != CUPTI_SUCCESS) {
    impl_->windowProblem =
        "CUPTI failed to flush its activity buffers (" + resultName(FLUSHED) + ")";
  } else if (droppedCount != CUPTI_SUCCESS) {
    impl_->windowProblem =
        "CUPTI could not count its dropped records (" + resultName(droppedCount) + ")";
  } else if (dropped > 0) {
    impl_->windowProblem = "CUPTI dropped " + std::to_string(dropped) +
                           (dropped == 1 ? " activity record" : " activity records");
  } else if (snapshot.empty()) {
    impl_->windowProblem = "CUPTI recorded no kernel launch";
  }

  stats_ = {};
  if (impl_->windowProblem.empty() && !snapshot.empty()) {
    stats_.kernelLaunches = snapshot.size();
    stats_.firstKernelName = snapshot.front().name;

    auto medianU16 = [](std::vector<std::uint16_t>& v) -> std::uint16_t {
      std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
      return v[v.size() / 2];
    };
    auto medianU32 = [](std::vector<std::uint32_t>& v) -> std::uint32_t {
      std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
      return v[v.size() / 2];
    };

    std::vector<std::uint16_t> regs;
    std::vector<std::uint32_t> ssmem;
    std::vector<std::uint32_t> dsmem;
    regs.reserve(snapshot.size());
    ssmem.reserve(snapshot.size());
    dsmem.reserve(snapshot.size());
    std::uint16_t regsMax = 0;
    for (const auto& r : snapshot) {
      regs.push_back(r.registersPerThread);
      ssmem.push_back(r.staticSmemBytes);
      dsmem.push_back(r.dynamicSmemBytes);
      if (r.registersPerThread > regsMax)
        regsMax = r.registersPerThread;
    }
    stats_.registersMedian = medianU16(regs);
    stats_.registersMax = regsMax;
    stats_.staticSmemBytes = medianU32(ssmem);
    stats_.dynamicSmemBytes = medianU32(dsmem);
  }
#endif

  running_ = false;
}

void CuptiCollector::reset() {
#if VERNIER_HAS_CUPTI
  {
    std::lock_guard<std::mutex> guard(aggregator().mtx);
    aggregator().records.clear();
    aggregator().dropped = 0;
    aggregator().droppedCount = CUPTI_SUCCESS;
  }
#endif
  stats_ = {};
  impl_->windowProblem.clear();
}

} // namespace bench
} // namespace vernier
