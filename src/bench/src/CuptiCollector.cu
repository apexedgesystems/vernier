/**
 * @file CuptiCollector.cu
 * @brief CUPTI Activity API implementation.
 *
 * Buffer-based model: register two callbacks; CUPTI fills our buffers with
 * activity records from kernel-launch threads; we walk the records in
 * stop() to populate the aggregate metrics. Every CUPTI call's result is
 * checked: a refusal to register, or to switch kernel records on or off,
 * leaves the collector unavailable with the reason; a refusal of the
 * unified-memory counters leaves the window without them, with the reason;
 * a failed flush, dropped records, records CUPTI could not read or no record
 * at all leave the window without stats, with the problem named.
 */

#include "src/bench/inc/CuptiCollector.hpp"

#include "src/bench/inc/PerfGpuStats.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// The build decides (src/bench/CMakeLists.txt): COMPAT_CUPTI_AVAILABLE is 1
// only when libcupti is linked. The presence of cupti.h decides nothing; it
// sits in the CUDA toolkit's main include directory with or without the
// library.
#if defined(COMPAT_CUPTI_AVAILABLE) && COMPAT_CUPTI_AVAILABLE
#define VERNIER_HAS_CUPTI 1
#include <cupti.h>
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif
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

// Unified-memory counter records are read through
// CUpti_ActivityUnifiedMemoryCounter2, which every CUPTI this file builds with
// declares. CUPTI 12.8 (API version 26) and later hand over
// CUpti_ActivityUnifiedMemoryCounter3, which appends to Counter2's fields; the
// assert holds that it keeps the ones read here where Counter2 has them.
/** @brief True when @p Newer keeps the fields read here where Counter2 has them. */
template <typename Newer> constexpr bool readsThroughCounter2() {
  return offsetof(CUpti_ActivityUnifiedMemoryCounter2, kind) == offsetof(Newer, kind) &&
         offsetof(CUpti_ActivityUnifiedMemoryCounter2, counterKind) ==
             offsetof(Newer, counterKind) &&
         offsetof(CUpti_ActivityUnifiedMemoryCounter2, value) == offsetof(Newer, value) &&
         offsetof(CUpti_ActivityUnifiedMemoryCounter2, start) == offsetof(Newer, start) &&
         offsetof(CUpti_ActivityUnifiedMemoryCounter2, end) == offsetof(Newer, end);
}
#if CUPTI_API_VERSION >= 26
static_assert(readsThroughCounter2<CUpti_ActivityUnifiedMemoryCounter3>(),
              "CUpti_ActivityUnifiedMemoryCounter3 moves a field read through "
              "CUpti_ActivityUnifiedMemoryCounter2");
#endif

constexpr std::size_t BUFFER_SIZE = 32 * 1024; // bytes per CUPTI activity buffer
constexpr std::size_t BUFFER_ALIGN = 8;        // CUPTI requires 8-byte aligned buffers
constexpr std::size_t RECORD_RESERVE = 1024;   // pre-allocate to avoid reallocation under callback

/// The unified-memory counters a window asks CUPTI for: the records of these
/// kinds are the ones UnifiedMemoryProfile is built from.
constexpr CUpti_ActivityUnifiedMemoryCounterKind UNIFIED_MEMORY_KINDS[] = {
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD,
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH,
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT,
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT,
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING,
};

struct KernelRecord {
  std::uint16_t registersPerThread{0};
  std::uint32_t staticSmemBytes{0};
  std::uint32_t dynamicSmemBytes{0};
  std::string name;
};

/**
 * @brief What one window's unified-memory counter records add up to, kind by
 *        kind, as UnifiedMemoryProfile documents them.
 */
struct UnifiedMemoryTally {
  std::uint64_t gpuFaultGroups{0};  ///< GPU page-fault records' values: fault groups per page
  std::uint64_t cpuFaults{0};       ///< CPU page-fault records: one fault each
  std::uint64_t h2dMigrations{0};   ///< Host-to-device transfer records
  std::uint64_t h2dBytes{0};        ///< Their values: bytes moved
  std::uint64_t d2hMigrations{0};   ///< Device-to-host transfer records
  std::uint64_t d2hBytes{0};        ///< Their values: bytes moved
  std::uint64_t transferNs{0};      ///< The transfers' end minus start, summed
  std::uint64_t thrashing{0};       ///< Thrashing records
  std::uint64_t endsBeforeStart{0}; ///< Transfers whose end CUPTI put before their start

  /** @brief Adds one record. A kind it does not count was configured elsewhere in the process. */
  void add(const CUpti_ActivityUnifiedMemoryCounter2& record) {
    switch (record.counterKind) {
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT:
      gpuFaultGroups += record.value;
      break;
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT:
      // The value is the faulting instruction's address, not a count.
      ++cpuFaults;
      break;
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
      ++h2dMigrations;
      h2dBytes += record.value;
      addTransferTime(record);
      break;
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
      ++d2hMigrations;
      d2hBytes += record.value;
      addTransferTime(record);
      break;
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING:
      ++thrashing;
      break;
    default:
      break;
    }
  }

  void addTransferTime(const CUpti_ActivityUnifiedMemoryCounter2& record) {
    if (record.end < record.start) {
      ++endsBeforeStart;
      return;
    }
    transferNs += record.end - record.start;
  }
};

// Aggregator state is global because CUPTI's buffer-completion callback is a
// free function. Guarded by a mutex; expected contention is low (one drain
// per measured window) and the mutex never appears in the hot path.
struct Aggregator {
  std::mutex mtx;
  std::vector<KernelRecord> records;
  UnifiedMemoryTally unifiedMemory;        ///< This window's unified-memory records
  std::size_t dropped{0};                  ///< Records CUPTI dropped in this window
  CUptiResult droppedCount{CUPTI_SUCCESS}; ///< The first failed count of them, if any
  CUptiResult walkError{CUPTI_SUCCESS};    ///< The first error that ended a buffer's walk early
  bool enabled{false};
  bool countsUnifiedMemory{false}; ///< The window recording the counters reads their records

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

  /** @brief Keeps the first error that left a buffer's records unread (call with mtx held). */
  void noteWalkError(CUptiResult ended) {
    if (walkError == CUPTI_SUCCESS) {
      walkError = ended;
    }
  }

  /** @brief Empties the window's records and counts (call with mtx held). */
  void clearWindow() {
    records.clear();
    unifiedMemory = {};
    dropped = 0;
    droppedCount = CUPTI_SUCCESS;
    walkError = CUPTI_SUCCESS;
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

/**
 * @brief Has this process use unified memory once, before CUPTI first
 *        records its counters here.
 *
 * In a process that never allocated managed memory, CUPTI switches
 * CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER on at once and takes about 2 s to
 * switch it off, every window; once the process has used unified memory, the
 * two take about 20 and 12 ms (CUDA 13.1, driver 580). One 4 KiB managed
 * allocation, freed at once, is that use. Built only where CUDA compiles this
 * file: the CUPTI stand-in's tests have no device.
 */
void engageUnifiedMemoryOnce() {
#if defined(__CUDACC__)
  static const bool ENGAGED = [] {
    void* page = nullptr;
    if (cudaMallocManaged(&page, 4096) == cudaSuccess) {
      cudaFree(page);
    } else {
      static_cast<void>(cudaGetLastError()); // no managed memory here: nothing to engage
    }
    return true;
  }();
  static_cast<void>(ENGAGED);
#endif
}

/** @brief The counts a window's records give, in the fields UnifiedMemoryProfile documents. */
UnifiedMemoryProfile profileOf(const UnifiedMemoryTally& tally) {
  UnifiedMemoryProfile profile;
  profile.pageFaults = tally.gpuFaultGroups + tally.cpuFaults;
  profile.h2dMigrations = tally.h2dMigrations;
  profile.d2hMigrations = tally.d2hMigrations;
  profile.migrationTimeUs = static_cast<double>(tally.transferNs) / 1000.0;
  profile.thrashingEvents = tally.thrashing;
  profile.h2dBytes = tally.h2dBytes;
  profile.d2hBytes = tally.d2hBytes;
  return profile;
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
    } else if (record->kind == CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER &&
               agg.countsUnifiedMemory) {
      agg.unifiedMemory.add(*reinterpret_cast<CUpti_ActivityUnifiedMemoryCounter2*>(record));
    }
  } while (status == CUPTI_SUCCESS);

  // CUPTI ends a buffer's records with CUPTI_ERROR_MAX_LIMIT_REACHED. Any
  // other error (CUPTI_ERROR_INVALID_KIND for an incomplete or invalid
  // record) leaves the rest of the buffer unread, so the window's records are
  // not complete, whatever was read before it.
  if (status != CUPTI_SUCCESS && status != CUPTI_ERROR_MAX_LIMIT_REACHED) {
    agg.noteWalkError(status);
  }

  std::free(buffer);
}

} // namespace

#endif // VERNIER_HAS_CUPTI

/** @brief What the collector says about itself; kept here, off the class's layout. */
struct CuptiCollector::Impl {
  std::string unavailableReason;    ///< Why the collector does not collect; empty while it can
  std::string windowProblem;        ///< What kept the last window's records from being complete
  bool unifiedMemoryAsked{false};   ///< collectUnifiedMemory(true)
  bool unifiedMemoryRunning{false}; ///< This window switched the counters' kind on
  std::string unifiedMemoryRefusal; ///< Why the last window that was to record them did not
  std::string unifiedMemoryStuck; ///< CUPTI's answer when the kind stayed on; no window asks again
  std::optional<UnifiedMemoryProfile> unifiedMemory; ///< The last window's counters

#if VERNIER_HAS_CUPTI
  /**
   * @brief Asks CUPTI for this window's unified-memory counters: the five
   *        kinds configured, then their activity kind switched on. A refusal
   *        leaves the window without them and says why; kernel records go on.
   */
  void startUnifiedMemory() {
    if (!unifiedMemoryStuck.empty()) {
      unifiedMemoryRefusal = unifiedMemoryStuck;
      return;
    }
    engageUnifiedMemoryOnce();
    CUpti_ActivityUnifiedMemoryCounterConfig config[std::size(UNIFIED_MEMORY_KINDS)];
    for (std::size_t i = 0; i < std::size(UNIFIED_MEMORY_KINDS); ++i) {
      // The scope and device id are deprecated (CUDA 7.0): the counters are
      // the process's, as the kernel records are.
      config[i].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
      config[i].kind = UNIFIED_MEMORY_KINDS[i];
      config[i].deviceId = 0;
      config[i].enable = 1;
    }
    const CUptiResult CONFIGURED = cuptiActivityConfigureUnifiedMemoryCounter(
        config, static_cast<uint32_t>(std::size(UNIFIED_MEMORY_KINDS)));
    if (CONFIGURED != CUPTI_SUCCESS) {
      unifiedMemoryRefusal =
          "CUPTI refused to configure the unified-memory counters (" + resultName(CONFIGURED) + ")";
      return;
    }
    {
      std::lock_guard<std::mutex> guard(aggregator().mtx);
      aggregator().countsUnifiedMemory = true;
    }
    const CUptiResult ENABLED = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER);
    if (ENABLED != CUPTI_SUCCESS) {
      {
        std::lock_guard<std::mutex> guard(aggregator().mtx);
        aggregator().countsUnifiedMemory = false;
      }
      unifiedMemoryRefusal =
          "CUPTI refused to record unified-memory counter activity (" + resultName(ENABLED) + ")";
      return;
    }
    unifiedMemoryRunning = true;
  }
#endif
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

void CuptiCollector::collectUnifiedMemory(bool on) noexcept { impl_->unifiedMemoryAsked = on; }

const std::optional<UnifiedMemoryProfile>& CuptiCollector::unifiedMemory() const noexcept {
  return impl_->unifiedMemory;
}

const std::string& CuptiCollector::unifiedMemoryRefusal() const noexcept {
  return impl_->unifiedMemoryRefusal;
}

const std::string& CuptiCollector::windowProblem() const noexcept { return impl_->windowProblem; }

void CuptiCollector::start() {
  if (running_)
    return;
  // A new window: nothing of the last one carries over, also when this one
  // collects nothing.
  stats_ = {};
  impl_->windowProblem.clear();
  impl_->unifiedMemory.reset();
  impl_->unifiedMemoryRefusal.clear();
  // A collector that stood down at construction, has no CUPTI, or was
  // refused by CUPTI is not available.
  if (!available_)
    return;
#if VERNIER_HAS_CUPTI
  {
    std::lock_guard<std::mutex> guard(aggregator().mtx);
    aggregator().clearWindow();
    aggregator().countsUnifiedMemory = false;
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
  if (impl_->unifiedMemoryAsked) {
    impl_->startUnifiedMemory();
  }
#endif
  running_ = true;
}

void CuptiCollector::stop() {
  if (!available_ || !running_)
    return;

#if VERNIER_HAS_CUPTI
  // 1 = force: partly filled buffers are handed over too. The flush comes
  // before either kind is switched off: switched off first, the
  // unified-memory kind loses the window's device-side records (the
  // host-to-device transfers and the GPU page faults).
  const CUptiResult FLUSHED = cuptiActivityFlushAll(1);
  const CUptiResult DISABLED = cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
  const CUptiResult UNIFIED_MEMORY_DISABLED =
      impl_->unifiedMemoryRunning ? cuptiActivityDisable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER)
                                  : CUPTI_SUCCESS;
  // Records dropped while CUPTI had no buffer to hand over are counted in the
  // same global queue the completion callback reads, but reach no callback.
  std::size_t droppedUnbuffered = 0;
  const CUptiResult COUNTED = cuptiActivityGetNumDroppedRecords(nullptr, 0, &droppedUnbuffered);

  // Aggregate under the mutex. A buffer CUPTI completes after this (when the
  // flush or the disable failed) finds the aggregator disabled and is freed
  // unread.
  std::vector<KernelRecord> snapshot;
  UnifiedMemoryTally unifiedMemory;
  std::size_t dropped = 0;
  CUptiResult droppedCount = CUPTI_SUCCESS;
  CUptiResult walkError = CUPTI_SUCCESS;
  {
    std::lock_guard<std::mutex> guard(aggregator().mtx);
    aggregator().noteDropped(COUNTED, droppedUnbuffered);
    aggregator().enabled = false;
    aggregator().countsUnifiedMemory = false;
    snapshot = std::move(aggregator().records);
    aggregator().records.clear();
    unifiedMemory = aggregator().unifiedMemory;
    dropped = aggregator().dropped;
    droppedCount = aggregator().droppedCount;
    walkError = aggregator().walkError;
  }

  // Kernel records still on would arrive between windows and be read into
  // the next one, so the collector collects no further window.
  if (DISABLED != CUPTI_SUCCESS) {
    available_ = false;
    impl_->unavailableReason =
        "CUPTI did not stop recording kernel activity (" + resultName(DISABLED) + ")";
  }
  // Unified-memory records still on reach no later window (the aggregator
  // reads them only in a window that switched their kind on itself), and no
  // later window switches it on: this one's counts were flushed first.
  if (UNIFIED_MEMORY_DISABLED != CUPTI_SUCCESS) {
    impl_->unifiedMemoryStuck = "CUPTI did not stop recording unified-memory counter activity (" +
                                resultName(UNIFIED_MEMORY_DISABLED) + ")";
  }

  // A count or a median from part of a window is not published: the window
  // reports no launch and no counter, and names what went wrong.
  if (FLUSHED != CUPTI_SUCCESS) {
    impl_->windowProblem =
        "CUPTI failed to flush its activity buffers (" + resultName(FLUSHED) + ")";
  } else if (droppedCount != CUPTI_SUCCESS) {
    impl_->windowProblem =
        "CUPTI could not count its dropped records (" + resultName(droppedCount) + ")";
  } else if (dropped > 0) {
    impl_->windowProblem = "CUPTI dropped " + std::to_string(dropped) +
                           (dropped == 1 ? " activity record" : " activity records");
  } else if (walkError != CUPTI_SUCCESS) {
    impl_->windowProblem =
        "CUPTI could not read all of its activity records (" + resultName(walkError) + ")";
  } else if (unifiedMemory.endsBeforeStart > 0) {
    impl_->windowProblem = "CUPTI recorded " + std::to_string(unifiedMemory.endsBeforeStart) +
                           (unifiedMemory.endsBeforeStart == 1
                                ? " unified-memory transfer that ends before it starts"
                                : " unified-memory transfers that end before they start");
  }
  // The counters need complete records, not a launch: a window that launched
  // nothing has them all.
  const bool COMPLETE = impl_->windowProblem.empty();
  if (COMPLETE && snapshot.empty()) {
    impl_->windowProblem = "CUPTI recorded no kernel launch";
  }
  if (COMPLETE && impl_->unifiedMemoryRunning) {
    impl_->unifiedMemory = profileOf(unifiedMemory);
  }
  impl_->unifiedMemoryRunning = false;

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
    aggregator().clearWindow();
  }
#endif
  stats_ = {};
  impl_->windowProblem.clear();
  impl_->unifiedMemory.reset();
  impl_->unifiedMemoryRefusal.clear();
}

} // namespace bench
} // namespace vernier
