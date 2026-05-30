/**
 * @file CuptiCollector.cu
 * @brief CUPTI Activity API implementation.
 *
 * Buffer-based model: register two callbacks; CUPTI fills our buffers with
 * activity records from kernel-launch threads; we walk the records in
 * stop() to populate the aggregate metrics.
 */

#include "src/bench/inc/CuptiCollector.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#if __has_include(<cupti.h>)
#define VERNIER_HAS_CUPTI 1
#include <cupti.h>
#else
#define VERNIER_HAS_CUPTI 0
#endif

namespace vernier {
namespace bench {

#if VERNIER_HAS_CUPTI

namespace {

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
  bool enabled{false};
};

Aggregator& aggregator() {
  static Aggregator g;
  return g;
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

extern "C" void CUPTIAPI cuptiBufferCompleted(CUcontext /*ctx*/, uint32_t /*streamId*/,
                                              uint8_t* buffer, size_t /*size*/, size_t validSize) {
  if (!buffer)
    return;

  Aggregator& agg = aggregator();
  CUpti_Activity* record = nullptr;
  CUptiResult status = CUPTI_SUCCESS;

  std::lock_guard<std::mutex> guard(agg.mtx);
  if (!agg.enabled) {
    std::free(buffer);
    return;
  }

  do {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status != CUPTI_SUCCESS || !record)
      break;

    // Tolerate both KERNEL and CONCURRENT_KERNEL activity kinds; payload
    // shape is identical for the fields we read.
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

struct CuptiCollector::Impl {
  bool subscribed{false};
};

CuptiCollector::CuptiCollector() {
  impl_ = new Impl();
  if (cuptiActivityRegisterCallbacks(cuptiBufferRequested, cuptiBufferCompleted) == CUPTI_SUCCESS) {
    aggregator().records.reserve(RECORD_RESERVE);
    available_ = true;
  }
}

CuptiCollector::~CuptiCollector() {
  if (running_)
    stop();
  delete impl_;
}

void CuptiCollector::start() {
  if (!available_ || running_)
    return;
  // Single-client gate: when VERNIER_DISABLE_CUPTI is set, skip in-process
  // CUPTI so an external Nsight session (nsys / ncu) can attach -- CUPTI
  // allows only one client per process. stop()/stats() stay safe no-ops
  // because running_ remains false.
  if (std::getenv("VERNIER_DISABLE_CUPTI") != nullptr)
    return;
  {
    std::lock_guard<std::mutex> guard(aggregator().mtx);
    aggregator().records.clear();
    aggregator().enabled = true;
  }
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  running_ = true;
}

void CuptiCollector::stop() {
  if (!available_ || !running_)
    return;

  cuptiActivityFlushAll(1); // 1 = force flush even partially-filled buffers
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
  cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);

  // Aggregate under the mutex; callbacks can no longer fire because the
  // activities are disabled and the buffers have been drained by flush.
  std::vector<KernelRecord> snapshot;
  {
    std::lock_guard<std::mutex> guard(aggregator().mtx);
    aggregator().enabled = false;
    snapshot = std::move(aggregator().records);
    aggregator().records.clear();
  }

  stats_ = {};
  stats_.kernelLaunches = snapshot.size();
  if (!snapshot.empty()) {
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

  running_ = false;
}

void CuptiCollector::reset() {
  std::lock_guard<std::mutex> guard(aggregator().mtx);
  aggregator().records.clear();
  stats_ = {};
}

#else // !VERNIER_HAS_CUPTI

struct CuptiCollector::Impl {};

CuptiCollector::CuptiCollector() { impl_ = nullptr; }
CuptiCollector::~CuptiCollector() {}
void CuptiCollector::start() {}
void CuptiCollector::stop() {}
void CuptiCollector::reset() {}

#endif // VERNIER_HAS_CUPTI

} // namespace bench
} // namespace vernier
