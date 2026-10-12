/**
 * @file NvtxRangeRecorder.cpp
 * @brief A stand-in NVTX tool: records every range the process pushes and pops.
 *
 * When a module first calls NVTX, nvtx3 loads the library NVTX_INJECTION64_PATH
 * names and calls its InitializeInjectionNvtx2, which may replace the module's
 * NVTX functions with its own. This one replaces the push and the pop with
 * functions that append to one process-wide log. The process links this library
 * and nvtx3 opens the same file, so both reach the same log.
 *
 * Built as a shared library of its own, for ProfilerNsightRange_uTest.cu.
 */

#include "src/bench/utst/NvtxRangeRecorder.hpp"

#include <nvtx3/nvToolsExt.h>

#include <atomic>
#include <mutex>

namespace vernier {
namespace bench {
namespace test {

namespace {

/* ----------------------------- The Log ----------------------------- */

std::mutex& logMutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<NvtxRangeEvent>& eventLog() {
  static std::vector<NvtxRangeEvent> events;
  return events;
}

std::atomic<std::size_t> eventCount{0};
std::atomic<int> modulesAttached{0};

/// Ranges open on this thread, which NVTX reports back as a push's level.
thread_local int openRanges = 0;

/* ----------------------------- NVTX Functions ----------------------------- */

int NVTX_API recordPushA(const char* message) {
  const std::lock_guard<std::mutex> LOCK(logMutex());
  eventLog().push_back({true, message != nullptr ? message : "", std::this_thread::get_id()});
  eventCount.fetch_add(1);
  return openRanges++;
}

int NVTX_API recordPop() {
  const std::lock_guard<std::mutex> LOCK(logMutex());
  eventLog().push_back({false, "", std::this_thread::get_id()});
  eventCount.fetch_add(1);
  return --openRanges;
}

} // namespace

/* --------------------------------- API --------------------------------- */

std::vector<NvtxRangeEvent> nvtxRangeEvents() {
  const std::lock_guard<std::mutex> LOCK(logMutex());
  return eventLog();
}

std::size_t nvtxRangeEventCount() { return eventCount.load(); }

void clearNvtxRangeEvents() {
  const std::lock_guard<std::mutex> LOCK(logMutex());
  eventLog().clear();
  eventCount.store(0);
}

int nvtxRecorderModules() { return modulesAttached.load(); }

} // namespace test
} // namespace bench
} // namespace vernier

/* ----------------------------- NVTX Entry Point ----------------------------- */

/**
 * @brief Called by nvtx3 once per module that calls NVTX, with that module's
 *        export table: installs the recording push and pop in its function
 *        table. Returning 0 leaves the module's NVTX calls as no-ops.
 */
extern "C" __attribute__((visibility("default"))) int
InitializeInjectionNvtx2(NvtxGetExportTableFunc_t getExportTable) {
  const auto* callbacks =
      static_cast<const NvtxExportTableCallbacks*>(getExportTable(NVTX_ETID_CALLBACKS));
  NvtxFunctionTable table = nullptr;
  unsigned int size = 0;
  if (callbacks == nullptr ||
      callbacks->GetModuleFunctionTable(NVTX_CB_MODULE_CORE, &table, &size) == 0 ||
      table == nullptr || size <= NVTX_CBID_CORE_RangePop) {
    return 0;
  }
  *table[NVTX_CBID_CORE_RangePushA] =
      reinterpret_cast<NvtxFunctionPointer>(&vernier::bench::test::recordPushA);
  *table[NVTX_CBID_CORE_RangePop] =
      reinterpret_cast<NvtxFunctionPointer>(&vernier::bench::test::recordPop);
  vernier::bench::test::modulesAttached.fetch_add(1);
  return 1;
}
