#ifndef VERNIER_NVTXRANGERECORDER_HPP
#define VERNIER_NVTXRANGERECORDER_HPP
/**
 * @file NvtxRangeRecorder.hpp
 * @brief Test helper: a stand-in NVTX tool that records the ranges a process
 *        pushes and pops.
 *
 * nvtx3 hands each module's NVTX calls to the library the
 * NVTX_INJECTION64_PATH variable names, the way Nsight Systems receives them.
 * NvtxRangeRecorder is such a library. A test that links it and sets that
 * variable to the library's path before anything calls NVTX sees, here, every
 * range the process pushes and pops, libbench_cuda's included.
 *
 * Test support for ProfilerNsightRange_uTest.cu; not part of the benchmark
 * library.
 */

#include <cstddef>

#include <string>
#include <thread>
#include <vector>

namespace vernier {
namespace bench {
namespace test {

/* ----------------------------- NvtxRangeEvent ----------------------------- */

/** @brief One push or pop, as NVTX delivered it. */
struct NvtxRangeEvent {
  bool push = false;     ///< True for a push, false for a pop.
  std::string name;      ///< The pushed range's name; empty for a pop.
  std::thread::id where; ///< The thread that pushed or popped.
};

/* --------------------------------- API --------------------------------- */

/** @brief Every push and pop since the last clearNvtxRangeEvents(), in call order. */
std::vector<NvtxRangeEvent> nvtxRangeEvents();

/**
 * @brief How many pushes and pops have been recorded since the last clear.
 * @note Safe to call from any thread, a kernel launch's included.
 */
std::size_t nvtxRangeEventCount();

/** @brief Forget every event recorded so far. */
void clearNvtxRangeEvents();

/**
 * @brief How many modules have initialized NVTX with this recorder as their
 *        tool: zero while nothing has called NVTX, or when another tool was
 *        named.
 */
int nvtxRecorderModules();

} // namespace test
} // namespace bench
} // namespace vernier

#endif // VERNIER_NVTXRANGERECORDER_HPP
