#ifndef VERNIER_CUDAHANDLERECORDER_HPP
#define VERNIER_CUDAHANDLERECORDER_HPP
/**
 * @file CudaHandleRecorder.hpp
 * @brief Test helper: records the CUDA streams and events a process makes and
 *        destroys, and makes one step of a device's setup fail on request.
 *
 * CudaHandleRecorder.cpp defines, in the test program itself, the CUDA runtime
 * calls that make and destroy streams and events and read a device's
 * properties. libbench_cuda imports those calls from libcudart, and the
 * dynamic linker binds an import to the program's own definition before any
 * library's, so the harness's calls arrive here; each one is recorded and
 * passed on to libcudart's own. The harness has no hook for this.
 *
 * Test support for PerfGpuHarness_uTest.cu; not part of the benchmark library.
 * It needs ELF symbol binding: elsewhere available() is false.
 */

#include <cuda_runtime_api.h>

#include <string>

namespace vernier {
namespace bench {
namespace test {

/* ----------------------------- Types ----------------------------- */

/** @brief The setup step a recording makes fail, once. */
enum class CudaSetupFailure {
  None,        ///< Nothing fails
  FirstEvent,  ///< The first cudaEventCreate of the recording
  SecondEvent, ///< The second cudaEventCreate of the recording
  Properties,  ///< The first cudaGetDeviceProperties of the recording
};

/** @brief What a recording saw of the streams and events made while it ran. */
struct CudaHandleCounts {
  int streamsMade = 0; ///< Streams made while recording
  int eventsMade = 0;  ///< Events made while recording
  int streamsLeft = 0; ///< Of the streams made while recording, those not destroyed
  int eventsLeft = 0;  ///< Of the events made while recording, those not destroyed
};

/* --------------------------------- API --------------------------------- */

/**
 * @brief True when the harness's runtime calls reach this recorder: the
 *        process resolves cudaStreamCreate to the test program's definition.
 */
bool cudaHandleRecorderAvailable();

/** @brief Why the recorder is not available; empty when it is. */
std::string cudaHandleRecorderUnavailableReason();

/**
 * @brief Starts a recording: forgets the last one's counts and makes @p failure
 *        fail once, with injectedCudaError().
 */
void startCudaHandleRecording(CudaSetupFailure failure);

/** @brief The counts so far; the recording goes on. */
CudaHandleCounts cudaHandleCounts();

/**
 * @brief Ends the recording and returns its counts. Handles made while it ran
 *        and destroyed later still count as destroyed until the next start.
 */
CudaHandleCounts stopCudaHandleRecording();

/** @brief The error the failing call returns: cudaErrorMemoryAllocation. */
cudaError_t injectedCudaError();

} // namespace test
} // namespace bench
} // namespace vernier

#endif // VERNIER_CUDAHANDLERECORDER_HPP
