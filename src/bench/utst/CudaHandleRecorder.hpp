#ifndef VERNIER_CUDAHANDLERECORDER_HPP
#define VERNIER_CUDAHANDLERECORDER_HPP
/**
 * @file CudaHandleRecorder.hpp
 * @brief Test helper: records the CUDA streams, events and device memory a
 *        process makes and releases, makes one step fail on request, and can
 *        show the process a second device that runs on the first.
 *
 * CudaHandleRecorder.cpp defines, in the test program itself, the CUDA runtime
 * calls that choose, pair and describe devices, make and release streams,
 * events and device memory, and record, copy and time on them. libbench_cuda
 * imports those calls from libcudart, and the dynamic linker binds an import
 * to the program's own definition before any library's, so the harness's
 * calls arrive here; each one is recorded and passed on to libcudart's own.
 * The harness has no hook for this. A call is defined, and passed on, under
 * the symbol the runtime's headers give it (CUDA 12's map
 * cudaGetDeviceProperties to cudaGetDeviceProperties_v2), the one the
 * harness's call is linked to; cudaHandleRecorderForwarding() shows that in
 * the running process.
 *
 * An emulated-peer recording (CudaDevices::EmulatedPeer) lets one GPU stand
 * in for two: the process sees a device 1 that runs on device 0. A thread
 * that makes device 1 current runs on device 0 and is told 1; devices 0 and 1
 * are each other's peers, and access between them is enabled once per
 * direction; a peer copy between them is a real copy on device 0. Each
 * stream, event and allocation made while recording belongs to the device
 * current in its thread, and, as CUDA's rules for two devices say, an event
 * is not recorded on a stream of the other device and two events of
 * different devices are not timed against each other: the call returns
 * cudaErrorInvalidResourceHandle, as the runtime answers for two contexts of
 * one device, without reaching it. Such an answer, "access already enabled", a
 * failed enable (CudaSetupFailure::PeerAccess, after which the pair stays
 * without access) and a failed switch to device 1
 * (CudaSetupFailure::PeerSwitch, after which the thread's device is unchanged)
 * are the thread's last error, as the runtime's own errors are, until
 * cudaGetLastError() reads it. The rules are the CUDA programming guide's;
 * that a driver applies them between two GPUs, and anything about a copy
 * between two GPUs, is outside what this shows.
 *
 * Test support for PerfGpuHarness_uTest.cu; not part of the benchmark library.
 * It needs ELF symbol binding: elsewhere available() is false.
 */

#include <cuda_runtime_api.h>

#include <string>
#include <vector>

namespace vernier {
namespace bench {
namespace test {

/* ----------------------------- Types ----------------------------- */

/** @brief The setup or timed step a recording makes fail, once. */
enum class CudaSetupFailure {
  None,             ///< Nothing fails
  FirstEvent,       ///< The first cudaEventCreate of the recording
  SecondEvent,      ///< The second cudaEventCreate of the recording
  Properties,       ///< The first cudaGetDeviceProperties of the recording
  FirstAllocation,  ///< The first cudaMalloc of the recording
  SecondAllocation, ///< The second cudaMalloc of the recording
  FirstStream,      ///< The first stream the recording sees made
  PeerCopy,         ///< The first cudaMemcpyPeerAsync of the recording
  ElapsedTime,      ///< The first cudaEventElapsedTime of the recording
  PeerAccess,       ///< The first enable between the emulated peers (cudaErrorTooManyPeers)
  PeerSwitch,       ///< The traced thread's first switch to device 1 (cudaErrorDevicesUnavailable)
};

/** @brief The devices a recording shows the process. */
enum class CudaDevices {
  Real,         ///< The runtime's own: every call is passed on unchanged
  EmulatedPeer, ///< Also a device 1 that runs on device 0 (see the file comment)
};

/**
 * @brief What a recording saw of the streams, events and allocations made
 *        while it ran. A handle leaves the counts when it is passed to its
 *        release call: whether libcudart then released it, or saw it twice, is
 *        not counted.
 */
struct CudaHandleCounts {
  int streamsMade = 0;       ///< Streams made while recording
  int eventsMade = 0;        ///< Events made while recording
  int streamsLeft = 0;       ///< Of the streams made while recording, those not passed to a destroy
  int eventsLeft = 0;        ///< Of the events made while recording, those not passed to a destroy
  int allocationsMade = 0;   ///< Device allocations (cudaMalloc) made while recording
  int allocationsLeft = 0;   ///< Of those, the ones not passed to cudaFree
  int releasedOffDevice = 0; ///< Streams, events and allocations made while recording and
                             ///< released with another device current
  int refusedCalls = 0;      ///< Records and elapsed-time reads an emulated-peer recording refused
  int peerAccessEnables = 0; ///< Peer-access enables between devices 0 and 1 it answered
};

/** @brief One of the recorder's runtime calls, as this process forwards it. */
struct CudaForwardedCall {
  std::string api;           ///< The call as code names it, e.g. "cudaGetDeviceProperties"
  std::string definedAs;     ///< The symbol the recorder's definition is exported under
  std::string definedIn;     ///< The file holding that definition (the test program)
  bool forwarded = false;    ///< The call has run, so its forward has been looked up
  std::string symbol;        ///< The symbol that lookup asked for
  bool resolvedHere = false; ///< This process resolves that symbol to the recorder's definition
  std::string forwardedTo;   ///< The symbol of the definition the lookup found
  std::string forwardFile;   ///< The file holding that definition (libcudart)
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
 * @brief Starts a recording: forgets the last one's counts and trace, shows
 *        the process @p devices, and makes @p failure fail once, with
 *        injectedCudaError() or the error its enumerator names. The calling
 *        thread's handle calls are traced.
 */
void startCudaHandleRecording(CudaSetupFailure failure, CudaDevices devices = CudaDevices::Real);

/** @brief The counts so far; the recording goes on. */
CudaHandleCounts cudaHandleCounts();

/**
 * @brief Ends the recording and returns its counts. Handles made while it ran
 *        and released later still count as released until the next start.
 */
CudaHandleCounts stopCudaHandleRecording();

/**
 * @brief The handle calls the thread that started the recording made while it
 *        ran, in order: each call's name, then its arguments in order. A
 *        handle made while recording is written as its name and the device it
 *        was made on (S1@0: the first stream, made on device 0; E for events,
 *        A for allocations), one made before as "unknown"; a release adds the
 *        device current then ("with 0 current"), and a call that failed ends
 *        with "-> " and its error's name.
 */
std::vector<std::string> cudaHandleTrace();

/** @brief What each traced cudaEventElapsedTime that succeeded returned (ms), in order. */
std::vector<float> cudaElapsedTimes();

/** @brief The error the failing call returns: cudaErrorMemoryAllocation. */
cudaError_t injectedCudaError();

/**
 * @brief The calls the recorder defines, each with the symbol it is exported
 *        under and, once it has run, the symbol its forward looked up and the
 *        definition found: the forward is right when all three name one
 *        symbol, this process resolves it to the recorder, and the definition
 *        found is in another file. Empty where available() is false for want
 *        of ELF binding.
 */
std::vector<CudaForwardedCall> cudaHandleRecorderForwarding();

} // namespace test
} // namespace bench
} // namespace vernier

#endif // VERNIER_CUDAHANDLERECORDER_HPP
