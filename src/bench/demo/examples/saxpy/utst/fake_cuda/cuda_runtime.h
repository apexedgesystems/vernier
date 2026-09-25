#ifndef VERNIER_DEMO_EXAMPLES_SAXPY_FAKE_CUDA_RUNTIME_H
#define VERNIER_DEMO_EXAMPLES_SAXPY_FAKE_CUDA_RUNTIME_H
/**
 * @file cuda_runtime.h
 * @brief A tracked stand-in for the CUDA runtime calls the SAXPY host code makes.
 *
 * Only for host tests: it lets the real SaxpyGpu.cpp, and demo 02's guard
 * timing (gpu/02_NsightProfiler_Timing.cpp), be built and run on a machine
 * without CUDA. Every acquisition (device memory, pinned memory, a stream, an
 * event) is host memory recorded in a live set and removed on release, and any
 * one acquisition, copy or launch check can be made to fail. Nothing here
 * models what the GPU computes; an event pair reports a set elapsed time.
 */

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <set>

/* ----------------------------- Types ----------------------------- */

using cudaError_t = int;
using cudaStream_t = void*;
using cudaEvent_t = void*;

enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 };

constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorInjected = 2;
constexpr unsigned cudaHostAllocDefault = 0;

/* ----------------------------- Tracking ----------------------------- */

namespace fake_cuda {

/** @brief What the fake has handed out, and which call is to fail. */
struct State {
  int acquisitions = 0;             ///< Acquisitions attempted so far (1-based count)
  int failAcquisition = 0;          ///< Fail the acquisition with this number (0: none)
  int copies = 0;                   ///< Copies attempted so far
  int failCopy = 0;                 ///< Fail the copy with this number (0: none)
  bool failNextLaunchCheck = false; ///< cudaGetLastError reports an error once
  int eventRecords = 0;             ///< cudaEventRecord calls so far
  float elapsedMs = 1.0F;           ///< What cudaEventElapsedTime reports
  /// Called with the acquisition count after each acquisition completes
  /// (null: none); a test can act at that exact point.
  void (*afterAcquisition)(int) = nullptr;
  std::set<void*> live; ///< Acquired and not yet released
};

/** @brief The one fake runtime state of the process. */
inline State& state() {
  static State s;
  return s;
}

/** @brief Forget everything; the next test starts from nothing live. */
inline void reset() {
  for (void* p : state().live) {
    std::free(p);
  }
  state() = State{};
}

/** @brief Hand out @p bytes of tracked memory, unless this is the call to fail. */
inline cudaError_t acquire(void** out, std::size_t bytes) {
  State& s = state();
  if (++s.acquisitions == s.failAcquisition) {
    return cudaErrorInjected;
  }
  *out = std::malloc(bytes != 0 ? bytes : 1);
  s.live.insert(*out);
  if (s.afterAcquisition != nullptr) {
    s.afterAcquisition(s.acquisitions);
  }
  return cudaSuccess;
}

/** @brief Give tracked memory back; releasing nothing is allowed, as in CUDA. */
inline cudaError_t release(void* p) {
  if (p != nullptr) {
    state().live.erase(p);
    std::free(p);
  }
  return cudaSuccess;
}

/** @brief Copy unless this is the copy to fail. */
inline cudaError_t copy(void* dst, const void* src, std::size_t bytes) {
  State& s = state();
  if (++s.copies == s.failCopy) {
    return cudaErrorInjected;
  }
  std::memcpy(dst, src, bytes);
  return cudaSuccess;
}

} // namespace fake_cuda

/* ----------------------------- Runtime calls ----------------------------- */

template <class T> cudaError_t cudaMalloc(T** ptr, std::size_t bytes) {
  return fake_cuda::acquire(reinterpret_cast<void**>(ptr), bytes);
}
template <class T> cudaError_t cudaHostAlloc(T** ptr, std::size_t bytes, unsigned /*flags*/) {
  return fake_cuda::acquire(reinterpret_cast<void**>(ptr), bytes);
}
inline cudaError_t cudaStreamCreate(cudaStream_t* stream) { return fake_cuda::acquire(stream, 1); }
inline cudaError_t cudaFree(void* p) { return fake_cuda::release(p); }
inline cudaError_t cudaFreeHost(void* p) { return fake_cuda::release(p); }
inline cudaError_t cudaStreamDestroy(cudaStream_t stream) { return fake_cuda::release(stream); }

inline cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind) {
  return fake_cuda::copy(dst, src, bytes);
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind,
                                   cudaStream_t) {
  return fake_cuda::copy(dst, src, bytes);
}
inline cudaError_t cudaGetLastError() {
  fake_cuda::State& s = fake_cuda::state();
  if (s.failNextLaunchCheck) {
    s.failNextLaunchCheck = false;
    return cudaErrorInjected;
  }
  return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaEventCreate(cudaEvent_t* event) { return fake_cuda::acquire(event, 1); }
inline cudaError_t cudaEventDestroy(cudaEvent_t event) { return fake_cuda::release(event); }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr) {
  ++fake_cuda::state().eventRecords;
  return cudaSuccess;
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) {
  *ms = fake_cuda::state().elapsedMs;
  return cudaSuccess;
}
inline const char* cudaGetErrorString(cudaError_t) { return "injected failure"; }

#endif // VERNIER_DEMO_EXAMPLES_SAXPY_FAKE_CUDA_RUNTIME_H
