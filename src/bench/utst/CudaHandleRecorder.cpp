/**
 * @file CudaHandleRecorder.cpp
 * @brief The CUDA runtime calls that make and destroy streams and events and
 *        read a device's properties, defined in the test program: each one is
 *        recorded, or made to fail on request, and passed on to libcudart's own.
 *
 * A C++ file, not CUDA: nvcc defines cudaStreamDestroy and cudaEventDestroy
 * for device code itself (cuda_device_runtime_api.h), so a .cu file cannot
 * define them. Linked into TestBenchGpu, whose runtime is the shared libcudart:
 * libbench_cuda imports these calls, and the program's own definitions come
 * first in the dynamic linker's search.
 */

#include "src/bench/utst/CudaHandleRecorder.hpp"

#if defined(__ELF__)
#include <dlfcn.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>

namespace vernier {
namespace bench {
namespace test {

namespace {

/* ----------------------------- The Record ----------------------------- */

/** @brief What the current recording saw, and which call it makes fail. */
struct Record {
  std::mutex mutex;
  bool recording = false;
  CudaSetupFailure failure = CudaSetupFailure::None;
  int eventCreates = 0;           ///< cudaEventCreate calls since the start
  int propertyReads = 0;          ///< cudaGetDeviceProperties calls since the start
  int streamsMade = 0;            ///< Streams made since the start
  int eventsMade = 0;             ///< Events made since the start
  std::set<cudaStream_t> streams; ///< Made since the start, not yet destroyed
  std::set<cudaEvent_t> events;   ///< Made since the start, not yet destroyed
};

/** @brief The process's one record; never destroyed, since CUDA calls can come at exit. */
Record& record() {
  static Record* const RECORD = new Record;
  return *RECORD;
}

CudaHandleCounts countsOf(const Record& r) {
  return CudaHandleCounts{r.streamsMade, r.eventsMade, static_cast<int>(r.streams.size()),
                          static_cast<int>(r.events.size())};
}

} // namespace

/* --------------------------------- API --------------------------------- */

std::string cudaHandleRecorderUnavailableReason() {
#if defined(__ELF__)
  if (dlsym(RTLD_DEFAULT, "cudaStreamCreate") != reinterpret_cast<void*>(&::cudaStreamCreate)) {
    return "this process resolves cudaStreamCreate to another definition than the test "
           "program's, so the recorder sees none of the harness's calls";
  }
  return "";
#else
  return "the CUDA handle recorder needs ELF symbol binding";
#endif
}

bool cudaHandleRecorderAvailable() { return cudaHandleRecorderUnavailableReason().empty(); }

void startCudaHandleRecording(CudaSetupFailure failure) {
  Record& r = record();
  const std::lock_guard<std::mutex> LOCK(r.mutex);
  r.recording = true;
  r.failure = failure;
  r.eventCreates = 0;
  r.propertyReads = 0;
  r.streamsMade = 0;
  r.eventsMade = 0;
  r.streams.clear();
  r.events.clear();
}

CudaHandleCounts cudaHandleCounts() {
  Record& r = record();
  const std::lock_guard<std::mutex> LOCK(r.mutex);
  return countsOf(r);
}

CudaHandleCounts stopCudaHandleRecording() {
  Record& r = record();
  const std::lock_guard<std::mutex> LOCK(r.mutex);
  r.recording = false;
  r.failure = CudaSetupFailure::None;
  return countsOf(r);
}

cudaError_t injectedCudaError() { return cudaErrorMemoryAllocation; }

} // namespace test
} // namespace bench
} // namespace vernier

#if defined(__ELF__)

namespace {

namespace vt = vernier::bench::test;

/** @brief libcudart's definition of @p name: the one after this program's. */
template <typename Fn> Fn libcudart(const char* name) {
  void* fn = dlsym(RTLD_NEXT, name);
  if (fn == nullptr) {
    std::fprintf(stderr, "CudaHandleRecorder: no %s after the test program's\n", name);
    std::abort();
  }
  return reinterpret_cast<Fn>(fn);
}

/** @brief Counts a stream made while recording. */
void madeStream(cudaStream_t stream) {
  vt::Record& r = vt::record();
  const std::lock_guard<std::mutex> LOCK(r.mutex);
  if (r.recording) {
    ++r.streamsMade;
    r.streams.insert(stream);
  }
}

} // namespace

// The runtime calls. Each keeps libcudart's declaration and result; the
// recorder's lock is never held across a call into libcudart.

extern "C" __attribute__((visibility("default"))) cudaError_t
cudaStreamCreate(cudaStream_t* pStream) {
  static const auto REAL = libcudart<cudaError_t (*)(cudaStream_t*)>("cudaStreamCreate");
  const cudaError_t RESULT = REAL(pStream);
  if (RESULT == cudaSuccess) {
    madeStream(*pStream);
  }
  return RESULT;
}

extern "C" __attribute__((visibility("default"))) cudaError_t
cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) {
  static const auto REAL =
      libcudart<cudaError_t (*)(cudaStream_t*, unsigned int, int)>("cudaStreamCreateWithPriority");
  const cudaError_t RESULT = REAL(pStream, flags, priority);
  if (RESULT == cudaSuccess) {
    madeStream(*pStream);
  }
  return RESULT;
}

extern "C" __attribute__((visibility("default"))) cudaError_t
cudaStreamDestroy(cudaStream_t stream) {
  static const auto REAL = libcudart<cudaError_t (*)(cudaStream_t)>("cudaStreamDestroy");
  {
    vt::Record& r = vt::record();
    const std::lock_guard<std::mutex> LOCK(r.mutex);
    r.streams.erase(stream);
  }
  return REAL(stream);
}

extern "C" __attribute__((visibility("default"))) cudaError_t cudaEventCreate(cudaEvent_t* event) {
  static const auto REAL = libcudart<cudaError_t (*)(cudaEvent_t*)>("cudaEventCreate");
  bool fail = false;
  {
    vt::Record& r = vt::record();
    const std::lock_guard<std::mutex> LOCK(r.mutex);
    if (r.recording) {
      ++r.eventCreates;
      fail = (r.failure == vt::CudaSetupFailure::FirstEvent && r.eventCreates == 1) ||
             (r.failure == vt::CudaSetupFailure::SecondEvent && r.eventCreates == 2);
    }
  }
  if (fail) {
    return vt::injectedCudaError();
  }
  const cudaError_t RESULT = REAL(event);
  if (RESULT == cudaSuccess) {
    vt::Record& r = vt::record();
    const std::lock_guard<std::mutex> LOCK(r.mutex);
    if (r.recording) {
      ++r.eventsMade;
      r.events.insert(*event);
    }
  }
  return RESULT;
}

extern "C" __attribute__((visibility("default"))) cudaError_t cudaEventDestroy(cudaEvent_t event) {
  static const auto REAL = libcudart<cudaError_t (*)(cudaEvent_t)>("cudaEventDestroy");
  {
    vt::Record& r = vt::record();
    const std::lock_guard<std::mutex> LOCK(r.mutex);
    r.events.erase(event);
  }
  return REAL(event);
}

extern "C" __attribute__((visibility("default"))) cudaError_t
cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
  static const auto REAL =
      libcudart<cudaError_t (*)(cudaDeviceProp*, int)>("cudaGetDeviceProperties");
  bool fail = false;
  {
    vt::Record& r = vt::record();
    const std::lock_guard<std::mutex> LOCK(r.mutex);
    if (r.recording) {
      ++r.propertyReads;
      fail = r.failure == vt::CudaSetupFailure::Properties && r.propertyReads == 1;
    }
  }
  if (fail) {
    return vt::injectedCudaError();
  }
  return REAL(prop, device);
}

#endif // __ELF__
