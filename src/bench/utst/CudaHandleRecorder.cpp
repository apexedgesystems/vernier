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
 *
 * A call's symbol is the name the runtime's headers give it, which can differ
 * from the name the code writes: CUDA 12's headers map cudaGetDeviceProperties
 * to cudaGetDeviceProperties_v2, CUDA 13's keep it. A definition here is named
 * through those headers, like the harness's call, and its forward is looked
 * up under the same mapped name (VERNIER_RECORDER_SYMBOL), never a literal.
 */

#include "src/bench/utst/CudaHandleRecorder.hpp"

#if defined(__ELF__)
#include <dlfcn.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <vector>

/// The symbol of the runtime call @p api, as a string: @p api expanded by the
/// runtime's headers first, so a mapped name gives the symbol it maps to.
#define VERNIER_RECORDER_SYMBOL(api) VERNIER_RECORDER_TEXT(api)
#define VERNIER_RECORDER_TEXT(name) #name

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
  std::set<cudaStream_t> streams; ///< Made since the start, not yet passed to a destroy
  std::set<cudaEvent_t> events;   ///< Made since the start, not yet passed to a destroy
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

#if defined(__ELF__)

/* ----------------------------- The Forwards ----------------------------- */

/** @brief A forward a definition below looked up: the call, the symbol, what was found. */
struct Forward {
  const char* api;    ///< The call as the code names it
  const char* symbol; ///< The symbol the lookup asked for
  void* target;       ///< The definition the lookup found
};

/** @brief The forwards looked up so far, one per call that has run; never destroyed. */
struct Forwards {
  std::mutex mutex;
  std::vector<Forward> list;
};

Forwards& forwards() {
  static Forwards* const FORWARDS = new Forwards;
  return *FORWARDS;
}

#endif // __ELF__

} // namespace

/* --------------------------------- API --------------------------------- */

std::string cudaHandleRecorderUnavailableReason() {
#if defined(__ELF__)
  if (dlsym(RTLD_DEFAULT, VERNIER_RECORDER_SYMBOL(cudaStreamCreate)) !=
      reinterpret_cast<void*>(&::cudaStreamCreate)) {
    return std::string("this process resolves ") + VERNIER_RECORDER_SYMBOL(cudaStreamCreate) +
           " to another definition than the test program's, so the recorder sees none of "
           "the harness's calls";
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

std::vector<CudaForwardedCall> cudaHandleRecorderForwarding() {
  std::vector<CudaForwardedCall> calls;
#if defined(__ELF__)
  struct Intercepted {
    const char* api;
    void* definition;
  };
  // The definitions' addresses, through the headers' names like the calls.
#define VERNIER_RECORDER_INTERCEPTED(api)                                                          \
  Intercepted { #api, reinterpret_cast<void*>(&::api) }
  const Intercepted INTERCEPTED[] = {
      VERNIER_RECORDER_INTERCEPTED(cudaStreamCreate),
      VERNIER_RECORDER_INTERCEPTED(cudaStreamCreateWithPriority),
      VERNIER_RECORDER_INTERCEPTED(cudaStreamDestroy),
      VERNIER_RECORDER_INTERCEPTED(cudaEventCreate),
      VERNIER_RECORDER_INTERCEPTED(cudaEventDestroy),
      VERNIER_RECORDER_INTERCEPTED(cudaGetDeviceProperties),
  };
#undef VERNIER_RECORDER_INTERCEPTED
  std::vector<Forward> looked;
  {
    Forwards& f = forwards();
    const std::lock_guard<std::mutex> LOCK(f.mutex);
    looked = f.list;
  }
  for (const Intercepted& call : INTERCEPTED) {
    CudaForwardedCall out;
    out.api = call.api;
    Dl_info self{};
    if (dladdr(call.definition, &self) != 0 && self.dli_saddr == call.definition) {
      out.definedAs = (self.dli_sname != nullptr) ? self.dli_sname : "";
      out.definedIn = (self.dli_fname != nullptr) ? self.dli_fname : "";
    }
    for (const Forward& fwd : looked) {
      if (std::strcmp(fwd.api, call.api) != 0) {
        continue;
      }
      out.forwarded = true;
      out.symbol = fwd.symbol;
      out.resolvedHere = dlsym(RTLD_DEFAULT, fwd.symbol) == call.definition;
      Dl_info target{};
      if (dladdr(fwd.target, &target) != 0 && target.dli_saddr == fwd.target) {
        out.forwardedTo = (target.dli_sname != nullptr) ? target.dli_sname : "";
        out.forwardFile = (target.dli_fname != nullptr) ? target.dli_fname : "";
      }
      break;
    }
    calls.push_back(out);
  }
#endif
  return calls;
}

} // namespace test
} // namespace bench
} // namespace vernier

#if defined(__ELF__)

namespace {

namespace vt = vernier::bench::test;

/**
 * @brief The definition of the runtime call @p api that comes after this
 *        program's, looked up under @p symbol and kept in the forwards.
 */
template <typename Fn> Fn forwardTo(const char* api, const char* symbol) {
  void* fn = dlsym(RTLD_NEXT, symbol);
  if (fn == nullptr) {
    std::fprintf(stderr, "CudaHandleRecorder: no %s after the test program's\n", symbol);
    std::abort();
  }
  {
    vt::Forwards& f = vt::forwards();
    const std::lock_guard<std::mutex> LOCK(f.mutex);
    f.list.push_back(vt::Forward{api, symbol, fn});
  }
  return reinterpret_cast<Fn>(fn);
}

/// The definition @p api forwards to: the same symbol, in the next object
/// (libcudart), with the type the runtime's headers declare for it.
#define VERNIER_RECORDER_FORWARD(api)                                                              \
  forwardTo<decltype(&::api)>(#api, VERNIER_RECORDER_SYMBOL(api))

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
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaStreamCreate);
  const cudaError_t RESULT = REAL(pStream);
  if (RESULT == cudaSuccess) {
    madeStream(*pStream);
  }
  return RESULT;
}

extern "C" __attribute__((visibility("default"))) cudaError_t
cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaStreamCreateWithPriority);
  const cudaError_t RESULT = REAL(pStream, flags, priority);
  if (RESULT == cudaSuccess) {
    madeStream(*pStream);
  }
  return RESULT;
}

extern "C" __attribute__((visibility("default"))) cudaError_t
cudaStreamDestroy(cudaStream_t stream) {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaStreamDestroy);
  {
    vt::Record& r = vt::record();
    const std::lock_guard<std::mutex> LOCK(r.mutex);
    r.streams.erase(stream);
  }
  return REAL(stream);
}

extern "C" __attribute__((visibility("default"))) cudaError_t cudaEventCreate(cudaEvent_t* event) {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaEventCreate);
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
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaEventDestroy);
  {
    vt::Record& r = vt::record();
    const std::lock_guard<std::mutex> LOCK(r.mutex);
    r.events.erase(event);
  }
  return REAL(event);
}

extern "C" __attribute__((visibility("default"))) cudaError_t
cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaGetDeviceProperties);
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
