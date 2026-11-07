/**
 * @file CudaHandleRecorder.cpp
 * @brief The CUDA runtime calls that choose, pair and describe devices, make
 *        and release streams, events and device memory, and record, copy and
 *        time on them, defined in the test program: each one is recorded, made
 *        to fail on request or emulated, and passed on to libcudart's own.
 *
 * A C++ file, not CUDA: nvcc defines several of these calls for device code
 * itself (cuda_device_runtime_api.h), so a .cu file cannot define them. Linked
 * into TestBenchGpu, whose runtime is the shared libcudart: libbench_cuda
 * imports these calls, and the program's own definitions come first in the
 * dynamic linker's search.
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

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
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

/** @brief A stream, event or allocation made while recording. */
struct Made {
  std::string name; ///< Its kind and place among them: "S1", "E2", "A1"
  int device = -1;  ///< The device current in its thread when it was made
};

/** @brief What the current recording saw, which call it makes fail, and what it shows. */
struct Record {
  std::mutex mutex;
  bool recording = false;
  CudaSetupFailure failure = CudaSetupFailure::None;
  CudaDevices devices = CudaDevices::Real;
  unsigned generation = 0;                 ///< Counts the starts: an emulated device belongs to one
  std::thread::id tracedThread;            ///< The thread that started the recording
  int eventCreates = 0;                    ///< cudaEventCreate calls since the start
  int propertyReads = 0;                   ///< cudaGetDeviceProperties calls since the start
  int streamCreates = 0;                   ///< Stream-making calls since the start
  int mallocs = 0;                         ///< cudaMalloc calls since the start
  int peerCopies = 0;                      ///< cudaMemcpyPeerAsync calls since the start
  int elapsedReads = 0;                    ///< cudaEventElapsedTime calls since the start
  int streamsMade = 0;                     ///< Streams made since the start
  int eventsMade = 0;                      ///< Events made since the start
  int allocationsMade = 0;                 ///< Allocations made since the start
  int releasedOffDevice = 0;               ///< Of all made since, released off their device
  int refusedCalls = 0;                    ///< Records and elapsed-time reads the emulation refused
  int peerAccessEnables = 0;               ///< Enables between the emulated peers it answered
  std::map<cudaStream_t, Made> streams;    ///< Made since the start, not yet passed to a destroy
  std::map<cudaEvent_t, Made> events;      ///< Made since the start, not yet passed to a destroy
  std::map<const void*, Made> allocations; ///< Made since the start, not yet passed to cudaFree
  std::set<std::pair<int, int>> peerAccess; ///< Emulated (device, peer) access enabled
  std::vector<std::string> trace;           ///< The traced thread's handle calls
  std::vector<float> elapsedMs;             ///< What its successful elapsed-time reads returned
};

/** @brief The process's one record; never destroyed, since CUDA calls can come at exit. */
Record& record() {
  static Record* const RECORD = new Record;
  return *RECORD;
}

CudaHandleCounts countsOf(const Record& r) {
  return CudaHandleCounts{r.streamsMade,
                          r.eventsMade,
                          static_cast<int>(r.streams.size()),
                          static_cast<int>(r.events.size()),
                          r.allocationsMade,
                          static_cast<int>(r.allocations.size()),
                          r.releasedOffDevice,
                          r.refusedCalls,
                          r.peerAccessEnables};
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

void startCudaHandleRecording(CudaSetupFailure failure, CudaDevices devices) {
  Record& r = record();
  const std::lock_guard<std::mutex> LOCK(r.mutex);
  r.recording = true;
  r.failure = failure;
  r.devices = devices;
  ++r.generation;
  r.tracedThread = std::this_thread::get_id();
  r.eventCreates = 0;
  r.propertyReads = 0;
  r.streamCreates = 0;
  r.mallocs = 0;
  r.peerCopies = 0;
  r.elapsedReads = 0;
  r.streamsMade = 0;
  r.eventsMade = 0;
  r.allocationsMade = 0;
  r.releasedOffDevice = 0;
  r.refusedCalls = 0;
  r.peerAccessEnables = 0;
  r.streams.clear();
  r.events.clear();
  r.allocations.clear();
  r.peerAccess.clear();
  r.trace.clear();
  r.elapsedMs.clear();
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

std::vector<std::string> cudaHandleTrace() {
  Record& r = record();
  const std::lock_guard<std::mutex> LOCK(r.mutex);
  return r.trace;
}

std::vector<float> cudaElapsedTimes() {
  Record& r = record();
  const std::lock_guard<std::mutex> LOCK(r.mutex);
  return r.elapsedMs;
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
      VERNIER_RECORDER_INTERCEPTED(cudaGetDevice),
      VERNIER_RECORDER_INTERCEPTED(cudaSetDevice),
      VERNIER_RECORDER_INTERCEPTED(cudaDeviceCanAccessPeer),
      VERNIER_RECORDER_INTERCEPTED(cudaDeviceEnablePeerAccess),
      VERNIER_RECORDER_INTERCEPTED(cudaMalloc),
      VERNIER_RECORDER_INTERCEPTED(cudaFree),
      VERNIER_RECORDER_INTERCEPTED(cudaMemcpyPeerAsync),
      VERNIER_RECORDER_INTERCEPTED(cudaEventRecord),
      VERNIER_RECORDER_INTERCEPTED(cudaEventElapsedTime),
      VERNIER_RECORDER_INTERCEPTED(cudaGetLastError),
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

/* ----------------------------- The Emulated Peer ----------------------------- */

/// The device an emulated-peer recording adds, and the device it runs on.
constexpr int EMULATED_DEVICE = 1;
constexpr int RUNTIME_DEVICE = 0;

/// What the emulation answers for an event and a stream, or two events, of
/// different devices: the runtime's answer for two contexts of one device.
constexpr cudaError_t REFUSED = cudaErrorInvalidResourceHandle;

/// What a failed enable between the emulated peers answers: the runtime's
/// answer when the resources peer access needs are exhausted.
constexpr cudaError_t ENABLE_FAILED = cudaErrorTooManyPeers;

/** @brief The device a thread made current in one emulated-peer recording. */
struct ThreadDevice {
  unsigned generation = 0; ///< The recording it was made current in
  int device = -1;         ///< The device, as the thread named it
};

thread_local ThreadDevice threadDevice;

/** @brief An error the emulation answered in the runtime's place, not yet read. */
struct AnsweredError {
  unsigned generation = 0;         ///< The recording it was answered in
  cudaError_t error = cudaSuccess; ///< The answer
};

thread_local AnsweredError answeredError;

/** @brief @p device as the runtime knows it: the emulated device runs on device 0. */
int runtimeDevice(int device) { return (device == EMULATED_DEVICE) ? RUNTIME_DEVICE : device; }

/** @brief True for devices 0 and 1, in either order: the emulated peers. */
bool emulatedPeers(int device, int peer) {
  return (device == RUNTIME_DEVICE && peer == EMULATED_DEVICE) ||
         (device == EMULATED_DEVICE && peer == RUNTIME_DEVICE);
}

/* ----------------------------- The Recording, As A Call Sees It ----------------------------- */

/** @brief The recording as one call sees it, read under the lock. */
struct RecordingView {
  bool recording = false;  ///< A recording runs
  bool emulating = false;  ///< An emulated-peer recording runs
  bool traced = false;     ///< The calling thread started it: its handle calls are traced
  unsigned generation = 0; ///< Which recording
};

RecordingView view() {
  vt::Record& r = vt::record();
  const std::lock_guard<std::mutex> LOCK(r.mutex);
  RecordingView m;
  m.recording = r.recording;
  m.emulating = r.recording && r.devices == vt::CudaDevices::EmulatedPeer;
  m.traced = r.recording && r.tracedThread == std::this_thread::get_id();
  m.generation = r.generation;
  return m;
}

/** @brief libcudart's cudaGetDevice, for the definition below and the recorder's own questions. */
cudaError_t runtimeGetDevice(int* device) {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaGetDevice);
  return REAL(device);
}

/** @brief libcudart's cudaGetLastError, for the definition below and the emulation's answers. */
cudaError_t runtimeGetLastError() {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaGetLastError);
  return REAL();
}

/**
 * @brief Returns @p error, answered in the runtime's place, and leaves it as
 *        the calling thread's last error, as the runtime leaves its own: an
 *        older error in the runtime's slot gives way to it, and one the
 *        runtime records after it is the newer.
 */
cudaError_t answer(const RecordingView& m, cudaError_t error) {
  static_cast<void>(runtimeGetLastError());
  answeredError = AnsweredError{m.generation, error};
  return error;
}

/** @brief The device current in the calling thread, as @p m shows it; -1 when unknown. */
int currentDevice(const RecordingView& m) {
  if (m.emulating && threadDevice.generation == m.generation && threadDevice.device >= 0) {
    return threadDevice.device;
  }
  int device = -1;
  return (runtimeGetDevice(&device) == cudaSuccess) ? device : -1;
}

/**
 * @brief Counts a call of one kind while recording; true when the recording
 *        makes this one fail: @p first fails the first such call, @p second
 *        the second.
 */
bool failsHere(int vt::Record::* calls, vt::CudaSetupFailure first,
               vt::CudaSetupFailure second = vt::CudaSetupFailure::None) {
  vt::Record& r = vt::record();
  const std::lock_guard<std::mutex> LOCK(r.mutex);
  if (!r.recording) {
    return false;
  }
  const int CALL = ++(r.*calls);
  return (r.failure == first && CALL == 1) ||
         (second != vt::CudaSetupFailure::None && r.failure == second && CALL == 2);
}

/** @brief " -> <error name>" after a call that failed; nothing after one that succeeded. */
std::string outcome(cudaError_t result) {
  return (result == cudaSuccess) ? std::string() : std::string(" -> ") + cudaGetErrorName(result);
}

/** @brief "S1@0": a handle's name and the device it was made on. */
std::string label(const vt::Made& made) { return made.name + "@" + std::to_string(made.device); }

/** @brief How the trace writes @p handle: its label, "null", or "unknown" if made before. */
template <typename Handle>
std::string labelOf(const std::map<Handle, vt::Made>& made, Handle handle) {
  if (handle == nullptr) {
    return "null";
  }
  const auto IT = made.find(handle);
  return (IT == made.end()) ? "unknown" : label(IT->second);
}

/** @brief True when @p a and @p b were both made while recording, on different devices. */
template <typename HandleA, typename HandleB>
bool onDifferentDevices(const std::map<HandleA, vt::Made>& madeA, HandleA a,
                        const std::map<HandleB, vt::Made>& madeB, HandleB b) {
  const auto A = madeA.find(a);
  const auto B = madeB.find(b);
  return A != madeA.end() && B != madeB.end() && A->second.device != B->second.device;
}

/** @brief Adds @p line to the trace when the calling thread's calls are traced. */
void trace(const RecordingView& m, const std::string& line) {
  if (!m.traced) {
    return;
  }
  vt::Record& r = vt::record();
  const std::lock_guard<std::mutex> LOCK(r.mutex);
  if (r.recording && r.generation == m.generation) {
    r.trace.push_back(line);
  }
}

/** @brief Records @p handle, just made on @p device by @p api, under the next name of its kind. */
template <typename Handle>
void madeHandle(const RecordingView& m, std::map<Handle, vt::Made> vt::Record::* all,
                int vt::Record::* made, char kind, Handle handle, int device, const char* api) {
  vt::Made entry;
  {
    vt::Record& r = vt::record();
    const std::lock_guard<std::mutex> LOCK(r.mutex);
    if (!r.recording) {
      return;
    }
    entry = vt::Made{kind + std::to_string(++(r.*made)), device};
    (r.*all)[handle] = entry;
  }
  trace(m, std::string(api) + " " + label(entry));
}

/**
 * @brief Forgets @p handle, passed to @p api with @p current current: one made
 *        while recording and released with another device current is counted.
 */
template <typename Handle>
void releasedHandle(const RecordingView& m, std::map<Handle, vt::Made> vt::Record::* all,
                    Handle handle, int current, const char* api) {
  std::string what;
  {
    vt::Record& r = vt::record();
    const std::lock_guard<std::mutex> LOCK(r.mutex);
    std::map<Handle, vt::Made>& made = r.*all;
    if (m.traced) {
      what = labelOf(made, handle);
    }
    const auto IT = made.find(handle);
    if (IT != made.end()) {
      if (IT->second.device != current) {
        ++r.releasedOffDevice;
      }
      made.erase(IT);
    }
  }
  if (m.traced) {
    trace(m, std::string(api) + " " + what + " with " + std::to_string(current) + " current");
  }
}

/** @brief Makes a stream through @p make unless the recording fails this one, and records it. */
template <typename Make> cudaError_t madeStream(const char* api, cudaStream_t* pStream, Make make) {
  const RecordingView VIEW = view();
  if (!VIEW.recording) {
    return make();
  }
  const cudaError_t RESULT =
      failsHere(&vt::Record::streamCreates, vt::CudaSetupFailure::FirstStream)
          ? vt::injectedCudaError()
          : make();
  if (RESULT == cudaSuccess) {
    madeHandle(VIEW, &vt::Record::streams, &vt::Record::streamsMade, 'S', *pStream,
               currentDevice(VIEW), api);
  } else {
    trace(VIEW, api + outcome(RESULT));
  }
  return RESULT;
}

} // namespace

/* ----------------------------- The Runtime Calls ----------------------------- */

// Each keeps libcudart's declaration and result, and passes straight on when
// no recording runs; the recorder's lock is never held across a call into
// libcudart.

extern "C" __attribute__((visibility("default"))) cudaError_t
cudaStreamCreate(cudaStream_t* pStream) {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaStreamCreate);
  return madeStream("cudaStreamCreate", pStream, [&] { return REAL(pStream); });
}

extern "C" __attribute__((visibility("default"))) cudaError_t
cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaStreamCreateWithPriority);
  return madeStream("cudaStreamCreateWithPriority", pStream,
                    [&] { return REAL(pStream, flags, priority); });
}

extern "C" __attribute__((visibility("default"))) cudaError_t
cudaStreamDestroy(cudaStream_t stream) {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaStreamDestroy);
  const RecordingView VIEW = view();
  releasedHandle(VIEW, &vt::Record::streams, stream, currentDevice(VIEW), "cudaStreamDestroy");
  return REAL(stream);
}

extern "C" __attribute__((visibility("default"))) cudaError_t cudaEventCreate(cudaEvent_t* event) {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaEventCreate);
  const RecordingView VIEW = view();
  if (!VIEW.recording) {
    return REAL(event);
  }
  const cudaError_t RESULT = failsHere(&vt::Record::eventCreates, vt::CudaSetupFailure::FirstEvent,
                                       vt::CudaSetupFailure::SecondEvent)
                                 ? vt::injectedCudaError()
                                 : REAL(event);
  if (RESULT == cudaSuccess) {
    madeHandle(VIEW, &vt::Record::events, &vt::Record::eventsMade, 'E', *event, currentDevice(VIEW),
               "cudaEventCreate");
  } else {
    trace(VIEW, "cudaEventCreate" + outcome(RESULT));
  }
  return RESULT;
}

extern "C" __attribute__((visibility("default"))) cudaError_t cudaEventDestroy(cudaEvent_t event) {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaEventDestroy);
  const RecordingView VIEW = view();
  releasedHandle(VIEW, &vt::Record::events, event, currentDevice(VIEW), "cudaEventDestroy");
  return REAL(event);
}

extern "C" __attribute__((visibility("default"))) cudaError_t
cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaGetDeviceProperties);
  if (failsHere(&vt::Record::propertyReads, vt::CudaSetupFailure::Properties)) {
    return vt::injectedCudaError();
  }
  return REAL(prop, device);
}

extern "C" __attribute__((visibility("default"))) cudaError_t cudaGetDevice(int* device) {
  const RecordingView VIEW = view();
  if (VIEW.emulating && device != nullptr && threadDevice.generation == VIEW.generation &&
      threadDevice.device >= 0) {
    *device = threadDevice.device;
    return cudaSuccess;
  }
  return runtimeGetDevice(device);
}

extern "C" __attribute__((visibility("default"))) cudaError_t cudaSetDevice(int device) {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaSetDevice);
  const RecordingView VIEW = view();
  if (!VIEW.emulating) {
    return REAL(device);
  }
  const cudaError_t RESULT = REAL(runtimeDevice(device));
  if (RESULT == cudaSuccess) {
    threadDevice = ThreadDevice{VIEW.generation, device};
  }
  return RESULT;
}

extern "C" __attribute__((visibility("default"))) cudaError_t
cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice) {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaDeviceCanAccessPeer);
  const RecordingView VIEW = view();
  if (!VIEW.emulating) {
    return REAL(canAccessPeer, device, peerDevice);
  }
  if (canAccessPeer != nullptr && emulatedPeers(device, peerDevice)) {
    *canAccessPeer = 1;
    return cudaSuccess;
  }
  return REAL(canAccessPeer, runtimeDevice(device), runtimeDevice(peerDevice));
}

extern "C" __attribute__((visibility("default"))) cudaError_t
cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaDeviceEnablePeerAccess);
  const RecordingView VIEW = view();
  if (!VIEW.emulating) {
    return REAL(peerDevice, flags);
  }
  const int DEVICE = currentDevice(VIEW);
  if (flags != 0 || !emulatedPeers(DEVICE, peerDevice)) {
    return REAL(runtimeDevice(peerDevice), flags);
  }
  bool fails = false;
  bool first = false;
  {
    vt::Record& r = vt::record();
    const std::lock_guard<std::mutex> LOCK(r.mutex);
    const int CALL = ++r.peerAccessEnables;
    fails = r.failure == vt::CudaSetupFailure::PeerAccess && CALL == 1;
    first = !fails && r.peerAccess.insert({DEVICE, peerDevice}).second;
  }
  if (fails) {
    return answer(VIEW, ENABLE_FAILED);
  }
  return first ? cudaSuccess : answer(VIEW, cudaErrorPeerAccessAlreadyEnabled);
}

extern "C" __attribute__((visibility("default"))) cudaError_t cudaMalloc(void** devPtr,
                                                                         size_t size) {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaMalloc);
  const RecordingView VIEW = view();
  if (!VIEW.recording) {
    return REAL(devPtr, size);
  }
  const cudaError_t RESULT = failsHere(&vt::Record::mallocs, vt::CudaSetupFailure::FirstAllocation,
                                       vt::CudaSetupFailure::SecondAllocation)
                                 ? vt::injectedCudaError()
                                 : REAL(devPtr, size);
  if (RESULT == cudaSuccess) {
    madeHandle(VIEW, &vt::Record::allocations, &vt::Record::allocationsMade, 'A',
               static_cast<const void*>(*devPtr), currentDevice(VIEW), "cudaMalloc");
  } else {
    trace(VIEW, "cudaMalloc" + outcome(RESULT));
  }
  return RESULT;
}

extern "C" __attribute__((visibility("default"))) cudaError_t cudaFree(void* devPtr) {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaFree);
  const RecordingView VIEW = view();
  releasedHandle(VIEW, &vt::Record::allocations, static_cast<const void*>(devPtr),
                 currentDevice(VIEW), "cudaFree");
  return REAL(devPtr);
}

extern "C" __attribute__((visibility("default"))) cudaError_t cudaMemcpyPeerAsync(
    void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream) {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaMemcpyPeerAsync);
  const RecordingView VIEW = view();
  if (!VIEW.recording) {
    return REAL(dst, dstDevice, src, srcDevice, count, stream);
  }
  std::string line;
  {
    vt::Record& r = vt::record();
    const std::lock_guard<std::mutex> LOCK(r.mutex);
    line = "cudaMemcpyPeerAsync " + labelOf(r.allocations, static_cast<const void*>(dst)) + " " +
           std::to_string(dstDevice) + " " + labelOf(r.allocations, src) + " " +
           std::to_string(srcDevice) + " " + std::to_string(count) + " " +
           labelOf(r.streams, stream);
  }
  cudaError_t result = vt::injectedCudaError();
  if (!failsHere(&vt::Record::peerCopies, vt::CudaSetupFailure::PeerCopy)) {
    result = VIEW.emulating
                 ? REAL(dst, runtimeDevice(dstDevice), src, runtimeDevice(srcDevice), count, stream)
                 : REAL(dst, dstDevice, src, srcDevice, count, stream);
  }
  trace(VIEW, line + outcome(result));
  return result;
}

extern "C" __attribute__((visibility("default"))) cudaError_t cudaEventRecord(cudaEvent_t event,
                                                                              cudaStream_t stream) {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaEventRecord);
  const RecordingView VIEW = view();
  if (!VIEW.recording) {
    return REAL(event, stream);
  }
  std::string line;
  bool refused = false;
  {
    vt::Record& r = vt::record();
    const std::lock_guard<std::mutex> LOCK(r.mutex);
    line = "cudaEventRecord " + labelOf(r.events, event) + " " + labelOf(r.streams, stream);
    refused = VIEW.emulating && onDifferentDevices(r.events, event, r.streams, stream);
    if (refused) {
      ++r.refusedCalls;
    }
  }
  const cudaError_t RESULT = refused ? answer(VIEW, REFUSED) : REAL(event, stream);
  trace(VIEW, line + outcome(RESULT));
  return RESULT;
}

extern "C" __attribute__((visibility("default"))) cudaError_t
cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
  static const auto REAL = VERNIER_RECORDER_FORWARD(cudaEventElapsedTime);
  const RecordingView VIEW = view();
  if (!VIEW.recording) {
    return REAL(ms, start, end);
  }
  const bool INJECTED = failsHere(&vt::Record::elapsedReads, vt::CudaSetupFailure::ElapsedTime);
  std::string line;
  bool refused = false;
  {
    vt::Record& r = vt::record();
    const std::lock_guard<std::mutex> LOCK(r.mutex);
    line = "cudaEventElapsedTime " + labelOf(r.events, start) + " " + labelOf(r.events, end);
    refused = !INJECTED && VIEW.emulating && onDifferentDevices(r.events, start, r.events, end);
    if (refused) {
      ++r.refusedCalls;
    }
  }
  const cudaError_t RESULT = INJECTED  ? vt::injectedCudaError()
                             : refused ? answer(VIEW, REFUSED)
                                       : REAL(ms, start, end);
  trace(VIEW, line + outcome(RESULT));
  if (VIEW.traced && RESULT == cudaSuccess && ms != nullptr) {
    vt::Record& r = vt::record();
    const std::lock_guard<std::mutex> LOCK(r.mutex);
    if (r.recording && r.generation == VIEW.generation) {
      r.elapsedMs.push_back(*ms);
    }
  }
  return RESULT;
}

extern "C" __attribute__((visibility("default"))) cudaError_t cudaGetLastError() {
  const RecordingView VIEW = view();
  const cudaError_t RUNTIME = runtimeGetLastError();
  if (!VIEW.emulating || answeredError.generation != VIEW.generation ||
      answeredError.error == cudaSuccess) {
    return RUNTIME;
  }
  const cudaError_t ANSWERED = std::exchange(answeredError.error, cudaSuccess);
  return (RUNTIME != cudaSuccess) ? RUNTIME : ANSWERED;
}

#endif // __ELF__
