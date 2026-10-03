/**
 * @file 02_NsightProfiler_Timing.cpp
 * @brief Demo 02's guard timing: CUDA events around batches of bare launches.
 *
 * Host code: it calls the CUDA runtime and the launch in the SAXPY example.
 * The stream and the two events belong to small scoped owners, as the SAXPY
 * example's buffers do, so a failure at any step, a thrown exception included,
 * releases exactly what had been created before it.
 */

#include "src/bench/demo/gpu/02_NsightProfiler_Timing.hpp"

#include "src/bench/demo/examples/saxpy/inc/Saxpy.hpp"
#include "src/bench/inc/PerfStats.hpp"

#include <cuda_runtime.h>

#include <vector>

namespace vernier {
namespace bench {
namespace demo {

namespace {

/* ----------------------------- Scoped owners ----------------------------- */

/** @brief A CUDA stream for the life of the owner; null when creation failed. */
class ScopedStream {
public:
  ScopedStream() {
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
      stream_ = nullptr;
    }
  }
  ~ScopedStream() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  ScopedStream(const ScopedStream&) = delete;
  ScopedStream& operator=(const ScopedStream&) = delete;

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
  cudaStream_t stream_ = nullptr;
};

/** @brief A CUDA event for the life of the owner; null when creation failed. */
class ScopedEvent {
public:
  ScopedEvent() {
    if (cudaEventCreate(&event_) != cudaSuccess) {
      event_ = nullptr;
    }
  }
  ~ScopedEvent() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }

  ScopedEvent(const ScopedEvent&) = delete;
  ScopedEvent& operator=(const ScopedEvent&) = delete;

  [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

private:
  cudaEvent_t event_ = nullptr;
};

} // namespace

/* --------------------------------- API --------------------------------- */

double medianLaunchUs(float a, const float* dX, float* dY, std::size_t n, int threadsPerBlock,
                      int launchesPerSample, int samples) {
  const ScopedStream STREAM;
  if (STREAM.get() == nullptr) {
    return -1.0;
  }
  const ScopedEvent START;
  if (START.get() == nullptr) {
    return -1.0;
  }
  const ScopedEvent STOP;
  if (STOP.get() == nullptr) {
    return -1.0;
  }

  std::vector<double> perLaunch;
  perLaunch.reserve(static_cast<std::size_t>(samples));
  launchSaxpy(a, dX, dY, n, threadsPerBlock, STREAM.get()); // untimed
  for (int sample = 0; sample < samples; ++sample) {
    cudaEventRecord(START.get(), STREAM.get());
    for (int launch = 0; launch < launchesPerSample; ++launch) {
      launchSaxpy(a, dX, dY, n, threadsPerBlock, STREAM.get());
    }
    cudaEventRecord(STOP.get(), STREAM.get());
    cudaEventSynchronize(STOP.get());
    float ms = 0.0F;
    cudaEventElapsedTime(&ms, START.get(), STOP.get());
    perLaunch.push_back(static_cast<double>(ms) * 1000.0 / launchesPerSample);
  }
  if (cudaGetLastError() != cudaSuccess) {
    return -1.0;
  }
  return summarize(perLaunch).median;
} // the events and the stream are released here, however the function leaves

} // namespace demo
} // namespace bench
} // namespace vernier
