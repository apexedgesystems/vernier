/**
 * @file SaxpyGpu.cpp
 * @brief The two GPU versions, driven badly (G0) and properly (G1).
 *
 * Host code: it calls the CUDA runtime and the launch in SaxpyKernel.cu, and
 * owns everything it acquires. Each device buffer, pinned host buffer and
 * stream belongs to a small scoped owner that releases it when the owner goes,
 * so a failure at any step, including a failed acquisition part-way through
 * setting up G1, releases exactly what had been acquired before it.
 */

#include "src/bench/demo/examples/saxpy/inc/Saxpy.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace vernier {
namespace bench {
namespace demo {

namespace {

/* ----------------------------- Helpers ----------------------------- */

/** @brief Turn a failed CUDA call into an exception naming the call. */
void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

/* ----------------------------- Scoped owners ----------------------------- */

/** @brief Device memory for the life of the owner. */
class DeviceBuffer {
public:
  /** @throws std::runtime_error naming @p what when the allocation fails. */
  DeviceBuffer(std::size_t bytes, const char* what) { check(cudaMalloc(&ptr_, bytes), what); }
  ~DeviceBuffer() { cudaFree(ptr_); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] float* get() const noexcept { return ptr_; }

private:
  float* ptr_ = nullptr;
};

/** @brief Pinned (page-locked) host memory for the life of the owner. */
class PinnedBuffer {
public:
  /** @throws std::runtime_error naming @p what when the allocation fails. */
  PinnedBuffer(std::size_t bytes, const char* what) {
    check(cudaHostAlloc(&ptr_, bytes, cudaHostAllocDefault), what);
  }
  ~PinnedBuffer() { cudaFreeHost(ptr_); }

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  [[nodiscard]] float* get() const noexcept { return ptr_; }

private:
  float* ptr_ = nullptr;
};

/** @brief A CUDA stream for the life of the owner. */
class Stream {
public:
  /** @throws std::runtime_error naming @p what when creation fails. */
  explicit Stream(const char* what) { check(cudaStreamCreate(&stream_), what); }
  ~Stream() { cudaStreamDestroy(stream_); }

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
  cudaStream_t stream_ = nullptr;
};

} // namespace

/* --------------------------------- API --------------------------------- */

void saxpyG0(float a, const std::vector<float>& x, std::vector<float>& y) {
  const std::size_t N = x.size();
  const std::size_t BYTES = N * sizeof(float);

  const DeviceBuffer D_X(BYTES, "cudaMalloc x"); // every call
  const DeviceBuffer D_Y(BYTES, "cudaMalloc y");
  // Pageable host memory: the copy cannot overlap anything and blocks here.
  check(cudaMemcpy(D_X.get(), x.data(), BYTES, cudaMemcpyHostToDevice), "HtoD x");
  check(cudaMemcpy(D_Y.get(), y.data(), BYTES, cudaMemcpyHostToDevice), "HtoD y");
  launchSaxpy(a, D_X.get(), D_Y.get(), N, /*threadsPerBlock=*/1, nullptr); // one thread per block
  check(cudaGetLastError(), "launch");
  check(cudaMemcpy(y.data(), D_Y.get(), BYTES, cudaMemcpyDeviceToHost), "DtoH y");
} // both buffers are freed here, every call, however the function leaves

/* ----------------------------- SaxpyG1 ----------------------------- */

/**
 * @brief Everything G1 keeps between calls.
 *
 * Members are acquired in declaration order and released in reverse. When one
 * acquisition throws, the members already built are released and the rest
 * were never acquired.
 */
struct SaxpyG1::Impl {
  explicit Impl(std::size_t count)
      : n(count), dX(count * sizeof(float), "cudaMalloc x"),
        dY(count * sizeof(float), "cudaMalloc y"), hX(count * sizeof(float), "pinned x"),
        hY(count * sizeof(float), "pinned y"), stream("stream") {}

  std::size_t n;
  DeviceBuffer dX;
  DeviceBuffer dY;
  PinnedBuffer hX;
  PinnedBuffer hY;
  Stream stream;
};

SaxpyG1::SaxpyG1(std::size_t n) : impl_(std::make_unique<Impl>(n)) {} // buffers once

SaxpyG1::~SaxpyG1() = default;

void SaxpyG1::apply(float a, const std::vector<float>& x, std::vector<float>& y) {
  Impl& s = *impl_;
  const std::size_t BYTES = s.n * sizeof(float);
  std::memcpy(s.hX.get(), x.data(), BYTES);
  std::memcpy(s.hY.get(), y.data(), BYTES);
  check(cudaMemcpyAsync(s.dX.get(), s.hX.get(), BYTES, cudaMemcpyHostToDevice, s.stream.get()),
        "HtoD x");
  check(cudaMemcpyAsync(s.dY.get(), s.hY.get(), BYTES, cudaMemcpyHostToDevice, s.stream.get()),
        "HtoD y");
  launchSaxpy(a, s.dX.get(), s.dY.get(), s.n, /*threadsPerBlock=*/256, s.stream.get());
  check(cudaGetLastError(), "launch");
  check(cudaMemcpyAsync(s.hY.get(), s.dY.get(), BYTES, cudaMemcpyDeviceToHost, s.stream.get()),
        "DtoH y");
  check(cudaStreamSynchronize(s.stream.get()), "sync"); // one wait, at the end
  std::memcpy(y.data(), s.hY.get(), BYTES);
}

} // namespace demo
} // namespace bench
} // namespace vernier
