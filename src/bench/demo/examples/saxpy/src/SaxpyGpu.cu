/**
 * @file SaxpyGpu.cu
 * @brief One kernel, driven badly (G0) and properly (G1), plus the bare launch.
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

/* ----------------------------- Kernel ----------------------------- */

/** @brief One element per thread: y[i] = a * x[i] + y[i]. */
__global__ void saxpyKernel(float a, const float* x, float* y, std::size_t n) {
  const std::size_t I = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (I < n) {
    y[I] = a * x[I] + y[I];
  }
}

/** @brief Turn a failed CUDA call into an exception naming the call. */
void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

} // namespace

/* --------------------------------- API --------------------------------- */

void launchSaxpy(float a, const float* dX, float* dY, std::size_t n, int threadsPerBlock,
                 void* stream) {
  const unsigned BLOCKS =
      static_cast<unsigned>((n + static_cast<std::size_t>(threadsPerBlock) - 1) /
                            static_cast<std::size_t>(threadsPerBlock));
  saxpyKernel<<<BLOCKS, threadsPerBlock, 0, static_cast<cudaStream_t>(stream)>>>(a, dX, dY, n);
}

void saxpyG0(float a, const std::vector<float>& x, std::vector<float>& y) {
  const std::size_t N = x.size();
  const std::size_t BYTES = N * sizeof(float);

  float* dX = nullptr;
  float* dY = nullptr;
  check(cudaMalloc(&dX, BYTES), "cudaMalloc x"); // every call
  check(cudaMalloc(&dY, BYTES), "cudaMalloc y");
  // Pageable host memory: the copy cannot overlap anything and blocks here.
  check(cudaMemcpy(dX, x.data(), BYTES, cudaMemcpyHostToDevice), "HtoD x");
  check(cudaMemcpy(dY, y.data(), BYTES, cudaMemcpyHostToDevice), "HtoD y");
  launchSaxpy(a, dX, dY, N, /*threadsPerBlock=*/1, nullptr); // one thread per block
  check(cudaGetLastError(), "launch");
  check(cudaMemcpy(y.data(), dY, BYTES, cudaMemcpyDeviceToHost), "DtoH y");
  cudaFree(dY); // every call
  cudaFree(dX);
}

/* ----------------------------- SaxpyG1 ----------------------------- */

struct SaxpyG1::Impl {
  std::size_t n = 0;
  float* dX = nullptr;
  float* dY = nullptr;
  float* hX = nullptr;
  float* hY = nullptr;
  cudaStream_t stream = nullptr;
};

SaxpyG1::SaxpyG1(std::size_t n) : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;
  s.n = n;
  const std::size_t BYTES = n * sizeof(float);
  check(cudaMalloc(&s.dX, BYTES), "cudaMalloc x"); // once
  check(cudaMalloc(&s.dY, BYTES), "cudaMalloc y");
  check(cudaHostAlloc(&s.hX, BYTES, cudaHostAllocDefault), "pinned x");
  check(cudaHostAlloc(&s.hY, BYTES, cudaHostAllocDefault), "pinned y");
  check(cudaStreamCreate(&s.stream), "stream");
}

SaxpyG1::~SaxpyG1() {
  Impl& s = *impl_;
  cudaStreamDestroy(s.stream);
  cudaFreeHost(s.hY);
  cudaFreeHost(s.hX);
  cudaFree(s.dY);
  cudaFree(s.dX);
}

void SaxpyG1::apply(float a, const std::vector<float>& x, std::vector<float>& y) {
  Impl& s = *impl_;
  const std::size_t BYTES = s.n * sizeof(float);
  std::memcpy(s.hX, x.data(), BYTES);
  std::memcpy(s.hY, y.data(), BYTES);
  check(cudaMemcpyAsync(s.dX, s.hX, BYTES, cudaMemcpyHostToDevice, s.stream), "HtoD x");
  check(cudaMemcpyAsync(s.dY, s.hY, BYTES, cudaMemcpyHostToDevice, s.stream), "HtoD y");
  launchSaxpy(a, s.dX, s.dY, s.n, /*threadsPerBlock=*/256, s.stream);
  check(cudaGetLastError(), "launch");
  check(cudaMemcpyAsync(s.hY, s.dY, BYTES, cudaMemcpyDeviceToHost, s.stream), "DtoH y");
  check(cudaStreamSynchronize(s.stream), "sync"); // one wait, at the end
  std::memcpy(y.data(), s.hY, BYTES);
}

} // namespace demo
} // namespace bench
} // namespace vernier
