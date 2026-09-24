/**
 * @file SaxpyKernel.cu
 * @brief The SAXPY kernel and the bare launch every GPU version goes through.
 *
 * Device code only. The host code that drives the kernel, and owns what it
 * allocates, is in SaxpyGpu.cpp, which a host compiler can build and test.
 */

#include "src/bench/demo/examples/saxpy/inc/Saxpy.hpp"

#include <cuda_runtime.h>

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

} // namespace

/* --------------------------------- API --------------------------------- */

void launchSaxpy(float a, const float* dX, float* dY, std::size_t n, int threadsPerBlock,
                 void* stream) {
  const unsigned BLOCKS =
      static_cast<unsigned>((n + static_cast<std::size_t>(threadsPerBlock) - 1) /
                            static_cast<std::size_t>(threadsPerBlock));
  saxpyKernel<<<BLOCKS, threadsPerBlock, 0, static_cast<cudaStream_t>(stream)>>>(a, dX, dY, n);
}

} // namespace demo
} // namespace bench
} // namespace vernier
