/**
 * @file 04_ComputeSanitizerProfiler_Unguarded.cu
 * @brief The SAXPY kernel without its bounds guard, for Compute Sanitizer
 *        to find.
 *
 * Compiled with device line information whatever the build type, so the
 * tool's report names the line of the access. Private to demo 04; the
 * header says what it is for and not to copy it.
 */

#include "src/bench/demo/gpu/04_ComputeSanitizerProfiler_Unguarded.hpp"

#include <cuda_runtime.h>

namespace vernier {
namespace bench {
namespace demo {
namespace sanitizer_demo {

namespace {

/* ----------------------------- Kernel ----------------------------- */

/**
 * @brief One element per thread, with no check that the thread has one: on
 *        a grid that rounds up, the last block's extra threads run past the
 *        end of x and y.
 */
__global__ void saxpyUnguarded(float a, const float* x, float* y) {
  const std::size_t I = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  y[I] = a * x[I] + y[I]; // the shared kernel's `if (I < n)` is missing
}

} // namespace

/* --------------------------------- API --------------------------------- */

void launchSaxpyUnguarded(float a, const float* dX, float* dY, std::size_t n, int threadsPerBlock,
                          void* stream) {
  const unsigned BLOCKS =
      static_cast<unsigned>((n + static_cast<std::size_t>(threadsPerBlock) - 1) /
                            static_cast<std::size_t>(threadsPerBlock));
  saxpyUnguarded<<<BLOCKS, threadsPerBlock, 0, static_cast<cudaStream_t>(stream)>>>(a, dX, dY);
}

} // namespace sanitizer_demo
} // namespace demo
} // namespace bench
} // namespace vernier
