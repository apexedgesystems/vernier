/**
 * @file 02_NsightProfiler_KernelColumns_Kernel.cu
 * @brief The SAXPY example's kernel source, compiled into walkthrough 19's
 *        column check, where its kernel can be named.
 *
 * The kernel sits in an unnamed namespace of SaxpyKernel.cu, so only that
 * translation unit can hand it to cudaFuncGetAttributes. Compiling the source
 * here gives the check the same kernel from the same source with the build's
 * flags: the registers and shared memory the demo binary's copy compiles to.
 * The check therefore links no example library: launchSaxpy is defined here.
 *
 * Test support for 02_NsightProfiler_KernelColumns_uTest.cpp; not part of the
 * demo.
 */

#include "src/bench/demo/examples/saxpy/src/SaxpyKernel.cu"

#include "src/bench/demo/gpu/utst/02_NsightProfiler_KernelColumns_Kernel.hpp"

namespace vernier {
namespace bench {
namespace demo {
namespace kernel_columns {

/* --------------------------------- API --------------------------------- */

cudaError_t saxpyKernelAttributes(cudaFuncAttributes& attributes) {
  return cudaFuncGetAttributes(&attributes, saxpyKernel);
}

} // namespace kernel_columns
} // namespace demo
} // namespace bench
} // namespace vernier
