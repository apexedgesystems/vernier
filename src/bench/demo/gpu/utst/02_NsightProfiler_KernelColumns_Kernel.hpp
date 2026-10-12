#ifndef VERNIER_DEMO_02_KERNEL_COLUMNS_KERNEL_HPP
#define VERNIER_DEMO_02_KERNEL_COLUMNS_KERNEL_HPP
/**
 * @file 02_NsightProfiler_KernelColumns_Kernel.hpp
 * @brief The SAXPY kernel's compiled resource use, read from the example's own
 *        source, for walkthrough 19's column check.
 *
 * Test support for 02_NsightProfiler_KernelColumns_uTest.cpp; not part of the
 * demo.
 */

#include <cuda_runtime.h>

namespace vernier {
namespace bench {
namespace demo {
namespace kernel_columns {

/* --------------------------------- API --------------------------------- */

/**
 * @brief cudaFuncGetAttributes for the SAXPY kernel, as the example's
 *        SaxpyKernel.cu compiles with this build's flags.
 * @param attributes Filled on success.
 * @return The CUDA runtime's result.
 */
cudaError_t saxpyKernelAttributes(cudaFuncAttributes& attributes);

} // namespace kernel_columns
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_02_KERNEL_COLUMNS_KERNEL_HPP
