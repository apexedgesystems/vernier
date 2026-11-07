/**
 * @file GpuMemoryStrategies_pTest.cu
 * @brief CUDA memory strategies, side by side and by the selected one
 *
 * ExplicitMemory, UnifiedMemory, PinnedMemory and MappedMemory each measure
 * one strategy with an allocation of their own, whatever --gpu-memory says:
 *  - Explicit: pageable host memory, copied to and from cudaMalloc() memory
 *  - Unified: cudaMallocManaged() memory the host and the kernel share
 *  - Pinned: page-locked host memory (cudaMallocHost()), copied to and from
 *    cudaMalloc() memory
 *  - Mapped: page-locked host memory the kernel reads and writes in place
 *    (cudaHostAlloc() with cudaHostAllocMapped)
 *
 * SelectedStrategy measures the strategy --gpu-memory selects (explicit
 * without the flag). The test reads the selection, not the harness: it reads
 * perf.gpuConfig().memStrategy and allocates its buffers by it with a helper
 * of its own (StrategyBuffer), declaring the copies the strategy needs. The
 * harness reads the strategy nowhere and allocates no test's memory, and the
 * choice asks for no unified-memory counters (--capture-um does). The case
 * prints the strategy it allocated by; its row compares the strategies on the
 * same work from one run to the next.
 *
 * Usage:
 *   @code{.sh}
 *   # The four strategies side by side, and the selected one (explicit)
 *   ./build/native-linux-release/bin/ptests/BenchmarkGPU_PTEST \
 *       --gtest_filter="GpuMemoryStrategies.*"
 *
 *   # The selected strategy alone, one CSV per strategy
 *   ./build/native-linux-release/bin/ptests/BenchmarkGPU_PTEST \
 *       --gtest_filter="GpuMemoryStrategies.SelectedStrategy" \
 *       --gpu-memory pinned --csv pinned.csv
 *   @endcode
 *
 * No case ranks the strategies: how their rows compare belongs to the
 * machine.
 *
 * @see PerfGpuCase
 * @see PerfGpuConfig::MemoryStrategy
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "src/bench/inc/Perf.hpp"
#include "src/bench/inc/PerfGpu.hpp"
#include "src/bench/ptst/gpu/UnifiedMemoryRefusal.hpp"

namespace ub = vernier::bench;

namespace {

/** @brief Simple computation kernel */
__global__ void computeKernel(const float* input, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = input[idx];
    // Some computation to make it non-trivial
    for (int i = 0; i < 10; ++i) {
      val = val * 0.99f + 0.01f;
    }
    output[idx] = val;
  }
}

using Strategy = ub::PerfGpuConfig::MemoryStrategy;

/// SelectedStrategy's buffers, MappedMemory's size: the mapped strategy's
/// launches reach the host's memory over the bus on a discrete GPU, and at this
/// size the default cycles and repeats finish in seconds.
constexpr int SELECTED_N = 64 * 1024;
constexpr size_t SELECTED_BYTES = SELECTED_N * sizeof(float);

/** @brief out[i] = 2 * in[i] + 1: exact in a float for the integers SelectedStrategy writes. */
__global__ void scaleKernel(const float* in, float* out, int n) {
  const int IDX = blockIdx.x * blockDim.x + threadIdx.x;
  if (IDX < n) {
    out[IDX] = 2.0f * in[IDX] + 1.0f;
  }
}

/**
 * @brief SelectedStrategy's allocation helper: n floats placed by one memory
 *        strategy, zeroed, and freed with the object however the test ends.
 *
 * host() is what the host writes or reads, kernel() what the kernel is given.
 * Explicit and Pinned give the kernel device memory and the host a buffer of
 * its own (pageable, or page-locked), which the test copies between
 * (needsCopy()); Unified and Mapped give both the same memory (managed, or
 * page-locked host memory the device maps), which nothing copies.
 */
class StrategyBuffer {
public:
  StrategyBuffer(Strategy strategy, int n) : strategy_(strategy) {
    const size_t BYTES = static_cast<size_t>(n) * sizeof(float);
    switch (strategy) {
    case Strategy::Explicit:
      pageable_.resize(n);
      host_ = pageable_.data();
      ok_ = cudaMalloc(&kernel_, BYTES) == cudaSuccess;
      break;
    case Strategy::Pinned:
      ok_ = cudaMallocHost(&host_, BYTES) == cudaSuccess &&
            cudaMalloc(&kernel_, BYTES) == cudaSuccess;
      break;
    case Strategy::Unified:
      ok_ = cudaMallocManaged(&host_, BYTES) == cudaSuccess;
      kernel_ = host_;
      break;
    case Strategy::Mapped:
      ok_ = cudaHostAlloc(&host_, BYTES, cudaHostAllocMapped) == cudaSuccess &&
            cudaHostGetDevicePointer(&kernel_, host_, 0) == cudaSuccess;
      break;
    }
    // Zeroed: the host's buffer by the host, a separate kernel buffer by CUDA.
    if (ok_) {
      std::fill_n(host_, n, 0.0f);
      if (needsCopy()) {
        ok_ = cudaMemset(kernel_, 0, BYTES) == cudaSuccess;
      }
    }
  }

  ~StrategyBuffer() {
    switch (strategy_) {
    case Strategy::Explicit: // the vector frees the host's buffer
      cudaFree(kernel_);
      break;
    case Strategy::Pinned:
      cudaFreeHost(host_);
      cudaFree(kernel_);
      break;
    case Strategy::Unified:
      cudaFree(host_);
      break;
    case Strategy::Mapped:
      cudaFreeHost(host_);
      break;
    }
  }

  StrategyBuffer(const StrategyBuffer&) = delete;
  StrategyBuffer& operator=(const StrategyBuffer&) = delete;

  /** @brief Every CUDA call succeeded. */
  [[nodiscard]] bool ok() const noexcept { return ok_; }
  /** @brief The host's buffer and the kernel's are separate, so the test copies between them. */
  [[nodiscard]] bool needsCopy() const noexcept {
    return strategy_ == Strategy::Explicit || strategy_ == Strategy::Pinned;
  }
  [[nodiscard]] float* host() const noexcept { return host_; }
  [[nodiscard]] float* kernel() const noexcept { return kernel_; }

private:
  Strategy strategy_;
  std::vector<float> pageable_; ///< Explicit's host buffer
  float* host_ = nullptr;
  float* kernel_ = nullptr;
  bool ok_ = false;
};

/** @brief The --gpu-memory value that selects @p strategy, and what SelectedStrategy allocates. */
const char* describe(Strategy strategy) {
  switch (strategy) {
  case Strategy::Explicit:
    return "explicit: pageable host buffers, copied to and from device buffers";
  case Strategy::Pinned:
    return "pinned: page-locked host buffers, copied to and from device buffers";
  case Strategy::Unified:
    return "unified: managed buffers the host and the kernel share, no copies";
  case Strategy::Mapped:
    return "mapped: page-locked host buffers the kernel reads and writes in place, no copies";
  }
  return "an unknown strategy";
}

/** @brief Why the device @p prop describes cannot use @p strategy; empty when it can. */
std::string unavailable(Strategy strategy, const cudaDeviceProp& prop) {
  if (strategy == Strategy::Unified && prop.managedMemory == 0) {
    return std::string("the unified strategy needs managed memory, which ") + prop.name +
           " does not support (managedMemory is 0)";
  }
  if (strategy == Strategy::Mapped && prop.canMapHostMemory == 0) {
    return std::string("the mapped strategy needs host memory the device can map, which ") +
           prop.name + " cannot (canMapHostMemory is 0)";
  }
  return {};
}

/** @brief What CUDA says the memory at @p p is. */
std::string kindOf(const void* p) {
  cudaPointerAttributes attributes{};
  if (cudaPointerGetAttributes(&attributes, p) != cudaSuccess) {
    return "unknown (cudaPointerGetAttributes failed)";
  }
  switch (attributes.type) {
  case cudaMemoryTypeUnregistered:
    return "unregistered";
  case cudaMemoryTypeHost:
    return "host";
  case cudaMemoryTypeDevice:
    return "device";
  case cudaMemoryTypeManaged:
    return "managed";
  }
  return "unknown";
}

/** @brief What SelectedStrategy expects of one strategy, whatever its helper does. */
struct Expected {
  const char* hostKind;   ///< What CUDA says host() is
  const char* kernelKind; ///< What CUDA says kernel() is
  size_t copiedBytes;     ///< The copies declared, each way
};

/** @brief What SelectedStrategy expects under @p strategy. */
Expected expectedOf(Strategy strategy) {
  switch (strategy) {
  case Strategy::Explicit:
    return {"unregistered", "device", SELECTED_BYTES};
  case Strategy::Pinned:
    return {"host", "device", SELECTED_BYTES};
  case Strategy::Unified:
    return {"managed", "managed", 0};
  case Strategy::Mapped:
    return {"host", "host", 0};
  }
  return {"", "", 0};
}

} // anonymous namespace

/**
 * @brief Explicit memory management test
 *
 * Validates explicit memory allocation with cudaMalloc and cudaMemcpy.
 * This is the standard CUDA memory management approach.
 *
 * @test ExplicitMemory
 *
 * Validates:
 *  - cudaMalloc works correctly
 *  - cudaMemcpy transfers work
 *  - Kernel execution works
 *  - Results are correct
 *
 * Expected performance:
 *  - The kernel as fast as PinnedMemory's; the copies slower, since pageable
 *    host memory is staged through a page-locked buffer
 */
PERF_GPU_TEST(GpuMemoryStrategies, ExplicitMemory) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 512 * 1024;
  const size_t SIZE = N * sizeof(float);

  // Allocate regular host memory
  std::vector<float> h_input(N, 1.0f);
  std::vector<float> h_output(N, 0.0f);

  // Allocate device memory explicitly
  float *d_input, *d_output;
  cudaMalloc(&d_input, SIZE);
  cudaMalloc(&d_output, SIZE);

  // Launch configuration
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid(blocksPerGrid);
  dim3 block(threadsPerBlock);

  // Warmup
  perf.cudaWarmup(
      [&](cudaStream_t s) { computeKernel<<<grid, block, 0, s>>>(d_input, d_output, N); });

  // Measure with explicit transfers
  auto result = perf.cudaKernel([&](cudaStream_t s) {
                      computeKernel<<<grid, block, 0, s>>>(d_input, d_output, N);
                    })
                    .withHostToDevice(h_input.data(), d_input, SIZE)
                    .withDeviceToHost(d_output, h_output.data(), SIZE)
                    .withLaunchConfig(grid, block)
                    .measure();

  // Validate execution
  EXPECT_GT(result.kernelTimeUs, 0.0) << "Kernel should execute";
  EXPECT_GT(result.transferTimeUs, 0.0) << "Transfers should occur";

  // Validate results
  EXPECT_GT(h_output[0], 0.0f) << "Output should be computed";
  EXPECT_LT(h_output[0], 2.0f) << "Output should be reasonable";

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
}

/**
 * @brief Unified Memory test
 *
 * Validates Unified Memory (managed memory) allocation and usage.
 * Unified Memory simplifies programming but may have performance trade-offs.
 *
 * @test UnifiedMemory
 *
 * Validates:
 *  - cudaMallocManaged works
 *  - Automatic migration works
 *  - Kernel can access managed memory
 *  - Results are correct
 *
 * Expected performance:
 *  - May be slower than explicit due to page migration
 */
PERF_GPU_TEST(GpuMemoryStrategies, UnifiedMemory) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 512 * 1024;
  const size_t SIZE = N * sizeof(float);

  // Allocate unified memory
  float *um_input, *um_output;
  cudaMallocManaged(&um_input, SIZE);
  cudaMallocManaged(&um_output, SIZE);

  // Initialize on host
  for (int i = 0; i < N; ++i) {
    um_input[i] = 1.0f;
    um_output[i] = 0.0f;
  }

  // Launch configuration
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid(blocksPerGrid);
  dim3 block(threadsPerBlock);

  // Warmup
  perf.cudaWarmup([&](cudaStream_t s) {
    computeKernel<<<grid, block, 0, s>>>(um_input, um_output, N);
    cudaStreamSynchronize(s);
  });

  // Measure - no explicit transfers needed
  auto result = perf.cudaKernel([&](cudaStream_t s) {
                      computeKernel<<<grid, block, 0, s>>>(um_input, um_output, N);
                    })
                    .withLaunchConfig(grid, block)
                    .measure();

  // Synchronize to ensure data available on host
  cudaDeviceSynchronize();

  // Validate execution
  EXPECT_GT(result.kernelTimeUs, 0.0) << "Kernel should execute";

  // Validate results (accessible directly on host)
  EXPECT_GT(um_output[0], 0.0f) << "Output should be computed";
  EXPECT_LT(um_output[0], 2.0f) << "Output should be reasonable";

  // Cleanup
  cudaFree(um_input);
  cudaFree(um_output);
}

/**
 * @brief Pinned (page-locked) host memory test
 *
 * Validates pinned host memory allocation which provides faster transfers
 * than regular pageable host memory.
 *
 * @test PinnedMemory
 *
 * Validates:
 *  - cudaMallocHost works
 *  - Pinned memory transfers are fast
 *  - Performance benefit over regular memory
 *
 * Expected performance:
 *  - Similar or better than explicit
 */
PERF_GPU_TEST(GpuMemoryStrategies, PinnedMemory) {
  UB_PERF_GPU_GUARD(perf);

  const int N = 512 * 1024;
  const size_t SIZE = N * sizeof(float);

  // Allocate pinned host memory
  float *h_input, *h_output;
  cudaMallocHost(&h_input, SIZE);
  cudaMallocHost(&h_output, SIZE);

  // Initialize
  for (int i = 0; i < N; ++i) {
    h_input[i] = 1.0f;
    h_output[i] = 0.0f;
  }

  // Allocate device memory
  float *d_input, *d_output;
  cudaMalloc(&d_input, SIZE);
  cudaMalloc(&d_output, SIZE);

  // Launch configuration
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid(blocksPerGrid);
  dim3 block(threadsPerBlock);

  // Warmup
  perf.cudaWarmup(
      [&](cudaStream_t s) { computeKernel<<<grid, block, 0, s>>>(d_input, d_output, N); });

  // Measure with pinned host memory
  auto result = perf.cudaKernel([&](cudaStream_t s) {
                      computeKernel<<<grid, block, 0, s>>>(d_input, d_output, N);
                    })
                    .withHostToDevice(h_input, d_input, SIZE)
                    .withDeviceToHost(d_output, h_output, SIZE)
                    .withLaunchConfig(grid, block)
                    .measure();

  // Validate execution
  EXPECT_GT(result.kernelTimeUs, 0.0) << "Kernel should execute";
  EXPECT_GT(result.transferTimeUs, 0.0) << "Transfers should occur";

  // Pinned memory should have reasonable transfer times
  const auto& transfers = result.stats.transfers;
  EXPECT_GT(transfers.h2dBandwidthGBs(), 1.0)
      << "Pinned memory should have decent transfer bandwidth";

  // Validate results
  EXPECT_GT(h_output[0], 0.0f) << "Output should be computed";

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFreeHost(h_input);
  cudaFreeHost(h_output);
}

/**
 * @brief Zero-copy mapped memory test
 *
 * Validates mapped memory where GPU accesses host memory directly.
 * Useful for small data or when transfers are impractical.
 *
 * @test MappedMemory
 *
 * Validates:
 *  - Mapped memory allocation works
 *  - GPU can access host memory
 *  - Results are correct
 *
 * Expected performance:
 *  - Slower for large data
 *  - Acceptable for small workloads
 */
PERF_GPU_TEST(GpuMemoryStrategies, MappedMemory) {
  UB_PERF_GPU_GUARD(perf);

  // Use smaller size for mapped memory
  const int N = 64 * 1024;
  const size_t SIZE = N * sizeof(float);

  // Allocate mapped host memory
  float *h_input, *h_output;
  cudaHostAlloc(&h_input, SIZE, cudaHostAllocMapped);
  cudaHostAlloc(&h_output, SIZE, cudaHostAllocMapped);

  // Initialize
  for (int i = 0; i < N; ++i) {
    h_input[i] = 1.0f;
    h_output[i] = 0.0f;
  }

  // Get device pointers
  float *d_input, *d_output;
  cudaHostGetDevicePointer(&d_input, h_input, 0);
  cudaHostGetDevicePointer(&d_output, h_output, 0);

  // Launch configuration
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid(blocksPerGrid);
  dim3 block(threadsPerBlock);

  // Warmup
  perf.cudaWarmup(
      [&](cudaStream_t s) { computeKernel<<<grid, block, 0, s>>>(d_input, d_output, N); });

  // Measure - GPU accesses host memory directly
  auto result = perf.cudaKernel([&](cudaStream_t s) {
                      computeKernel<<<grid, block, 0, s>>>(d_input, d_output, N);
                    })
                    .withLaunchConfig(grid, block)
                    .measure();

  // Validate execution
  EXPECT_GT(result.kernelTimeUs, 0.0) << "Kernel should execute";

  // Validate results (available immediately on host)
  EXPECT_GT(h_output[0], 0.0f) << "Output should be computed";

  // Cleanup
  cudaFreeHost(h_input);
  cudaFreeHost(h_output);
}

/**
 * @brief The strategy --gpu-memory selects, read by the test and allocated by
 *        its own helper
 *
 * @test SelectedStrategy
 *
 * The test, not the harness, consumes the selection: it reads
 * perf.gpuConfig().memStrategy (explicit without the flag), prints it, and
 * allocates its input and output with StrategyBuffer. Explicit and pinned
 * declare a copy in and a copy out, so each repeat times the copy in, the
 * launches and the copy out; unified and mapped declare none, the kernel
 * reaching the host's data itself: the first launch moves a managed input to
 * the device, and every launch reads and writes mapped host memory in place.
 * The host writes the input after the warmup, so for every strategy it starts
 * on the host.
 *
 * Validates:
 *  - CUDA reports the kinds of memory the strategy names, for the host's
 *    buffers and the kernel's
 *  - Every element the host reads back is right
 *  - The copies declared are the strategy's: the buffer's bytes each way for
 *    explicit and pinned, none for unified and mapped
 *  - Without --capture-um the result has no unified-memory counters; under it,
 *    where the request was honoured, the unified strategy's window moved at
 *    least the input to the device
 *
 * Skipped, with the reason, on a device without managed memory (unified) or
 * that cannot map host memory (mapped). No timing is asserted.
 */
PERF_GPU_TEST(GpuMemoryStrategies, SelectedStrategy) {
  PERF_GPU_GUARD(perf);

  // The test reads the selected strategy and allocates by it.
  const Strategy STRATEGY = perf.gpuConfig().memStrategy;
  std::printf("[%s] memory strategy %s\n", perf.testName().c_str(), describe(STRATEGY));

  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, perf.gpuConfig().deviceId), cudaSuccess);
  const std::string UNAVAILABLE = unavailable(STRATEGY, prop);
  if (!UNAVAILABLE.empty()) {
    GTEST_SKIP() << UNAVAILABLE;
  }

  StrategyBuffer in(STRATEGY, SELECTED_N);
  StrategyBuffer out(STRATEGY, SELECTED_N);
  ASSERT_TRUE(in.ok() && out.ok()) << "a CUDA call failed while placing the buffers";
  const Expected WANT = expectedOf(STRATEGY);
  for (const StrategyBuffer* buffer : {&in, &out}) {
    EXPECT_EQ(kindOf(buffer->host()), WANT.hostKind) << "the host's buffer";
    EXPECT_EQ(kindOf(buffer->kernel()), WANT.kernelKind) << "the kernel's buffer";
  }

  const dim3 BLOCK(256);
  const dim3 GRID((SELECTED_N + 255) / 256);
  const auto LAUNCH = [&](cudaStream_t s) {
    scaleKernel<<<GRID, BLOCK, 0, s>>>(in.kernel(), out.kernel(), SELECTED_N);
  };
  perf.cudaWarmup(LAUNCH);

  // The input starts on the host, whatever the strategy.
  for (int i = 0; i < SELECTED_N; ++i) {
    in.host()[i] = static_cast<float>(i);
  }

  // Explicit and pinned copy the input in and the output out in every repeat;
  // unified and mapped declare no copy.
  ub::CudaKernelBuilder kernel = perf.cudaKernel(LAUNCH);
  kernel.withLaunchConfig(GRID, BLOCK);
  if (in.needsCopy()) {
    kernel.withHostToDevice(in.host(), in.kernel(), SELECTED_BYTES)
        .withDeviceToHost(out.kernel(), out.host(), SELECTED_BYTES);
  }
  const ub::PerfGpuResult RESULT = kernel.measure();

  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  int wrong = -1;
  for (int i = 0; i < SELECTED_N && wrong < 0; ++i) {
    if (out.host()[i] != 2.0f * static_cast<float>(i) + 1.0f) {
      wrong = i;
    }
  }
  EXPECT_EQ(wrong, -1) << "first wrong element the host read";
  EXPECT_EQ(RESULT.stats.transfers.h2dBytes, WANT.copiedBytes) << "bytes copied to the device";
  EXPECT_EQ(RESULT.stats.transfers.d2hBytes, WANT.copiedBytes) << "bytes copied to the host";

  // The counters only on request. Where the request was honoured, the unified
  // strategy's window moved the input the host wrote to the device.
  if (!perf.gpuConfig().captureUnifiedMemory) {
    EXPECT_FALSE(RESULT.stats.unifiedMemory.has_value())
        << "unified-memory counters in a run that did not ask for them";
  } else if (STRATEGY == Strategy::Unified && ub::test::unifiedMemoryRefusal().empty()) {
    ASSERT_TRUE(RESULT.stats.unifiedMemory.has_value())
        << "no unified-memory counters, and the request was not unsupported";
    EXPECT_GE(RESULT.stats.unifiedMemory->h2dBytes, SELECTED_BYTES)
        << "less than the input the host wrote reached the device";
  }
}
