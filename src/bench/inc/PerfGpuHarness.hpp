#ifndef VERNIER_PERFGPUHARNESS_HPP
#define VERNIER_PERFGPUHARNESS_HPP
/**
 * @file PerfGpuHarness.hpp
 * @brief GPU performance test harness with multi-GPU support and profiler hook integration.
 */

#include <functional>
#include <string>
#include <vector>
#include <memory>
#include <cuda_runtime.h>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfGpuConfig.hpp"
#include "src/bench/inc/PerfGpuStats.hpp"
#include "src/bench/inc/PerfHarness.hpp"
#include "src/bench/inc/PerfRegistry.hpp"
#include "src/bench/inc/Profiler.hpp"
#include "src/bench/inc/ProfilerNsight.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- Forward Declarations ----------------------------- */

class PerfGpuCaseImpl;

/* ----------------------------- Result Types ----------------------------- */

struct PerfGpuResult {
  GpuStats stats{};
  double kernelTimeUs{};
  double transferTimeUs{};
  double totalTimeUs{};
  double callsPerSecond{};
  double speedupVsCpu{};
  std::string label;
  int deviceId{-1}; // device tracking
};

struct MultiGpuResult {
  std::vector<PerfGpuResult> perDevice;
  GpuStats aggregatedStats{};
  double totalSpeedupVsCpu{};
  std::string label;
};

/* ----------------------------- CudaKernelBuilder ----------------------------- */

class CudaKernelBuilder {
public:
  using KernelFn = std::function<void(cudaStream_t)>;

  struct Transfer {
    const void* src;
    void* dst;
    size_t bytes;
  };

  inline CudaKernelBuilder(PerfGpuCaseImpl* impl, KernelFn kernel, std::string label);

  inline CudaKernelBuilder& withHostToDevice(const void* src, void* dst, size_t bytes);
  inline CudaKernelBuilder& withDeviceToHost(const void* src, void* dst, size_t bytes);
  inline CudaKernelBuilder& withLaunchConfig(dim3 grid, dim3 block, size_t sharedMemBytes = 0);
  inline CudaKernelBuilder& withDeviceId(int deviceId);

  PerfGpuResult measure();

private:
  PerfGpuCaseImpl* impl_;
  KernelFn kernel_;
  std::string label_;
  std::vector<Transfer> h2d_;
  std::vector<Transfer> d2h_;
  dim3 grid_{};
  dim3 block_{};
  size_t sharedMemBytes_ = 0;
  bool hasLaunchConfig_ = false;
  int deviceId_ = -1;

  friend class PerfGpuCaseImpl;
};

/* ----------------------------- MultiGpuKernelBuilder ----------------------------- */

class MultiGpuKernelBuilder {
public:
  using MultiGpuKernelFn = std::function<void(int deviceId, cudaStream_t stream)>;

  inline MultiGpuKernelBuilder(PerfGpuCaseImpl* impl, int deviceCount, MultiGpuKernelFn kernel,
                               std::string label);

  inline MultiGpuKernelBuilder& withLaunchConfig(dim3 grid, dim3 block, size_t sharedMemBytes = 0);
  inline MultiGpuKernelBuilder& withP2PAccess();
  inline MultiGpuKernelBuilder& measureP2PBandwidth(int srcDevice, int dstDevice, size_t bytes);

  MultiGpuResult measure();

private:
  PerfGpuCaseImpl* impl_;
  int deviceCount_;
  MultiGpuKernelFn kernel_;
  std::string label_;
  dim3 grid_{};
  dim3 block_{};
  size_t sharedMemBytes_ = 0;
  bool hasLaunchConfig_ = false;
  bool enableP2P_ = false;
  int p2pSrcDevice_ = -1;
  int p2pDstDevice_ = -1;
  size_t p2pTestBytes_ = 0;

  friend class PerfGpuCaseImpl;
};

/* ----------------------------- PerfGpuCase ----------------------------- */

class PerfGpuCase {
public:
  using CpuFn = std::function<void()>;
  using KernelFn = std::function<void(cudaStream_t)>;
  using MultiGpuKernelFn = std::function<void(int deviceId, cudaStream_t stream)>;
  using BeforeHook = std::function<void(const PerfGpuCase&)>;
  using AfterHook = std::function<void(const PerfGpuCase&, const GpuStats&)>;

  PerfGpuCase(std::string testName, PerfConfig cpuCfg);
  ~PerfGpuCase();

  PerfGpuCase(const PerfGpuCase&) = delete;
  PerfGpuCase& operator=(const PerfGpuCase&) = delete;

  PerfResult cpuBaseline(CpuFn fn, std::string label = "cpu_baseline");
  CudaKernelBuilder cudaKernel(KernelFn kernel, std::string label = "cuda_kernel");

  // Multi-GPU support
  MultiGpuKernelBuilder cudaKernelMultiGpu(int deviceCount, MultiGpuKernelFn kernel,
                                           std::string label = "multi_gpu_kernel");

  void cudaWarmup(KernelFn kernel);
  void setBeforeMeasureHook(BeforeHook h);
  void setAfterMeasureHook(AfterHook h);

  [[nodiscard]] int cycles() const noexcept;
  [[nodiscard]] int repeats() const noexcept;
  [[nodiscard]] const PerfConfig& cpuConfig() const noexcept;
  [[nodiscard]] const PerfGpuConfig& gpuConfig() const noexcept;
  [[nodiscard]] const std::string& testName() const noexcept;
  [[nodiscard]] cudaStream_t stream() const noexcept;

private:
  std::unique_ptr<PerfGpuCaseImpl> impl_;
};

/* ----------------------------- Inline Implementations ----------------------------- */

inline CudaKernelBuilder::CudaKernelBuilder(PerfGpuCaseImpl* impl, KernelFn kernel,
                                            std::string label)
    : impl_(impl), kernel_(std::move(kernel)), label_(std::move(label)) {}

inline CudaKernelBuilder& CudaKernelBuilder::withHostToDevice(const void* src, void* dst,
                                                              size_t bytes) {
  h2d_.push_back({src, dst, bytes});
  return *this;
}

inline CudaKernelBuilder& CudaKernelBuilder::withDeviceToHost(const void* src, void* dst,
                                                              size_t bytes) {
  d2h_.push_back({src, dst, bytes});
  return *this;
}

inline CudaKernelBuilder& CudaKernelBuilder::withLaunchConfig(dim3 grid, dim3 block,
                                                              size_t sharedMemBytes) {
  grid_ = grid;
  block_ = block;
  sharedMemBytes_ = sharedMemBytes;
  hasLaunchConfig_ = true;
  return *this;
}

inline CudaKernelBuilder& CudaKernelBuilder::withDeviceId(int deviceId) {
  deviceId_ = deviceId;
  return *this;
}

inline MultiGpuKernelBuilder::MultiGpuKernelBuilder(PerfGpuCaseImpl* impl, int deviceCount,
                                                    MultiGpuKernelFn kernel, std::string label)
    : impl_(impl), deviceCount_(deviceCount), kernel_(std::move(kernel)), label_(std::move(label)) {
}

inline MultiGpuKernelBuilder& MultiGpuKernelBuilder::withLaunchConfig(dim3 grid, dim3 block,
                                                                      size_t sharedMemBytes) {
  grid_ = grid;
  block_ = block;
  sharedMemBytes_ = sharedMemBytes;
  hasLaunchConfig_ = true;
  return *this;
}

inline MultiGpuKernelBuilder& MultiGpuKernelBuilder::withP2PAccess() {
  enableP2P_ = true;
  return *this;
}

inline MultiGpuKernelBuilder&
MultiGpuKernelBuilder::measureP2PBandwidth(int srcDevice, int dstDevice, size_t bytes) {
  p2pSrcDevice_ = srcDevice;
  p2pDstDevice_ = dstDevice;
  p2pTestBytes_ = bytes;
  return *this;
}

/* ----------------------------- GPU Profiler Hooks ----------------------------- */

/**
 * @brief Attach profiler hooks to a GPU test case.
 *
 * Supports Nsight Systems/Compute profiling via --profile nsight.
 */
inline void attachGpuProfilerHooks(PerfGpuCase& pc, const PerfConfig& cfg) {
  std::shared_ptr<Profiler> prof;

  if (cfg.profileTool == "nsight") {
    prof = std::shared_ptr<Profiler>(makeNsightProfiler(cfg, pc.testName()).release());
  } else {
    prof = std::shared_ptr<Profiler>(Profiler::make(cfg, pc.testName()).release());
  }

  pc.setBeforeMeasureHook([prof](const PerfGpuCase&) { prof->beforeMeasure(); });

  pc.setAfterMeasureHook([prof](const PerfGpuCase&, const GpuStats& s) {
    prof->afterMeasure(s.cpuStats);
    PerfRegistry::instance().updateProfileMeta(prof->toolName(), prof->artifactDir());
  });
}

} // namespace bench
} // namespace vernier

#endif // VERNIER_PERFGPUHARNESS_HPP