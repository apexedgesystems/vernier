#ifndef VERNIER_DEMO_EXAMPLES_SAXPY_HPP
#define VERNIER_DEMO_EXAMPLES_SAXPY_HPP
/**
 * @file Saxpy.hpp
 * @brief y = a*x + y, as a CPU loop and as two ways of driving one CUDA kernel.
 *
 * One multiply and one add per element, every element independent: nothing
 * about the arithmetic can be slow, so what a measurement finds is how the
 * GPU is being driven, not what it is being asked to compute. The three
 * versions produce the same answers (see the unit test), which is what makes
 * their timings comparable.
 */

#include <cstddef>
#include <memory>
#include <vector>

namespace vernier {
namespace bench {
namespace demo {

/* --------------------------------- API --------------------------------- */

/**
 * @brief The reference: a plain loop on one CPU core.
 * @param a Scalar multiplier.
 * @param x Input vector.
 * @param y Vector read and overwritten with a*x + y; same size as @p x.
 * @note RT-safe (no allocation, no I/O).
 */
void saxpyCpu(float a, const std::vector<float>& x, std::vector<float>& y);

/**
 * @brief The naive port: everything per call.
 *
 * Allocates device memory, copies x and y from pageable host memory, launches
 * with one thread per block, copies y back and frees, on every call.
 *
 * @param a Scalar multiplier.
 * @param x Input vector.
 * @param y Vector read and overwritten with a*x + y; same size as @p x.
 * @note NOT RT-safe (device allocation, synchronous copies).
 * @throws std::runtime_error when a CUDA call fails; the buffers it had
 *         allocated are freed first.
 */
void saxpyG0(float a, const std::vector<float>& x, std::vector<float>& y);

/**
 * @brief The same kernel driven properly: buffers once, 256 threads per block.
 *
 * Device buffers, pinned host staging and the stream are created once, so a
 * call is two asynchronous copies, a launch, one copy back and a single wait.
 *
 * @note NOT RT-safe (the constructor allocates; apply() copies and waits).
 */
class SaxpyG1 {
public:
  /**
   * @brief Allocate the device and pinned host buffers for @p n elements.
   * @throws std::runtime_error when a CUDA call fails; whatever had been
   *         acquired before the failure is released.
   */
  explicit SaxpyG1(std::size_t n);
  ~SaxpyG1();

  SaxpyG1(const SaxpyG1&) = delete;
  SaxpyG1& operator=(const SaxpyG1&) = delete;

  /**
   * @brief Compute a*x + y into @p y through the preallocated buffers.
   * @param a Scalar multiplier.
   * @param x Input vector of the size given to the constructor.
   * @param y Vector read and overwritten with a*x + y.
   * @throws std::runtime_error when a CUDA call fails.
   */
  void apply(float a, const std::vector<float>& x, std::vector<float>& y);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief The bare launch, for a benchmark that owns the buffers and the stream.
 *
 * This is what a `cudaKernel(...)` lambda calls: no allocation, no copy and no
 * synchronization, so the harness times the launch and nothing else.
 *
 * @param a               Scalar multiplier.
 * @param dX              Device pointer to x.
 * @param dY              Device pointer to y, read and written.
 * @param n               Element count.
 * @param threadsPerBlock Block size; the grid follows from @p n.
 * @param stream          A `cudaStream_t` as `void*`, so callers that do not
 *                        include the CUDA headers can still declare it.
 * @note NOT RT-safe (a kernel launch enqueues on a stream).
 */
void launchSaxpy(float a, const float* dX, float* dY, std::size_t n, int threadsPerBlock,
                 void* stream);

} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_EXAMPLES_SAXPY_HPP
