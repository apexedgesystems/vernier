#ifndef VERNIER_DEMO_GPU_02_NSIGHT_PROFILER_TIMING_HPP
#define VERNIER_DEMO_GPU_02_NSIGHT_PROFILER_TIMING_HPP
/**
 * @file 02_NsightProfiler_Timing.hpp
 * @brief Demo 02's own timing of the bare SAXPY kernel, for its launch-shape guard.
 *
 * Private to demo 02. It is kept out of the demo source so the host tests can
 * build it against the tracked stand-in runtime (examples/saxpy/utst/fake_cuda)
 * and check, on any machine, that the stream and events it acquires are
 * released however it leaves, an exception included.
 */

#include <cstddef>

namespace vernier {
namespace bench {
namespace demo {

/* --------------------------------- API --------------------------------- */

/**
 * @brief Median device microseconds per launch of the bare SAXPY kernel.
 *
 * One untimed launch, then @p samples batches of @p launchesPerSample launches
 * of @p threadsPerBlock threads per block on a stream of its own, each batch
 * timed by a CUDA event pair.
 *
 * @return The median over the batches, or a negative value when creating the
 *         stream or an event fails, or the launches report an error.
 * @throws std::bad_alloc when the host cannot allocate the sample buffer; the
 *         stream and both events are released first.
 * @note NOT RT-safe (creates a stream and two events, allocates).
 */
double medianLaunchUs(float a, const float* dX, float* dY, std::size_t n, int threadsPerBlock,
                      int launchesPerSample, int samples);

} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_GPU_02_NSIGHT_PROFILER_TIMING_HPP
