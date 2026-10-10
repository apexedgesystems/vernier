#ifndef VERNIER_DEMO_04_COMPUTESANITIZER_UNGUARDED_HPP
#define VERNIER_DEMO_04_COMPUTESANITIZER_UNGUARDED_HPP
/**
 * @file 04_ComputeSanitizerProfiler_Unguarded.hpp
 * @brief The SAXPY kernel without its bounds guard, for Compute Sanitizer
 *        to find.
 *
 * A copy of the shared example's kernel with its `if (I < n)` removed: a
 * grid rounds up to whole blocks, and without the guard the last block's
 * extra threads read and write past the end of both vectors. Private to
 * demo 04, launched only under compute-sanitizer. Do not copy this pattern;
 * it is here so the tool has something to find.
 */

#include <cstddef>

namespace vernier {
namespace bench {
namespace demo {
namespace sanitizer_demo {

/* --------------------------------- API --------------------------------- */

/**
 * @brief Launch the unguarded kernel: y = a*x + y by one thread per element,
 *        on the grid that covers @p n, with no check that a thread has an
 *        element.
 *
 * Every thread past @p n reads x and y beyond their ends and writes y there.
 * Whether the hardware notices depends on what follows the allocations; the
 * result is wrong or right by luck. Under compute-sanitizer's memcheck the
 * first such access is reported and the kernel stopped.
 *
 * @param a               Scalar multiplier.
 * @param dX              Device pointer to x, @p n elements.
 * @param dY              Device pointer to y, @p n elements, read and written.
 * @param n               Element count.
 * @param threadsPerBlock Block size; the grid is the smallest that covers @p n.
 * @param stream          A `cudaStream_t` as `void*`, as the example's launch
 *                        takes it.
 * @note NOT RT-safe (a kernel launch enqueues on a stream).
 */
void launchSaxpyUnguarded(float a, const float* dX, float* dY, std::size_t n, int threadsPerBlock,
                          void* stream);

} // namespace sanitizer_demo
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_04_COMPUTESANITIZER_UNGUARDED_HPP
