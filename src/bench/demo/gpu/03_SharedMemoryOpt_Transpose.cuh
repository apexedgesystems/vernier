#ifndef VERNIER_DEMO_03_SHARED_MEMORY_OPT_TRANSPOSE_CUH
#define VERNIER_DEMO_03_SHARED_MEMORY_OPT_TRANSPOSE_CUH
/**
 * @file 03_SharedMemoryOpt_Transpose.cuh
 * @brief Demo 03's three transpose kernels: through global memory alone,
 *        through a shared-memory tile whose column reads conflict, and
 *        through the same tile padded so they do not.
 *
 * Every kernel maps one thread to one element and reads the input along a
 * row, so its reads are coalesced. They differ in how the transposed
 * element reaches the output: written straight to its column, so a warp's
 * writes stride through the output; or staged through a tile in shared
 * memory and written along a row. The two tiled kernels are the same code
 * but for the tile's row pitch, and the pitch decides whether a warp's reads
 * of one tile column land in one bank or in thirty-two. All three write the
 * same answer for any dim (utst/03_SharedMemoryOpt_uTest.cu holds them to
 * it).
 *
 * Declarations for the demo and its tests; the kernels are in
 * 03_SharedMemoryOpt_Transpose.cu.
 */

#include "src/bench/demo/gpu/03_SharedMemoryOpt_Workload.hpp"

#include <cuda_runtime.h>

namespace vernier {
namespace bench {
namespace demo {
namespace shared_memory_demo {

/* ----------------------------- Kernels ----------------------------- */

/**
 * @brief Transpose through global memory alone: coalesced reads, and writes
 *        that stride through the output a column at a time.
 */
__global__ void transposeNaive(const float* input, float* output, int dim);

/**
 * @brief Transpose through a TILE_DIM x TILE_DIM tile in shared memory. The
 *        global reads and writes are both coalesced; every warp's reads of
 *        one tile column land in one shared-memory bank.
 */
__global__ void transposeSharedConflict(const float* input, float* output, int dim);

/**
 * @brief The same transpose through a tile whose rows are one float longer,
 *        so every warp's reads of one tile column land in TILE_DIM banks.
 */
__global__ void transposeSharedPadded(const float* input, float* output, int dim);

/* ----------------------------- Launch Shape ----------------------------- */

/** @brief One thread per tile element: a TILE_DIM x TILE_DIM block. */
inline dim3 transposeBlock() { return dim3(TILE_DIM, TILE_DIM); }

/** @brief Enough blocks to cover a dim x dim matrix, one tile each. */
inline dim3 transposeGrid(int dim) {
  const unsigned TILES = static_cast<unsigned>((dim + TILE_DIM - 1) / TILE_DIM);
  return dim3(TILES, TILES);
}

} // namespace shared_memory_demo
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_03_SHARED_MEMORY_OPT_TRANSPOSE_CUH
