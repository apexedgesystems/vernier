#ifndef VERNIER_DEMO_03_SHARED_MEMORY_OPT_WORKLOAD_HPP
#define VERNIER_DEMO_03_SHARED_MEMORY_OPT_WORKLOAD_HPP
/**
 * @file 03_SharedMemoryOpt_Workload.hpp
 * @brief The shape of demo 03's transpose: the matrix, the tile, and the
 *        static shared memory each tiled kernel declares.
 *
 * Shared by the demo, its kernels (03_SharedMemoryOpt_Transpose.cuh) and the
 * checks beside it (utst/), so the sizes a check reasons about are the sizes
 * the demo runs. Plain C++; the CUDA declarations are in the .cuh.
 */

#include <cstddef>

namespace vernier {
namespace bench {
namespace demo {
namespace shared_memory_demo {

/* ----------------------------- Constants ----------------------------- */

/// One side of the square matrix, in floats: 4 MiB in, 4 MiB out.
constexpr int MATRIX_DIM = 1024;

/// One side of the square tile a thread block transposes: a 32 x 32 block,
/// one thread per element, 32 warps.
constexpr int TILE_DIM = 32;

/// Static shared memory transposeSharedConflict declares: one TILE_DIM x
/// TILE_DIM tile of floats.
constexpr std::size_t TILE_BYTES = static_cast<std::size_t>(TILE_DIM) * TILE_DIM * sizeof(float);

/// Static shared memory transposeSharedPadded declares: TILE_DIM rows of
/// TILE_DIM + 1 floats.
constexpr std::size_t TILE_PADDED_BYTES =
    static_cast<std::size_t>(TILE_DIM) * (TILE_DIM + 1) * sizeof(float);

} // namespace shared_memory_demo
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_03_SHARED_MEMORY_OPT_WORKLOAD_HPP
