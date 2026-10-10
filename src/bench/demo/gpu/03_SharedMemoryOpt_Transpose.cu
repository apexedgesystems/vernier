/**
 * @file 03_SharedMemoryOpt_Transpose.cu
 * @brief Demo 03's transpose kernels, declared in 03_SharedMemoryOpt_Transpose.cuh.
 *
 * Device code only. The demo launches these; its tests hold them to a CPU
 * transpose and read what Nsight Compute counts in them.
 */

#include "src/bench/demo/gpu/03_SharedMemoryOpt_Transpose.cuh"

namespace vernier {
namespace bench {
namespace demo {
namespace shared_memory_demo {

/* ----------------------------- Kernels ----------------------------- */

__global__ void transposeNaive(const float* input, float* output, int dim) {
  const int X = blockIdx.x * TILE_DIM + threadIdx.x;
  const int Y = blockIdx.y * TILE_DIM + threadIdx.y;
  if (X < dim && Y < dim) {
    // A warp reads 32 consecutive floats of one row and writes them dim
    // floats apart, down one column: one transaction in, 32 out.
    output[X * dim + Y] = input[Y * dim + X];
  }
}

__global__ void transposeSharedConflict(const float* input, float* output, int dim) {
  __shared__ float tile[TILE_DIM][TILE_DIM];

  const int X = blockIdx.x * TILE_DIM + threadIdx.x;
  const int Y = blockIdx.y * TILE_DIM + threadIdx.y;
  if (X < dim && Y < dim) {
    // Each warp fills one row of the tile from one row of the input.
    tile[threadIdx.y][threadIdx.x] = input[Y * dim + X];
  }
  __syncthreads();

  const int OUT_X = blockIdx.y * TILE_DIM + threadIdx.x;
  const int OUT_Y = blockIdx.x * TILE_DIM + threadIdx.y;
  if (OUT_X < dim && OUT_Y < dim) {
    // Each warp reads one column of the tile and writes it as one row of
    // the output. The 32 elements of a column are TILE_DIM floats apart,
    // and TILE_DIM is a multiple of the bank count, so all 32 lanes read the
    // same bank: the read is served 32 times over.
    output[OUT_Y * dim + OUT_X] = tile[threadIdx.x][threadIdx.y];
  }
}

__global__ void transposeSharedPadded(const float* input, float* output, int dim) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // rows TILE_DIM + 1 floats apart

  const int X = blockIdx.x * TILE_DIM + threadIdx.x;
  const int Y = blockIdx.y * TILE_DIM + threadIdx.y;
  if (X < dim && Y < dim) {
    tile[threadIdx.y][threadIdx.x] = input[Y * dim + X];
  }
  __syncthreads();

  const int OUT_X = blockIdx.y * TILE_DIM + threadIdx.x;
  const int OUT_Y = blockIdx.x * TILE_DIM + threadIdx.y;
  if (OUT_X < dim && OUT_Y < dim) {
    // The same column read, but the elements are TILE_DIM + 1 floats apart:
    // each lands one bank further than the last, so the 32 lanes read 32
    // banks and the read is served once.
    output[OUT_Y * dim + OUT_X] = tile[threadIdx.x][threadIdx.y];
  }
}

} // namespace shared_memory_demo
} // namespace demo
} // namespace bench
} // namespace vernier
