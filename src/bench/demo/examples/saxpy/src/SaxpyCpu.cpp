/**
 * @file SaxpyCpu.cpp
 * @brief The CPU reference the GPU versions are checked and compared against.
 */

#include "src/bench/demo/examples/saxpy/inc/Saxpy.hpp"

namespace vernier {
namespace bench {
namespace demo {

/* --------------------------------- API --------------------------------- */

void saxpyCpu(float a, const std::vector<float>& x, std::vector<float>& y) {
  const std::size_t N = x.size();
  const float* __restrict__ px = x.data();
  float* __restrict__ py = y.data();
  for (std::size_t i = 0; i < N; ++i) {
    py[i] = a * px[i] + py[i];
  }
}

} // namespace demo
} // namespace bench
} // namespace vernier
