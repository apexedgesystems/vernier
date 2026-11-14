#ifndef VERNIER_UNIFIEDMEMORYREFUSAL_HPP
#define VERNIER_UNIFIEDMEMORYREFUSAL_HPP
/**
 * @file UnifiedMemoryRefusal.hpp
 * @brief Why a run reported its request for unified-memory counters as
 *        unsupported, read from the run's recorded profile failures.
 *
 * Test support shared by the GPU samples whose counter checks hold only where
 * the request was honoured: GpuUnifiedMemory_pTest.cu and
 * GpuMemoryStrategies_pTest.cu.
 */

#include <string>

#include "src/bench/inc/ProfilerRegistry.hpp"

namespace vernier {
namespace bench {
namespace test {

/**
 * @brief Why the run reported its request for unified-memory counters as
 *        unsupported (CUPTI cannot record them, or the collector stood
 *        down); empty if it did not, the request then honoured. A window
 *        that lost its records is not a refusal: it is a failure of the test.
 */
inline std::string unifiedMemoryRefusal() {
  for (const ProfileFailure& failure : ProfilerRegistry::instance().failures()) {
    if (failure.backend == "capture-um" && failure.test.empty() &&
        failure.result.cause == ReadinessCause::UNSUPPORTED) {
      return failure.result.report.message;
    }
  }
  return {};
}

} // namespace test
} // namespace bench
} // namespace vernier

#endif // VERNIER_UNIFIEDMEMORYREFUSAL_HPP
