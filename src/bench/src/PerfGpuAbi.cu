/**
 * @file PerfGpuAbi.cu
 * @brief libbench_cuda's side of the benchmark/library layout check.
 */

#include "src/bench/inc/PerfAbi.hpp"

namespace vernier {
namespace bench {

/* ----------------------------- API ----------------------------- */

void checkBenchGpuAbi(std::uint32_t abiVersion, std::size_t sizeofPerfConfig,
                      std::size_t sizeofPerfGpuConfig, std::size_t sizeofStats) noexcept {
  // This library stores its own PerfConfig and PerfGpuConfig copies, hands out
  // references to them, and embeds Stats in the results the benchmark reads.
  const detail::AbiField FIELDS[] = {
      {"ABI version", abiVersion, BENCH_ABI_VERSION},
      {"sizeof(PerfConfig)", sizeofPerfConfig, sizeof(PerfConfig)},
      {"sizeof(PerfGpuConfig)", sizeofPerfGpuConfig, sizeof(PerfGpuConfig)},
      {"sizeof(Stats)", sizeofStats, sizeof(Stats)},
  };
  detail::requireAbiMatch("libbench_cuda", FIELDS, sizeof(FIELDS) / sizeof(FIELDS[0]));
}

} // namespace bench
} // namespace vernier
