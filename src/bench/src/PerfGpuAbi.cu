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
  // Stats is embedded in the GpuStats and PerfRow objects this library fills
  // in for the benchmark to read, so its size must match exactly.
  const detail::AbiField FIELDS[] = {
      {"ABI version", abiVersion, BENCH_ABI_VERSION, detail::AbiRule::EQUAL},
      {"sizeof(PerfConfig)", sizeofPerfConfig, sizeof(PerfConfig),
       detail::AbiRule::BENCHMARK_AT_LEAST_LIBRARY},
      {"sizeof(PerfGpuConfig)", sizeofPerfGpuConfig, sizeof(PerfGpuConfig),
       detail::AbiRule::BENCHMARK_AT_LEAST_LIBRARY},
      {"sizeof(Stats)", sizeofStats, sizeof(Stats), detail::AbiRule::EQUAL},
  };
  detail::requireAbiMatch("libbench_cuda", FIELDS, sizeof(FIELDS) / sizeof(FIELDS[0]));
}

} // namespace bench
} // namespace vernier
