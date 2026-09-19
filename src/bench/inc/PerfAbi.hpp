#ifndef VERNIER_PERFABI_HPP
#define VERNIER_PERFABI_HPP
/**
 * @file PerfAbi.hpp
 * @brief Start-up check that a benchmark and the bench libraries it loaded
 * were built from compatible headers.
 *
 * Header-inline code compiled into the benchmark hands PerfConfig, Stats and
 * PerfGpuConfig objects to libbench and libbench_cuda. Their layout is
 * therefore shared between two separately built binaries, and a library from
 * another build reads the wrong bytes. The benchmark reports what it was
 * compiled with; the library compares that with what it was compiled with and,
 * on a mismatch, prints one message and ends the process instead of running
 * into undefined behavior.
 *
 * The rule per value:
 *  - the ABI version must be equal;
 *  - a struct the library only reads (PerfConfig, PerfGpuConfig, and Stats for
 *    libbench) may be larger in the benchmark than in the library, because
 *    members are appended and the library reads the part it knows;
 *  - a struct the library embeds in results it hands back (Stats for
 *    libbench_cuda) must have the same size on both sides.
 */

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfGpuConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"

#include <cstddef>
#include <cstdint>

#include <string>

namespace vernier {
namespace bench {

/* ----------------------------- Constants ----------------------------- */

/// Version of the layout shared with the bench libraries. Appending a member
/// to a shared struct needs no change here (the sizes carry it). Raise it,
/// together with the libraries' SOVERSION, for any change the sizes cannot
/// show or that an older library cannot tolerate: a reordered, retyped or
/// removed member.
inline constexpr std::uint32_t BENCH_ABI_VERSION = 1;

/// Exit status of a process ended by a failed check.
inline constexpr int BENCH_ABI_MISMATCH_EXIT_CODE = 3;

/* ----------------------------- AbiField ----------------------------- */

namespace detail {

/// How the benchmark's value must relate to the library's.
enum class AbiRule : std::uint8_t {
  EQUAL,                     ///< Must match exactly.
  BENCHMARK_AT_LEAST_LIBRARY ///< Benchmark may be larger (appended members).
};

/// One compared value, as each side was compiled.
struct AbiField {
  const char* name;      ///< What is compared, e.g. "sizeof(PerfConfig)".
  std::size_t benchmark; ///< Value compiled into the benchmark.
  std::size_t library;   ///< Value compiled into the library.
  AbiRule rule;          ///< Relation that must hold.
};

/**
 * @brief Describe every violated field, or return an empty string when all hold.
 * @param libraryName Library doing the comparison, e.g. "libbench".
 * @note NOT RT-safe (heap allocation).
 */
[[nodiscard]] std::string abiMismatchMessage(const char* libraryName, const AbiField* fields,
                                             std::size_t count);

/**
 * @brief Return when every field holds; otherwise print abiMismatchMessage()
 * to stderr and end the process with BENCH_ABI_MISMATCH_EXIT_CODE.
 * @note Ends the process without unwinding: after a mismatch no object shared
 * with the library can be trusted, including during static destruction.
 */
void requireAbiMatch(const char* libraryName, const AbiField* fields, std::size_t count) noexcept;

} // namespace detail

/* ----------------------------- API ----------------------------- */

/**
 * @brief libbench's side of the check. Arguments are the benchmark's values.
 * @note Returns only when the benchmark is compatible with this libbench.
 */
void checkBenchAbi(std::uint32_t abiVersion, std::size_t sizeofPerfConfig,
                   std::size_t sizeofStats) noexcept;

/**
 * @brief libbench_cuda's side of the check. Arguments are the benchmark's values.
 * @note Returns only when the benchmark is compatible with this libbench_cuda.
 */
void checkBenchGpuAbi(std::uint32_t abiVersion, std::size_t sizeofPerfConfig,
                      std::size_t sizeofPerfGpuConfig, std::size_t sizeofStats) noexcept;

/**
 * @brief Run the libbench check once per process, with this translation
 * unit's values. Called by PERF_MAIN, PERF_GPU_MAIN and Profiler::make(), so a
 * benchmark with its own main() is checked before its first profiler is built.
 * @note NOT RT-safe on the first call (may print and exit); a flag test after.
 */
inline void ensureBenchAbi() noexcept {
  static const bool CHECKED =
      (checkBenchAbi(BENCH_ABI_VERSION, sizeof(PerfConfig), sizeof(Stats)), true);
  static_cast<void>(CHECKED);
}

/**
 * @brief Run the libbench and libbench_cuda checks once per process. Called by
 * PERF_GPU_MAIN and the GPU guard, before the first PerfGpuCase is built.
 * @note NOT RT-safe on the first call (may print and exit); a flag test after.
 */
inline void ensureBenchGpuAbi() noexcept {
  ensureBenchAbi();
  static const bool CHECKED = (checkBenchGpuAbi(BENCH_ABI_VERSION, sizeof(PerfConfig),
                                                sizeof(PerfGpuConfig), sizeof(Stats)),
                               true);
  static_cast<void>(CHECKED);
}

} // namespace bench
} // namespace vernier

#endif // VERNIER_PERFABI_HPP
