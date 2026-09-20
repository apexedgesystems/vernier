#ifndef VERNIER_PERFABI_HPP
#define VERNIER_PERFABI_HPP
/**
 * @file PerfAbi.hpp
 * @brief Start-up check that a benchmark and the bench libraries it loaded
 * were built from compatible headers.
 *
 * Header-inline code compiled into the benchmark hands PerfConfig, Stats and
 * PerfGpuConfig objects to libbench and libbench_cuda, which copy them into
 * storage of their own size, destroy the copies, embed them in results and
 * hand back references to them. The layout is therefore shared between two
 * separately built binaries. The benchmark reports what it was compiled with;
 * the library compares that with what it was compiled with and, on any
 * difference, prints one message and ends the process before an object of
 * these types is passed across.
 *
 * Policy: the ABI version and the size of every compared struct must be equal
 * on both sides. There is no tolerated difference, a member appended at the
 * end included. Every layout change raises BENCH_ABI_VERSION, because a member
 * that fits into existing padding leaves the size unchanged and only the
 * version can show it. PerfAbi_uTest.cpp pins the member count, order, types
 * and offsets that belong to the current version, so a layout change without a
 * version change fails there.
 *
 * Not covered: a benchmark built from headers older than this check never
 * calls it, and one that constructs a PerfGpuCase without PERF_GPU_MAIN or the
 * GPU guard reaches libbench_cuda unchecked. The SONAME of libbench and
 * libbench_cuda covers that gap from outside the process: a benchmark whose
 * headers predate this check asks the loader for the SONAME those headers
 * shipped with, so it never reaches a library built from a later layout.
 *
 * BENCH_ABI_VERSION and the SONAME are separate numbers for the same layout.
 * The SONAME is set in src/bench/CMakeLists.txt, names the file the loader
 * picks, and also has to move when an exported signature changes, which this
 * check cannot see. BENCH_ABI_VERSION is compared inside the process and is
 * what catches a layout change that leaves both the size and the file name
 * alone. Raise both for a layout change; the numbers do not have to agree,
 * and they do not today.
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

/// Version of the layout shared with the bench libraries. Raise it for every
/// change to the members of PerfConfig, Stats, PerfGpuConfig or of a struct the
/// libraries hand back (GpuStats, PerfGpuResult, MultiGpuResult, PerfRow):
/// added (also at the end), removed, reordered or retyped.
inline constexpr std::uint32_t BENCH_ABI_VERSION = 1;

/// Exit status of a process ended by a failed check.
inline constexpr int BENCH_ABI_MISMATCH_EXIT_CODE = 3;

/* ----------------------------- AbiField ----------------------------- */

namespace detail {

/// One compared value, as each side was compiled. The two must be equal.
struct AbiField {
  const char* name;      ///< What is compared, e.g. "sizeof(PerfConfig)".
  std::size_t benchmark; ///< Value compiled into the benchmark.
  std::size_t library;   ///< Value compiled into the library.
};

/**
 * @brief Describe every field that differs, or return an empty string when none does.
 * @param libraryName Library doing the comparison, e.g. "libbench".
 * @note NOT RT-safe (heap allocation).
 */
[[nodiscard]] std::string abiMismatchMessage(const char* libraryName, const AbiField* fields,
                                             std::size_t count);

/**
 * @brief Return when every field is equal; otherwise print abiMismatchMessage()
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
