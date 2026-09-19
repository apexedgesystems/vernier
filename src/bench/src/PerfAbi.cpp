/**
 * @file PerfAbi.cpp
 * @brief libbench's side of the benchmark/library layout check.
 */

#include "src/bench/inc/PerfAbi.hpp"

#include <cstdio>
#include <cstdlib>

namespace vernier {
namespace bench {

/* ----------------------------- File Helpers ----------------------------- */

namespace {

bool holds(const detail::AbiField& field) noexcept {
  if (field.rule == detail::AbiRule::EQUAL) {
    return field.benchmark == field.library;
  }
  return field.benchmark >= field.library;
}

} // namespace

/* ----------------------------- API ----------------------------- */

namespace detail {

std::string abiMismatchMessage(const char* libraryName, const AbiField* fields, std::size_t count) {
  std::string differences;
  for (std::size_t i = 0; i < count; ++i) {
    if (holds(fields[i])) {
      continue;
    }
    if (!differences.empty()) {
      differences += "; ";
    }
    differences += fields[i].name;
    differences += ": benchmark " + std::to_string(fields[i].benchmark);
    differences += ", library " + std::to_string(fields[i].library);
  }
  if (differences.empty()) {
    return {};
  }
  const std::string LIB = libraryName;
  return "[bench] ABI mismatch: this benchmark and the " + LIB +
         " it loaded were built from different vernier headers (" + differences +
         "). Rebuild the benchmark against this " + LIB + ", or load the " + LIB +
         " that matches the benchmark's headers. Exiting.\n";
}

void requireAbiMatch(const char* libraryName, const AbiField* fields, std::size_t count) noexcept {
  const std::string MESSAGE = abiMismatchMessage(libraryName, fields, count);
  if (MESSAGE.empty()) {
    return;
  }
  std::fputs(MESSAGE.c_str(), stderr);
  std::fflush(stderr);
  std::_Exit(BENCH_ABI_MISMATCH_EXIT_CODE);
}

} // namespace detail

void checkBenchAbi(std::uint32_t abiVersion, std::size_t sizeofPerfConfig,
                   std::size_t sizeofStats) noexcept {
  const detail::AbiField FIELDS[] = {
      {"ABI version", abiVersion, BENCH_ABI_VERSION, detail::AbiRule::EQUAL},
      {"sizeof(PerfConfig)", sizeofPerfConfig, sizeof(PerfConfig),
       detail::AbiRule::BENCHMARK_AT_LEAST_LIBRARY},
      {"sizeof(Stats)", sizeofStats, sizeof(Stats), detail::AbiRule::BENCHMARK_AT_LEAST_LIBRARY},
  };
  detail::requireAbiMatch("libbench", FIELDS, sizeof(FIELDS) / sizeof(FIELDS[0]));
}

} // namespace bench
} // namespace vernier
