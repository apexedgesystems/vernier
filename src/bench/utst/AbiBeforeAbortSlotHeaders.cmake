# ==============================================================================
# AbiBeforeAbortSlotHeaders.cmake - Header tree of a bench library built
# before Profiler had its abortMeasure() slot
#
# Run with: cmake -DSRC=<vernier root> -DDST=<output root> -P AbiBeforeAbortSlotHeaders.cmake
#
# Copies src/bench/inc to <output root>/src/bench/inc, takes the
# abortMeasure() slot out of Profiler (and the hook that calls it), and sets
# BENCH_ABI_VERSION to 2, the version every library built before the slot
# carries: the library a benchmark built from this tree's headers may still
# find installed.
# ==============================================================================

cmake_minimum_required(VERSION 3.24)

file(REMOVE_RECURSE "${DST}/src/bench/inc")
file(COPY "${SRC}/src/bench/inc" DESTINATION "${DST}/src/bench")

# Replace the one occurrence of <old> in <file> with <new>; fail when it is
# not there exactly once, so a change to the header shows here.
function (replace_once file old new)
  file(READ "${file}" _text)
  string(FIND "${_text}" "${old}" _first)
  string(FIND "${_text}" "${old}" _last REVERSE)
  if (_first EQUAL -1 OR NOT _first EQUAL _last)
    message(FATAL_ERROR "expected exactly one '${old}' in ${file}; update this script")
  endif ()
  string(REPLACE "${old}" "${new}" _text "${_text}")
  file(WRITE "${file}" "${_text}")
endfunction ()

set(_inc "${DST}/src/bench/inc")
replace_once("${_inc}/Profiler.hpp" "  virtual void abortMeasure() noexcept {}\n" "")
replace_once(
  "${_inc}/Profiler.hpp"
  "  pc.setAbortMeasureHook([prof](const PerfCase&) { prof->abortMeasure(); });\n" ""
)

file(READ "${_inc}/PerfAbi.hpp" _abi)
# Matched without its ';', which a CMake list would split on.
string(REGEX MATCHALL "inline constexpr std::uint32_t BENCH_ABI_VERSION = [0-9]+" _versions
             "${_abi}"
)
list(LENGTH _versions _count)
if (NOT _count EQUAL 1)
  message(FATAL_ERROR "expected exactly one BENCH_ABI_VERSION definition in PerfAbi.hpp")
endif ()
string(REGEX REPLACE "inline constexpr std::uint32_t BENCH_ABI_VERSION = [0-9]+;"
                     "inline constexpr std::uint32_t BENCH_ABI_VERSION = 2;" _abi "${_abi}"
)
file(WRITE "${_inc}/PerfAbi.hpp" "${_abi}")
