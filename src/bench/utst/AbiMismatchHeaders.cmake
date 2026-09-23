# ==============================================================================
# AbiMismatchHeaders.cmake - Header tree of a benchmark built against a
# different PerfConfig
#
# Run with: cmake -DSRC=<vernier root> -DDST=<output root> -P AbiMismatchHeaders.cmake
#
# Copies src/bench/inc to <output root>/src/bench/inc and appends one
# std::string member to PerfConfig, leaving BENCH_ABI_VERSION alone: the change
# a developer makes when adding an option and forgetting the version.
# ==============================================================================

cmake_minimum_required(VERSION 3.24)

file(REMOVE_RECURSE "${DST}/src/bench/inc")
file(COPY "${SRC}/src/bench/inc" DESTINATION "${DST}/src/bench")

set(_header "${DST}/src/bench/inc/PerfConfig.hpp")
file(READ "${_header}" _text)
set(_last_member "  int targetTimeUs = 0[;]")
string(FIND "${_text}" "  int targetTimeUs = 0;" _first)
string(FIND "${_text}" "  int targetTimeUs = 0;" _last REVERSE)
if (_first EQUAL -1 OR NOT _first EQUAL _last)
  message(FATAL_ERROR "expected exactly one 'int targetTimeUs = 0;' line in PerfConfig.hpp; "
                      "update this script to the struct's last member"
  )
endif ()
string(REGEX REPLACE "(${_last_member}[^\n]*\n)" "\\1  std::string appendedForAbiTest;\n" _text
                     "${_text}"
)
file(WRITE "${_header}" "${_text}")
