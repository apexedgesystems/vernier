# ==============================================================================
# AbiGperfLayout.cmake - GperfProfiler's layout as two benchmarks see it
#
# Included from the bench unit-test CMakeLists.txt. AbiGperfConsumer.cpp is
# compiled twice against the same libbench: with the default include path, and
# with one in which each of the compiler's own include directories that holds
# gperftools/ is replaced by a copy of its entries without it, as on a host
# with the gperftools library but not its development package. Each reports
# its sizeof(GperfProfiler) and the size libbench was compiled with; the test
# requires them to agree, and the second program to see no gperftools header.
#
# Targets: AbiGperfConsumerDefault, AbiGperfConsumerHidden
# Test:    AbiGperfLayoutTest
# ==============================================================================

# The hidden include path needs the compiler's own directory list and the
# GCC/Clang flags that drop it: a native build with either compiler.
if (CMAKE_CROSSCOMPILING OR NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  return()
endif ()

# ------------------------------------------------------------------------------
# Hidden-Headers Include Path
# ------------------------------------------------------------------------------
# The compiler's own include directories, in its order; each one holding
# gperftools/ is replaced by a directory of links to its other entries. Each
# "-isystem <dir>" pair is one SHELL: option, so that CMake's de-duplication of
# compile options does not drop the repeated flag.
# ------------------------------------------------------------------------------

set(_gperf_hidden_root "${CMAKE_CURRENT_BINARY_DIR}/abi_gperf_hidden_include")
file(REMOVE_RECURSE "${_gperf_hidden_root}")
set(_gperf_hidden_flags -nostdinc -nostdinc++)
set(_index 0)
foreach (_dir IN LISTS CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
  if (IS_DIRECTORY "${_dir}/gperftools")
    set(_copy "${_gperf_hidden_root}/${_index}")
    file(MAKE_DIRECTORY "${_copy}")
    file(
      GLOB _entries
      LIST_DIRECTORIES true
      "${_dir}/*"
    )
    foreach (_entry IN LISTS _entries)
      get_filename_component(_name "${_entry}" NAME)
      if (NOT _name STREQUAL "gperftools")
        file(CREATE_LINK "${_entry}" "${_copy}/${_name}" SYMBOLIC)
      endif ()
    endforeach ()
    list(APPEND _gperf_hidden_flags "SHELL:-isystem ${_copy}")
  else ()
    list(APPEND _gperf_hidden_flags "SHELL:-isystem ${_dir}")
  endif ()
  math(EXPR _index "${_index} + 1")
endforeach ()

# ------------------------------------------------------------------------------
# Consumer Programs (2)
# ------------------------------------------------------------------------------
# Test fixtures, not products: never installed or exported, no UPX copy.
# ------------------------------------------------------------------------------

foreach (_variant Default Hidden)
  vernier_add_app(
    NAME
    AbiGperfConsumer${_variant}
    SRC
    "${CMAKE_CURRENT_LIST_DIR}/AbiGperfConsumer.cpp"
    LINK
    bench
    NO_INSTALL
    NO_UPX
  )
endforeach ()

# The helper skips apps on platforms without POSIX; the test goes with them.
if (NOT TARGET AbiGperfConsumerDefault OR NOT TARGET AbiGperfConsumerHidden)
  return()
endif ()

target_compile_options(
  AbiGperfConsumerHidden
  PRIVATE ${_gperf_hidden_flags}
)

# ------------------------------------------------------------------------------
# Test (1)
# ------------------------------------------------------------------------------
# It runs the two consumers built against this tree's library: a component
# test, as the other consumer checks are.
# ------------------------------------------------------------------------------

add_test(
  NAME AbiGperfLayoutTest
  COMMAND
    "${CMAKE_COMMAND}" -DDEFAULT=$<TARGET_FILE:AbiGperfConsumerDefault>
    -DHIDDEN=$<TARGET_FILE:AbiGperfConsumerHidden> -P
    "${CMAKE_CURRENT_LIST_DIR}/AbiGperfLayout_test.cmake"
)
set_tests_properties(AbiGperfLayoutTest PROPERTIES LABELS "benchmarking;abi;component")
