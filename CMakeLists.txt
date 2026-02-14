cmake_minimum_required(VERSION 3.24)

# ==============================================================================
# Vernier -- Performance Benchmarking Framework
#
# Build examples:
#   cmake --preset native-linux-debug
#   cmake --preset native-linux-release
#   cmake --preset jetson-aarch64-release
#
# Options:
#   -DCUDA_ARCHS="89"           Ada only (fast)
#   -DCUDA_ARCHS="80;86;89"     Ampere + Ada
#   -DENABLE_COVERAGE=ON        Enable coverage (Debug + Clang only)
#   -DENABLE_CLANG_TIDY=ON      Enable clang-tidy
#   -DVERNIER_BUILD_TOOLS=ON     Build CLI tools (Rust + Python)
#   -DSANITIZER=asan|tsan|ubsan Enable sanitizers
#
# ==============================================================================

# ------------------------------------------------------------------------------
# Project
# ------------------------------------------------------------------------------
project(
  vernier
  VERSION 1.0.0
  LANGUAGES C CXX
  DESCRIPTION "Performance benchmarking framework with profiler integrations"
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Build acceleration (ccache, fast linker, split DWARF)
include(vernier/BuildAcceleration)

# Coverage initialization
include(vernier/Coverage)
vernier_coverage_init()

# ------------------------------------------------------------------------------
# Language standard
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------------------------
# Warnings (C++ only to avoid NVCC noise)
# ------------------------------------------------------------------------------
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)

# ------------------------------------------------------------------------------
# Shared libraries
# ------------------------------------------------------------------------------
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# ------------------------------------------------------------------------------
# Feature toggles
# ------------------------------------------------------------------------------
option(VERNIER_BUILD_GPU "Build GPU/CUDA support (requires CUDA toolkit)" ON)
option(VERNIER_BUILD_TOOLS "Build CLI tools (Rust + Python, requires cargo and poetry)" OFF)
option(PROJECT_BUILD_DOCS "Enable Doxygen doc targets" ON)
option(ENABLE_CLANG_TIDY "Enable clang-tidy integration" OFF)

# ------------------------------------------------------------------------------
# CUDA (optional)
# ------------------------------------------------------------------------------
if (VERNIER_BUILD_GPU)
  find_package(CUDAToolkit QUIET)

  if (CUDAToolkit_FOUND AND (NOT CMAKE_CROSSCOMPILING OR DEFINED CMAKE_CUDA_COMPILER))
    enable_language(CUDA)

    set(CUDA_ARCHS
        "89"
        CACHE STRING "CUDA SM architectures"
    )
    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHS}")
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    option(CUDA_KEEP_INTERMEDIATES "Keep NVCC intermediate files" ON)
    set(_nvcc_keep "")
    if (CUDA_KEEP_INTERMEDIATES)
      set(NVCC_TMP_DIR "${CMAKE_BINARY_DIR}/nvcc")
      file(MAKE_DIRECTORY "${NVCC_TMP_DIR}")
      set(_nvcc_keep "--keep --keep-dir=${NVCC_TMP_DIR}")
    endif ()

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr ${_nvcc_keep}")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -lineinfo -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_MINSIZEREL "${CMAKE_CUDA_FLAGS_MINSIZEREL} -DNDEBUG")
  endif ()
endif ()

# ------------------------------------------------------------------------------
# Sanitizers
# ------------------------------------------------------------------------------
set(SANITIZER
    ""
    CACHE STRING "Enable sanitizer: asan | tsan | ubsan"
)

if (SANITIZER AND NOT CMAKE_CROSSCOMPILING)
  if (SANITIZER STREQUAL "asan")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address -fno-omit-frame-pointer)
  elseif (SANITIZER STREQUAL "tsan")
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
  elseif (SANITIZER STREQUAL "ubsan")
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
  endif ()
endif ()

# ------------------------------------------------------------------------------
# Build parallelism and output directories
# ------------------------------------------------------------------------------
include(ProcessorCount)
processorcount(_N)
if (_N GREATER 0 AND NOT DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
  set(CMAKE_BUILD_PARALLEL_LEVEL ${_N})
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# ------------------------------------------------------------------------------
# RPATH configuration
# ------------------------------------------------------------------------------
set(CMAKE_SKIP_BUILD_RPATH OFF)
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_BUILD_RPATH "$ORIGIN/../lib")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

# ------------------------------------------------------------------------------
# Configure summary
# ------------------------------------------------------------------------------
include(vernier/Summary)
vernier_print_config_summary()

# ------------------------------------------------------------------------------
# External dependencies
# ------------------------------------------------------------------------------
include("${PROJECT_SOURCE_DIR}/ExternalDependencies.cmake" OPTIONAL)

# ------------------------------------------------------------------------------
# Project-wide includes
# ------------------------------------------------------------------------------
include_directories("${PROJECT_SOURCE_DIR}")

# ------------------------------------------------------------------------------
# clang-tidy
# ------------------------------------------------------------------------------
if (ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if (CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
  else ()
    message(WARNING "ENABLE_CLANG_TIDY=ON but clang-tidy not found")
  endif ()
endif ()

# ------------------------------------------------------------------------------
# Subdirectories
# ------------------------------------------------------------------------------

add_subdirectory(src)

if (VERNIER_BUILD_TOOLS AND NOT CMAKE_SYSTEM_NAME STREQUAL "Generic")
  add_subdirectory(tools)
endif ()

# ------------------------------------------------------------------------------
# Package configuration (find_package support)
# ------------------------------------------------------------------------------

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/vernierConfig.cmake.in" "${CMAKE_BINARY_DIR}/vernierConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/vernier"
)

write_basic_package_version_file(
  "${CMAKE_BINARY_DIR}/vernierConfigVersion.cmake" COMPATIBILITY SameMajorVersion
)

install(FILES "${CMAKE_BINARY_DIR}/vernierConfig.cmake"
              "${CMAKE_BINARY_DIR}/vernierConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/vernier"
)

vernier_install_docs("${PROJECT_SOURCE_DIR}/README.md")
