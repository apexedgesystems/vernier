include_guard(GLOBAL)

# ==============================================================================
# BuildAcceleration.cmake
# ------------------------------------------------------------------------------
# Optional build speed optimizations. Include early in root CMakeLists.txt
# (after project() but before add_subdirectory calls).
#
# Features:
#   - ccache/sccache for compiler caching
#   - mold/lld for faster linking
#   - Split DWARF for faster debug builds
#   - Unity build support hints
#
# All features auto-detect and are non-fatal if tools are missing.
# ==============================================================================

# ------------------------------------------------------------------------------
# Compiler Cache (ccache or sccache)
# ------------------------------------------------------------------------------
option(VERNIER_USE_CCACHE "Use ccache/sccache if available" ON)

if (VERNIER_USE_CCACHE)
  # Prefer sccache (Rust-based, better for distributed builds), fall back to ccache
  find_program(_CCACHE_PROGRAM NAMES sccache ccache)

  if (_CCACHE_PROGRAM)
    # Determine which one we found
    get_filename_component(_CCACHE_NAME "${_CCACHE_PROGRAM}" NAME)

    set(CMAKE_C_COMPILER_LAUNCHER
        "${_CCACHE_PROGRAM}"
        CACHE STRING ""
    )
    set(CMAKE_CXX_COMPILER_LAUNCHER
        "${_CCACHE_PROGRAM}"
        CACHE STRING ""
    )

    # CUDA support: ccache 4.4+, sccache 0.2.14+
    # Check if nvcc wrapper works (some versions have issues)
    if (CMAKE_CUDA_COMPILER)
      set(CMAKE_CUDA_COMPILER_LAUNCHER
          "${_CCACHE_PROGRAM}"
          CACHE STRING ""
      )
    endif ()

    message(STATUS "Compiler cache: ${_CCACHE_NAME} (${_CCACHE_PROGRAM})")
  endif ()

  unset(_CCACHE_PROGRAM CACHE)
endif ()

# ------------------------------------------------------------------------------
# Fast Linker (mold > lld > gold > default)
# ------------------------------------------------------------------------------
option(VERNIER_USE_FAST_LINKER "Use mold/lld if available" ON)

if (VERNIER_USE_FAST_LINKER AND NOT CMAKE_CROSSCOMPILING)
  # Cross-compilation uses the toolchain's linker
  set(_LINKER_FOUND FALSE)

  # Try mold first (fastest, open source)
  if (NOT _LINKER_FOUND)
    find_program(_MOLD_LINKER mold)
    if (_MOLD_LINKER)
      # Verify compiler supports -fuse-ld=mold
      include(CheckCXXCompilerFlag)
      set(CMAKE_REQUIRED_LINK_OPTIONS "-fuse-ld=mold")
      check_cxx_compiler_flag("" _MOLD_WORKS)
      unset(CMAKE_REQUIRED_LINK_OPTIONS)

      if (_MOLD_WORKS)
        add_link_options("-fuse-ld=mold")
        set(_LINKER_FOUND TRUE)
        set(_FAST_LINKER_AVAILABLE TRUE)
        message(STATUS "Linker: mold (${_MOLD_LINKER})")
      endif ()
    endif ()
    unset(_MOLD_LINKER CACHE)
  endif ()

  # Try lld (LLVM linker, very fast)
  if (NOT _LINKER_FOUND AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    find_program(_LLD_LINKER ld.lld)
    if (_LLD_LINKER)
      add_link_options("-fuse-ld=lld")
      set(_LINKER_FOUND TRUE)
      set(_FAST_LINKER_AVAILABLE TRUE)
      message(STATUS "Linker: lld (${_LLD_LINKER})")
    endif ()
    unset(_LLD_LINKER CACHE)
  endif ()

  # gold is slower than mold/lld but faster than bfd
  if (NOT _LINKER_FOUND)
    find_program(_GOLD_LINKER ld.gold)
    if (_GOLD_LINKER)
      include(CheckCXXCompilerFlag)
      set(CMAKE_REQUIRED_LINK_OPTIONS "-fuse-ld=gold")
      check_cxx_compiler_flag("" _GOLD_WORKS)
      unset(CMAKE_REQUIRED_LINK_OPTIONS)

      if (_GOLD_WORKS)
        add_link_options("-fuse-ld=gold")
        set(_LINKER_FOUND TRUE)
        set(_FAST_LINKER_AVAILABLE TRUE)
        message(STATUS "Linker: gold (${_GOLD_LINKER})")
      endif ()
    endif ()
    unset(_GOLD_LINKER CACHE)
  endif ()

  if (NOT _LINKER_FOUND)
    message(STATUS "Linker: system default (install mold for faster builds)")
  endif ()

  unset(_LINKER_FOUND)
endif ()

# ------------------------------------------------------------------------------
# Split DWARF (faster Debug linking)
# ------------------------------------------------------------------------------
option(VERNIER_USE_SPLIT_DWARF "Use split DWARF for Debug builds" ON)

if (VERNIER_USE_SPLIT_DWARF)
  if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      # -gsplit-dwarf puts debug info in .dwo files, speeds up linking
      add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-gsplit-dwarf> $<$<COMPILE_LANGUAGE:C>:-gsplit-dwarf>
      )

      # --gdb-index creates index for faster debugger startup.
      # Only supported by gold/lld/mold, not the standard GNU ld used in cross-compilation.
      if (NOT CMAKE_CROSSCOMPILING AND _FAST_LINKER_AVAILABLE)
        add_link_options(-Wl,--gdb-index)
        message(STATUS "Split DWARF: enabled with gdb-index (faster debug linking)")
      else ()
        message(
          STATUS "Split DWARF: enabled (gdb-index skipped - cross-compiling or no fast linker)"
        )
      endif ()
    endif ()
  endif ()
endif ()

# ------------------------------------------------------------------------------
# Parallel Compilation Hints
# ------------------------------------------------------------------------------
# These are informational - actual parallelism comes from -j flag or
# CMAKE_BUILD_PARALLEL_LEVEL

# Ninja already parallelizes well; Make needs explicit -j
if (CMAKE_GENERATOR MATCHES "Unix Makefiles")
  message(STATUS "Tip: Use 'make -j\$(nproc)' or set CMAKE_BUILD_PARALLEL_LEVEL")
endif ()
