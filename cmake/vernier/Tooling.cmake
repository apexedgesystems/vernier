# ==============================================================================
# vernier/Tooling.cmake - Documentation, compression, and static analysis
# ==============================================================================

include_guard(GLOBAL)

# ==============================================================================
# Doxygen Documentation
# ==============================================================================

# Global docs target (individual *_docs targets register as dependencies)
if (PROJECT_BUILD_DOCS AND NOT TARGET docs)
  add_custom_target(docs COMMENT "Building all documentation")
endif ()

# ------------------------------------------------------------------------------
# vernier_add_doxygen(<target>
#                     [README <path>] [SRC <dir>] [INC <dir>] [TST <dir>]
#                     [DOCS <dir>] [TOOLS <dir>] [TEMPLATES <dir>]
#                     [EXTRA_DIRS <dirs...>]
#                     [HEADERS_ONLY] [NO_TST])
#
# Generate API documentation for <target> using Doxygen.
# Output: ${CMAKE_BINARY_DIR}/docs/lib<target>/doxygen/html
#
# Auto-detected directories (if they exist):
#   src, inc, tst, docs, tools, templates
#
# For custom directories, use EXTRA_DIRS.
# Respects PROJECT_BUILD_DOCS toggle.
# ------------------------------------------------------------------------------
function (vernier_add_doxygen _target)
  if (NOT PROJECT_BUILD_DOCS)
    return()
  endif ()

  vernier_guard(${_target})

  cmake_parse_arguments(
    DOX "HEADERS_ONLY;NO_TST" "README;SRC;INC;TST;DOCS;TOOLS;TEMPLATES" "EXTRA_DIRS" ${ARGN}
  )

  # Defaults - auto-detect standard directories
  if (DOX_HEADERS_ONLY)
    set(DOX_SRC "")
  elseif (NOT DOX_SRC AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    set(DOX_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src")
  endif ()

  if (NOT DOX_INC)
    set(DOX_INC "${CMAKE_CURRENT_SOURCE_DIR}/inc")
  endif ()

  if (DOX_NO_TST)
    set(DOX_TST "")
  elseif (NOT DOX_TST AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tst")
    set(DOX_TST "${CMAKE_CURRENT_SOURCE_DIR}/tst")
  endif ()

  if (NOT DOX_DOCS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/docs")
    set(DOX_DOCS "${CMAKE_CURRENT_SOURCE_DIR}/docs")
  endif ()

  if (NOT DOX_TOOLS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools")
    set(DOX_TOOLS "${CMAKE_CURRENT_SOURCE_DIR}/tools")
  endif ()

  if (NOT DOX_TEMPLATES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/templates")
    set(DOX_TEMPLATES "${CMAKE_CURRENT_SOURCE_DIR}/templates")
  endif ()

  if (NOT DOX_README AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    set(DOX_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
  endif ()

  find_package(Doxygen QUIET)
  if (NOT DOXYGEN_FOUND)
    message(WARNING "Doxygen not found; skipping docs for '${_target}'")
    return()
  endif ()

  # Template variables
  set(LIB_NAME "${_target}")
  set(DOCS_BASE "${CMAKE_BINARY_DIR}/docs/lib${_target}")
  set(DOCS_DIR "${DOCS_BASE}/doxygen/html")
  set(SRC_DIR "${DOX_SRC}")
  set(INC_DIR "${DOX_INC}")
  set(TST_DIR "${DOX_TST}")
  set(DOCS_SRC_DIR "${DOX_DOCS}")
  set(TOOLS_DIR "${DOX_TOOLS}")
  set(TEMPLATES_DIR "${DOX_TEMPLATES}")
  set(EXTRA_DIRS "${DOX_EXTRA_DIRS}")
  set(README_FILE "${DOX_README}")

  configure_file("${CMAKE_SOURCE_DIR}/docs/Doxyfile.common.in" "${DOCS_BASE}/Doxyfile" @ONLY)

  set(_docs_stamp "${DOCS_BASE}/.doxygen.stamp")
  set(_deps "${DOCS_BASE}/Doxyfile")

  if (README_FILE AND NOT README_FILE STREQUAL "")
    list(APPEND _deps "${README_FILE}")
  endif ()

  get_target_property(_tgt_type "${_target}" TYPE)
  if (NOT _tgt_type STREQUAL "INTERFACE_LIBRARY" AND NOT _tgt_type STREQUAL "OBJECT_LIBRARY")
    list(APPEND _deps "$<TARGET_FILE:${_target}>")
  endif ()

  if (README_FILE)
    add_custom_command(
      OUTPUT "${_docs_stamp}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${DOCS_DIR}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${README_FILE}" "${DOCS_BASE}/README.md"
      COMMAND "${DOXYGEN_EXECUTABLE}" "${DOCS_BASE}/Doxyfile"
      COMMAND ${CMAKE_COMMAND} -E touch "${_docs_stamp}"
      DEPENDS ${_deps}
      COMMENT "Doxygen: ${_target} -> ${DOCS_DIR}"
      VERBATIM
    )
  else ()
    add_custom_command(
      OUTPUT "${_docs_stamp}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${DOCS_DIR}"
      COMMAND "${DOXYGEN_EXECUTABLE}" "${DOCS_BASE}/Doxyfile"
      COMMAND ${CMAKE_COMMAND} -E touch "${_docs_stamp}"
      DEPENDS ${_deps}
      COMMENT "Doxygen: ${_target} -> ${DOCS_DIR}"
      VERBATIM
    )
  endif ()

  add_custom_target(${_target}_docs DEPENDS "${_docs_stamp}")
  add_dependencies(${_target}_docs ${_target})

  # Add to global docs target
  if (TARGET docs)
    add_dependencies(docs ${_target}_docs)
  endif ()

  set_property(
    TARGET ${_target}_docs
    APPEND
    PROPERTY ADDITIONAL_CLEAN_FILES
             "${DOCS_DIR};${DOCS_BASE}/Doxyfile;${DOCS_BASE}/README.md;${_docs_stamp}"
  )
endfunction ()

# ==============================================================================
# Documentation Install (production markdown only, release builds)
# ==============================================================================

# ------------------------------------------------------------------------------
# vernier_install_docs(<path> [DESTINATION <subdir>])
#
# Install a markdown file or directory to CMAKE_INSTALL_DOCDIR.
# Only runs for Release/MinSizeRel/RelWithDebInfo builds.
# Directories are installed recursively. Files are installed as-is.
# DESTINATION places content under a subdirectory of DOCDIR.
# ------------------------------------------------------------------------------
function (vernier_install_docs _path)
  if (CMAKE_BUILD_TYPE)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt)
    if (NOT _bt MATCHES "^(RELEASE|MINSIZEREL|RELWITHDEBINFO)$")
      return()
    endif ()
  endif ()

  cmake_parse_arguments(DOC "" "DESTINATION" "" ${ARGN})

  set(_dest "${CMAKE_INSTALL_DOCDIR}")
  if (DOC_DESTINATION)
    set(_dest "${CMAKE_INSTALL_DOCDIR}/${DOC_DESTINATION}")
  endif ()

  if (IS_DIRECTORY "${_path}")
    install(DIRECTORY "${_path}/" DESTINATION "${_dest}")
  else ()
    install(FILES "${_path}" DESTINATION "${_dest}")
  endif ()
endfunction ()

# ==============================================================================
# UPX Compression
# ==============================================================================

option(UPX_COMPRESS_SHARED "Allow UPX on shared libraries" OFF)

# ------------------------------------------------------------------------------
# vernier_add_upx_copy(<target>)
#
# Create UPX-compressed copy under <artifact_dir>/upx/.
# Respects ENABLE_UPX toggle. Only compresses executables by default.
# Only runs for Release/MinSizeRel builds (skips Debug/RelWithDebInfo).
# ------------------------------------------------------------------------------
function (vernier_add_upx_copy _target)
  if (NOT ENABLE_UPX)
    return()
  endif ()

  # Skip for Debug builds - no point compressing debug binaries
  if (CMAKE_BUILD_TYPE)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type_upper)
    if (NOT _build_type_upper MATCHES "^(RELEASE|MINSIZEREL)$")
      return()
    endif ()
  endif ()

  vernier_guard(${_target})

  get_target_property(_tgt_type ${_target} TYPE)
  if (NOT _tgt_type OR _tgt_type STREQUAL "STATIC_LIBRARY")
    return()
  endif ()

  set(_do_upx FALSE)
  if (_tgt_type STREQUAL "EXECUTABLE")
    set(_do_upx TRUE)
  elseif ((_tgt_type STREQUAL "SHARED_LIBRARY" OR _tgt_type STREQUAL "MODULE_LIBRARY")
          AND UPX_COMPRESS_SHARED
  )
    set(_do_upx TRUE)
  endif ()

  if (NOT _do_upx)
    return()
  endif ()

  find_program(UPX_EXECUTABLE upx QUIET)
  if (NOT UPX_EXECUTABLE)
    message(WARNING "ENABLE_UPX=ON but upx not found; skipping '${_target}'")
    return()
  endif ()

  set(_upx_dir "$<TARGET_FILE_DIR:${_target}>/upx")
  set(_upx_file "${_upx_dir}/$<TARGET_FILE_NAME:${_target}>.upx")

  add_custom_command(
    TARGET ${_target}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_upx_dir}"
    COMMAND ${CMAKE_COMMAND} -E rm -f "${_upx_file}"
    COMMAND
      /usr/bin/env sh -c
      "'${UPX_EXECUTABLE}' -q --best --lzma -o '${_upx_file}' '$<TARGET_FILE:${_target}>' >/dev/null 2>&1 || cp '$<TARGET_FILE:${_target}>' '${_upx_file}'"
    COMMENT "UPX: ${_target}"
    VERBATIM
  )

  set_property(
    TARGET ${_target}
    APPEND
    PROPERTY ADDITIONAL_CLEAN_FILES "${_upx_file}"
  )
endfunction ()

# ==============================================================================
# Clang-Tidy for CUDA
# ==============================================================================

# Cache options
if (NOT DEFINED VERNIER_CLANG_TIDY_CUDA_STUBS)
  set(VERNIER_CLANG_TIDY_CUDA_STUBS
      ON
      CACHE BOOL "Generate stub headers for clang-tidy CUDA"
  )
endif ()
if (NOT DEFINED VERNIER_CLANG_TIDY_CUDA_STUB_HEADERS)
  set(VERNIER_CLANG_TIDY_CUDA_STUB_HEADERS
      "texture_fetch_functions.h"
      CACHE STRING "CUDA headers to stub (semicolon-separated)"
  )
endif ()
if (NOT DEFINED VERNIER_CLANG_TIDY_CUDA_MUTE_FRONTEND)
  set(VERNIER_CLANG_TIDY_CUDA_MUTE_FRONTEND
      ON
      CACHE BOOL "Silence frontend warnings during tidy"
  )
endif ()
if (NOT DEFINED VERNIER_CLANG_TIDY_CUDA_HEADER_FILTER)
  string(REPLACE "\\" "\\\\" _vernier_src_root "${CMAKE_SOURCE_DIR}")
  set(VERNIER_CLANG_TIDY_CUDA_HEADER_FILTER
      "^${_vernier_src_root}/src/"
      CACHE STRING "Header filter regex for clang-tidy"
  )
endif ()

# ------------------------------------------------------------------------------
# vernier_clang_tidy_cuda(<target> FILES <cu...> [ALLOW_FAILURE])
#
# Run clang-tidy on CUDA sources via Clang's CUDA frontend.
# No-op when CUDA not found, ENABLE_CLANG_TIDY=OFF, or no FILES.
# ------------------------------------------------------------------------------
function (vernier_clang_tidy_cuda _target)
  if (NOT TARGET "${_target}")
    return()
  endif ()

  get_target_property(_tgt_type "${_target}" TYPE)
  if (_tgt_type STREQUAL "INTERFACE_LIBRARY")
    return()
  endif ()

  if (NOT CUDAToolkit_FOUND)
    return()
  endif ()

  if (DEFINED ENABLE_CLANG_TIDY AND NOT ENABLE_CLANG_TIDY)
    return()
  endif ()

  cmake_parse_arguments(CT "ALLOW_FAILURE" "" "FILES" ${ARGN})
  if (NOT CT_FILES)
    return()
  endif ()

  # Find clang-tidy
  set(_tidy_exe "${CMAKE_CXX_CLANG_TIDY}")
  if (NOT _tidy_exe)
    find_program(_tidy_exe NAMES clang-tidy clang-tidy-21 clang-tidy-20)
  endif ()
  if (NOT _tidy_exe)
    return()
  endif ()

  # CUDA root
  if (DEFINED CUDAToolkit_ROOT)
    set(_cuda_root "${CUDAToolkit_ROOT}")
  elseif (CUDAToolkit_BIN_DIR)
    get_filename_component(_cuda_root "${CUDAToolkit_BIN_DIR}/.." ABSOLUTE)
  else ()
    set(_cuda_root "/usr/local/cuda")
  endif ()

  # C++ standard
  set(_std_flag "")
  if (CMAKE_CUDA_STANDARD)
    set(_std_flag "-std=c++${CMAKE_CUDA_STANDARD}")
  elseif (CMAKE_CXX_STANDARD)
    set(_std_flag "-std=c++${CMAKE_CXX_STANDARD}")
  endif ()

  # SM architecture
  if (DEFINED VERNIER_CLANG_TIDY_CUDA_ARCH)
    set(_sm "${VERNIER_CLANG_TIDY_CUDA_ARCH}")
  elseif (CMAKE_CUDA_ARCHITECTURES)
    list(GET CMAKE_CUDA_ARCHITECTURES 0 _sm)
  else ()
    set(_sm "89")
  endif ()

  # Include flags
  set(_iflags)
  get_target_property(_tgt_incs ${_target} INCLUDE_DIRECTORIES)
  foreach (_d IN LISTS _tgt_incs)
    if (_d)
      list(APPEND _iflags "-I\"${_d}\"")
    endif ()
  endforeach ()

  # Stub headers
  if (VERNIER_CLANG_TIDY_CUDA_STUBS)
    set(_stub_dir "${CMAKE_BINARY_DIR}/clang_tidy_cuda_stubs")
    file(MAKE_DIRECTORY "${_stub_dir}")
    foreach (_hdr IN LISTS VERNIER_CLANG_TIDY_CUDA_STUB_HEADERS)
      if (NOT _hdr)
        continue()
      endif ()
      get_filename_component(_hdr_base "${_hdr}" NAME)
      set(_dst "${_stub_dir}/${_hdr_base}")
      if (NOT EXISTS "${_dst}")
        file(
          WRITE "${_dst}"
          "// clang-tidy CUDA stub: ${_hdr_base}
#pragma once
#if !defined(__CUDA_ARCH__)
#endif
"
        )
      endif ()
    endforeach ()
    list(INSERT _iflags 0 "-isystem" "\"${_stub_dir}\"")
  endif ()

  # CUDA includes
  set(_cuda_incs)
  if (TARGET CUDA::cudart)
    get_target_property(_cuda_incs CUDA::cudart INTERFACE_INCLUDE_DIRECTORIES)
  endif ()
  if (NOT _cuda_incs)
    set(_cuda_incs ${CUDAToolkit_INCLUDE_DIRS})
  endif ()
  foreach (_cd IN LISTS _cuda_incs)
    if (_cd)
      list(APPEND _iflags "-isystem" "\"${_cd}\"")
    endif ()
  endforeach ()

  # Compile definitions
  set(_dflags)
  get_target_property(_tgt_defs ${_target} COMPILE_DEFINITIONS)
  foreach (_def IN LISTS _tgt_defs)
    if (NOT _def)
      continue()
    endif ()
    if ("${_def}" MATCHES "^-D")
      list(APPEND _dflags "\"${_def}\"")
    else ()
      list(APPEND _dflags "-D\"${_def}\"")
    endif ()
  endforeach ()

  # Compile database
  set(_p_flag "")
  if (EXISTS "${CMAKE_BINARY_DIR}/compile_commands.json")
    set(_p_flag "-p \"${CMAKE_BINARY_DIR}\"")
  endif ()

  # Sysroot for cross-compilation
  set(_sysroot_flag "")
  if (CMAKE_CROSSCOMPILING AND CMAKE_SYSROOT)
    set(_sysroot_flag "--extra-arg=--sysroot=${CMAKE_SYSROOT}")
  endif ()

  set(_filter_re "^[0-9]+ warnings? generated( when compiling for sm_[0-9]+)?\\.")

  foreach (_src IN LISTS CT_FILES)
    if (NOT EXISTS "${_src}")
      continue()
    endif ()

    set(_args)
    list(APPEND _args "-quiet")
    if (_p_flag)
      list(APPEND _args "${_p_flag}")
    endif ()
    list(APPEND _args "--header-filter='${VERNIER_CLANG_TIDY_CUDA_HEADER_FILTER}'")
    if (VERNIER_CLANG_TIDY_CUDA_MUTE_FRONTEND)
      list(APPEND _args "--extra-arg=-Wno-everything")
    endif ()
    if (_sysroot_flag)
      list(APPEND _args "${_sysroot_flag}")
    endif ()
    list(APPEND _args "\"${_src}\"" "--" "-x" "cuda")
    if (_std_flag)
      list(APPEND _args "${_std_flag}")
    endif ()
    list(APPEND _args "--cuda-path=${_cuda_root}" "--cuda-gpu-arch=sm_${_sm}")
    list(APPEND _args ${_iflags} ${_dflags})

    string(JOIN " " _joined ${_tidy_exe} ${_args})
    set(_bash_cmd "set -o pipefail; ${_joined} 2>&1 | (grep -v -E '${_filter_re}' || :)")

    if (CT_ALLOW_FAILURE)
      add_custom_command(
        TARGET ${_target}
        POST_BUILD
        COMMAND /usr/bin/env bash -lc "${_bash_cmd} ; exit 0"
        COMMENT "clang-tidy (CUDA) ${_src}"
        VERBATIM
      )
    else ()
      add_custom_command(
        TARGET ${_target}
        POST_BUILD
        COMMAND /usr/bin/env bash -lc "${_bash_cmd}"
        COMMENT "clang-tidy (CUDA) ${_src}"
        VERBATIM
      )
    endif ()
  endforeach ()
endfunction ()
