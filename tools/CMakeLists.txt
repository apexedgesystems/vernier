# ==============================================================================
# tools/CMakeLists.txt - CLI tools (C++, Rust, Python)
#
# Discovers and builds tool subdirectories by their native toolchains:
#   cpp/   - Standard CMake (add_subdirectory)
#   rust/  - Cargo (requires cargo on PATH)
#   py/    - Poetry + pip (requires poetry and pip on PATH)
#
# Each section is gated on the directory existing and the toolchain being
# available. Safe to copy to any project that follows the same layout.
# ==============================================================================

# ------------------------------------------------------------------------------
# Helper script (generated at configure time)
# ------------------------------------------------------------------------------
# Configures the env-helper.sh.in template into the build dir. The helper
# runs at build time and appends absolute PATH / PYTHONPATH entries for
# the CLI tools into the build .env file. Absolute paths mean the .env
# works regardless of cwd: `. build/<preset>/.env` from the project root
# puts every tool on PATH. Append-if-missing semantics let downstream
# projects that FetchContent this one and write their own .env rows
# coexist with these entries.

set(_ENV_FILE "${CMAKE_BINARY_DIR}/.env")
set(_ENV_HELPER "${CMAKE_BINARY_DIR}/${PROJECT_NAME}-env-helper.sh")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/env-helper.sh.in" "${_ENV_HELPER}" @ONLY)

# ------------------------------------------------------------------------------
# C++ tools
# ------------------------------------------------------------------------------

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/CMakeLists.txt")
  add_subdirectory(cpp)
endif ()

# ------------------------------------------------------------------------------
# Rust tools
# ------------------------------------------------------------------------------

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rust/Cargo.toml")
  find_program(CARGO_EXECUTABLE cargo)

  if (CARGO_EXECUTABLE)
    set(_RUST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/rust")
    set(_RUST_OUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tools/rust")
    # Inherit CARGO_HOME from the environment (the dev image bakes a pinned,
    # offline-ready registry cache at /opt/rust/cargo; locally it is ~/.cargo).
    # Sharing the registry cache -- not a fresh per-build one -- lets a clean
    # build reuse the baked crates and lets the release build run offline
    # (hermetic). The target dir stays per-build so compiled artifacts isolate.
    set(_CARGO_TARGET "${CMAKE_BINARY_DIR}/${PROJECT_NAME}-rust-target")
    set(_RUST_STAMP "${CMAKE_BINARY_DIR}/${PROJECT_NAME}-stamps/rust-tools.built")

    # Auto-enable cuda feature if nvcc is available
    set(_CARGO_FEATURES "")
    find_program(_NVCC_EXE nvcc)
    if (_NVCC_EXE)
      set(_CARGO_FEATURES "--features cuda")
    endif ()

    # Generate build script (copies all cargo binaries)
    set(_RUST_BUILD_SCRIPT "${CMAKE_BINARY_DIR}/${PROJECT_NAME}-build-rust-tools.sh")
    file(
      WRITE "${_RUST_BUILD_SCRIPT}"
      "#!/bin/bash\nset -e\n"
      # Build offline against the dev image's baked cargo cache when the image
      # opts in (VERNIER_RUST_OFFLINE, set in the build-base tier). Scoped to this
      # build so local dev (no such env) still resolves crates online.
      "OFFLINE=\n"
      "if [ -n \"\${VERNIER_RUST_OFFLINE:-}\" ]; then OFFLINE=--offline; fi\n"
      "'${CARGO_EXECUTABLE}' build --release \${OFFLINE}"
      " --manifest-path '${_RUST_SRC}/Cargo.toml' --target-dir '${_CARGO_TARGET}'"
      " ${_CARGO_FEATURES}\n"
      "mkdir -p '${_RUST_OUT}'\n"
      "find '${_CARGO_TARGET}/release' -maxdepth 1 -type f -executable ! -name '*.*'"
      " -exec cp -f {} '${_RUST_OUT}/' \\;\n"
      "mkdir -p '${CMAKE_BINARY_DIR}/${PROJECT_NAME}-stamps'\n"
      "touch '${_RUST_STAMP}'\n"
    )

    # Track the rust sources so editing them dirties the stamp: without this,
    # ninja considers the tools up to date forever and ships stale binaries.
    # CONFIGURE_DEPENDS re-globs on build so new files are picked up too.
    file(GLOB_RECURSE _RUST_SOURCES CONFIGURE_DEPENDS "${_RUST_SRC}/src/*.rs")

    add_custom_command(
      OUTPUT "${_RUST_STAMP}"
      COMMAND bash "${_RUST_BUILD_SCRIPT}"
      COMMAND bash "${_ENV_HELPER}"
      DEPENDS ${_RUST_SOURCES} "${_RUST_SRC}/Cargo.toml" "${_RUST_SRC}/Cargo.lock"
      COMMENT "Building ${PROJECT_NAME} Rust tools"
      VERBATIM
    )

    add_custom_target(${PROJECT_NAME}_rust_tools DEPENDS "${_RUST_STAMP}")
    list(APPEND _TOOLS_DEPS ${PROJECT_NAME}_rust_tools)
  else ()
    message(WARNING "${PROJECT_NAME}: cargo not found -- Rust tools skipped")
  endif ()
endif ()

# ------------------------------------------------------------------------------
# Python tools
# ------------------------------------------------------------------------------

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/py/pyproject.toml")
  find_program(POETRY_EXECUTABLE poetry)
  find_program(PIP_EXECUTABLE pip3 pip)

  if (POETRY_EXECUTABLE AND PIP_EXECUTABLE)
    set(_PY_SRC "${CMAKE_CURRENT_SOURCE_DIR}/py")
    set(_PY_OUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tools/py")
    set(_PY_LIB "${CMAKE_BINARY_DIR}/lib/python")
    set(_PY_WHEELS "${CMAKE_BINARY_DIR}/${PROJECT_NAME}-wheels")
    set(_PY_STAMP "${CMAKE_BINARY_DIR}/${PROJECT_NAME}-stamps/py-tools.built")

    # Generate build script (handles shell globs that VERBATIM mode cannot expand)
    set(_PY_BUILD_SCRIPT "${CMAKE_BINARY_DIR}/${PROJECT_NAME}-build-py-tools.sh")
    file(
      WRITE "${_PY_BUILD_SCRIPT}"
      "#!/bin/bash\nset -e\n"
      # Install the dependencies offline from the dev image's baked wheelhouse
      # when present (VERNIER_PIP_WHEELHOUSE, set in the build-base tier). poetry
      # build itself is already offline (poetry-core is bundled). Local dev (no
      # such env) resolves online from PyPI.
      "FINDLINKS=\n"
      "if [ -n \"\${VERNIER_PIP_WHEELHOUSE:-}\" ]; then FINDLINKS=\"--no-index --find-links \${VERNIER_PIP_WHEELHOUSE}\"; fi\n"
      "mkdir -p '${_PY_OUT}' '${_PY_LIB}' '${_PY_WHEELS}'\n"
      "cd '${_PY_SRC}'\n"
      "'${POETRY_EXECUTABLE}' build --format wheel --output '${_PY_WHEELS}'\n"
      "'${PIP_EXECUTABLE}' install --quiet --no-deps --target '${_PY_LIB}' --upgrade ${_PY_WHEELS}/*.whl\n"
      "rm -rf '${_PY_OUT}'\n"
      "mv '${_PY_LIB}/bin' '${_PY_OUT}'\n"
      "'${PIP_EXECUTABLE}' install --quiet \${FINDLINKS} --target '${_PY_LIB}' --upgrade ${_PY_WHEELS}/*.whl\n"
      "rm -rf '${_PY_LIB}/bin'\n"
      "mkdir -p '${CMAKE_BINARY_DIR}/${PROJECT_NAME}-stamps'\n"
      "touch '${_PY_STAMP}'\n"
    )

    add_custom_command(
      OUTPUT "${_PY_STAMP}"
      COMMAND bash "${_PY_BUILD_SCRIPT}"
      COMMAND bash "${_ENV_HELPER}"
      COMMENT "Building ${PROJECT_NAME} Python tools"
      VERBATIM
    )

    add_custom_target(${PROJECT_NAME}_py_tools DEPENDS "${_PY_STAMP}")
    list(APPEND _TOOLS_DEPS ${PROJECT_NAME}_py_tools)
  else ()
    message(WARNING "${PROJECT_NAME}: poetry/pip not found -- Python tools skipped")
  endif ()
endif ()

# ------------------------------------------------------------------------------
# Aggregate target
# ------------------------------------------------------------------------------

if (_TOOLS_DEPS)
  add_custom_target(${PROJECT_NAME}_tools ALL DEPENDS ${_TOOLS_DEPS})
endif ()
