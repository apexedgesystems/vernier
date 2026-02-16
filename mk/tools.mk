# ==============================================================================
# mk/tools.mk - Developer utilities and tool builds
#
# Static analysis, profiling, and CLI tool build targets.
# ==============================================================================

ifndef TOOLS_MK_GUARD
TOOLS_MK_GUARD := 1

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# Note: Static analysis runs on native host builds. Cross-compiled targets
# are typically analyzed using the native toolchain.

# Uses BUILD_DIR and NUM_JOBS from common.mk

# ------------------------------------------------------------------------------
# Tool Build Targets
# ------------------------------------------------------------------------------

# Build all tools (C++, Rust, Python)
tools: tools-cpp tools-rust tools-py
	$(call log,tools,All tools built)

# Build C++ tools only (skips if no vernier_cpp_tools target exists)
tools-cpp: prep
	@test -f "$(BUILD_DIR)/CMakeCache.txt" || cmake --preset $(HOST_DEBUG_PRESET) $(CMAKE_VERBOSE_FLAG)
	@cd "$(BUILD_DIR)" && if ninja -t query vernier_cpp_tools >/dev/null 2>&1; then \
	  printf '[tools] Building C++ tools\n'; ninja vernier_cpp_tools; \
	else printf '[tools] No C++ tools to build\n'; fi

# Build Rust tools only (skips if no target exists)
tools-rust: prep
	@test -f "$(BUILD_DIR)/CMakeCache.txt" || cmake --preset $(HOST_DEBUG_PRESET) $(CMAKE_VERBOSE_FLAG)
	@cd "$(BUILD_DIR)" && if ninja -t query vernier_rust_tools >/dev/null 2>&1; then \
	  printf '[tools] Building Rust tools\n'; ninja vernier_rust_tools; \
	else printf '[tools] No Rust tools to build\n'; fi

# Build Python tools only (skips if no target exists)
tools-py: prep
	@test -f "$(BUILD_DIR)/CMakeCache.txt" || cmake --preset $(HOST_DEBUG_PRESET) $(CMAKE_VERBOSE_FLAG)
	@cd "$(BUILD_DIR)" && if ninja -t query vernier_py_tools >/dev/null 2>&1; then \
	  printf '[tools] Building Python tools\n'; ninja vernier_py_tools; \
	else printf '[tools] No Python tools to build\n'; fi

# ------------------------------------------------------------------------------
# Static Analysis
# ------------------------------------------------------------------------------

# Static analysis with Clang's scan-build
static: prep
	$(call log,static,Configuring for static analysis)
	@cmake -DCMAKE_BUILD_TYPE=Debug -B"$(BUILD_DIR)" -S. -GNinja
	$(call log,static,Running scan-build)
	@cd "$(BUILD_DIR)" && scan-build --status-bugs ninja -j$(NUM_JOBS)
	$(call log,static,Running tests to verify)
	@cd "$(BUILD_DIR)" && $(call with_lib_path,ctest --output-on-failure)

# ------------------------------------------------------------------------------
# Phony Declarations
# ------------------------------------------------------------------------------

.PHONY: static tools tools-cpp tools-py tools-rust

endif  # TOOLS_MK_GUARD
