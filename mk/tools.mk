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

# Python tools source directory
PY_TOOLS_DIR := tools/py

# ------------------------------------------------------------------------------
# Tool Build Targets
# ------------------------------------------------------------------------------

# Build all tools (C++, Python, and Rust)
tools: tools-cpp tools-py tools-rust
	$(call log,tools,All tools built)

# Build C++ tools only (includes library dependencies)
tools-cpp: prep
	$(call log,tools,Building C++ tools)
	@test -f "$(BUILD_DIR)/CMakeCache.txt" || cmake --preset $(HOST_DEBUG_PRESET) $(CMAKE_VERBOSE_FLAG)
	@cd "$(BUILD_DIR)" && ninja cpp_tools

# Build Python tools only (no CMake dependency)
# Note: Uses absolute paths because poetry runs from tools/py/ subdirectory
tools-py:
	$(call log,tools,Building Python tools)
	@mkdir -p "$(BUILD_DIR)/bin/tools/py" "$(BUILD_DIR)/lib/python" "$(BUILD_DIR)/wheels"
	@cd "$(PY_TOOLS_DIR)" && poetry build --format wheel --output "$(CURDIR)/$(BUILD_DIR)/wheels"
	@python3 -m pip install --quiet --no-deps \
	  --target "$(BUILD_DIR)/lib/python" --upgrade "$(BUILD_DIR)/wheels/vernier_py_tools-"*.whl
	@rm -rf "$(BUILD_DIR)/bin/tools/py"
	@mv "$(BUILD_DIR)/lib/python/bin" "$(BUILD_DIR)/bin/tools/py"
	@python3 -m pip install --quiet \
	  --target "$(BUILD_DIR)/lib/python" --upgrade "$(BUILD_DIR)/wheels/vernier_py_tools-"*.whl
	@rm -rf "$(BUILD_DIR)/lib/python/bin"
	@grep -q 'PYTHONPATH.*lib/python' "$(BUILD_DIR)/.env" 2>/dev/null || \
	  printf 'export PYTHONPATH="$$PWD/lib/python:$$PYTHONPATH"\nexport PATH="$$PWD/bin/tools/py:$$PATH"\n' >> "$(BUILD_DIR)/.env"
	$(call log,tools,Python tools ready - source .env from build directory to use)

# Build Rust tools only (no CMake dependency)
# Auto-enables cuda feature if nvcc is available
# Note: Uses absolute paths because cargo runs from tools/rust/ subdirectory
tools-rust:
	$(call log,tools,Building Rust tools)
	@mkdir -p "$(BUILD_DIR)/bin/tools/rust"
	@cd tools/rust && CARGO_HOME="$(CURDIR)/$(BUILD_DIR)/rust-cargo" \
	  cargo build --release --target-dir "$(CURDIR)/$(BUILD_DIR)/rust-target" \
	  $$(command -v nvcc >/dev/null 2>&1 && echo "--features cuda")
	@find "$(BUILD_DIR)/rust-target/release" -maxdepth 1 -type f -executable \
	  ! -name "*.so" ! -name "*.d" -exec cp {} "$(BUILD_DIR)/bin/tools/rust/" \;
	@rm -rf "$(BUILD_DIR)/rust-cargo" "$(BUILD_DIR)/rust-target"
	@grep -q 'bin/tools/rust' "$(BUILD_DIR)/.env" 2>/dev/null || \
	  printf 'export PATH="$$PWD/bin/tools/rust:$$PATH"\n' >> "$(BUILD_DIR)/.env"
	$(call log,tools,Rust tools ready - source .env from build directory to use)

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
