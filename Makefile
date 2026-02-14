# ==============================================================================
# Makefile - Vernier project entry point
#
# Performance benchmarking framework. Run `make help` for available targets.
# ==============================================================================

.DEFAULT_GOAL := debug
MAKEFLAGS += --no-print-directory

# ==============================================================================
# Includes
# ==============================================================================

include mk/common.mk
include mk/build.mk
include mk/test.mk
include mk/coverage.mk
include mk/sanitizers.mk
include mk/tools.mk
include mk/format.mk
include mk/docker.mk
include mk/compose.mk
include mk/clean.mk

# ==============================================================================
# Configuration
# ==============================================================================

# CMake verbosity (set VERBOSE=1 for per-target details)
VERBOSE ?= 0
ifeq ($(VERBOSE),1)
  CMAKE_VERBOSE_FLAG := -DVERNIER_TARGETS_VERBOSE=ON
else
  CMAKE_VERBOSE_FLAG :=
endif

# Extra CMake args (e.g., make debug CMAKE_EXTRA_ARGS="-DVERNIER_BUILD_GPU=OFF")
CMAKE_EXTRA_ARGS ?=

# Install pre-commit hooks during `prep` (make PRE_COMMIT_INSTALL=yes prep)
PRE_COMMIT_INSTALL ?= no

# ------------------------------------------------------------------------------
# CMake Preset Names (must match CMakePresets.json)
# ------------------------------------------------------------------------------

# Native x86_64
HOST_DEBUG_PRESET   ?= native-linux-debug
HOST_RELEASE_PRESET ?= native-linux-release

# Jetson (aarch64 + CUDA)
JETSON_DEBUG_PRESET   ?= jetson-aarch64-debug
JETSON_RELEASE_PRESET ?= jetson-aarch64-release

# Raspberry Pi (aarch64)
RPI_DEBUG_PRESET   ?= rpi-aarch64-debug
RPI_RELEASE_PRESET ?= rpi-aarch64-release

# RISC-V 64
RISCV_DEBUG_PRESET   ?= riscv64-linux-debug
RISCV_RELEASE_PRESET ?= riscv64-linux-release

# ------------------------------------------------------------------------------
# Build Directories (derived from preset names)
# ------------------------------------------------------------------------------

HOST_DEBUG_DIR     := build/$(HOST_DEBUG_PRESET)
HOST_RELEASE_DIR   := build/$(HOST_RELEASE_PRESET)
JETSON_DEBUG_DIR   := build/$(JETSON_DEBUG_PRESET)
JETSON_RELEASE_DIR := build/$(JETSON_RELEASE_PRESET)
RPI_DEBUG_DIR      := build/$(RPI_DEBUG_PRESET)
RPI_RELEASE_DIR    := build/$(RPI_RELEASE_PRESET)
RISCV_DEBUG_DIR    := build/$(RISCV_DEBUG_PRESET)
RISCV_RELEASE_DIR  := build/$(RISCV_RELEASE_PRESET)

# ==============================================================================
# Build Macros
# ==============================================================================

# _build: Configure and build a CMake preset
# Usage: $(call _build,display_name,preset,build_dir)
define _build
	$(call log,build,Configuring $(1))
	@cmake --preset $(2) $(CMAKE_VERBOSE_FLAG) $(CMAKE_EXTRA_ARGS)
	$(call log,build,Building $(1))
	@cmake --build --preset $(2) -j$(NUM_JOBS)
	@ln -sf $(3)/compile_commands.json compile_commands.json
endef

# _configure: Configure a CMake preset (no build)
# Usage: $(call _configure,preset,build_dir)
define _configure
	@cmake --preset $(1) $(CMAKE_VERBOSE_FLAG) $(CMAKE_EXTRA_ARGS)
	@ln -sf $(2)/compile_commands.json compile_commands.json
endef

# ==============================================================================
# Help
# ==============================================================================

help:
	@printf '%s\n' "Vernier -- Performance Benchmarking Framework"
	@printf '%s\n' "============================================="
	@printf '\n'
	@printf '%s\n' "Native Builds:"
	@printf '  %-28s %s\n' "make debug" "Build native debug (default)"
	@printf '  %-28s %s\n' "make release" "Build native release"
	@printf '  %-28s %s\n' "make docs" "Build Doxygen documentation"
	@printf '  %-28s %s\n' "make configure" "Configure only (no build)"
	@printf '\n'
	@printf '%s\n' "Install (release + cmake --install):"
	@printf '  %-28s %s\n' "make install" "Build + install native (libs only)"
	@printf '  %-28s %s\n' "make install-tools" "Build + install CLI tools (Rust + Python wheel)"
	@printf '  %-28s %s\n' "make install-jetson" "Build + install Jetson (libs only)"
	@printf '  %-28s %s\n' "make install-rpi" "Build + install RPi (libs only)"
	@printf '  %-28s %s\n' "make install-riscv" "Build + install RISC-V (libs only)"
	@printf '\n'
	@printf '%s\n' "Cross-Compilation:"
	@printf '  %-28s %s\n' "make jetson-debug" "Build for Jetson (aarch64 + CUDA)"
	@printf '  %-28s %s\n' "make jetson-release" "Build for Jetson release"
	@printf '  %-28s %s\n' "make rpi-debug" "Build for Raspberry Pi (aarch64)"
	@printf '  %-28s %s\n' "make rpi-release" "Build for Raspberry Pi release"
	@printf '  %-28s %s\n' "make riscv-debug" "Build for RISC-V 64"
	@printf '  %-28s %s\n' "make riscv-release" "Build for RISC-V 64 release"
	@printf '\n'
	@printf '%s\n' "Testing:"
	@printf '  %-28s %s\n' "make test" "Run all tests (serial)"
	@printf '  %-28s %s\n' "make testp" "Run tests (parallel + timing serial)"
	@printf '  %-28s %s\n' "make test-py" "Run Python tools tests"
	@printf '  %-28s %s\n' "make test-rust" "Run Rust tools tests"
	@printf '\n'
	@printf '%s\n' "Quality:"
	@printf '  %-28s %s\n' "make format" "Auto-fix formatting issues"
	@printf '  %-28s %s\n' "make format-check" "Check formatting (no fixes)"
	@printf '  %-28s %s\n' "make coverage" "Generate code coverage report"
	@printf '  %-28s %s\n' "make static" "Run static analysis (scan-build)"
	@printf '  %-28s %s\n' "make asan" "Build + test with AddressSanitizer"
	@printf '  %-28s %s\n' "make tsan" "Build + test with ThreadSanitizer"
	@printf '  %-28s %s\n' "make ubsan" "Build + test with UBSanitizer"
	@printf '\n'
	@printf '%s\n' "Tools:"
	@printf '  %-28s %s\n' "make tools" "Build all tools (C++, Python, Rust)"
	@printf '  %-28s %s\n' "make tools-cpp" "Build C++ tools only"
	@printf '  %-28s %s\n' "make tools-py" "Build Python tools only"
	@printf '  %-28s %s\n' "make tools-rust" "Build Rust tools only"
	@printf '\n'
	@printf '%s\n' "Compose (build via Docker Compose):"
	@printf '  %-28s %s\n' "make compose-debug" "Native debug via dev-cuda"
	@printf '  %-28s %s\n' "make compose-release" "Native release via dev-cuda"
	@printf '  %-28s %s\n' "make compose-docs" "Documentation via dev-cuda"
	@printf '  %-28s %s\n' "make compose-test" "Run tests via dev-cuda"
	@printf '  %-28s %s\n' "make compose-testp" "Run tests (parallel) via dev-cuda"
	@printf '  %-28s %s\n' "make compose-coverage" "Coverage report via dev-cuda"
	@printf '  %-28s %s\n' "make compose-format" "Format code via dev-cuda"
	@printf '  %-28s %s\n' "make compose-static" "Static analysis via dev-cuda"
	@printf '  %-28s %s\n' "make compose-asan" "AddressSanitizer via dev-cuda"
	@printf '  %-28s %s\n' "make compose-tsan" "ThreadSanitizer via dev-cuda"
	@printf '  %-28s %s\n' "make compose-ubsan" "UBSanitizer via dev-cuda"
	@printf '  %-28s %s\n' "make compose-tools" "Build all tools via dev-cuda"
	@printf '  %-28s %s\n' "make compose-jetson-debug" "Jetson debug via dev-jetson"
	@printf '  %-28s %s\n' "make compose-jetson-release" "Jetson release via dev-jetson"
	@printf '  %-28s %s\n' "make compose-rpi-debug" "RPi debug via dev-rpi"
	@printf '  %-28s %s\n' "make compose-rpi-release" "RPi release via dev-rpi"
	@printf '  %-28s %s\n' "make compose-riscv-debug" "RISC-V debug via dev-riscv64"
	@printf '  %-28s %s\n' "make compose-riscv-release" "RISC-V release via dev-riscv64"
	@printf '  %-28s %s\n' "make compose-install" "Install native via dev-cuda"
	@printf '  %-28s %s\n' "make compose-install-jetson" "Install Jetson via dev-jetson"
	@printf '  %-28s %s\n' "make compose-install-rpi" "Install RPi via dev-rpi"
	@printf '  %-28s %s\n' "make compose-install-riscv" "Install RISC-V via dev-riscv64"
	@printf '\n'
	@printf '%s\n' "Docker:"
	@printf '  %-28s %s\n' "make shell-dev" "Enter CPU development shell"
	@printf '  %-28s %s\n' "make shell-dev-cuda" "Enter CUDA development shell"
	@printf '  %-28s %s\n' "make shell-dev-jetson" "Enter Jetson cross-compile shell"
	@printf '  %-28s %s\n' "make shell-dev-rpi" "Enter Raspberry Pi shell"
	@printf '  %-28s %s\n' "make shell-dev-riscv64" "Enter RISC-V shell"
	@printf '  %-28s %s\n' "make docker-all" "Build all Docker images"
	@printf '  %-28s %s\n' "make artifacts" "Extract release artifacts"
	@printf '\n'
	@printf '%s\n' "Cleanup:"
	@printf '  %-28s %s\n' "make clean" "Clean build artifacts"
	@printf '  %-28s %s\n' "make distclean" "Remove build/ entirely"
	@printf '  %-28s %s\n' "make docker-clean" "Clean Docker dangling images"
	@printf '  %-28s %s\n' "make docker-prune" "Remove all vernier.* images"
	@printf '\n'
	@printf '%s\n' "Utilities:"
	@printf '  %-28s %s\n' "make ccache-stats" "Show ccache statistics"
	@printf '  %-28s %s\n' "make ccache-clear" "Clear ccache"
	@printf '  %-28s %s\n' "make docker-disk-usage" "Show Docker disk usage"
	@printf '  %-28s %s\n' "make docker-lint" "Lint all Dockerfiles"
	@printf '  %-28s %s\n' "make docker-validate" "Validate docker compose config"
	@printf '\n'
	@printf '%s\n' "Variables:"
	@printf '  %-28s %s\n' "VERBOSE=1" "Enable verbose CMake output"
	@printf '  %-28s %s\n' "NUM_JOBS=N" "Override parallel job count (current: $(NUM_JOBS))"
	@printf '  %-28s %s\n' "CMAKE_EXTRA_ARGS=\"...\"" "Pass extra CMake arguments"
	@printf '  %-28s %s\n' "BUILD_DIR=path" "Override build directory"
	@printf '  %-28s %s\n' "PRE_COMMIT_INSTALL=yes" "Install pre-commit hooks during prep"

# ==============================================================================
# Prep
# ==============================================================================

prep:
	@mkdir -p "$(BUILD_DIR)"
	@touch "$(BUILD_DIR)/.env"
	@if [ "$(PRE_COMMIT_INSTALL)" = "yes" ]; then \
	  printf '[prep] Installing pre-commit hooks\n'; \
	  pre-commit install; \
	fi

# ==============================================================================
# Utilities
# ==============================================================================

ccache-stats:
	@if command -v ccache >/dev/null 2>&1; then \
	  ccache -s; \
	else \
	  printf '[ccache] Not installed\n'; \
	fi

ccache-clear:
	@if command -v ccache >/dev/null 2>&1; then \
	  ccache -C; \
	  printf '[ccache] Cache cleared\n'; \
	else \
	  printf '[ccache] Not installed\n'; \
	fi

# ==============================================================================
# Phony Declarations
# ==============================================================================

.PHONY: help prep
.PHONY: ccache-stats ccache-clear
