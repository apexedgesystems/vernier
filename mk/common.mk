# ==============================================================================
# mk/common.mk - Shared configuration and utilities
#
# Provides common variables, logging helpers, and environment wrappers used
# across all Vernier Makefile modules.
# ==============================================================================

ifndef COMMON_MK_GUARD
COMMON_MK_GUARD := 1

# ------------------------------------------------------------------------------
# Build Configuration
# ------------------------------------------------------------------------------

# Parallel jobs (auto-detect CPU count)
# Use nproc --all to ignore OMP_NUM_THREADS (set to 1 by NVIDIA containers)
NUM_JOBS ?= $(shell nproc --all 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# LLVM toolchain version
LLVM_VER ?= 21

# Default build directory (native debug)
BUILD_DIR ?= build/native-linux-debug

# Project version (extracted from CMakeLists.txt)
PROJECT_VERSION := $(shell sed -n 's/.*VERSION \([0-9]*\.[0-9]*\.[0-9]*\).*/\1/p' \
  CMakeLists.txt | head -1)
export PROJECT_VERSION

# ------------------------------------------------------------------------------
# Logging Utilities
# ------------------------------------------------------------------------------
# Usage: $(call log,tag,message)
#   $(call log,build,Compiling sources...)
#   $(call log,test,Running unit tests)

define log
	@printf '[%s] %s\n' '$(1)' '$(2)'
endef

# Colored variants (optional, degrades gracefully)
# Usage: $(call log_ok,tag,message) - green
#        $(call log_warn,tag,message) - yellow
#        $(call log_err,tag,message) - red

TERM_GREEN  := $(shell printf '\033[32m')
TERM_YELLOW := $(shell printf '\033[33m')
TERM_RED    := $(shell printf '\033[31m')
TERM_RESET  := $(shell printf '\033[0m')

define log_ok
	@printf '$(TERM_GREEN)[%s]$(TERM_RESET) %s\n' '$(1)' '$(2)'
endef

define log_warn
	@printf '$(TERM_YELLOW)[%s]$(TERM_RESET) %s\n' '$(1)' '$(2)'
endef

define log_err
	@printf '$(TERM_RED)[%s]$(TERM_RESET) %s\n' '$(1)' '$(2)'
endef

# ------------------------------------------------------------------------------
# Environment Wrappers
# ------------------------------------------------------------------------------
# Usage: $(call with_lib_path,command)
#   Runs command with LD_LIBRARY_PATH set to include build/lib

define with_lib_path
env LD_LIBRARY_PATH="$$PWD/lib:$$LD_LIBRARY_PATH" $(1)
endef

# ------------------------------------------------------------------------------
# Path Utilities
# ------------------------------------------------------------------------------

# Check if a command exists
# Usage: $(call cmd_exists,command)
define cmd_exists
$(shell command -v $(1) >/dev/null 2>&1 && echo yes || echo no)
endef

endif  # COMMON_MK_GUARD
