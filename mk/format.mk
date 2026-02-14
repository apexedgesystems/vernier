# ==============================================================================
# mk/format.mk - Code formatting and linting
#
# Git-agnostic formatting using pre-commit. Scans filesystem (tracked and
# untracked files), prunes build/cache directories, and batches paths to
# avoid argument limits.
# ==============================================================================

ifndef FORMAT_MK_GUARD
FORMAT_MK_GUARD := 1

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Directories to scan (override: make format PC_SCOPE="src tools")
PC_SCOPE ?= .

# Find command with pruning for build/cache directories
PRECOMMIT_FIND := find $(PC_SCOPE) \
  \( -path ./build -o -path './cmake-build*' -o -path ./dist -o -path ./out \
     -o -path ./node_modules -o -path ./.git -o -path ./.hg -o -path ./.venv \
     -o -path ./.mypy_cache -o -path ./.pytest_cache -o -path ./.ruff_cache \
     -o -path ./.cache \) -prune -o \
  -type f -not -name 'compile_commands.json' -print0

# ------------------------------------------------------------------------------
# Internal Helpers
# ------------------------------------------------------------------------------

# Run pre-commit on all files at once
# Usage: $(call _run_precommit,extra-args)
define _run_precommit
	@$(PRECOMMIT_FIND) | xargs -0 -r pre-commit run --hook-stage manual $(1) --files
endef

# ------------------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------------------

# Auto-fix formatting issues
format:
	$(call log,format,Running formatters with auto-fix)
	$(call _run_precommit,)

# Check-only mode (no fixes), show diffs on failure
format-check:
	$(call log,format,Checking formatting (no fixes))
	$(call _run_precommit,--show-diff-on-failure)

# ------------------------------------------------------------------------------
# Phony Declarations
# ------------------------------------------------------------------------------

.PHONY: format format-check

endif  # FORMAT_MK_GUARD
