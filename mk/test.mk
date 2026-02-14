# ==============================================================================
# mk/test.mk - Test execution
#
# CTest wrappers for unit tests and timing-sensitive tests.
# Supports both serial and parallel execution modes with TTY output and logging.
#
# Note: Performance tests (ptst/) are not in CTest. Run directly:
#   ./build/native-linux-debug/bin/ptests/Benchmarking_PTEST
# ==============================================================================

ifndef TEST_MK_GUARD
TEST_MK_GUARD := 1

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# Note: Tests run on native host builds only. Cross-compiled targets (Jetson,
# RPi) require on-device execution.

# Uses BUILD_DIR and NUM_JOBS from common.mk

# Test labels (must match CMake test properties)
COVERAGE_LABEL ?= Coverage
TIMING_LABEL   ?= Timing

# Log files
TEST_LOG ?= ctest.log

# Python tools directory
PY_TOOLS_DIR := tools/py
PY_LIB_DIR   := $(BUILD_DIR)/lib/python

# ------------------------------------------------------------------------------
# CTest Command Presets
# ------------------------------------------------------------------------------

# All tests except Coverage, serial execution
# Note: Perf tests are not in CTest (use bin/ptests/* directly)
CTEST_ALL_SERIAL := ctest \
  -LE "$(COVERAGE_LABEL)" \
  -j1 --no-tests=ignore --output-on-failure

# All tests except Coverage and Timing, parallel execution
CTEST_ALL_PARALLEL := ctest \
  -LE "$(COVERAGE_LABEL)|$(TIMING_LABEL)" \
  -j$(NUM_JOBS) --no-tests=ignore

# Timing tests only, serial execution
CTEST_TIMING_SERIAL := ctest \
  -L "$(TIMING_LABEL)" \
  -j1 --no-tests=ignore --output-on-failure

# ------------------------------------------------------------------------------
# Internal Helpers
# ------------------------------------------------------------------------------

# Log section header (outputs to stdout, caller handles tee)
# Usage: $(call _test_header,title)
define _test_header
printf '%s\nctest: %s\ndir: %s\n%s\n' "============================================================" "$(1)" "$$PWD" "------------------------------------------------------------"
endef

# Log section footer (outputs to stdout, caller handles tee)
# Usage: $(call _test_footer,summary,logfile)
define _test_footer
printf '%s\ndone: %s\nlog: %s\n%s\n' "------------------------------------------------------------" "$(1)" "$(2)" "============================================================"
endef

# ------------------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------------------

# All tests (serial) - excludes Coverage and Perf labels
test: debug
	$(call log,test,Running all tests (serial))
	@cd "$(BUILD_DIR)" && : > "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call _test_header,ALL (serial) - excluding: $(COVERAGE_LABEL)) | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call with_lib_path,$(CTEST_ALL_SERIAL)) 2>&1 | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call _test_footer,all (serial),$(TEST_LOG)) | tee -a "$(TEST_LOG)"

# Parallel tests - runs non-timing parallel, then timing serial
testp: debug
	$(call log,test,Running tests (parallel + timing serial))
	@cd "$(BUILD_DIR)" && : > "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call _test_header,NON-TIMING (parallel -j$(NUM_JOBS))) | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call with_lib_path,$(CTEST_ALL_PARALLEL)) 2>&1 | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && printf '\n' | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call _test_header,TIMING (serial)) | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call with_lib_path,$(CTEST_TIMING_SERIAL)) 2>&1 | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call _test_footer,parallel + timing,$(TEST_LOG)) | tee -a "$(TEST_LOG)"

# Python tools unit tests (runs from source, no build dependency)
# Note: Run 'cd tools/py && poetry install' first if deps are missing
test-py:
	$(call log,test,Running Python tools tests)
	@cd "$(PY_TOOLS_DIR)" && poetry install --quiet && (poetry run pytest -v || test $$? -eq 5)

# Rust tools unit tests
test-rust:
	$(call log,test,Running Rust tools tests)
	@cd tools/rust && CARGO_HOME="$(CURDIR)/$(BUILD_DIR)/rust-cargo" cargo test

# ------------------------------------------------------------------------------
# Phony Declarations
# ------------------------------------------------------------------------------

.PHONY: test testp test-py test-rust

endif  # TEST_MK_GUARD
