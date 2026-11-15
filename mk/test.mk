# ==============================================================================
# mk/test.mk - Test execution
#
# CTest wrappers by test class. Every CTest entry is labelled with one class,
# unit or component, and rig beside it where it needs a rig
# (cmake/vernier/Testing.cmake):
#   unit       one unit in isolation, its collaborators values or stand-ins the
#              test controls: about 1 s a test at most, a whole unit program in
#              seconds. make test (serial) and make testp (parallel) run them,
#              and so do make verify and CI on every pull request.
#   component  a component through its real protocol, or an integration (real
#              tools, consumer builds, a benchmark end to end): about 10 s a
#              test. make test-component runs them.
#   rig        needs hardware, privileges or a device a hosted runner lacks.
#              make test-rig runs them, on a host that has it; no other lane.
# The times are guidance for review, never enforced: no test fails for its
# time, and each lane ends by listing its slowest tests. Supports serial and
# parallel execution with TTY output and logging.
#
# Note: Performance tests (ptst/) are not in CTest. Run them directly:
#   ./build/native-linux-release/bin/ptests/BenchmarkCPU_PTEST
#   ./build/native-linux-release/bin/ptests/BenchmarkGPU_PTEST
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
COVERAGE_LABEL  ?= Coverage
TIMING_LABEL    ?= Timing
COMPONENT_LABEL ?= component
RIG_LABEL       ?= rig

# Log files
TEST_LOG ?= ctest.log

# How many of its slowest tests a lane lists at its end
TEST_SLOWEST ?= 10

# Python tools directory
PY_TOOLS_DIR := tools/py
PY_LIB_DIR   := $(BUILD_DIR)/lib/python

# Heap-profiling lane: the debug preset with tcmalloc linked, in a tree of its
# own. It is the one configuration where gperf captures the CPU and the heap at
# once and where heaptrack's doctor row names tcmalloc; the default tree runs
# everything else.
HEAP_BUILD_DIR   ?= build/native-linux-debug-heap
HEAP_TEST_FILTER ?= GperfLifecycleTest.*:GperfProfileReader.*
HEAP_RUN_FILTER  ?= ReadinessFixtureRun.HeaptrackUnwrappedFails

# ------------------------------------------------------------------------------
# CTest Command Presets
# ------------------------------------------------------------------------------
# A lane whose selection matches no test fails (--no-tests=error: CTest exits
# 8): a mistyped label or a tree without tests is not a passing run. The timing
# pass may match none; the lane's selection is both passes. Each label regex is
# anchored, so a label is matched whole.

# The unit class leaves out the other classes' labels and the coverage runs, so
# a test with no class label runs with the unit tests.
UNIT_EXCLUDE := $(COVERAGE_LABEL)|$(COMPONENT_LABEL)|$(RIG_LABEL)

# The unit class, serial execution
# Note: Perf tests are not in CTest (use bin/ptests/* directly)
CTEST_ALL_SERIAL := ctest \
  -LE "^($(UNIT_EXCLUDE))$$" \
  -j1 --no-tests=error --output-on-failure

# The unit class except Timing, parallel execution
CTEST_ALL_PARALLEL := ctest \
  -LE "^($(UNIT_EXCLUDE)|$(TIMING_LABEL))$$" \
  -j$(NUM_JOBS) --no-tests=error --output-on-failure

# The unit class's Timing tests, serial execution
CTEST_TIMING_SERIAL := ctest \
  -L "^$(TIMING_LABEL)$$" -LE "^($(UNIT_EXCLUDE))$$" \
  -j1 --no-tests=ignore --output-on-failure

# The component class, rig tests aside, parallel execution
CTEST_COMPONENT := ctest \
  -L "^$(COMPONENT_LABEL)$$" -LE "^($(COVERAGE_LABEL)|$(RIG_LABEL))$$" \
  -j$(NUM_JOBS) --no-tests=error --output-on-failure

# The rig tests of both classes, serial execution
CTEST_RIG := ctest \
  -L "^$(RIG_LABEL)$$" -LE "^$(COVERAGE_LABEL)$$" \
  -j1 --no-tests=error --output-on-failure

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

# The slowest tests in a lane's log, slowest first, with their times
# (outputs to stdout, caller handles tee). Information for review only: no
# time fails a test, and the listing's own status is never the lane's.
# Usage: $(call _test_slowest,logfile)
define _test_slowest
{ printf 'slowest %s tests (information only):\n' "$(TEST_SLOWEST)"; awk '/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: / { printf "  %8.2f s  %s\n", $$(NF-1), $$4 }' "$(1)" | sort -rn | head -n $(TEST_SLOWEST) || true; }
endef

# ------------------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------------------

# Unit tests (serial) - excludes the component, rig and Coverage labels; ends
# with the slowest
test: debug
	$(call log,test,Running the unit tests (serial))
	@cd "$(BUILD_DIR)" && : > "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call _test_header,UNIT (serial) - excluding: $(UNIT_EXCLUDE)) | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call with_lib_path,$(CTEST_ALL_SERIAL)) 2>&1 | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call _test_slowest,$(TEST_LOG)) | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call _test_footer,unit (serial),$(TEST_LOG)) | tee -a "$(TEST_LOG)"

# Unit tests in parallel - runs the non-timing ones parallel, then the timing
# ones serial; ends with the slowest of both
testp: debug
	$(call log,test,Running the unit tests (parallel + timing serial))
	@cd "$(BUILD_DIR)" && : > "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call _test_header,UNIT NON-TIMING (parallel -j$(NUM_JOBS))) | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call with_lib_path,$(CTEST_ALL_PARALLEL)) 2>&1 | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && printf '\n' | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call _test_header,UNIT TIMING (serial)) | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call with_lib_path,$(CTEST_TIMING_SERIAL)) 2>&1 | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call _test_slowest,$(TEST_LOG)) | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call _test_footer,unit parallel + timing,$(TEST_LOG)) | tee -a "$(TEST_LOG)"

# Component tests (parallel) - excludes the rig and Coverage labels; ends with
# the slowest. No pull request runs them.
test-component: debug
	$(call log,test,Running the component tests (parallel))
	@cd "$(BUILD_DIR)" && : > "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call _test_header,COMPONENT (parallel -j$(NUM_JOBS)) - excluding: $(RIG_LABEL)|$(COVERAGE_LABEL)) | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call with_lib_path,$(CTEST_COMPONENT)) 2>&1 | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call _test_slowest,$(TEST_LOG)) | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call _test_footer,component (parallel),$(TEST_LOG)) | tee -a "$(TEST_LOG)"

# Rig tests (serial) - the tests labelled rig, of either class, which need
# hardware, privileges or a device a hosted runner lacks; where it is missing
# they skip. Run on a rig host; ends with the slowest.
test-rig: debug
	$(call log,test,Running the rig tests (serial))
	@cd "$(BUILD_DIR)" && : > "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call _test_header,RIG (serial) - excluding: $(COVERAGE_LABEL)) | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call with_lib_path,$(CTEST_RIG)) 2>&1 | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call _test_slowest,$(TEST_LOG)) | tee -a "$(TEST_LOG)"
	@cd "$(BUILD_DIR)" && $(call _test_footer,rig (serial),$(TEST_LOG)) | tee -a "$(TEST_LOG)"

# Heap-profiling tests: gperf's lifecycle and reader cases, the captures of
# both kinds at once included, and the heaptrack fixture run, built with
# tcmalloc (VERNIER_LINK_TCMALLOC=ON). Builds only what they need.
test-heap:
	$(call log,test,Configuring the heap-profiling tree ($(HEAP_BUILD_DIR)))
	@cmake --preset $(HOST_DEBUG_PRESET) -B "$(HEAP_BUILD_DIR)" $(CMAKE_VERBOSE_FLAG) \
	  -DVERNIER_LINK_TCMALLOC=ON -DVERNIER_BUILD_GPU=OFF -DVERNIER_BUILD_TOOLS=OFF
	$(call log,test,Building TestBench and the readiness fixture run with tcmalloc)
	@cmake --build "$(HEAP_BUILD_DIR)" -j$(NUM_JOBS) --target TestBench TestBenchReadinessRun \
	  ReadinessFixtureTarget ReadinessCustomMainTarget
	$(call log,test,Running the heap-profiling tests)
	@cd "$(HEAP_BUILD_DIR)" && $(call with_lib_path,bin/tests/TestBench --gtest_filter='$(HEAP_TEST_FILTER)')
	@cd "$(HEAP_BUILD_DIR)" && $(call with_lib_path,bin/tests/TestBenchReadinessRun --gtest_filter='$(HEAP_RUN_FILTER)')

# Python tools unit tests (runs from source, no build dependency)
# Note: Run 'cd tools/py && poetry install' first if deps are missing
test-py:
	$(call log,test,Running Python tools tests)
	@cd "$(PY_TOOLS_DIR)" && poetry install --quiet && (poetry run pytest -v || test $$? -eq 5)

# Rust tools unit tests. Inherit CARGO_HOME (the dev image bakes an offline-ready
# registry cache; locally it is ~/.cargo) so tests reuse the shared crate cache
# instead of re-downloading into a per-build throwaway. The compiled test
# artifacts still isolate to the per-build target dir.
test-rust:
	$(call log,test,Running Rust tools tests)
	@cd tools/rust && cargo test --target-dir "$(CURDIR)/$(BUILD_DIR)/rust-test-target"

# ------------------------------------------------------------------------------
# Phony Declarations
# ------------------------------------------------------------------------------

.PHONY: test testp test-component test-rig test-heap test-py test-rust

endif  # TEST_MK_GUARD
