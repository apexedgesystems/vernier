# ==============================================================================
# mk/test.mk - Test execution
#
# CTest wrappers for unit tests and timing-sensitive tests.
# Supports both serial and parallel execution modes with TTY output and logging.
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
COVERAGE_LABEL ?= Coverage
TIMING_LABEL   ?= Timing

# Log files
TEST_LOG ?= ctest.log

# Python tools directory
PY_TOOLS_DIR := tools/py
PY_LIB_DIR   := $(BUILD_DIR)/lib/python

# Heap-profiling lane: the debug preset with tcmalloc linked, in a tree of its
# own. It is the one configuration where gperf captures the CPU and the heap at
# once and where heaptrack's doctor row names tcmalloc. It runs every case whose
# expectation depends on whether tcmalloc is linked; the default tree runs the
# same cases without it, and everything else. The selection is space-separated:
# gtest patterns for TestBench and for the readiness fixture run, then CTest
# names. Each must select a test, so a renamed suite fails the lane instead of
# leaving it, and none of the three lists may be empty.
HEAP_BUILD_DIR    ?= build/native-linux-debug-heap
HEAP_TEST_FILTERS ?= GperfLifecycleTest.* GperfProfileReader.* TcmallocOptInTest.* \
                     TcmallocOptInDeathTest.* GperfCheckTest.* \
                     GperfSamplingCheck.HeapRefusalsInTheBuildsOrder
HEAP_RUN_FILTERS  ?= ReadinessFixtureRun.HeaptrackUnwrappedFails ReadinessFixtureRun.Gperf*
HEAP_CTEST_TESTS  ?= ReadinessCli.GperfHeapWithoutSupport

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
  -j$(NUM_JOBS) --no-tests=ignore --output-on-failure

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

_empty :=
_space := $(_empty) $(_empty)

# Space-separated gtest patterns as one filter field, and as its negative part
# (empty when there are none).
_gtest_join       = $(subst $(_space),:,$(strip $(1)))
_gtest_exclusions = $(if $(strip $(1)),-$(call _gtest_join,$(1)))

# Fail, naming the variable, when a selection list is empty: a target would
# otherwise run nothing from it and pass. Checked before anything is built.
# Usage: $(call _require_lists,VAR...)
_require_lists = $(foreach v,$(1),$(if $(strip $($(v))),,printf '%s is empty: the selection names nothing\n' '$(v)' >&2; exit 1;)) true

# Run a GoogleTest program on a named selection, from inside its build tree.
# Each pattern must select a test once the exclusions are applied, so a renamed
# suite fails the run instead of leaving it; the program then runs once on the
# whole selection. No pattern at all fails too: gtest runs no test for an empty
# filter, and every test but the exclusions for exclusions alone. Each failure
# exits explicitly, since errexit does not hold where a caller tests the run's
# status (a teed run followed by ||). With a fourth argument, no-skips, a
# skipped test fails the run too: the program writes gtest's JSON report, and
# that report, the parent's own, decides, which no child's output can fake.
# Usage: $(call _gtest_selection,program,patterns[,exclusions[,no-skips]])
define _gtest_selection
$(if $(strip $(2)),,printf '%s: the selection names no pattern\n' '$(notdir $(1))' >&2; exit 1;) \
for p in $(foreach p,$(2),'$(p)'); do \
  listed=$$($(call with_lib_path,$(1)) --gtest_list_tests \
    --gtest_filter="$$p$(call _gtest_exclusions,$(3))") || exit $$?; \
  if [ "$$(printf '%s\n' "$$listed" | grep -c '^  ' || true)" -eq 0 ]; then \
    printf '%s: %s selects no test\n' '$(notdir $(1))' "$$p" >&2; \
    exit 1; \
  fi; \
done; \
$(if $(4),rm -f $(notdir $(1)).json; )$(call with_lib_path,$(1)) \
  --gtest_filter='$(call _gtest_join,$(2))$(call _gtest_exclusions,$(3))'$(if $(4), \
  --gtest_output=json:$(notdir $(1)).json) || exit $$?$(if $(4),$(call _gtest_refuse_skips,$(notdir $(1))))
endef

# The end of a run that refuses skips, for the program named: its JSON report,
# removed before the run, must exist and record no skipped test.
_gtest_refuse_skips = ; [ -s $(1).json ] || { printf '%s: gtest wrote no JSON report\n' '$(1)' >&2; exit 1; }; if grep -q '"result": "SKIPPED"' $(1).json; then printf '%s: a selected test skipped, and this selection refuses skips\n' '$(1)' >&2; exit 1; fi

# Run program=pattern[:pattern...] words (programs under bin/tests), each
# through _gtest_selection with the same exclusions and skip rule, from inside
# the build tree; each run ends with a semicolon.
# Usage: $(call _gtest_selections,words[,exclusions[,no-skips]])
_selection_program  = bin/tests/$(firstword $(subst =, ,$(1)))
_selection_patterns = $(subst :, ,$(word 2,$(subst =, ,$(1))))
_gtest_selections   = $(foreach t,$(1),$(call _gtest_selection,$(call \
  _selection_program,$(t)),$(call _selection_patterns,$(t)),$(2),$(3));)

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

# Allocator-dependent tests, built with tcmalloc (VERNIER_LINK_TCMALLOC=ON):
# gperf's lifecycle, reader, check and sampling cases (the captures of both
# kinds at once included), the tcmalloc opt-in cases, and the gperf and
# heaptrack fixture runs, in the gtest selections above, and the command-line
# heap case through CTest. Builds only what they need.
test-heap:
	@$(call _require_lists,HEAP_TEST_FILTERS HEAP_RUN_FILTERS HEAP_CTEST_TESTS)
	$(call log,test,Configuring the heap-profiling tree ($(HEAP_BUILD_DIR)))
	@cmake --preset $(HOST_DEBUG_PRESET) -B "$(HEAP_BUILD_DIR)" $(CMAKE_VERBOSE_FLAG) \
	  -DVERNIER_LINK_TCMALLOC=ON -DVERNIER_BUILD_GPU=OFF -DVERNIER_BUILD_TOOLS=OFF
	$(call log,test,Building TestBench and the readiness fixture run with tcmalloc)
	@cmake --build "$(HEAP_BUILD_DIR)" -j$(NUM_JOBS) --target TestBench TestBenchReadinessRun \
	  ReadinessFixtureTarget ReadinessCustomMainTarget
	$(call log,test,Running the heap-profiling tests)
	@cd "$(HEAP_BUILD_DIR)" && $(call _gtest_selection,bin/tests/TestBench,$(HEAP_TEST_FILTERS))
	@cd "$(HEAP_BUILD_DIR)" && $(call _gtest_selection,bin/tests/TestBenchReadinessRun,$(HEAP_RUN_FILTERS))
	@cd "$(HEAP_BUILD_DIR)" && for t in $(foreach t,$(HEAP_CTEST_TESTS),'^$(subst .,\.,$(t))$$'); do \
	  $(call with_lib_path,ctest --output-on-failure --no-tests=error -R "$$t") || exit $$?; \
	done

# ------------------------------------------------------------------------------
# CUDA lane
# ------------------------------------------------------------------------------
# Every CUDA target compiled and linked in the shipped configuration (the debug
# preset, the GPU, CUPTI and NVML on, the tools off) in build/cuda-check, which
# must have compiled the CUPTI and NVML collection paths; then the cases that
# need no device, in build/cuda-check-nonvml, built without NVML: a program
# linked to NVML cannot start without the driver's libnvidia-ml.so.1, and the
# CUDA images carry only its link stub. A selected case that skips fails the
# lane. Runs in the CUDA build image; the cases that need a device run on a GPU
# host. The CUDA launcher is named on the configure line because the preset's
# build acceleration sets it only once CUDA is known, which a new tree is not.
CUDA_TARGETS ?= bench_cuda NvtxRangeRecorder demo_example_saxpy TestBenchGpu TestBenchNsight \
  TestBenchNsightRange TestDemoKernelColumns TestDemoComputeSanitizer TestDemoNvtxRanges \
  TestDemoTranspose TestDemoBankConflicts TestDemoSaxpy BenchDemo_Gpu_01_GpuBasicWorkflow \
  BenchDemo_Gpu_02_NsightProfiler BenchDemo_Gpu_03_SharedMemoryOpt \
  BenchDemo_Gpu_04_ComputeSanitizerProfiler BenchDemo_Gpu_05_NvtxAnnotation BenchmarkGPU_PTEST
CUDA_NODEVICE_TARGETS ?= TestBenchNsight TestBenchNsightRange TestDemoComputeSanitizer \
  BenchDemo_Gpu_04_ComputeSanitizerProfiler
CUDA_NODEVICE_TESTS ?= TestBenchNsight=* TestBenchNsightRange=NsightRangeTest.* \
  TestDemoComputeSanitizer=ComputeSanitizer.UnguardedSkipsOutsideTheTool

_cuda_check_needs_nvcc = cuda-check needs the CUDA toolkit (nvcc), which this environment \
  lacks: run it in the CUDA build image, docker compose -f docker-compose.yml -f \
  docker-compose.ci-cache.yml run --rm -T ci-build-cuda make cuda-check

cuda-check:
	@$(call _require_lists,CUDA_TARGETS CUDA_NODEVICE_TARGETS CUDA_NODEVICE_TESTS)
	@command -v nvcc >/dev/null 2>&1 || { printf '%s\n' '$(_cuda_check_needs_nvcc)' >&2; exit 1; }
	$(call log,cuda,Configuring build/cuda-check in the shipped configuration)
	@cmake --preset $(HOST_DEBUG_PRESET) -B build/cuda-check --fresh $(CMAKE_VERBOSE_FLAG) \
	  -DVERNIER_BUILD_TOOLS=OFF -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
	$(call log,cuda,Building every CUDA target)
	@cmake --build build/cuda-check -j$(NUM_JOBS) --target $(CUDA_TARGETS)
	@grep -qE '"command": .*-DCOMPAT_CUPTI_AVAILABLE=1 .*CuptiCollector\.cu' \
	  build/cuda-check/compile_commands.json \
	  || { printf '%s\n' 'cuda-check: CuptiCollector.cu was not compiled with CUPTI' >&2; exit 1; }
	@grep -qE '"command": .*-DCOMPAT_NVML_AVAILABLE=1 .*PerfGpuHarness\.cu' \
	  build/cuda-check/compile_commands.json \
	  || { printf '%s\n' 'cuda-check: PerfGpuHarness.cu was not compiled with NVML' >&2; exit 1; }
	$(call log,cuda,Configuring build/cuda-check-nonvml without NVML)
	@cmake --preset $(HOST_DEBUG_PRESET) -B build/cuda-check-nonvml --fresh $(CMAKE_VERBOSE_FLAG) \
	  -DVERNIER_BUILD_TOOLS=OFF -DVERNIER_USE_NVML=OFF -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
	@cmake --build build/cuda-check-nonvml -j$(NUM_JOBS) --target $(CUDA_NODEVICE_TARGETS)
	$(call log,cuda,Running the cases that need no device)
	@cd build/cuda-check-nonvml && { $(call _gtest_selections,$(CUDA_NODEVICE_TESTS),,no-skips) }

.PHONY: cuda-check

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

.PHONY: test testp test-heap test-py test-rust

endif  # TEST_MK_GUARD
