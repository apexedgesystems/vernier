# ==============================================================================
# mk/sanitizers.mk - Runtime sanitizer builds
#
# AddressSanitizer (ASan), ThreadSanitizer (TSan), and UndefinedBehaviorSanitizer
# (UBSan) for detecting memory errors, data races, and undefined behavior.
#
# Each target configures the debug preset into a tree of its own (build/asan,
# build/tsan, build/ubsan) with the GPU and the tools off, builds the programs of
# one named selection, and runs it there. A failing test, an empty list, a
# pattern that selects no test, or any sanitizer report, whatever the caller
# sets in the runtime's options, fails the target; the run's output is kept in
# the tree's sanitizer.log. The debug tree is not touched. For the whole suite
# under a sanitizer, build the tree and run ctest in it.
# ==============================================================================

ifndef SANITIZERS_MK_GUARD
SANITIZERS_MK_GUARD := 1

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# Note: Sanitizers are native-only. Cross-compiled targets (Jetson, RPi)
# don't support runtime sanitizer instrumentation. The GPU stays off: nvcc
# compiles the host code of CUDA sources with its default host compiler (GCC),
# whose sanitizer runtimes do not pair with clang's.

# The selection's programs, and the fixture programs the readiness run starts.
SANITIZER_TARGETS ?= TestBench TestBenchReadinessRun ReadinessFixtureTarget \
                     ReadinessCustomMainTarget TestMonitor

# The selection: program=pattern[:pattern...] for programs under bin/tests, with
# SANITIZER_EXCLUDE taken out of every program's patterns.
SANITIZER_TESTS ?= TestBench=* TestBenchReadinessRun=* TestMonitor=*

# TestBench's process-lifecycle scenarios (fake tracers and tools started and
# stopped through the real code, signals escalated, children reaped, each
# waiting out real timeouts) and the six tests that run every backend's real
# check on the machine: nearly all of the program's run time. This list goes
# when those tests leave TestBench for a program of their own.
SANITIZER_EXCLUDE ?= BpfCheckTest.* OffCpuCaptureTest.* ReadinessProbes.* \
  ReadinessOwnedHelper.* ReadinessChildProcesses.* ZombieCount.* PerfLifecycleTest.* \
  ProfilerReadinessRouting.AnalysisStageErrorFailsEveryViewAndItsTwinIsOk \
  ProfilerReadinessRouting.DoctorJsonKeys \
  ProfilerReadinessRouting.DoctorLabelsRowsAndDecidesAfresh \
  ProfilerReadinessRouting.DoctorSelectedRow \
  ProfilerRegistryTest.RunAllChecksMatchesBackendNames \
  ProfilerRegistryTest.RunCheckReturnsStatus

# ------------------------------------------------------------------------------
# Internal Helpers
# ------------------------------------------------------------------------------

# A report fails the target whatever options the caller gives the runtimes, by
# two guards. The run keeps the caller's options and appends the failure policy,
# the last value of an option winning: halt_on_error=1 where a runtime would
# report and carry on (UBSan by default; ASan only when built to recover), and
# the runtime's own nonzero exitcode, which a caller's exitcode=0 would replace.
# exitcode goes into every variable the runtime reads after its own, since a
# later one overrides it: ASan reads LSAN_OPTIONS and UBSAN_OPTIONS, TSan reads
# UBSAN_OPTIONS. TSan keeps reporting every race and sets that status at exit;
# UBSan also gets a stack trace when the caller sets nothing. Then the log is
# searched for report lines, which catches a report whose process still exits
# 0. No clean run prints such a line.
_sanitizer_append    = export $(1)="$${$(1):+$$$(1):}$(2)";
_sanitizer_env_asan  = $(call _sanitizer_append,ASAN_OPTIONS,halt_on_error=1:exitcode=1) \
  $(call _sanitizer_append,LSAN_OPTIONS,exitcode=1) \
  $(call _sanitizer_append,UBSAN_OPTIONS,exitcode=1)
_sanitizer_env_tsan  = $(call _sanitizer_append,TSAN_OPTIONS,exitcode=66) \
  $(call _sanitizer_append,UBSAN_OPTIONS,exitcode=66)
_sanitizer_env_ubsan = export UBSAN_OPTIONS="$${UBSAN_OPTIONS-print_stacktrace=1}"; \
  $(call _sanitizer_append,UBSAN_OPTIONS,halt_on_error=1:exitcode=1)
_sanitizer_reports   = ^==[0-9]+==ERROR: [A-Za-z]+Sanitizer|^WARNING: ThreadSanitizer:|: runtime error: |^SUMMARY: [A-Za-z]+Sanitizer:

# A runtime that cannot start here prints its own fatal line; the target then
# names the limitation and the remedy, and still fails.
_sanitizer_limit_tsan = ThreadSanitizer: unexpected memory mapping|incompatible memory layout
_sanitizer_remedy_tsan = ThreadSanitizer could not run here: it needs ASLR off for its own \
  process, and this process may not turn it off. Run make compose-tsan (the dev service allows \
  the personality call that does it), or setarch -R make tsan where that call is allowed.

# The whole selection, run from inside a sanitizer tree.
_sanitizer_selection = $(call _gtest_selections,$(SANITIZER_TESTS),$(SANITIZER_EXCLUDE))

# _sanitizer_build: Configure, build, and run the selection with a sanitizer
# Usage: $(call _sanitizer_build,name,display_name)
define _sanitizer_build
	@$(call _require_lists,SANITIZER_TARGETS SANITIZER_TESTS)
	$(call log,$(1),Configuring build/$(1) with $(2))
	@cmake --preset $(HOST_DEBUG_PRESET) -B "build/$(1)" --fresh $(CMAKE_VERBOSE_FLAG) \
	  -DSANITIZER=$(1) -DVERNIER_BUILD_GPU=OFF -DVERNIER_BUILD_TOOLS=OFF
	$(call log,$(1),Building the programs of the selection)
	@cmake --build "build/$(1)" -j$(NUM_JOBS) --target $(SANITIZER_TARGETS)
	$(call log,$(1),Running the selection under $(2) (log: build/$(1)/sanitizer.log))
	@cd "build/$(1)" && { $(_sanitizer_env_$(1)) $(_sanitizer_selection) } 2>&1 \
	  | tee sanitizer.log || { \
	  $(if $(_sanitizer_limit_$(1)),grep -qE '$(_sanitizer_limit_$(1))' sanitizer.log \
	    && printf '[%s] %s\n' '$(1)' '$(_sanitizer_remedy_$(1))' >&2;) \
	  exit 1; }
	@cd "build/$(1)" && if grep -qE '$(_sanitizer_reports)' sanitizer.log; then \
	  printf '[%s] %s\n' '$(1)' 'A sanitizer report is in build/$(1)/sanitizer.log:' >&2; \
	  grep -m 20 -E '$(_sanitizer_reports)' sanitizer.log >&2; \
	  exit 1; \
	fi
endef

# ------------------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------------------

# AddressSanitizer - detects memory errors (use-after-free, buffer overflow)
asan:
	$(call _sanitizer_build,asan,AddressSanitizer)

# ThreadSanitizer - detects data races
tsan:
	$(call _sanitizer_build,tsan,ThreadSanitizer)

# UndefinedBehaviorSanitizer - detects undefined behavior
ubsan:
	$(call _sanitizer_build,ubsan,UBSanitizer)

# ------------------------------------------------------------------------------
# Phony Declarations
# ------------------------------------------------------------------------------

.PHONY: asan tsan ubsan

endif  # SANITIZERS_MK_GUARD
