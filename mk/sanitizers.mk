# ==============================================================================
# mk/sanitizers.mk - Runtime sanitizer builds
#
# AddressSanitizer (ASan), ThreadSanitizer (TSan), and UndefinedBehaviorSanitizer
# (UBSan) for detecting memory errors, data races, and undefined behavior.
#
# Each target configures the debug preset into a tree of its own (build/asan,
# build/tsan, build/ubsan) with the GPU and the tools off, builds the programs of
# one named selection, and runs it there. A failing test, an empty list, a
# program with no case of the selection's class, a case of a program that the
# tree's CTest inventory gives no class, or any sanitizer report, whatever the
# caller sets in the runtime's options, fails the target; the run's output is
# kept in the tree's sanitizer.log (sanitizer-component.log for the component
# selection). The debug tree is not touched. make asan, make tsan and make ubsan
# run the unit selection, make asan-component the component one; every pull
# request runs make asan, and the nightly workflow (mk/checks.mk) runs make
# asan-component, make tsan and make ubsan. For the whole suite under a
# sanitizer, build the tree and run ctest in it.
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

# A selection names its programs: a program under bin/tests, or
# dtests/<program> for one under bin/dtests. Each runs on its cases of the
# selection's class, as the tree's CTest inventory labels them once the
# programs are built: a case's class is the one its module's CMake gives it
# (cmake/vernier/Testing.cmake), and no case is named here.

# The unit selection, the unit tests of the CPU unit programs: the targets it
# builds and the programs it runs.
SANITIZER_TARGETS  ?= TestBench TestMonitor
SANITIZER_PROGRAMS ?= TestBench TestMonitor

# The component selection, the targets it builds and the programs it runs, with
# the fixture programs the readiness run starts: TestBench's component cases,
# the readiness fixture run (each case a benchmark run end to end) and the
# component test program (the backends' and readiness processes' lifecycles
# through their real protocol).
SANITIZER_COMPONENT_TARGETS  ?= TestBench TestBenchReadinessRun ReadinessFixtureTarget \
                                ReadinessCustomMainTarget TestBenchComponent
SANITIZER_COMPONENT_PROGRAMS ?= TestBench TestBenchReadinessRun dtests/TestBenchComponent

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

# A selection's variables, log and CTest labels, by its class: the unit
# selection's names carry no class, the component one's COMPONENT_. Its labels
# are the CTest lanes' (mk/test.mk): the unit selection's cases are labelled
# unit and not component, rig or Coverage, the component one's component and
# not rig or Coverage, as a rig case runs under make test-rig alone. A case
# labelled either class is classed.
_sanitizer_log_unit         = sanitizer.log
_sanitizer_log_component    = sanitizer-component.log
_sanitizer_var_unit         = SANITIZER_
_sanitizer_var_component    = SANITIZER_COMPONENT_
_sanitizer_labels_unit      = -L '^$(UNIT_LABEL)$$' -LE '^($(UNIT_EXCLUDE))$$'
_sanitizer_labels_component = -L '^$(COMPONENT_LABEL)$$' -LE '^($(COVERAGE_LABEL)|$(RIG_LABEL))$$'
_sanitizer_labels_classed   = -L '^($(UNIT_LABEL)|$(COMPONENT_LABEL))$$'

# The cases of a program that a CTest listing holds, as one GoogleTest filter:
# the --gtest_filter value of each entry whose command runs the program, read
# from ctest --show-only=json-v1 on standard input and joined by colons. CTest
# gives an entry its command once the program is built; an entry that runs the
# whole program (a coverage run) adds nothing.
# Usage: <listing> | $(call _ctest_gtest_filter,program)
_ctest_gtest_filter = python3 -c 'import json, os, sys; p = os.path.realpath(sys.argv[1]); \
  print(":".join(a.partition("=")[2] for t in json.load(sys.stdin)["tests"] \
  if "command" in t and os.path.realpath(t["command"][0]) == p \
  for a in t["command"][1:] if a.startswith("--gtest_filter=")))' '$(1)'

# Run a program on its cases of a selection's class, from inside its built
# tree, as the tree's CTest inventory labels them, so a case added to the
# program or given another class in its module's CMake joins that class's
# selection with no edit here. It fails when the inventory labels none of the
# program's cases with the class, and when the program holds a case that the
# inventory gives no class, which no selection would run: the program lists
# its cases outside every classed entry's filter (<program>.unclassed.flags),
# and each one listed is named. The filter reaches the program in
# <program>.<class>.flags, kept as the record of what ran; the program writes
# gtest's JSON report, <program>.json, removed before the run. Each failure
# exits explicitly, as in _gtest_selection.
# Usage: $(call _sanitizer_class_run,program,class)
_sanitizer_unclassed = has no class label in the CTest inventory, so no selection runs it
define _sanitizer_class_run
sel=$$(ctest --show-only=json-v1 $(_sanitizer_labels_$(2)) | $(call _ctest_gtest_filter,$(1))) \
  || exit $$?; \
[ -n "$$sel" ] || { printf '%s: the CTest inventory labels none of its cases %s\n' \
  '$(notdir $(1))' '$(2)' >&2; exit 1; }; \
classed=$$(ctest --show-only=json-v1 $(_sanitizer_labels_classed) \
  | $(call _ctest_gtest_filter,$(1))) || exit $$?; \
printf '%s%s\n' '--gtest_filter=*-' "$$classed" > $(notdir $(1)).unclassed.flags; \
listed=$$($(call with_lib_path,$(1)) --gtest_list_tests \
  --gtest_flagfile=$(notdir $(1)).unclassed.flags) || exit $$?; \
unclassed=$$(printf '%s\n' "$$listed" | awk '/^[^ ]/ { s = $$1 } /^  / { print s $$1 }'); \
if [ -n "$$unclassed" ]; then \
  printf '%s\n' "$$unclassed" | sed 's/.*/$(notdir $(1)): & $(_sanitizer_unclassed)/' >&2; \
  exit 1; \
fi; \
printf '%s%s\n' '--gtest_filter=' "$$sel" > $(notdir $(1)).$(2).flags; \
printf '%s: the cases the CTest inventory labels %s (%s)\n' '$(notdir $(1))' '$(2)' \
  '$(notdir $(1)).$(2).flags'; \
rm -f $(notdir $(1)).json; $(call with_lib_path,$(1)) --gtest_flagfile=$(notdir $(1)).$(2).flags \
  --gtest_output=json:$(notdir $(1)).json || exit $$?
endef

# _sanitizer_build: Configure, build, and run one selection with a sanitizer,
# from inside its tree, then search the run's log for reports
# Usage: $(call _sanitizer_build,name,display_name,class)
define _sanitizer_build
	@$(call _require_lists,$(_sanitizer_var_$(3))TARGETS $(_sanitizer_var_$(3))PROGRAMS)
	$(call log,$(1),Configuring build/$(1) with $(2))
	@cmake --preset $(HOST_DEBUG_PRESET) -B "build/$(1)" --fresh $(CMAKE_VERBOSE_FLAG) \
	  -DSANITIZER=$(1) -DVERNIER_BUILD_GPU=OFF -DVERNIER_BUILD_TOOLS=OFF
	$(call log,$(1),Building the programs of the $(3) selection)
	@cmake --build "build/$(1)" -j$(NUM_JOBS) --target $($(_sanitizer_var_$(3))TARGETS)
	$(call log,$(1),Running the $(3) selection under $(2) (log: build/$(1)/$(_sanitizer_log_$(3))))
	@cd "build/$(1)" && { $(_sanitizer_env_$(1)) $(foreach p,$($(_sanitizer_var_$(3))PROGRAMS),$(call \
	  _sanitizer_class_run,$(call _selection_program,$(p)),$(3));) } 2>&1 \
	  | tee $(_sanitizer_log_$(3)) || { \
	  $(if $(_sanitizer_limit_$(1)),grep -qE '$(_sanitizer_limit_$(1))' $(_sanitizer_log_$(3)) \
	    && printf '[%s] %s\n' '$(1)' '$(_sanitizer_remedy_$(1))' >&2;) \
	  exit 1; }
	@cd "build/$(1)" && if grep -qE '$(_sanitizer_reports)' $(_sanitizer_log_$(3)); then \
	  printf '[%s] %s\n' '$(1)' 'A sanitizer report is in build/$(1)/$(_sanitizer_log_$(3)):' >&2; \
	  grep -m 20 -E '$(_sanitizer_reports)' $(_sanitizer_log_$(3)) >&2; \
	  exit 1; \
	fi
	@cd "build/$(1)" && $(call _selection_slowest,$(call \
	  _selection_reports,$($(_sanitizer_var_$(3))PROGRAMS))) | tee -a $(_sanitizer_log_$(3))
endef

# ------------------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------------------

# AddressSanitizer - detects memory errors (use-after-free, buffer overflow)
asan:
	$(call _sanitizer_build,asan,AddressSanitizer,unit)

# AddressSanitizer on the component selection, in the same tree
asan-component:
	$(call _sanitizer_build,asan,AddressSanitizer,component)

# ThreadSanitizer - detects data races
tsan:
	$(call _sanitizer_build,tsan,ThreadSanitizer,unit)

# UndefinedBehaviorSanitizer - detects undefined behavior
ubsan:
	$(call _sanitizer_build,ubsan,UBSanitizer,unit)

# ------------------------------------------------------------------------------
# Phony Declarations
# ------------------------------------------------------------------------------

.PHONY: asan asan-component tsan ubsan

endif  # SANITIZERS_MK_GUARD
