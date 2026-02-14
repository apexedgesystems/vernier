# ==============================================================================
# mk/coverage.mk - Code coverage instrumentation and reporting
#
# LLVM source-based coverage using Clang. Generates per-library HTML reports,
# text summaries, and LCOV exports for CI integration.
# ==============================================================================

ifndef COVERAGE_MK_GUARD
COVERAGE_MK_GUARD := 1

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# Note: Coverage is native-only. Cross-compiled targets (Jetson, RPi)
# don't support host-side coverage instrumentation.

# Uses BUILD_DIR and LLVM_VER from common.mk

# ------------------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------------------

coverage: prep
	$(call log,coverage,Configuring with coverage instrumentation)
	@cmake -DCMAKE_BUILD_TYPE=Debug \
	       -DCMAKE_C_COMPILER=clang-$(LLVM_VER) \
	       -DCMAKE_CXX_COMPILER=clang++-$(LLVM_VER) \
	       -DENABLE_COVERAGE=ON \
	       -B"$(BUILD_DIR)" -S. -GNinja
	$(call log,coverage,Building)
	@cd "$(BUILD_DIR)" && ninja -j$(NUM_JOBS)
	$(call log,coverage,Running coverage tests)
	@cd "$(BUILD_DIR)" && $(call with_lib_path,ctest -L Coverage -j1 --no-tests=ignore --output-on-failure) || true
	$(call log,coverage,Generating reports)
	@cmake --build "$(BUILD_DIR)" --target coverage-report

coverage-clean:
	$(call log,coverage,Cleaning coverage artifacts)
	@cmake --build "$(BUILD_DIR)" --target coverage-clean 2>/dev/null || true
	@find "$(BUILD_DIR)" -name '*.profraw' -delete 2>/dev/null || true

# ------------------------------------------------------------------------------
# Phony Declarations
# ------------------------------------------------------------------------------

.PHONY: coverage coverage-clean

endif  # COVERAGE_MK_GUARD
