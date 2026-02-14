# ==============================================================================
# mk/sanitizers.mk - Runtime sanitizer builds
#
# AddressSanitizer (ASan), ThreadSanitizer (TSan), and UndefinedBehaviorSanitizer
# (UBSan) for detecting memory errors, data races, and undefined behavior.
# ==============================================================================

ifndef SANITIZERS_MK_GUARD
SANITIZERS_MK_GUARD := 1

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# Note: Sanitizers are native-only. Cross-compiled targets (Jetson, RPi)
# don't support runtime sanitizer instrumentation.

# Uses BUILD_DIR and NUM_JOBS from common.mk

# ------------------------------------------------------------------------------
# Internal Helpers
# ------------------------------------------------------------------------------

# _sanitizer_build: Configure, build, and test with a sanitizer
# Usage: $(call _sanitizer_build,name,display_name)
define _sanitizer_build
	$(call log,$(1),Configuring with $(2))
	@cmake -DCMAKE_BUILD_TYPE=Debug -DSANITIZER=$(1) \
	       -B"$(BUILD_DIR)" -S. -GNinja
	$(call log,$(1),Building)
	@cd "$(BUILD_DIR)" && ninja -j$(NUM_JOBS)
	$(call log,$(1),Running tests)
	@cd "$(BUILD_DIR)" && $(call with_lib_path,ctest --output-on-failure)
endef

# ------------------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------------------

# AddressSanitizer - detects memory errors (use-after-free, buffer overflow)
asan: prep
	$(call _sanitizer_build,asan,AddressSanitizer)

# ThreadSanitizer - detects data races
tsan: prep
	$(call _sanitizer_build,tsan,ThreadSanitizer)

# UndefinedBehaviorSanitizer - detects undefined behavior
ubsan: prep
	$(call _sanitizer_build,ubsan,UBSanitizer)

# ------------------------------------------------------------------------------
# Phony Declarations
# ------------------------------------------------------------------------------

.PHONY: asan tsan ubsan

endif  # SANITIZERS_MK_GUARD
