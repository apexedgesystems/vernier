# ==============================================================================
# mk/compose.mk - Docker Compose build wrappers
#
# Runs make targets inside the correct Docker Compose service so developers
# don't have to remember which service maps to which build.
#
# Usage:
#   make compose-debug                          Native debug via dev-cuda
#   make compose-jetson-release                 Jetson cross-compile via dev-jetson
# ==============================================================================

ifndef COMPOSE_MK_GUARD
COMPOSE_MK_GUARD := 1

# ------------------------------------------------------------------------------
# Internal Helpers
# ------------------------------------------------------------------------------

# _compose_run: Run a make target inside a docker compose service
# Usage: $(call _compose_run,display_name,service,target[,extra_make_args])
define _compose_run
	$(call log,compose,$(1) [$(2)])
	@docker compose run --rm -T $(2) make $(3) \
	  VERBOSE=$(VERBOSE) CMAKE_EXTRA_ARGS="$(CMAKE_EXTRA_ARGS)" $(4)
endef

# ------------------------------------------------------------------------------
# Native Builds (dev-cuda)
# ------------------------------------------------------------------------------

compose-debug:
	$(call _compose_run,native debug,dev-cuda,debug)

compose-release:
	$(call _compose_run,native release,dev-cuda,release)

compose-docs:
	$(call _compose_run,documentation,dev-cuda,docs)

# ------------------------------------------------------------------------------
# Testing and Quality (dev-cuda)
# ------------------------------------------------------------------------------

compose-test:
	$(call _compose_run,tests (serial),dev-cuda,test)

compose-testp:
	$(call _compose_run,tests (parallel),dev-cuda,testp)

compose-coverage:
	$(call _compose_run,coverage,dev-cuda,coverage)

compose-format:
	$(call _compose_run,format (auto-fix),dev-cuda,format)

compose-format-check:
	$(call _compose_run,format (check only),dev-cuda,format-check)

compose-static:
	$(call _compose_run,static analysis,dev-cuda,static)

compose-asan:
	$(call _compose_run,AddressSanitizer,dev-cuda,asan)

compose-tsan:
	$(call _compose_run,ThreadSanitizer,dev-cuda,tsan)

compose-ubsan:
	$(call _compose_run,UBSanitizer,dev-cuda,ubsan)

# ------------------------------------------------------------------------------
# Tools (dev-cuda)
# ------------------------------------------------------------------------------

compose-tools:
	$(call _compose_run,all tools,dev-cuda,tools)

compose-tools-cpp:
	$(call _compose_run,C++ tools,dev-cuda,tools-cpp)

compose-tools-py:
	$(call _compose_run,Python tools,dev-cuda,tools-py)

compose-tools-rust:
	$(call _compose_run,Rust tools,dev-cuda,tools-rust)

# ------------------------------------------------------------------------------
# Install (artifact packaging)
# ------------------------------------------------------------------------------

compose-install:
	$(call _compose_run,install (native),dev-cuda,install)

compose-install-jetson:
	$(call _compose_run,install (Jetson),dev-jetson,install-jetson)

compose-install-rpi:
	$(call _compose_run,install (RPi),dev-rpi,install-rpi)

compose-install-riscv:
	$(call _compose_run,install (RISC-V),dev-riscv64,install-riscv)

# ------------------------------------------------------------------------------
# Cross-Compilation
# ------------------------------------------------------------------------------

compose-jetson-debug:
	$(call _compose_run,Jetson debug,dev-jetson,jetson-debug)

compose-jetson-release:
	$(call _compose_run,Jetson release,dev-jetson,jetson-release)

compose-rpi-debug:
	$(call _compose_run,Raspberry Pi debug,dev-rpi,rpi-debug)

compose-rpi-release:
	$(call _compose_run,Raspberry Pi release,dev-rpi,rpi-release)

compose-riscv-debug:
	$(call _compose_run,RISC-V 64 debug,dev-riscv64,riscv-debug)

compose-riscv-release:
	$(call _compose_run,RISC-V 64 release,dev-riscv64,riscv-release)

# ------------------------------------------------------------------------------
# Phony Declarations
# ------------------------------------------------------------------------------

.PHONY: compose-debug compose-release compose-docs
.PHONY: compose-test compose-testp compose-coverage
.PHONY: compose-format compose-format-check compose-static
.PHONY: compose-asan compose-tsan compose-ubsan
.PHONY: compose-tools compose-tools-cpp compose-tools-py compose-tools-rust
.PHONY: compose-install compose-install-jetson compose-install-rpi compose-install-riscv
.PHONY: compose-jetson-debug compose-jetson-release
.PHONY: compose-rpi-debug compose-rpi-release
.PHONY: compose-riscv-debug compose-riscv-release

endif  # COMPOSE_MK_GUARD
