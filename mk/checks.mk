# ==============================================================================
# mk/checks.mk - Registry of the checks CI runs off the pull-request path
#
# One place that names the checks the nightly workflow runs
# (.github/workflows/nightly.yml): each is a make target defined in its own
# mk/ file, run on main in the CPU dev image, one matrix leg each. A pull
# request runs ci.yml's jobs, which make verify mirrors; what is too slow or
# too broad for every pull request goes here.
#
# Adding a nightly check:
#   1. define its `make <name>` target in the appropriate mk/ file
#   2. add <name> to CHECKS_NIGHTLY below
# The workflow reads the list from `make print-nightly-checks`, so it needs no
# edit.
#
# Run them all: make checks-nightly
# Discover:     make list-checks
# ==============================================================================

ifndef CHECKS_MK_GUARD
CHECKS_MK_GUARD := 1

# ------------------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------------------
# nightly = scheduled on main, and on demand (workflow_dispatch)

# test-component: the component tests (mk/test.mk), which no pull request runs.
# test-heap-component, asan-component: the component cases of the heap tree
# (mk/test.mk) and of the AddressSanitizer selection (mk/sanitizers.mk), whose
# unit cases every pull request runs.
CHECKS_NIGHTLY := test-component test-heap-component asan-component

# ------------------------------------------------------------------------------
# Aggregate Targets
# ------------------------------------------------------------------------------

checks-nightly: $(CHECKS_NIGHTLY)

# ------------------------------------------------------------------------------
# Discovery
# ------------------------------------------------------------------------------

list-checks:
	@printf 'CI tiers:\n'
	@printf '  nightly : %s\n' "$(CHECKS_NIGHTLY)"

# A JSON array for the workflow's matrix (the registry above is its one
# source), built in make to keep the quotes out of the shell.
_comma := ,
# _json_array: space-separated list -> ["a","b","c"]
_json_array = [$(subst $(_space),$(_comma),$(patsubst %,"%",$(strip $(1))))]

print-nightly-checks:
	@printf '%s\n' '$(call _json_array,$(CHECKS_NIGHTLY))'

# ------------------------------------------------------------------------------
# Phony Declarations
# ------------------------------------------------------------------------------

.PHONY: checks-nightly list-checks print-nightly-checks

endif  # CHECKS_MK_GUARD
