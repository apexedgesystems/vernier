# ==============================================================================
# verify -- answer "would CI pass?" before pushing.
#
# Runs the same lanes ci.yml runs, with the same compose invocation (the
# ci-cache overlay defines the lean ci-build tier and the registry
# cache_from sources), in the same modes (hosted lane parity: GPU off).
# Strictly stronger than CI's scoped runs -- every lane executes
# regardless of what changed. A branch is not "verified" until this is
# green; the target exists because ad-hoc local checks kept missing lanes
# (cargo build compiles no test modules, and ninja does not invoke cargo
# at all unless the stamp is dirty).
#
# Image parity is part of the contract: CI builds the dev tier against
# registry cache and pulls the lean build tier every run. A stale local
# dev image fails configure on the baked /opt/vernier-deps tier
# (observed), so the prelude mirrors CI's image steps before any lane.
# ==============================================================================

# The exact compose invocation ci.yml uses.
_CI_COMPOSE := docker compose -f docker-compose.yml -f docker-compose.ci-cache.yml
_CI_REGISTRY ?= ghcr.io/apexedgesystems/vernier

verify:
	$(call log,verify,Image parity: build base+dev tiers and pull the lean build tier)
	@$(_CI_COMPOSE) build base
	@$(_CI_COMPOSE) build dev
	@docker pull --quiet "$(_CI_REGISTRY)/vernier.build.cpu:latest" \
	  && docker tag "$(_CI_REGISTRY)/vernier.build.cpu:latest" vernier.build.cpu \
	  || true
	$(call log,verify,Format lane [dev])
	@$(_CI_COMPOSE) run --rm -T dev make format-check
	$(call log,verify,C++ lane: debug build + testp [dev with GPU off])
	@$(_CI_COMPOSE) run --rm -T dev make debug CMAKE_EXTRA_ARGS="-DVERNIER_BUILD_GPU=OFF"
	@$(_CI_COMPOSE) run --rm -T dev make testp
	$(call log,verify,Rust lane [ci-build])
	@$(_CI_COMPOSE) run --rm -T ci-build make test-rust
	$(call log,verify,Python lane [ci-build])
	@$(_CI_COMPOSE) run --rm -T ci-build make test-py
	$(call log,verify,All CI lanes green locally)

# ------------------------------------------------------------------------------
# Phony Declarations
# ------------------------------------------------------------------------------

.PHONY: verify
