# ==============================================================================
# verify -- answer "would CI pass?" before pushing.
#
# Runs the lanes of ci.yml's test jobs, each with its job's make target,
# compose service and image (the ci-cache overlay defines the lean tiers and
# the registry cache_from sources), whatever changed: CI runs a job only when
# its files change. The lanes run the unit tests; the component tests run in
# the nightly workflow (mk/checks.mk), which verify does not run. A branch is
# not "verified" until this is green; the target exists because ad-hoc local
# checks kept missing lanes (cargo build compiles no test modules, and ninja
# does not invoke cargo at all unless the stamp is dirty).
#
#   Format  make format-check in dev (vernier.dev.cpu): the formatters and
#           linters, Rust's included, find nothing to change.
#   C++     make debug, testp, test-heap in dev: the debug tree builds with the
#           GPU off and its unit tests pass, and the allocator-dependent unit
#           cases pass with tcmalloc linked.
#   Tools   make tools-check in dev: the debug build made every CLI tool.
#   ASan    make asan in dev: the unit selection passes under AddressSanitizer
#           with no report.
#   CUDA    make cuda-check in ci-build-cuda (vernier.build.cuda), no GPU:
#           every CUDA target compiles and links, and the unit cases that need
#           no device pass.
#   Rust    make test-rust in ci-build (vernier.build.cpu): the tools' Rust
#           tests pass.
#   Python  make test-py in ci-build: the tools' Python tests pass.
#
# What no lane proves, and where it is proven:
#   - The component tests, the heap tree's and the AddressSanitizer
#     selection's component cases among them, and the unit selection under
#     ThreadSanitizer and UndefinedBehaviorSanitizer: the nightly workflow
#     (make test-component, test-heap-component, asan-component, tsan and
#     ubsan).
#   - GPU execution, the CUDA cases that need a device and the two-GPU paths,
#     real bpftrace and off-CPU captures (a container's PID namespace prevents
#     them), the hardware-counter checks (they need counter access), and every
#     other rig test: make test-rig on a host that has what they need, dev-cuda
#     on a GPU host for the GPU's.
#   - The image leaves (the CUDA, cross and toolchain images): CI's
#     image-graph job.
#
# Images: the prelude builds the base and dev tiers here, against registry
# cache as CI does, and pulls the published lean build tiers
# (vernier.build.cpu, vernier.build.cuda); the lanes then run the same Make
# targets CI runs, in the same services. Hosted CI also rebuilds the lean tiers
# when a pull request changes their inputs (a gate image's, or
# docker/dev/cuda.Dockerfile for the CUDA tier); verify does not, so such a
# change is proven by the hosted run, not here. A stale local dev image fails
# configure on the baked /opt/vernier-deps tier (observed), which is why the
# prelude mirrors CI's image steps before any lane.
#
# The C++ and Python lanes run with POETRY_VIRTUALENVS_IN_PROJECT=false, as
# ci.yml's cpp and python jobs do. A host tools build can leave tools/py/.venv
# in the checkout (Poetry with in-project environments); without the setting,
# the image's Poetry adopts that environment, finds its host interpreter
# missing, and deletes and recreates it with the image's Python, which the
# host cannot run. The C++ lane reaches Poetry through the debug preset's
# Python tools build, and make testp builds through its debug prerequisite.
# ==============================================================================

# The exact compose invocation ci.yml uses.
_CI_COMPOSE := docker compose -f docker-compose.yml -f docker-compose.ci-cache.yml
_CI_REGISTRY ?= ghcr.io/apexedgesystems/vernier

verify:
	$(call log,verify,Image parity: build base+dev tiers and pull the lean build tiers)
	@$(_CI_COMPOSE) build base
	@$(_CI_COMPOSE) build dev
	@docker pull --quiet "$(_CI_REGISTRY)/vernier.build.cpu:latest" \
	  && docker tag "$(_CI_REGISTRY)/vernier.build.cpu:latest" vernier.build.cpu \
	  || true
	@docker pull --quiet "$(_CI_REGISTRY)/vernier.build.cuda:latest" \
	  && docker tag "$(_CI_REGISTRY)/vernier.build.cuda:latest" vernier.build.cuda \
	  || true
	$(call log,verify,Format lane [dev])
	@$(_CI_COMPOSE) run --rm -T dev make format-check
	$(call log,verify,C++ lane: debug build + testp + test-heap [dev with GPU off])
	@$(_CI_COMPOSE) run --rm -T -e POETRY_VIRTUALENVS_IN_PROJECT=false dev make debug CMAKE_EXTRA_ARGS="-DVERNIER_BUILD_GPU=OFF"
	@$(_CI_COMPOSE) run --rm -T -e POETRY_VIRTUALENVS_IN_PROJECT=false dev make testp
	@$(_CI_COMPOSE) run --rm -T -e POETRY_VIRTUALENVS_IN_PROJECT=false dev make test-heap
	$(call log,verify,Rust lane [ci-build])
	@$(_CI_COMPOSE) run --rm -T ci-build make test-rust
	$(call log,verify,Python lane [ci-build])
	@$(_CI_COMPOSE) run --rm -T -e POETRY_VIRTUALENVS_IN_PROJECT=false ci-build make test-py
	$(call log,verify,Tools lane: the debug build made every CLI tool [dev])
	@$(_CI_COMPOSE) run --rm -T dev make tools-check
	$(call log,verify,ASan lane: the unit selection in build/asan [dev])
	@$(_CI_COMPOSE) run --rm -T dev make asan
	$(call log,verify,CUDA lane: every CUDA target and the device-free unit cases [ci-build-cuda])
	@$(_CI_COMPOSE) run --rm -T ci-build-cuda make cuda-check
	$(call log,verify,Every pull-request lane green locally)

# ------------------------------------------------------------------------------
# Phony Declarations
# ------------------------------------------------------------------------------

.PHONY: verify
