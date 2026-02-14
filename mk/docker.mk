# ==============================================================================
# mk/docker.mk - Container builds and management
#
# Wrapper around docker compose for building development shells, toolchain
# images, builder images, and extracting release artifacts.
# ==============================================================================

ifndef DOCKER_MK_GUARD
DOCKER_MK_GUARD := 1

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Enable BuildKit for cache mounts and improved layer caching
export DOCKER_BUILDKIT := 1
export COMPOSE_DOCKER_CLI_BUILD := 1

# Export UID/GID for docker compose (UID is readonly in bash, won't export normally)
export USER := $(shell id -un)
export UID  := $(shell id -u)
export GID  := $(shell id -g)

# Artifact output directory
DOCKER_OUT_DIR := output

# ------------------------------------------------------------------------------
# Internal Helpers
# ------------------------------------------------------------------------------

# _docker_build: Build a docker compose service
# Usage: $(call _docker_build,tag,display_name,service)
define _docker_build
	$(call log,docker,Building $(2))
	@docker compose build $(3)
endef

# _docker_shell: Run an interactive shell in a docker compose service
# Usage: $(call _docker_shell,service)
define _docker_shell
	@docker compose run --rm $(1)
endef

# ------------------------------------------------------------------------------
# Aggregate Targets
# ------------------------------------------------------------------------------

docker-all: docker-base docker-toolchains docker-devs docker-builders docker-final
	$(call log,docker,All images built)

docker-toolchains: docker-toolchain-aarch64 docker-toolchain-rpi docker-toolchain-riscv64
	$(call log,docker,All toolchain images built)

docker-devs: docker-dev docker-dev-cuda docker-dev-jetson docker-dev-rpi docker-dev-riscv64
	$(call log,docker,All dev images built)

docker-builders: docker-builder-cpu docker-builder-cuda docker-builder-jetson \
                 docker-builder-rpi docker-builder-riscv64
	$(call log,docker,All builder images built)

# ------------------------------------------------------------------------------
# Base Image
# ------------------------------------------------------------------------------

docker-base:
	$(call _docker_build,base,base image,base)

# ------------------------------------------------------------------------------
# Toolchain Images
# ------------------------------------------------------------------------------

docker-toolchain-aarch64: docker-base
	$(call _docker_build,toolchain,aarch64 toolchain,toolchain-aarch64)

docker-toolchain-rpi: docker-toolchain-aarch64
	$(call _docker_build,toolchain,Raspberry Pi toolchain,toolchain-rpi)

docker-toolchain-riscv64: docker-base
	$(call _docker_build,toolchain,RISC-V 64 toolchain,toolchain-riscv64)

# ------------------------------------------------------------------------------
# Development Shell Images
# ------------------------------------------------------------------------------

docker-dev: docker-base
	$(call _docker_build,dev,CPU dev shell,dev)

docker-dev-cuda: docker-base
	$(call _docker_build,dev,CUDA dev shell,dev-cuda)

docker-dev-jetson: docker-dev-cuda docker-toolchain-aarch64
	$(call _docker_build,dev,Jetson dev shell,dev-jetson)

docker-dev-rpi: docker-dev docker-toolchain-rpi
	$(call _docker_build,dev,Raspberry Pi dev shell,dev-rpi)

docker-dev-riscv64: docker-dev docker-toolchain-riscv64
	$(call _docker_build,dev,RISC-V 64 dev shell,dev-riscv64)

# ------------------------------------------------------------------------------
# Builder Images (CI Artifact Generation)
# ------------------------------------------------------------------------------

docker-builder-cpu: docker-dev
	$(call _docker_build,builder,CPU builder,builder-cpu)

docker-builder-cuda: docker-dev-cuda
	$(call _docker_build,builder,CUDA builder,builder-cuda)

docker-builder-jetson: docker-dev-jetson
	$(call _docker_build,builder,Jetson builder,builder-jetson)

docker-builder-rpi: docker-dev-rpi
	$(call _docker_build,builder,Raspberry Pi builder,builder-rpi)

docker-builder-riscv64: docker-dev-riscv64
	$(call _docker_build,builder,RISC-V 64 builder,builder-riscv64)

# ------------------------------------------------------------------------------
# Final Image (Artifact Packaging)
# ------------------------------------------------------------------------------

docker-final: docker-builders
	$(call _docker_build,final,final artifact image,final)

# ------------------------------------------------------------------------------
# Artifact Extraction
# ------------------------------------------------------------------------------

artifacts: docker-final
	$(call log,docker,Extracting artifacts to $(DOCKER_OUT_DIR)/)
	@mkdir -p $(DOCKER_OUT_DIR)
	@CID=$$(docker create vernier.final) && \
	  docker cp $$CID:/output/vernier-$(PROJECT_VERSION)-x86_64-linux.tar.gz $(DOCKER_OUT_DIR)/ && \
	  docker cp $$CID:/output/vernier-$(PROJECT_VERSION)-x86_64-linux-cuda.tar.gz $(DOCKER_OUT_DIR)/ && \
	  docker cp $$CID:/output/vernier-$(PROJECT_VERSION)-aarch64-jetson.tar.gz $(DOCKER_OUT_DIR)/ && \
	  docker cp $$CID:/output/vernier-$(PROJECT_VERSION)-aarch64-rpi.tar.gz $(DOCKER_OUT_DIR)/ && \
	  docker cp $$CID:/output/vernier-$(PROJECT_VERSION)-riscv64-linux.tar.gz $(DOCKER_OUT_DIR)/ && \
	  docker cp $$CID:/output/vernier-tools-$(PROJECT_VERSION)-x86_64-linux.tar.gz $(DOCKER_OUT_DIR)/ && \
	  docker cp $$CID:/output/tools-py-staging/. $(DOCKER_OUT_DIR)/ && \
	  docker rm $$CID
	$(call log,docker,Artifacts ready in $(DOCKER_OUT_DIR)/)

# ------------------------------------------------------------------------------
# Interactive Shells
# ------------------------------------------------------------------------------

shell-dev: docker-dev
	$(call _docker_shell,dev)

shell-dev-cuda: docker-dev-cuda
	$(call _docker_shell,dev-cuda)

shell-dev-jetson: docker-dev-jetson
	$(call _docker_shell,dev-jetson)

shell-dev-rpi: docker-dev-rpi
	$(call _docker_shell,dev-rpi)

shell-dev-riscv64: docker-dev-riscv64
	$(call _docker_shell,dev-riscv64)

# ------------------------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------------------------

docker-clean:
	$(call log,docker,Cleaning dangling images and stopped containers)
	@docker image prune -f
	@docker container prune -f
	@docker network prune -f

docker-clean-deep:
	$(call log,docker,Deep cleanup including unused images and volumes)
	@docker system prune -af --volumes

docker-prune:
	$(call log,docker,Removing vernier project images)
	@docker images --format '{{.Repository}}:{{.Tag}}' | grep '^vernier\.' | xargs -r docker rmi -f 2>/dev/null || true
	@docker image prune -f

docker-disk-usage:
	@echo "Docker System Disk Usage:"
	@docker system df
	@echo ""
	@echo "Vernier Image Sizes:"
	@docker images --format "  {{.Repository}}:{{.Tag}} => {{.Size}}" | grep vernier | sort

# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------

docker-lint:
	$(call log,docker,Running hadolint on all Dockerfiles)
	@find docker -name "*.Dockerfile" -exec hadolint {} \;

docker-validate: docker-lint
	$(call log,docker,Validating docker compose configuration)
	@docker compose config --quiet

# ------------------------------------------------------------------------------
# Phony Declarations
# ------------------------------------------------------------------------------

.PHONY: docker-all docker-toolchains docker-devs docker-builders docker-final
.PHONY: docker-base artifacts
.PHONY: docker-toolchain-aarch64 docker-toolchain-rpi docker-toolchain-riscv64
.PHONY: docker-dev docker-dev-cuda docker-dev-jetson docker-dev-rpi docker-dev-riscv64
.PHONY: docker-builder-cpu docker-builder-cuda docker-builder-jetson
.PHONY: docker-builder-rpi docker-builder-riscv64
.PHONY: shell-dev shell-dev-cuda shell-dev-jetson shell-dev-rpi shell-dev-riscv64
.PHONY: docker-clean docker-clean-deep docker-prune docker-disk-usage
.PHONY: docker-lint docker-validate

endif  # DOCKER_MK_GUARD
