# ==============================================================================
# mk/build.mk - Native and cross-compilation build targets
# ==============================================================================

ifndef BUILD_MK_GUARD
BUILD_MK_GUARD := 1

# ------------------------------------------------------------------------------
# Native Builds
# ------------------------------------------------------------------------------

debug: prep
	$(call _build,native debug,$(HOST_DEBUG_PRESET),$(HOST_DEBUG_DIR))

release: prep
	$(call _build,native release,$(HOST_RELEASE_PRESET),$(HOST_RELEASE_DIR))

docs: prep
	$(call log,build,Building documentation)
	@cmake --preset $(HOST_DEBUG_PRESET) $(CMAKE_VERBOSE_FLAG)
	@cmake --build --preset $(HOST_DEBUG_PRESET) --target docs -j$(NUM_JOBS)

# ------------------------------------------------------------------------------
# Cross-Compilation Builds
# ------------------------------------------------------------------------------

jetson-debug: prep
	$(call _build,Jetson debug,$(JETSON_DEBUG_PRESET),$(JETSON_DEBUG_DIR))

jetson-release: prep
	$(call _build,Jetson release,$(JETSON_RELEASE_PRESET),$(JETSON_RELEASE_DIR))

rpi-debug: prep
	$(call _build,Raspberry Pi debug,$(RPI_DEBUG_PRESET),$(RPI_DEBUG_DIR))

rpi-release: prep
	$(call _build,Raspberry Pi release,$(RPI_RELEASE_PRESET),$(RPI_RELEASE_DIR))

riscv-debug: prep
	$(call _build,RISC-V 64 debug,$(RISCV_DEBUG_PRESET),$(RISCV_DEBUG_DIR))

riscv-release: prep
	$(call _build,RISC-V 64 release,$(RISCV_RELEASE_PRESET),$(RISCV_RELEASE_DIR))

# ------------------------------------------------------------------------------
# Install Targets
# ------------------------------------------------------------------------------

# Install to <build_dir>/install/ for artifact packaging.
# Native targets include Rust/Python CLI tools alongside C++ libraries.
# Cross-compile targets install C++ libraries only (no host tools).
#
# Usage: make install            (native release + tools)
#        make install-jetson     (Jetson cross-compile, libs only)

INSTALL_PREFIX       = $(HOST_RELEASE_DIR)/install
INSTALL_TOOLS_PREFIX = $(HOST_RELEASE_DIR)/install-tools

install: release
	$(call log,install,Installing native release artifacts)
	@cmake --install $(HOST_RELEASE_DIR) --prefix $(INSTALL_PREFIX)
	@printf '%s\n' \
	  '# Vernier environment setup' \
	  'export LD_LIBRARY_PATH="$$PWD/lib:$$LD_LIBRARY_PATH"' \
	  > $(INSTALL_PREFIX)/.env
	$(call log,install,Install tree ready at $(INSTALL_PREFIX))

install-tools: release
	$(call log,install,Building and installing CLI tools)
	@$(MAKE) tools-rust BUILD_DIR=$(HOST_RELEASE_DIR)
	@$(MAKE) tools-py BUILD_DIR=$(HOST_RELEASE_DIR)
	@mkdir -p $(INSTALL_TOOLS_PREFIX)/rust $(INSTALL_TOOLS_PREFIX)/py
	@cp -r $(HOST_RELEASE_DIR)/bin/tools/rust/* $(INSTALL_TOOLS_PREFIX)/rust/
	@cp LICENSE tools/README.md $(INSTALL_TOOLS_PREFIX)/rust/
	@cp $(HOST_RELEASE_DIR)/vernier-wheels/*.whl $(INSTALL_TOOLS_PREFIX)/py/
	$(call log,install,Tools ready at $(INSTALL_TOOLS_PREFIX))

install-jetson: jetson-release
	$(call log,install,Installing Jetson release artifacts)
	@cmake --install $(JETSON_RELEASE_DIR) --prefix $(JETSON_RELEASE_DIR)/install

install-rpi: rpi-release
	$(call log,install,Installing Raspberry Pi release artifacts)
	@cmake --install $(RPI_RELEASE_DIR) --prefix $(RPI_RELEASE_DIR)/install

install-riscv: riscv-release
	$(call log,install,Installing RISC-V 64 release artifacts)
	@cmake --install $(RISCV_RELEASE_DIR) --prefix $(RISCV_RELEASE_DIR)/install

# ------------------------------------------------------------------------------
# Configure-Only Targets
# ------------------------------------------------------------------------------

configure: prep
	$(call _configure,$(HOST_DEBUG_PRESET),$(HOST_DEBUG_DIR))

configure-release: prep
	$(call _configure,$(HOST_RELEASE_PRESET),$(HOST_RELEASE_DIR))

configure-jetson: prep
	$(call _configure,$(JETSON_DEBUG_PRESET),$(JETSON_DEBUG_DIR))

configure-rpi: prep
	$(call _configure,$(RPI_DEBUG_PRESET),$(RPI_DEBUG_DIR))

configure-riscv: prep
	$(call _configure,$(RISCV_DEBUG_PRESET),$(RISCV_DEBUG_DIR))

configure-jetson-release: prep
	$(call _configure,$(JETSON_RELEASE_PRESET),$(JETSON_RELEASE_DIR))

configure-rpi-release: prep
	$(call _configure,$(RPI_RELEASE_PRESET),$(RPI_RELEASE_DIR))

configure-riscv-release: prep
	$(call _configure,$(RISCV_RELEASE_PRESET),$(RISCV_RELEASE_DIR))

# ------------------------------------------------------------------------------
# Phony Declarations
# ------------------------------------------------------------------------------

.PHONY: debug release docs
.PHONY: install install-tools install-jetson install-rpi install-riscv
.PHONY: jetson-debug jetson-release
.PHONY: rpi-debug rpi-release
.PHONY: riscv-debug riscv-release
.PHONY: configure configure-release
.PHONY: configure-jetson configure-jetson-release
.PHONY: configure-rpi configure-rpi-release
.PHONY: configure-riscv configure-riscv-release

endif  # BUILD_MK_GUARD
