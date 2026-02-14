# ==============================================================================
# vernier/All.cmake - Single entry point for Vernier CMake infrastructure
# ==============================================================================
#
# Usage:
#   include(vernier/All)
#
# Provides all vernier_* functions. Include this once per CMakeLists.txt.
# ==============================================================================

include_guard(GLOBAL)

# Foundation utilities (must be first)
include(vernier/Core)

# Build acceleration (ccache, fast linker, split DWARF)
include(vernier/BuildAcceleration)

# CUDA integration (before Targets, which depends on it)
include(vernier/Cuda)

# Target factories
include(vernier/Targets)

# Coverage infrastructure
include(vernier/Coverage)

# Testing infrastructure
include(vernier/Testing)

# Tooling (docs, UPX, clang-tidy)
include(vernier/Tooling)

# Configure-time summary
include(vernier/Summary)
