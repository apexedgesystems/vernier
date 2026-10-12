# ==============================================================================
# Module: demo_example_filter
#
# Copy the values above a threshold, two ways: filterBranchy with a conditional
# store per value, filterBranchless storing every value and advancing the
# cursor by the test. Demo 02 links it from ../../CMakeLists.txt.
#
# Libraries:
#   demo_example_filter  - the two versions
# ==============================================================================

include(vernier/All)
vernier_module(demo_example_filter)

# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------

set(CPP_SOURCES "${SRC_DIR}/Filter.cpp")

# ------------------------------------------------------------------------------
# Library
# ------------------------------------------------------------------------------

add_library(${LIB_NAME} STATIC ${CPP_SOURCES})
target_include_directories(${LIB_NAME} PUBLIC "${PROJECT_SOURCE_DIR}")

# ------------------------------------------------------------------------------
# Subdirectories
# ------------------------------------------------------------------------------

if (EXISTS "${UTST_DIR}/CMakeLists.txt")
  add_subdirectory(utst)
endif ()
