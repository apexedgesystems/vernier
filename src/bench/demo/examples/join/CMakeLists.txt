# ==============================================================================
# Module: demo_example_join
#
# Join a list of strings with a separator, two ways: joinV0 rebuilds the
# string through temporaries for every part, joinV1 measures, reserves once
# and appends in place. The demos link it from ../../CMakeLists.txt.
#
# Libraries:
#   demo_example_join  - the two versions and the input the demos measure
# ==============================================================================

include(vernier/All)
vernier_module(demo_example_join)

# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------

set(CPP_SOURCES "${SRC_DIR}/Join.cpp")

# ------------------------------------------------------------------------------
# Library
# ------------------------------------------------------------------------------

add_library(${LIB_NAME} STATIC ${CPP_SOURCES})
target_include_directories(${LIB_NAME} PUBLIC "${PROJECT_SOURCE_DIR}")
# Line tables in every build type, so a profiler can name the example's source
# lines in an optimized build; -g adds debug sections and changes no code.
target_compile_options(
  ${LIB_NAME}
  PRIVATE -g
)

# ------------------------------------------------------------------------------
# Subdirectories
# ------------------------------------------------------------------------------

if (EXISTS "${UTST_DIR}/CMakeLists.txt")
  add_subdirectory(utst)
endif ()
if (EXISTS "${PTST_DIR}/CMakeLists.txt")
  add_subdirectory(ptst)
endif ()
