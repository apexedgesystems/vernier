# ==============================================================================
# ExternalDependencies.cmake - Third-party library fetching
# ==============================================================================

# Include the FetchContent module to manage external dependencies.
include(FetchContent)

# Declare and make the GoogleTest library available.
fetchcontent_declare(
  googletest
  SYSTEM
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.16.0
)
# Prevent GoogleTest from installing with your project
set(INSTALL_GTEST
    OFF
    CACHE BOOL "" FORCE
)
fetchcontent_makeavailable(googletest)

# GTest 1.16 has a char8_t -> char32_t implicit conversion in its printer code
# that Clang flags with -Wcharacter-conversion. Suppress until upstream fixes it.
if (TARGET gtest AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(gtest PRIVATE -Wno-character-conversion)
endif ()

# Include CTest to enable testing capabilities.
include(CTest)
