# ==============================================================================
# ExternalDependencies.cmake - Third-party library fetching
# ==============================================================================

# Include the FetchContent module to manage external dependencies.
include(FetchContent)

# Hermetic builds: when the dev image has baked the dependency sources at their
# pinned tags (docker/scripts/bake-external-deps.sh into $VERNIER_DEPS_DIR), point
# FetchContent at the local copies so configure never reaches GitHub. Absent the
# env (local dev), FetchContent clones normally. Names match the declarations
# below; FetchContent uppercases them for FETCHCONTENT_SOURCE_DIR_<NAME>.
if (DEFINED ENV{VERNIER_DEPS_DIR})
  foreach (_dep googletest)
    string(TOUPPER "${_dep}" _dep_uc)
    if (EXISTS "$ENV{VERNIER_DEPS_DIR}/${_dep}")
      set(FETCHCONTENT_SOURCE_DIR_${_dep_uc}
          "$ENV{VERNIER_DEPS_DIR}/${_dep}"
          CACHE PATH "Baked source for ${_dep} (hermetic build)" FORCE
      )
    endif ()
  endforeach ()
  unset(_dep)
  unset(_dep_uc)
endif ()

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
