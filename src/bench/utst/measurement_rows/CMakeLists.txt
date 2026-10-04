# ==============================================================================
# Unit Tests: bench measurement rows
#
# The rows a benchmark publishes for its completed measurements, checked end to
# end through a fixture benchmark with its own PERF_MAIN whose tests measure in
# each shape that decides how rows are named. The checks run the fixture as a
# child; its own Rows.* cases are never registered.
#
# Targets: MeasurementRowsProbe (fixture), TestMeasurementRows
# Tests:   MeasurementRows.Publication, MeasurementRows.CliAgreement
# ==============================================================================

include(vernier/All)

# ------------------------------------------------------------------------------
# Fixture
# ------------------------------------------------------------------------------
# Never installed or compressed, and not registered. Everything that runs it
# takes its path from the target.
# ------------------------------------------------------------------------------

vernier_add_app(
  NAME
  MeasurementRowsProbe
  SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/MeasurementRowsProbe.cpp"
  LINK
  bench
  GTest::gtest
  NO_INSTALL
  NO_UPX
)

# The helper skips programs where POSIX is missing; the checks go with them.
if (NOT TARGET MeasurementRowsProbe)
  return()
endif ()

# ------------------------------------------------------------------------------
# Publication (1 test)
# ------------------------------------------------------------------------------
# Runs the fixture and checks its CSV, end-of-run table and footer against the
# measurements it reports; needs no CLI tool.
# ------------------------------------------------------------------------------

vernier_add_gtest(
  TARGET
  TestMeasurementRows
  SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/MeasurementRows_uTest.cpp"
  LABELS
  benchmarking
  rows
)
target_compile_definitions(
  TestMeasurementRows PRIVATE MEASUREMENT_ROWS_PROBE="$<TARGET_FILE:MeasurementRowsProbe>"
)
add_dependencies(TestMeasurementRows MeasurementRowsProbe)

# ------------------------------------------------------------------------------
# CLI Agreement (1 test, where the CLI tools are built)
# ------------------------------------------------------------------------------
# A test of the CLI's own suite, run by cargo with the fixture's path: what
# bench summary and bench compare read from the fixture's CSV. It passes only
# on that test's pass line, so a missing cargo or fixture, or a test that no
# longer matches its name, fails it. The nested build's jobs are reserved from
# ctest's parallel slots.
# ------------------------------------------------------------------------------

if (VERNIER_BUILD_TOOLS)
  find_program(CARGO_EXECUTABLE cargo)
  set(_rows_cargo_jobs 4)
  add_test(
    NAME MeasurementRows.CliAgreement
    COMMAND
      "${CARGO_EXECUTABLE}" test --manifest-path "${PROJECT_SOURCE_DIR}/tools/rust/Cargo.toml"
      --target-dir "${CMAKE_BINARY_DIR}/rust-test-target" --jobs ${_rows_cargo_jobs} --test bench
      measurement_rows_cli_agreement -- --ignored --exact
  )
  set_tests_properties(
    MeasurementRows.CliAgreement
    PROPERTIES LABELS
               "benchmarking;rows"
               ENVIRONMENT
               "MEASUREMENT_ROWS_PROBE=$<TARGET_FILE:MeasurementRowsProbe>"
               PASS_REGULAR_EXPRESSION
               "test measurement_rows_cli_agreement \\.\\.\\. ok"
               PROCESSORS
               ${_rows_cargo_jobs}
  )
endif ()
