# ==============================================================================
# MeasurementRows.cmake - The measurement-rows probe and its tests
#
# Included from the bench unit-test CMakeLists.txt. MeasurementRowsProbe is a
# benchmark whose tests measure in each shape that decides how a test's rows
# are named; MeasurementRows_test.cmake runs it and checks the CSV, the
# end-of-run table, and what the bench CLI reads from the CSV (that case skips
# when the CLI was not built).
#
# Target: MeasurementRowsProbe (test fixture: never installed, no UPX copy)
# Tests:  MeasurementRows.Publication, MeasurementRows.CliAgreement
# ==============================================================================

vernier_add_app(
  NAME
  MeasurementRowsProbe
  SRC
  "${CMAKE_CURRENT_LIST_DIR}/MeasurementRowsProbe.cpp"
  LINK
  bench
  GTest::gtest
  NO_INSTALL
  NO_UPX
)

# The helper skips apps on platforms without POSIX; the tests go with it.
if (NOT TARGET MeasurementRowsProbe)
  return()
endif ()

foreach (_case Publication CliAgreement)
  add_test(
    NAME MeasurementRows.${_case}
    COMMAND
      "${CMAKE_COMMAND}" -DPROBE=$<TARGET_FILE:MeasurementRowsProbe> -DCASE=${_case}
      -DBENCH_CLI=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tools/rust/bench
      -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/measurement_rows/${_case} -P
      "${CMAKE_CURRENT_LIST_DIR}/MeasurementRows_test.cmake"
  )
  set_tests_properties(
    MeasurementRows.${_case} PROPERTIES LABELS "benchmarking;rows" SKIP_REGULAR_EXPRESSION
                                        "MEASUREMENT_ROWS_SKIPPED"
  )
endforeach ()
