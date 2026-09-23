# ==============================================================================
# Examples: monitor
#
# Demonstrates env-var-driven Monitor configuration: the same binary reports
# to the console, to a file, to neither, or records nothing at all, depending
# on the VERNIER_MONITOR_* env vars an operator sets.
# ==============================================================================

# ------------------------------------------------------------------------------
# Executables
# ------------------------------------------------------------------------------

add_executable(MonitorEnvVarExample MonitorEnvVarExample.cpp)
target_link_libraries(
  MonitorEnvVarExample
  PRIVATE monitor
)
target_include_directories(MonitorEnvVarExample PRIVATE "${CMAKE_SOURCE_DIR}")
set_target_properties(
  MonitorEnvVarExample PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples"
)
