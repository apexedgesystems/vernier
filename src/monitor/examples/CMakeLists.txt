# ==============================================================================
# Examples: monitor
#
# Demonstrates env-var-driven Monitor configuration: the same binary stays
# silent in production and emits a full report when an operator sets the
# relevant VERNIER_MONITOR_* env var.
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
