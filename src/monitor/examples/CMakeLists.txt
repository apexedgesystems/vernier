# ==============================================================================
# Monitor examples
# ==============================================================================

add_executable(MonitorEnvVarExample MonitorEnvVarExample.cpp)
target_link_libraries(MonitorEnvVarExample PRIVATE monitor)
target_include_directories(MonitorEnvVarExample PRIVATE "${CMAKE_SOURCE_DIR}")
set_target_properties(
  MonitorEnvVarExample
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples"
)
