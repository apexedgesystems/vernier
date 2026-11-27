# ==============================================================================
# Module: monitor
#
# Lightweight runtime performance monitor for application instrumentation.
# Provides scoped timers, counters, gauges, and threshold alerts with
# lock-free async output to console and file sinks.
#
# Libraries:
#   monitor - Runtime performance monitoring (shared library)
# ==============================================================================

include(vernier/All)
vernier_module(monitor)

# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------

set(CPP_SOURCES "${SRC_DIR}/MonitorFileSink.cpp")

# ------------------------------------------------------------------------------
# Library
# ------------------------------------------------------------------------------
# Monitor.hpp, compiled into the program, allocates the MonitorFileSink that
# libmonitor constructs and writes through, so the sink's layout crosses the
# library boundary: raise MONITOR_ABI_SOVERSION whenever it changes, or an
# exported signature does, so the loader refuses a program built against the
# previous layout instead of binding it to this one. 2: the file sink's path
# and record counts.
# ------------------------------------------------------------------------------

set(MONITOR_ABI_SOVERSION 2)

vernier_add_library(
  NAME
  ${LIB_NAME}
  TYPE
  SHARED
  INC
  "${INC_DIR}"
  SRC
  ${CPP_SOURCES}
  REQUIRES
  POSIX
  ABI_VERSION
  ${MONITOR_ABI_SOVERSION}
)

target_link_libraries(
  ${LIB_NAME}
  PRIVATE pthread
)

# ------------------------------------------------------------------------------
# Opt-ins (coverage, doxygen, UPX)
# ------------------------------------------------------------------------------

vernier_standard_optins(${LIB_NAME})

# ------------------------------------------------------------------------------
# Production documentation (release builds only)
# ------------------------------------------------------------------------------

vernier_install_docs("${CMAKE_CURRENT_SOURCE_DIR}/docs")

# ------------------------------------------------------------------------------
# Subdirectories
# ------------------------------------------------------------------------------

if (PROJECT_IS_TOP_LEVEL)
  if (EXISTS "${UTST_DIR}/CMakeLists.txt")
    add_subdirectory(utst)
  endif ()
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
    add_subdirectory(examples)
  endif ()
endif ()
