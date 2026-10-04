# ==============================================================================
# Module: demo_example_saxpy
#
# y = a*x + y as a CPU loop and two ways of driving one CUDA kernel: saxpyG0
# allocates and copies on every call with one thread per block, SaxpyG1 keeps
# its buffers and launches 256 threads per block. The GPU demos link it from
# ../../CMakeLists.txt.
#
# Libraries:
#   demo_example_saxpy  - the three versions; built only with the GPU demos
# ==============================================================================

include(vernier/All)
vernier_module(demo_example_saxpy)

# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------

set(CPP_SOURCES "${SRC_DIR}/SaxpyCpu.cpp" "${SRC_DIR}/SaxpyGpu.cpp")
set(CUDA_SOURCES "${SRC_DIR}/SaxpyKernel.cu")

# ------------------------------------------------------------------------------
# CUDA Library (requires CUDA toolkit)
# ------------------------------------------------------------------------------
# The header is one API whose GPU half cannot link without the toolkit, and
# nothing CPU-only uses it, so the library is built with the GPU demos or not
# at all. Its host tests on a stand-in runtime (utst) are built everywhere.
# ------------------------------------------------------------------------------

if (TARGET bench_cuda)
  add_library(${LIB_NAME} STATIC ${CPP_SOURCES} ${CUDA_SOURCES})
  target_include_directories(${LIB_NAME} PUBLIC "${PROJECT_SOURCE_DIR}")
  # SaxpyGpu.cpp calls the CUDA runtime from host code.
  target_link_libraries(${LIB_NAME} PUBLIC CUDA::cudart)
endif ()

# ------------------------------------------------------------------------------
# Subdirectories
# ------------------------------------------------------------------------------

if (EXISTS "${UTST_DIR}/CMakeLists.txt")
  add_subdirectory(utst)
endif ()
