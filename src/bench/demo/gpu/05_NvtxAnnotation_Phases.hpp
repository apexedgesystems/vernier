#ifndef VERNIER_DEMO_05_NVTXANNOTATION_PHASES_HPP
#define VERNIER_DEMO_05_NVTXANNOTATION_PHASES_HPP
/**
 * @file 05_NvtxAnnotation_Phases.hpp
 * @brief The names of the three NVTX ranges demo 05 opens around each G1 call,
 *        in the order it opens them.
 *
 * The demo names its ranges with these, and its check
 * (utst/05_NvtxAnnotation_Ranges_uTest.cpp) looks for these names in the
 * Nsight Systems report, so the two read one list.
 */

namespace vernier {
namespace bench {
namespace demo {
namespace nvtx_annotation {

/* ----------------------------- Constants ----------------------------- */

/// x and y into pinned staging, then both copied to the device.
inline constexpr const char* COPY_IN = "copy_in";

/// The SAXPY launch.
inline constexpr const char* KERNEL = "kernel";

/// y copied back to pinned staging, then out of it.
inline constexpr const char* COPY_OUT = "copy_out";

/// The three, in the order each call opens them.
inline constexpr const char* PHASES[] = {COPY_IN, KERNEL, COPY_OUT};

} // namespace nvtx_annotation
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_05_NVTXANNOTATION_PHASES_HPP
