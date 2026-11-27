#ifndef VERNIER_DEMO_SKIPUNLESSPROBE_HPP
#define VERNIER_DEMO_SKIPUNLESSPROBE_HPP
/**
 * @file SkipUnlessProbe.hpp
 * @brief What the skip helpers' probe cases print once their helper lets them
 *        run: the probes (SkipUnlessUnder*_Probe.cpp) print it, and the tests
 *        that run them as children look for it.
 *
 * Test-only; not part of the installed library.
 */

namespace vernier {
namespace bench {
namespace demo {

/// What a probe prints once its helper has let it run.
inline constexpr const char* PROBE_RAN = "probe: the helper let this case run";

} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_SKIPUNLESSPROBE_HPP
