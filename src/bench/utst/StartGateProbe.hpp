#ifndef VERNIER_STARTGATEPROBE_HPP
#define VERNIER_STARTGATEPROBE_HPP
/**
 * @file StartGateProbe.hpp
 * @brief The start-gate probes' names and harness settings, shared by
 *        StartGateHelgrind_uTest.cpp (the locked probe and the tests that run
 *        both probes under helgrind) and StartGateHelgrind_Probe.cpp (the racy
 *        probe).
 *
 * Test-only; not part of the installed library.
 */

#include "src/bench/inc/PerfConfig.hpp"

namespace vernier {
namespace bench {
namespace test {

/// Threads each probe starts, and calls each thread makes.
inline constexpr int PROBE_THREADS = 4;
inline constexpr int PROBE_CYCLES = 2;

/// The probe cases, as the helgrind runs select them.
inline constexpr const char* LOCKED_PROBE = "StartGateProbe.LockedWorkers";
inline constexpr const char* RACY_PROBE = "StartGateProbe.RacyWorkers";

/// The probes' harness settings: PROBE_THREADS threads of PROBE_CYCLES calls,
/// one repeat.
inline PerfConfig probeConfig() {
  PerfConfig cfg;
  cfg.threads = PROBE_THREADS;
  cfg.cycles = PROBE_CYCLES;
  cfg.repeats = 1;
  return cfg;
}

} // namespace test
} // namespace bench
} // namespace vernier

#endif // VERNIER_STARTGATEPROBE_HPP
