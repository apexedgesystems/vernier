/**
 * @file ProfilerRegistry_uTest.cpp
 * @brief Unit tests for the self-registering profiler backend registry.
 *
 * Covers the registry surface that `--profile`, `bench doctor`, and
 * `Profiler::make()` rely on: every CPU backend compiled into libbench
 * self-registers on load, name lookups are stable, make() never returns
 * nullptr, and each backend's environment check is callable.
 */

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/Profiler.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace {

using vernier::bench::EnvReport;
using vernier::bench::PerfConfig;
using vernier::bench::ProfilerRegistry;

// Backends compiled into libbench that self-register on load. compute-sanitizer
// and nsight self-register too, but they live in the separate CUDA library and
// so are absent from this CPU-only test binary.
const std::vector<std::string> CPU_BACKENDS = {"perf",   "gperf",     "callgrind", "bpftrace",
                                               "rapl",   "massif",    "memcheck",  "helgrind",
                                               "offcpu", "heaptrack", "jemalloc",  "rocprof"};

/** @test Every CPU profiler backend self-registers. */
TEST(ProfilerRegistryTest, CpuBackendsRegistered) {
  const ProfilerRegistry& reg = ProfilerRegistry::instance();
  for (const std::string& NAME : CPU_BACKENDS) {
    EXPECT_TRUE(reg.hasBackend(NAME)) << "backend not registered: " << NAME;
  }
}

/** @test backendNames() is non-empty, sorted, and lists the v1.0.2 additions. */
TEST(ProfilerRegistryTest, BackendNamesSortedAndComplete) {
  const std::vector<std::string> names = ProfilerRegistry::instance().backendNames();
  EXPECT_FALSE(names.empty());
  EXPECT_TRUE(std::is_sorted(names.begin(), names.end()));
  for (const char* NAME : {"helgrind", "heaptrack", "massif", "memcheck", "offcpu"}) {
    EXPECT_NE(std::find(names.begin(), names.end(), NAME), names.end())
        << "backendNames() missing: " << NAME;
  }
}

/** @test An unregistered name reports as absent. */
TEST(ProfilerRegistryTest, UnknownBackendNotRegistered) {
  EXPECT_FALSE(ProfilerRegistry::instance().hasBackend("not-a-real-backend"));
}

/** @test make() never returns nullptr: real backend or named no-op. */
TEST(ProfilerRegistryTest, MakeNeverReturnsNull) {
  const ProfilerRegistry& reg = ProfilerRegistry::instance();
  const PerfConfig cfg{};
  EXPECT_NE(reg.make("perf", cfg, "RegistryTest"), nullptr);
  EXPECT_NE(reg.make("not-a-real-backend", cfg, "RegistryTest"), nullptr);
}

/** @test runCheck() yields a valid status per backend, Error for unknown. */
TEST(ProfilerRegistryTest, RunCheckReturnsStatus) {
  const ProfilerRegistry& reg = ProfilerRegistry::instance();
  for (const std::string& NAME : CPU_BACKENDS) {
    const EnvReport report = reg.runCheck(NAME);
    EXPECT_TRUE(report.status == EnvReport::Status::Ok ||
                report.status == EnvReport::Status::Warning ||
                report.status == EnvReport::Status::Error);
  }
  EXPECT_EQ(reg.runCheck("not-a-real-backend").status, EnvReport::Status::Error);
}

/** @test runAllChecks() reports one entry per registered backend, name-aligned. */
TEST(ProfilerRegistryTest, RunAllChecksMatchesBackendNames) {
  const ProfilerRegistry& reg = ProfilerRegistry::instance();
  const std::vector<std::string> names = reg.backendNames();
  const std::vector<std::pair<std::string, EnvReport>> checks = reg.runAllChecks();
  ASSERT_EQ(checks.size(), names.size());
  for (std::size_t i = 0; i < names.size(); ++i) {
    EXPECT_EQ(checks[i].first, names[i]);
  }
}

} // namespace
