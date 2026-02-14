#ifndef VERNIER_PERFLISTENER_HPP
#define VERNIER_PERFLISTENER_HPP
/**
 * @file PerfListener.hpp
 * @brief GoogleTest event listener for centralized perf result emission.
 *
 * Handles multi-GPU CSV columns and Unified Memory CSV columns.
 * Passes all UM fields to writeCsvRow for complete data capture.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <optional>
#include <string>

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfCsv.hpp"
#include "src/bench/inc/PerfRegistry.hpp"

namespace vernier {
namespace bench {

/* --------------------------------- API --------------------------------- */

/**
 * @brief Print end-of-run summary table from accumulated results.
 *
 * Format:
 * @code
 * ================================ BENCHMARK SUMMARY ================================
 * Test                                       Median (us)   CV%     Calls/s   Status
 * -----------------------------------------------------------------------------------
 * CoreFeatures.BasicThroughput                    0.034    4.2%      29.4M   OK
 * CoreFeatures.WarmupStabilization                0.089   12.3%      11.2M   UNSTABLE
 * -----------------------------------------------------------------------------------
 * 2 tests | 1 stable | 1 unstable
 * @endcode
 *
 * @note NOT RT-safe (console I/O, heap allocation).
 */
inline void printSummaryTable(const std::vector<PerfSummaryEntry>& entries) {
  if (entries.empty()) {
    return;
  }

  // Find longest test name for column alignment
  std::size_t maxNameLen = 4; // "Test"
  for (const auto& e : entries) {
    maxNameLen = std::max(maxNameLen, e.testName.size());
  }
  // Cap at 50 to prevent absurd widths
  if (maxNameLen > 50) {
    maxNameLen = 50;
  }

  const int TOTAL_WIDTH = static_cast<int>(maxNameLen) + 50; // name + numeric columns

  // Header
  std::fprintf(stdout, "\n");
  for (int i = 0; i < TOTAL_WIDTH; ++i) {
    std::fputc('=', stdout);
  }
  std::fprintf(stdout, "\n%-*s  %12s  %6s  %10s  %s\n", static_cast<int>(maxNameLen), "Test",
               "Median (us)", "CV%", "Calls/s", "Status");
  for (int i = 0; i < TOTAL_WIDTH; ++i) {
    std::fputc('-', stdout);
  }
  std::fprintf(stdout, "\n");

  // Rows
  int stableCount = 0;
  int unstableCount = 0;
  for (const auto& e : entries) {
    // Truncate long names
    std::string displayName = e.testName;
    if (displayName.size() > maxNameLen) {
      displayName = displayName.substr(0, maxNameLen - 2) + "..";
    }

    // Format calls/s compactly
    char cpsBuf[16];
    if (e.callsPerSecond >= 1e9) {
      std::snprintf(cpsBuf, sizeof(cpsBuf), "%.1fG", e.callsPerSecond / 1e9);
    } else if (e.callsPerSecond >= 1e6) {
      std::snprintf(cpsBuf, sizeof(cpsBuf), "%.1fM", e.callsPerSecond / 1e6);
    } else if (e.callsPerSecond >= 1e3) {
      std::snprintf(cpsBuf, sizeof(cpsBuf), "%.1fK", e.callsPerSecond / 1e3);
    } else {
      std::snprintf(cpsBuf, sizeof(cpsBuf), "%.0f", e.callsPerSecond);
    }

    const char* STATUS = e.stable ? "OK" : "UNSTABLE";
    if (e.stable) {
      ++stableCount;
    } else {
      ++unstableCount;
    }

    std::fprintf(stdout, "%-*s  %12.3f  %5.1f%%  %10s  %s\n", static_cast<int>(maxNameLen),
                 displayName.c_str(), e.medianUs, e.cv * 100.0, cpsBuf, STATUS);
  }

  // Footer
  for (int i = 0; i < TOTAL_WIDTH; ++i) {
    std::fputc('-', stdout);
  }
  std::fprintf(stdout, "\n%d tests | %d stable | %d unstable\n", static_cast<int>(entries.size()),
               stableCount, unstableCount);
}

inline void installPerfEventListener(const PerfConfig& cfg, ::testing::UnitTest* ut = nullptr) {
  if (!ut) {
    ut = ::testing::UnitTest::GetInstance();
  }

  // Always install summary listener (prints end-of-run table)
  class SummaryListener : public ::testing::EmptyTestEventListener {
  public:
    void OnTestProgramEnd(const ::testing::UnitTest& /*ut*/) override {
      const auto& ENTRIES = PerfRegistry::instance().summary();
      if (ENTRIES.size() >= 2) {
        printSummaryTable(ENTRIES);
      }
    }
  };

  auto& listeners = ut->listeners();
  listeners.Append(new SummaryListener());

  // CSV listener (only if --csv provided)
  if (!cfg.csv) {
    return;
  }

  class CsvListener : public ::testing::EmptyTestEventListener {
  public:
    explicit CsvListener(std::string path, bool includeProfile, bool includeGpu)
        : path_(std::move(path)), includeProfile_(includeProfile), includeGpu_(includeGpu) {
      out_.open(path_, std::ios::out | std::ios::trunc);
      if (out_) {
        writeCsvHeader(out_, includeProfile_, /*includeMetadata=*/true, includeGpu_);
      }
    }
    ~CsvListener() override {
      if (out_) {
        out_.flush();
        out_.close();
      }
    }

    void OnTestEnd(const ::testing::TestInfo& /*info*/) override {
      if (!out_) {
        return;
      }

      if (auto row = PerfRegistry::instance().take()) {
        // Ensure row fields reflect the *parsed* flags used by this process
        if (const PerfConfig* cfgPtr = globalPerfConfig()) {
          row->cycles = cfgPtr->cycles;
          row->repeats = cfgPtr->repeats;
          row->threads = cfgPtr->threads;
          row->msgBytes = cfgPtr->msgBytes;
          row->console = cfgPtr->console;
          row->nonBlocking = cfgPtr->nonBlocking;
          row->minLevel = cfgPtr->minLevel;
        }

        // Clear profile metadata if not including profile columns
        if (!includeProfile_) {
          row->profileTool.reset();
          row->profileDir.reset();
        }

        writeCsvRow(out_, *row);
      }
    }

  private:
    std::string path_;
    std::ofstream out_{};
    bool includeProfile_{false};
    bool includeGpu_{false};
  };

  const bool INCLUDE_PROFILE = !cfg.profileTool.empty();

  // Auto-detect GPU tests by scanning test names
  bool hasGpuTests = false;
  for (int i = 0; i < ut->total_test_suite_count(); ++i) {
    const auto* suite = ut->GetTestSuite(i);
    for (int j = 0; j < suite->total_test_count(); ++j) {
      const auto* info = suite->GetTestInfo(j);
      std::string name = std::string(suite->name()) + "." + info->name();
      if (name.find("Gpu") != std::string::npos || name.find("CUDA") != std::string::npos ||
          name.find("GPU") != std::string::npos) {
        hasGpuTests = true;
        break;
      }
    }
    if (hasGpuTests)
      break;
  }

  listeners.Append(new CsvListener(*cfg.csv, INCLUDE_PROFILE, hasGpuTests));
}

} // namespace bench
} // namespace vernier

#endif // VERNIER_PERFLISTENER_HPP