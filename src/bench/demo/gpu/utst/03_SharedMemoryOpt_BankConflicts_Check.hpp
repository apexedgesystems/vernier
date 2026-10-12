#ifndef VERNIER_DEMO_GPU_03_BANK_CONFLICTS_CHECK_HPP
#define VERNIER_DEMO_GPU_03_BANK_CONFLICTS_CHECK_HPP
/**
 * @file 03_SharedMemoryOpt_BankConflicts_Check.hpp
 * @brief The reading of ncu's CSV and the verdict of demo 03's bank-conflict
 *        check
 *
 * BankConflicts.CountedByNsightCompute runs demo 03 under ncu with the three
 * metrics walkthrough 12 reads and holds every launch to the counts the
 * walkthrough gives. What it does with ncu's CSV log is here: the metrics
 * and kernels it reads, the counts, the reading of the log into each
 * launch's readings, and expectWalkthroughCounts, which records every
 * departure from the counts as a failure of the running test. None of it
 * needs CUDA or ncu, so the tests that hold it to logs as ncu prints them
 * (03_SharedMemoryOpt_BankConflictsReport_uTest.cpp) build and run in every
 * configuration, and the check that runs ncu on a GPU
 * (03_SharedMemoryOpt_BankConflicts_uTest.cpp) uses the same code.
 *
 * Test support for those two programs; not part of the demo.
 */

#include "src/bench/demo/gpu/03_SharedMemoryOpt_Workload.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace vernier {
namespace bench {
namespace demo {
namespace bank_conflicts_check {

namespace sm = vernier::bench::demo::shared_memory_demo;

/* ----------------------------- Constants ----------------------------- */

/// The three metrics, as the walkthrough reads them: shared-load
/// instructions per launch, the wavefronts those instructions took at the
/// SASS level, and ncu's headline bank-conflict counter for shared loads.
inline constexpr const char* METRIC_INSTRUCTIONS = "smsp__inst_executed_op_shared_ld.sum";
inline constexpr const char* METRIC_WAVEFRONTS =
    "smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ld.sum";
inline constexpr const char* METRIC_CONFLICTS =
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum";

/// The kernels, as ncu names them once their namespace and parameters are
/// stripped.
inline constexpr const char* NAIVE = "transposeNaive";
inline constexpr const char* CONFLICT = "transposeSharedConflict";
inline constexpr const char* PADDED = "transposeSharedPadded";

/// The lanes of a warp; the tile's row is one warp wide.
inline constexpr long WARP_SIZE = 32;

/// One shared-load instruction per warp per launch: every warp reads one
/// column of its block's tile, and the demo's matrix is a grid of tiles.
inline constexpr long WARPS_PER_LAUNCH =
    (static_cast<long>(sm::MATRIX_DIM) / sm::TILE_DIM) *
    (static_cast<long>(sm::MATRIX_DIM) / sm::TILE_DIM) *
    (static_cast<long>(sm::TILE_DIM) * sm::TILE_DIM / WARP_SIZE);

/// A column read of a TILE_DIM-wide tile lands every lane in one bank, so
/// the instruction is served once per lane: TILE_DIM wavefronts, one per row
/// of the tile, where the padded tile's read takes one.
inline constexpr long CONFLICTED_WAVEFRONTS_PER_LOAD = sm::TILE_DIM;

/// The floor for ncu's bank-conflict counter on the conflicted kernel: the
/// conflicts of one load are its wavefronts beyond the first, TILE_DIM - 1,
/// and the counter is allowed a little under that (the reference rig reads
/// 31.03 per load).
inline constexpr long MIN_CONFLICTS_PER_CONFLICTED_LOAD = sm::TILE_DIM - 2;

/// The ceiling for the counter on the padded kernel, as a share of the
/// conflicted kernel's smallest reading: the reference rig reads 0.04%, a
/// discrete GPU 0.4%.
inline constexpr double MAX_PADDED_CONFLICT_SHARE = 0.02;

/// ncu's words when the user may not read the GPU's performance counters,
/// and the value it prints for a metric it has no reading of on this GPU
/// or in this version (it prints no error for one and exits 0).
inline constexpr const char* NCU_NO_PERMISSION = "ERR_NVGPUCTRPERM";
inline constexpr const char* NCU_NO_VALUE = "n/a";

/* ----------------------------- Reading ncu's CSV ----------------------------- */

/// One launch's three readings; -1 where the report has none.
struct LaunchMetrics {
  long instructions = -1;
  long wavefronts = -1;
  long conflicts = -1;
};

/// Every launch's readings, by the kernel's short name, in launch order.
using Readings = std::map<std::string, std::map<long, LaunchMetrics>>;

/// What ncu's CSV log holds: the readings, and the first row in which ncu
/// had no value for one of the three metrics, as it printed it.
struct Report {
  Readings readings;
  std::string unavailableRow;
};

/// The fields of one CSV row as ncu writes it: every field quoted, a quote
/// inside a field doubled, commas inside quotes part of the field.
inline std::vector<std::string> splitCsvRow(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char CH = line[i];
    if (quoted) {
      if (CH == '"' && i + 1 < line.size() && line[i + 1] == '"') {
        field += '"';
        ++i;
      } else if (CH == '"') {
        quoted = false;
      } else {
        field += CH;
      }
    } else if (CH == '"') {
      quoted = true;
    } else if (CH == ',') {
      fields.push_back(field);
      field.clear();
    } else {
      field += CH;
    }
  }
  fields.push_back(field);
  return fields;
}

/// A count as ncu prints it, with or without thousands separators; -1 when
/// the text is not a whole number.
inline long parseCount(const std::string& text) {
  long value = 0;
  bool any = false;
  for (const char CH : text) {
    if (CH >= '0' && CH <= '9') {
      value = value * 10 + (CH - '0');
      any = true;
    } else if (CH != ',') {
      return -1;
    }
  }
  return any ? value : -1;
}

/// A kernel's name without its namespaces and parameter list:
/// "<unnamed>::transposeNaive(const float *, float *, int)" is "transposeNaive".
inline std::string kernelShortName(const std::string& kernelName) {
  const std::size_t PAREN = kernelName.find('(');
  const std::string QUALIFIED = kernelName.substr(0, PAREN);
  const std::size_t SCOPE = QUALIFIED.rfind("::");
  return SCOPE == std::string::npos ? QUALIFIED : QUALIFIED.substr(SCOPE + 2);
}

/// The readings in ncu's CSV log: the header names the columns, the
/// "==PROF==" lines above it and any metric the check does not read are
/// ignored, and each row files one metric of one launch under its kernel.
inline Report readReport(const std::string& csv) {
  Report report;
  Readings& readings = report.readings;
  std::istringstream in(csv);
  std::string line;
  long idCol = -1;
  long kernelCol = -1;
  long metricCol = -1;
  long valueCol = -1;
  while (std::getline(in, line)) {
    if (line.rfind("==", 0) == 0 || line.empty()) {
      continue;
    }
    const std::vector<std::string> FIELDS = splitCsvRow(line);
    if (idCol < 0) {
      for (std::size_t i = 0; i < FIELDS.size(); ++i) {
        if (FIELDS[i] == "ID") {
          idCol = static_cast<long>(i);
        } else if (FIELDS[i] == "Kernel Name") {
          kernelCol = static_cast<long>(i);
        } else if (FIELDS[i] == "Metric Name") {
          metricCol = static_cast<long>(i);
        } else if (FIELDS[i] == "Metric Value") {
          valueCol = static_cast<long>(i);
        }
      }
      if (idCol < 0 || kernelCol < 0 || metricCol < 0 || valueCol < 0) {
        return report; // not ncu's CSV
      }
      continue;
    }
    const std::size_t NEEDED =
        static_cast<std::size_t>(std::max({idCol, kernelCol, metricCol, valueCol}));
    if (FIELDS.size() <= NEEDED) {
      continue;
    }
    const long ID = parseCount(FIELDS[static_cast<std::size_t>(idCol)]);
    if (ID < 0) {
      continue;
    }
    const std::string& METRIC = FIELDS[static_cast<std::size_t>(metricCol)];
    const bool READ =
        METRIC == METRIC_INSTRUCTIONS || METRIC == METRIC_WAVEFRONTS || METRIC == METRIC_CONFLICTS;
    if (!READ) {
      continue;
    }
    const std::string& TEXT = FIELDS[static_cast<std::size_t>(valueCol)];
    if (TEXT == NCU_NO_VALUE && report.unavailableRow.empty()) {
      report.unavailableRow = line;
    }
    LaunchMetrics& launch =
        readings[kernelShortName(FIELDS[static_cast<std::size_t>(kernelCol)])][ID];
    const long VALUE = parseCount(TEXT);
    if (METRIC == METRIC_INSTRUCTIONS) {
      launch.instructions = VALUE;
    } else if (METRIC == METRIC_WAVEFRONTS) {
      launch.wavefronts = VALUE;
    } else {
      launch.conflicts = VALUE;
    }
  }
  return report;
}

/// The first line of @p text that contains @p marker, as ncu printed it;
/// empty when none does. A skip quotes this line, so what it rests on is
/// ncu's text, not the check's.
inline std::string ncuLineWith(const std::string& text, const char* marker) {
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    if (line.find(marker) != std::string::npos) {
      return line;
    }
  }
  return "";
}

/// One kernel's launches, in one line, for the check's output: "no launch"
/// where the readings name none.
inline std::string describeLaunches(const Readings& readings, const char* kernel) {
  const auto FOUND = readings.find(kernel);
  if (FOUND == readings.end() || FOUND->second.empty()) {
    return "no launch";
  }
  std::string out;
  for (const auto& entry : FOUND->second) {
    const LaunchMetrics& m = entry.second;
    out += (out.empty() ? "" : "; ") + std::to_string(m.instructions) + " loads, " +
           std::to_string(m.wavefronts) + " wavefronts, " + std::to_string(m.conflicts) +
           " conflicts";
  }
  return out;
}

/* ----------------------------- The Verdict ----------------------------- */

/**
 * @brief Holds every launch in @p readings to the counts walkthrough 12
 *        gives, each departure a failure of the running test.
 *
 * Every kernel has a launch; the naive kernel executes no shared-load
 * instruction and reports no wavefront and no conflict; each tiled kernel
 * executes one shared-load instruction per warp of the launch; the
 * conflicting kernel's SASS-level wavefronts are exactly TILE_DIM times its
 * instructions and its L1TEX counter at least TILE_DIM - 2 times them; the
 * padded kernel's wavefronts equal its instructions, and its counter is
 * read and stays below MAX_PADDED_CONFLICT_SHARE of the conflicting
 * kernel's smallest reading.
 *
 * A reading the report lacks, or gives as other than a whole number, is -1
 * (readReport). Each exact count and the floor fail it; the padded kernel's
 * ceiling would pass it, so that counter must be read before the ceiling
 * applies.
 */
inline void expectWalkthroughCounts(const Readings& readings) {
  for (const char* kernel : {NAIVE, CONFLICT, PADDED}) {
    ASSERT_TRUE(readings.count(kernel) != 0 && !readings.at(kernel).empty())
        << "ncu's report names no launch of " << kernel;
  }

  for (const auto& [id, m] : readings.at(NAIVE)) {
    EXPECT_EQ(m.instructions, 0) << NAIVE << " launch " << id << " loads from shared memory";
    EXPECT_EQ(m.wavefronts, 0) << NAIVE << " launch " << id;
    EXPECT_EQ(m.conflicts, 0) << NAIVE << " launch " << id << " reports bank conflicts";
  }

  long fewestConflicts = -1;
  for (const auto& [id, m] : readings.at(CONFLICT)) {
    EXPECT_EQ(m.instructions, WARPS_PER_LAUNCH)
        << CONFLICT << " launch " << id << " does not load one tile column per warp";
    EXPECT_EQ(m.wavefronts, CONFLICTED_WAVEFRONTS_PER_LOAD * m.instructions)
        << CONFLICT << " launch " << id << ": its column reads no longer take " << sm::TILE_DIM
        << " wavefronts each, so the tile no longer conflicts";
    EXPECT_GE(m.conflicts, MIN_CONFLICTS_PER_CONFLICTED_LOAD * m.instructions)
        << CONFLICT << " launch " << id << ": ncu counts fewer bank conflicts than a "
        << sm::TILE_DIM << "-way conflict makes";
    if (fewestConflicts < 0 || m.conflicts < fewestConflicts) {
      fewestConflicts = m.conflicts;
    }
  }

  for (const auto& [id, m] : readings.at(PADDED)) {
    EXPECT_EQ(m.instructions, WARPS_PER_LAUNCH)
        << PADDED << " launch " << id << " does not load one tile column per warp";
    EXPECT_EQ(m.wavefronts, m.instructions)
        << PADDED << " launch " << id << ": its column reads take more than one wavefront "
        << "each, so the padding no longer spreads them over the banks";
    if (m.conflicts < 0) {
      ADD_FAILURE() << PADDED << " launch " << id << ": ncu's report has no count of "
                    << METRIC_CONFLICTS << " for it (no row, or a value that is not a whole "
                    << "number), so nothing shows the padding took the conflicts away";
      continue;
    }
    EXPECT_LT(static_cast<double>(m.conflicts),
              MAX_PADDED_CONFLICT_SHARE * static_cast<double>(fewestConflicts))
        << PADDED << " launch " << id << ": ncu's counter is not far below the conflicting "
        << "kernel's " << fewestConflicts;
  }
}

} // namespace bank_conflicts_check
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_GPU_03_BANK_CONFLICTS_CHECK_HPP
