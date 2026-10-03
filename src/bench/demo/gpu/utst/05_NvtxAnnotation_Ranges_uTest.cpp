/**
 * @file 05_NvtxAnnotation_Ranges_uTest.cpp
 * @brief The check behind walkthrough 13: under Nsight Systems, every call of
 *        demo 05's G1Phases carries its three named NVTX ranges, in order,
 *        inside the test's own range, each holding its phase's GPU work.
 *
 * Not part of demo 05, whose source shows only what it teaches. This program
 * runs the demo binary (BenchDemo_Gpu_05_NvtxAnnotation, whose path the build
 * passes in) under nsys with --profile nsight, reads the report's push/pop
 * trace and GPU trace through `nsys stats`, and holds them to what the
 * walkthrough shows:
 *  - one range per test, named after it, NvtxAnnotation.G1 and
 *    NvtxAnnotation.G1Phases, neither inside the other;
 *  - G1's range holds its measured calls' kernels and no range: the
 *    unannotated view;
 *  - G1Phases's range holds copy_in, kernel and copy_out once per measured
 *    call, in that order, one after another, directly inside it, and the
 *    warmup call's three lie outside every test's range;
 *  - in every call, copy_in holds its two copies to the device, kernel its one
 *    SAXPY kernel and copy_out its copy back, each from start to end.
 * Every count follows from the flags the check passes; nothing compares a
 * duration. The process plumbing is walkthrough 15's
 * (12_MemcheckProfiler_Check.hpp); the reading of nsys's CSV is here. ctest
 * runs it under the demo and nsight labels.
 *
 * Usage:
 *   @code{.sh}
 *   ctest --test-dir build -L nsight
 *   ./build/bin/tests/TestDemoNvtxRanges      # the same, by hand
 *   @endcode
 */

#include "src/bench/demo/cpu/utst/12_MemcheckProfiler_Check.hpp"
#include "src/bench/demo/gpu/05_NvtxAnnotation_Phases.hpp"
#include "src/bench/inc/Nvtx.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <filesystem>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

namespace check = vernier::bench::demo::memcheck_check;
namespace phase = vernier::bench::demo::nvtx_annotation;
namespace fs = std::filesystem;

namespace {

/* ----------------------------- Constants ----------------------------- */

/// The demo binary the check runs; the build passes its path.
constexpr const char* DEMO_BINARY = VERNIER_DEMO_GPU_05_BINARY;

/// The two tests, as their per-test ranges are named.
constexpr const char* UNANNOTATED = "NvtxAnnotation.G1";
constexpr const char* ANNOTATED = "NvtxAnnotation.G1Phases";

/// The run: a few calls per test, all traced, after one warmup call.
constexpr int CYCLES = 4;
constexpr int REPEATS = 2;
constexpr int WARMUP_CALLS = 1;
constexpr int MEASURED_CALLS = CYCLES * REPEATS;

/// How the GPU trace names the work: the example's kernel, the copy directions.
constexpr const char* KERNEL_NAME = "saxpyKernel";
constexpr const char* TO_DEVICE = "Host-to-Device";
constexpr const char* TO_HOST = "Device-to-Host";

/// At most this many problems are listed; the first ones name the cause.
constexpr std::size_t MAX_PROBLEMS = 12;

/* ----------------------------- The Trace ----------------------------- */

/// One push/pop range of nsys's nvtx_pushpop_trace.
struct NvtxRange {
  long long start = 0;
  long long end = 0;
  std::string name;      ///< Without nsys's domain prefix
  int level = 0;         ///< 0 at the bottom of its thread's stack
  long long id = -1;     ///< nsys's RangeId
  long long parent = -1; ///< The enclosing range's id; -1 at level 0
};

/// One operation of nsys's cuda_gpu_trace.
struct GpuOp {
  enum class Kind { KERNEL, TO_DEVICE, TO_HOST, OTHER };
  Kind kind = Kind::OTHER;
  long long start = 0;
  long long end = 0;
};

/// The fields of one CSV row as nsys writes it: a field with a comma or a
/// quote in it is quoted, a quote inside doubled.
std::vector<std::string> splitCsvRow(const std::string& line) {
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
    } else if (CH != '\r') {
      field += CH;
    }
  }
  fields.push_back(field);
  return fields;
}

/// A whole number as nsys prints it; false when the text is not one.
bool parseInteger(const std::string& text, long long& value) {
  if (text.empty()) {
    return false;
  }
  std::size_t used = 0;
  try {
    value = std::stoll(text, &used);
  } catch (const std::exception&) {
    return false;
  }
  return used == text.size();
}

/// The rows of a CSV report with the columns @p wanted, by name, in that
/// order; @p error names what is missing when the header lacks one.
std::vector<std::vector<std::string>>
readColumns(const std::string& csv, const std::vector<std::string>& wanted, std::string& error) {
  std::vector<std::vector<std::string>> rows;
  std::istringstream in(csv);
  std::string line;
  std::vector<std::size_t> index;
  while (std::getline(in, line)) {
    if (line.empty() || line == "\r") {
      continue;
    }
    const std::vector<std::string> FIELDS = splitCsvRow(line);
    if (index.empty()) {
      for (const std::string& name : wanted) {
        const auto IT = std::find(FIELDS.begin(), FIELDS.end(), name);
        if (IT == FIELDS.end()) {
          error = "the report has no column '" + name + "'; its header is: " + line;
          return {};
        }
        index.push_back(static_cast<std::size_t>(IT - FIELDS.begin()));
      }
      continue;
    }
    std::vector<std::string> row;
    for (const std::size_t I : index) {
      row.push_back(I < FIELDS.size() ? FIELDS[I] : std::string());
    }
    rows.push_back(row);
  }
  if (index.empty()) {
    error = "the report is empty";
  }
  return rows;
}

/// A range's name without nsys's domain prefix: ":copy_in" is "copy_in".
std::string withoutDomain(const std::string& name) {
  const std::size_t COLON = name.find(':');
  return COLON == std::string::npos ? name : name.substr(COLON + 1);
}

/// The ranges of an nvtx_pushpop_trace CSV; @p error says why it cannot be read.
std::vector<NvtxRange> readRanges(const std::string& csv, std::string& error) {
  const auto ROWS =
      readColumns(csv, {"Start (ns)", "End (ns)", "Name", "Lvl", "RangeId", "ParentId"}, error);
  std::vector<NvtxRange> ranges;
  for (const auto& row : ROWS) {
    NvtxRange range;
    long long level = 0;
    if (!parseInteger(row[0], range.start) || !parseInteger(row[1], range.end) ||
        !parseInteger(row[3], level) || !parseInteger(row[4], range.id) ||
        (!row[5].empty() && !parseInteger(row[5], range.parent))) {
      error = "a push/pop row does not read: " + row[0] + "," + row[1] + "," + row[2] + "," +
              row[3] + "," + row[4] + "," + row[5];
      return {};
    }
    range.name = withoutDomain(row[2]);
    range.level = static_cast<int>(level);
    ranges.push_back(range);
  }
  return ranges;
}

/// The operations of a cuda_gpu_trace CSV; @p error says why it cannot be read.
std::vector<GpuOp> readGpuOps(const std::string& csv, std::string& error) {
  const auto ROWS = readColumns(csv, {"Start (ns)", "Duration (ns)", "Name"}, error);
  std::vector<GpuOp> ops;
  for (const auto& row : ROWS) {
    GpuOp op;
    long long duration = 0;
    if (!parseInteger(row[0], op.start) || !parseInteger(row[1], duration)) {
      error = "a GPU trace row does not read: " + row[0] + "," + row[1] + "," + row[2];
      return {};
    }
    op.end = op.start + duration;
    if (row[2].find(KERNEL_NAME) != std::string::npos) {
      op.kind = GpuOp::Kind::KERNEL;
    } else if (row[2].find(TO_DEVICE) != std::string::npos) {
      op.kind = GpuOp::Kind::TO_DEVICE;
    } else if (row[2].find(TO_HOST) != std::string::npos) {
      op.kind = GpuOp::Kind::TO_HOST;
    }
    ops.push_back(op);
  }
  return ops;
}

/* ----------------------------- The Assessment ----------------------------- */

/// How many operations of each kind lie wholly inside [start, end].
struct Contents {
  int kernels = 0;
  int toDevice = 0;
  int toHost = 0;
  int other = 0;
};

Contents contentsOf(const std::vector<GpuOp>& ops, long long start, long long end) {
  Contents inside;
  for (const GpuOp& op : ops) {
    if (op.start < start || op.end > end) {
      continue;
    }
    switch (op.kind) {
    case GpuOp::Kind::KERNEL:
      ++inside.kernels;
      break;
    case GpuOp::Kind::TO_DEVICE:
      ++inside.toDevice;
      break;
    case GpuOp::Kind::TO_HOST:
      ++inside.toHost;
      break;
    case GpuOp::Kind::OTHER:
      ++inside.other;
      break;
    }
  }
  return inside;
}

std::string describe(const Contents& c) {
  return std::to_string(c.kernels) + " kernel(s), " + std::to_string(c.toDevice) +
         " copy(ies) to the device, " + std::to_string(c.toHost) + " copy(ies) back, " +
         std::to_string(c.other) + " other";
}

bool isPhase(const std::string& name) {
  return std::find_if(std::begin(phase::PHASES), std::end(phase::PHASES),
                      [&](const char* p) { return name == p; }) != std::end(phase::PHASES);
}

/// What phase @p name holds, per call: two copies in, one kernel, one copy back.
bool holdsItsWork(const std::string& name, const Contents& c) {
  if (name == phase::COPY_IN) {
    return c.toDevice == 2 && c.kernels == 0 && c.toHost == 0 && c.other == 0;
  }
  if (name == phase::KERNEL) {
    return c.kernels == 1 && c.toDevice == 0 && c.toHost == 0 && c.other == 0;
  }
  return c.toHost == 1 && c.kernels == 0 && c.toDevice == 0 && c.other == 0;
}

/**
 * @brief What the trace must show, as the problems it does not: empty when it
 *        shows everything the walkthrough says.
 */
std::vector<std::string> assessRanges(std::vector<NvtxRange> ranges, const std::vector<GpuOp>& ops,
                                      int measuredCalls, int warmupCalls) {
  std::vector<std::string> problems;
  std::sort(ranges.begin(), ranges.end(),
            [](const NvtxRange& a, const NvtxRange& b) { return a.start < b.start; });

  std::map<std::string, std::vector<const NvtxRange*>> byTest;
  for (const NvtxRange& range : ranges) {
    if (range.name == UNANNOTATED || range.name == ANNOTATED) {
      byTest[range.name].push_back(&range);
    }
  }
  for (const char* test : {UNANNOTATED, ANNOTATED}) {
    if (byTest[test].size() != 1) {
      problems.push_back("expected one range named " + std::string(test) + ", found " +
                         std::to_string(byTest[test].size()));
    }
  }
  if (!problems.empty()) {
    return problems;
  }
  const NvtxRange& g1 = *byTest[UNANNOTATED].front();
  const NvtxRange& phased = *byTest[ANNOTATED].front();

  for (const NvtxRange* test : {&g1, &phased}) {
    if (test->level != 0 || test->parent != -1) {
      problems.push_back(test->name + "'s range is inside another range (level " +
                         std::to_string(test->level) + "): the range before it was still open");
    }
  }
  if (g1.end > phased.start && phased.end > g1.start) {
    problems.push_back("the two tests' ranges overlap");
  }

  // G1: the unannotated view.
  for (const NvtxRange& range : ranges) {
    if (range.parent == g1.id) {
      problems.push_back(std::string(UNANNOTATED) + "'s range holds a range, " + range.name +
                         ", where it should hold none");
    }
  }
  const Contents G1_INSIDE = contentsOf(ops, g1.start, g1.end);
  if (G1_INSIDE.kernels != measuredCalls) {
    problems.push_back(std::string(UNANNOTATED) + "'s range holds " +
                       std::to_string(G1_INSIDE.kernels) + " kernels, not the " +
                       std::to_string(measuredCalls) + " of its measured calls");
  }

  // G1Phases: the three phases of every measured call, directly inside.
  std::vector<const NvtxRange*> children;
  std::vector<const NvtxRange*> outside;
  for (const NvtxRange& range : ranges) {
    if (range.parent == phased.id) {
      children.push_back(&range);
    } else if (isPhase(range.name) && range.parent == -1) {
      outside.push_back(&range);
    }
  }
  const std::size_t EXPECTED_CHILDREN = 3 * static_cast<std::size_t>(measuredCalls);
  if (children.size() != EXPECTED_CHILDREN) {
    problems.push_back(std::string(ANNOTATED) + "'s range holds " +
                       std::to_string(children.size()) + " ranges, not the " +
                       std::to_string(EXPECTED_CHILDREN) + " of " + std::to_string(measuredCalls) +
                       " calls' three phases");
  }
  for (std::size_t i = 0; i < children.size(); ++i) {
    const NvtxRange& range = *children[i];
    const std::string WHERE = "call " + std::to_string(i / 3) + ", range " + range.name;
    if (range.name != phase::PHASES[i % 3]) {
      problems.push_back(WHERE + ": expected " + std::string(phase::PHASES[i % 3]) +
                         " here (the order is copy_in, kernel, copy_out)");
      continue;
    }
    if (range.level != phased.level + 1) {
      problems.push_back(WHERE + ": not directly inside the test's range");
    }
    if (i + 1 < children.size() && range.end > children[i + 1]->start) {
      problems.push_back(WHERE + ": still open when the next range opened");
    }
    const Contents INSIDE = contentsOf(ops, range.start, range.end);
    if (!holdsItsWork(range.name, INSIDE)) {
      problems.push_back(WHERE + " holds " + describe(INSIDE));
    }
  }
  const Contents PHASED_INSIDE = contentsOf(ops, phased.start, phased.end);
  const int EXPECTED_OPS = 4 * measuredCalls;
  const int PHASED_OPS =
      PHASED_INSIDE.kernels + PHASED_INSIDE.toDevice + PHASED_INSIDE.toHost + PHASED_INSIDE.other;
  if (PHASED_OPS != EXPECTED_OPS) {
    problems.push_back(std::string(ANNOTATED) + "'s range holds " + std::to_string(PHASED_OPS) +
                       " GPU operations, not " + std::to_string(EXPECTED_OPS));
  }

  // The warmup's phases: outside every test's range, before G1Phases's.
  const std::size_t EXPECTED_OUTSIDE = 3 * static_cast<std::size_t>(warmupCalls);
  if (outside.size() != EXPECTED_OUTSIDE) {
    problems.push_back(std::to_string(outside.size()) +
                       " phase ranges lie outside every test's range, not the " +
                       std::to_string(EXPECTED_OUTSIDE) + " of the warmup");
  } else {
    for (std::size_t i = 0; i < outside.size(); ++i) {
      if (outside[i]->name != phase::PHASES[i % 3] || outside[i]->end > phased.start) {
        problems.push_back("warmup range " + outside[i]->name +
                           " is out of order or after the test's range opened");
      }
    }
  }

  const std::size_t EXPECTED_TOTAL = 2 + EXPECTED_CHILDREN + EXPECTED_OUTSIDE;
  if (ranges.size() != EXPECTED_TOTAL) {
    problems.push_back("the report has " + std::to_string(ranges.size()) + " ranges, not " +
                       std::to_string(EXPECTED_TOTAL));
  }
  if (problems.size() > MAX_PROBLEMS) {
    problems.resize(MAX_PROBLEMS);
  }
  return problems;
}

/// The smallest margins of the kernel ranges around their kernels, in
/// nanoseconds: from the push to the kernel's start, and from the kernel's
/// end to the pop. Information for the output; nothing is asserted on them.
std::pair<long long, long long> kernelMargins(const std::vector<NvtxRange>& ranges,
                                              const std::vector<GpuOp>& ops) {
  long long before = -1;
  long long after = -1;
  for (const NvtxRange& range : ranges) {
    if (range.name != phase::KERNEL) {
      continue;
    }
    for (const GpuOp& op : ops) {
      if (op.kind == GpuOp::Kind::KERNEL && op.start >= range.start && op.end <= range.end) {
        const long long B = op.start - range.start;
        const long long A = range.end - op.end;
        before = before < 0 ? B : std::min(before, B);
        after = after < 0 ? A : std::min(after, A);
      }
    }
  }
  return {before, after};
}

/// The check's temporary directory: removed when the test passes or skips,
/// kept, and named, when it fails.
class RunDirectory {
public:
  explicit RunDirectory(fs::path dir) : dir_(std::move(dir)) {}
  ~RunDirectory() {
    if (::testing::Test::HasFailure()) {
      std::printf("nsys output kept in %s\n", dir_.c_str());
      return;
    }
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }
  RunDirectory(const RunDirectory&) = delete;
  RunDirectory& operator=(const RunDirectory&) = delete;

  [[nodiscard]] fs::path operator/(const char* name) const { return dir_ / name; }

private:
  fs::path dir_;
};

/// The demo's arguments, in the order the check and the walkthrough give them.
/// Started by hand under nsys, the backend makes an empty folder per test
/// (<Suite.Case>.nsight); @p folders keeps them in the check's own directory.
std::vector<std::string> demoArgs(const std::string& demo, const fs::path& folders) {
  return {demo,
          "--profile",
          "nsight",
          "--profile-output-dir",
          folders.string(),
          "--gtest_filter=NvtxAnnotation.*",
          "--gtest_print_time=0",
          "--cycles",
          std::to_string(CYCLES),
          "--repeats",
          std::to_string(REPEATS),
          "--warmup",
          std::to_string(WARMUP_CALLS)};
}

} // namespace

/* ----------------------------- Tests ----------------------------- */

/**
 * @test Nsight Systems records each measured call of G1Phases as copy_in,
 *       kernel, copy_out inside the test's range, each around its own GPU
 *       work, and G1's range with no range inside
 *
 * Runs the demo's two tests under nsys, reads the report with nsys stats and
 * checks it with assessRanges(). Skipped where nsys is not on PATH, no CUDA
 * device is present, or the build has no NVTX headers, decided before
 * anything runs; and where the demo, which lists its tests on its own, does
 * not start under nsys, quoting nsys's last lines. The demo's two tests must
 * pass under nsys and nsys must exit 0; anything else fails with the run's
 * last lines, and a failing run keeps its files and says where.
 */
TEST(NvtxRanges, RecordedByNsightSystems) {
#if !VERNIER_NVTX_USABLE
  GTEST_SKIP() << "this build has no NVTX headers (the CUDA toolkit's nvtx3), so the demo's "
                  "ranges and the Nsight backend's compile to nothing";
#else
  if (!vernier::bench::profiler_env::isOnPath("nsys")) {
    GTEST_SKIP() << "nsys is not on PATH; this test runs the demo under Nsight Systems";
  }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP() << "no CUDA device available; this test runs the demo's kernels under nsys";
  }

  std::error_code found;
  const std::string DEMO = fs::canonical(DEMO_BINARY, found).string();
  ASSERT_FALSE(found) << "the demo binary is missing: " << DEMO_BINARY;

  std::string dirTemplate = (fs::temp_directory_path() / "vernier-demo-gpu05-XXXXXX").string();
  ASSERT_NE(::mkdtemp(dirTemplate.data()), nullptr) << "cannot create a temporary directory";
  const RunDirectory DIR{fs::path(dirTemplate)};

  // The demo starts on its own, so a start that fails under nsys is nsys's.
  const check::ChildExit LISTED = check::runLogged({DEMO, "--gtest_list_tests"}, DIR / "list.txt");
  const std::string LIST = check::readText(DIR / "list.txt");
  ASSERT_TRUE(check::exitedWith(LISTED, 0) && LIST.find("G1Phases") != std::string::npos)
      << "the demo does not list its tests (it " << check::describe(LISTED) << "):\n"
      << check::lastLines(LIST);

  std::vector<std::string> args = {"nsys",
                                   "profile",
                                   "-o",
                                   (DIR / "capture").string(),
                                   "-t",
                                   "cuda,nvtx",
                                   "--sample=none",
                                   "--cpuctxsw=none",
                                   "--force-overwrite",
                                   "true"};
  const std::vector<std::string> DEMO_ARGS = demoArgs(DEMO, DIR / "folders");
  args.insert(args.end(), DEMO_ARGS.begin(), DEMO_ARGS.end());
  const check::ChildExit END = check::runLogged(args, DIR / "program.txt");
  const std::string PROGRAM = check::readText(DIR / "program.txt");

  if (!check::testsStarted(PROGRAM)) {
    GTEST_SKIP() << "the demo starts on its own but not under nsys (nsys " << check::describe(END)
                 << "). nsys printed:\n"
                 << check::lastLines(PROGRAM, 8);
  }
  ASSERT_NE(PROGRAM.find("[  PASSED  ] 2 tests."), std::string::npos)
      << "the demo's two tests did not both pass under nsys:\n"
      << check::lastLines(PROGRAM, 40);
  const fs::path REPORT = DIR / "capture.nsys-rep";
  ASSERT_TRUE(check::exitedWith(END, 0)) << "nsys " << check::describe(END) << ":\n"
                                         << check::lastLines(PROGRAM, 20);
  ASSERT_TRUE(fs::exists(REPORT)) << "nsys wrote no report at " << REPORT << ":\n"
                                  << check::lastLines(PROGRAM, 20);

  const check::ChildExit STATS = check::runLogged(
      {"nsys", "stats", "--force-export=true", "--report", "nvtx_pushpop_trace", "--report",
       "cuda_gpu_trace", "--format", "csv", "--output", (DIR / "trace").string(), REPORT.string()},
      DIR / "stats.txt");
  ASSERT_TRUE(check::exitedWith(STATS, 0))
      << "nsys stats " << check::describe(STATS) << ":\n"
      << check::lastLines(check::readText(DIR / "stats.txt"), 20);

  std::string error;
  const std::vector<NvtxRange> RANGES =
      readRanges(check::readText(DIR / "trace_nvtx_pushpop_trace.csv"), error);
  const std::vector<GpuOp> OPS =
      error.empty() ? readGpuOps(check::readText(DIR / "trace_cuda_gpu_trace.csv"), error)
                    : std::vector<GpuOp>{};
  ASSERT_TRUE(error.empty()) << error << "\nnsys stats printed:\n"
                             << check::lastLines(check::readText(DIR / "stats.txt"), 20);

  const auto [BEFORE, AFTER] = kernelMargins(RANGES, OPS);
  std::printf("[NvtxRanges.RecordedByNsightSystems]  %zu ranges, %zu GPU operations; kernel "
              "ranges open %.3f us before their kernels at least and close %.3f us after\n",
              RANGES.size(), OPS.size(), static_cast<double>(BEFORE) / 1e3,
              static_cast<double>(AFTER) / 1e3);

  const std::vector<std::string> PROBLEMS = assessRanges(RANGES, OPS, MEASURED_CALLS, WARMUP_CALLS);
  for (const std::string& problem : PROBLEMS) {
    ADD_FAILURE() << problem;
  }
#endif
}

/* ----------------------------- Reading Tests ----------------------------- */

// What RecordedByNsightSystems's reading rests on, checked without nsys or a
// device: a real report of demo 05 from each nsys version the check has run
// with, and traces built the way the demo's calls lay them out.

namespace {

/**
 * @brief A real report of demo 05, as `nsys stats` wrote its two CSVs, from a
 *        capture the way the check takes it but with one warmup call and one
 *        measured call per test (--cycles 1 --repeats 1 --warmup 1).
 *
 * The push/pop trace: G1's range, the warmup's three phases outside any test's
 * range, G1Phases's range and its call's three phases inside it. The GPU
 * trace: four calls of two copies to the device, the kernel, whose name nsys
 * quotes, and the copy back.
 */
struct NsysReport {
  const char* version;  ///< The nsys that wrote it, as a test name
  const char* pushPop;  ///< nvtx_pushpop_trace
  const char* gpuTrace; ///< cuda_gpu_trace
};

/// How a failure names a report: by the nsys that wrote it.
void PrintTo(const NsysReport& report, std::ostream* out) { *out << report.version; }

/// The calls in a report: G1's warmup and measured call, then G1Phases's.
constexpr std::size_t REPORT_CALLS = 4;

/// nsys 2026.3.1, in the project's CUDA image on an RTX 5000 Ada laptop GPU.
constexpr NsysReport NSYS_2026_3_1 = {
    "Nsys2026_3_1",
    "Start (ns),End (ns),Duration (ns),DurChild (ns),DurNonChild (ns),Name,PID,TID,Lvl,NumChild,"
    "RangeId,ParentId,RangeStack,NameTree\n"
    "1137544193,1485711471,348167278,0,348167278,:NvtxAnnotation.G1,53,53,0,0,1,,:1,"
    ":NvtxAnnotation.G1\n"
    "1496163242,1498711076,2547834,0,2547834,:copy_in,53,53,0,0,2,,:2,:copy_in\n"
    "1498713787,1498780483,66696,0,66696,:kernel,53,53,0,0,3,,:3,:kernel\n"
    "1498781433,1499967543,1186110,0,1186110,:copy_out,53,53,0,0,4,,:4,:copy_out\n"
    "1499971563,1503940278,3968715,3934391,34324,:NvtxAnnotation.G1Phases,53,53,0,3,5,,:5,"
    ":NvtxAnnotation.G1Phases\n"
    "1499980066,1502730617,2750551,0,2750551,:copy_in,53,53,1,0,6,5,:5:6,--:copy_in\n"
    "1502732596,1502821610,89014,0,89014,:kernel,53,53,1,0,7,5,:5:7,--:kernel\n"
    "1502822338,1503917164,1094826,0,1094826,:copy_out,53,53,1,0,8,5,:5:8,--:copy_out\n",
    "Start (ns),Duration (ns),CorrId,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Reg/Trd,StcSMem (MB),"
    "DymSMem (MB),Bytes (MB),Throughput (MB/s),SrcMemKd,DstMemKd,Device,Ctx,GreenCtx,Strm,Name\n"
    "1134711990,661398,123,,,,,,,,,,4.194,6337.593,Pinned,Device,"
    "NVIDIA RTX 5000 Ada Generation Laptop GPU (0),1,,13,[CUDA memcpy Host-to-Device]\n"
    "1135375308,660534,124,,,,,,,,,,4.194,6345.982,Pinned,Device,"
    "NVIDIA RTX 5000 Ada Generation Laptop GPU (0),1,,13,[CUDA memcpy Host-to-Device]\n"
    "1136125733,7072,128,4096,1,1,256,1,1,16,0.000,0.000,,,,,"
    "NVIDIA RTX 5000 Ada Generation Laptop GPU (0),1,,13,"
    "\"vernier::bench::demo::<unnamed>::saxpyKernel(float, const float *, float *,"
    " unsigned long)\"\n"
    "1136151973,542802,130,,,,,,,,,,4.194,7725.908,Device,Pinned,"
    "NVIDIA RTX 5000 Ada Generation Laptop GPU (0),1,,13,[CUDA memcpy Device-to-Host]\n"
    "1138878399,520818,132,,,,,,,,,,4.194,8053.064,Pinned,Device,"
    "NVIDIA RTX 5000 Ada Generation Laptop GPU (0),1,,13,[CUDA memcpy Host-to-Device]\n"
    "1139400913,526641,133,,,,,,,,,,4.194,7960.789,Pinned,Device,"
    "NVIDIA RTX 5000 Ada Generation Laptop GPU (0),1,,13,[CUDA memcpy Host-to-Device]\n"
    "1139935682,6945,135,4096,1,1,256,1,1,16,0.000,0.000,,,,,"
    "NVIDIA RTX 5000 Ada Generation Laptop GPU (0),1,,13,"
    "\"vernier::bench::demo::<unnamed>::saxpyKernel(float, const float *, float *,"
    " unsigned long)\"\n"
    "1139948227,520497,137,,,,,,,,,,4.194,8057.258,Device,Pinned,"
    "NVIDIA RTX 5000 Ada Generation Laptop GPU (0),1,,13,[CUDA memcpy Device-to-Host]\n"
    "1497544834,638613,149,,,,,,,,,,4.194,6564.086,Pinned,Device,"
    "NVIDIA RTX 5000 Ada Generation Laptop GPU (0),1,,14,[CUDA memcpy Host-to-Device]\n"
    "1498185367,512241,150,,,,,,,,,,4.194,8187.281,Pinned,Device,"
    "NVIDIA RTX 5000 Ada Generation Laptop GPU (0),1,,14,[CUDA memcpy Host-to-Device]\n"
    "1498770795,6816,153,4096,1,1,256,1,1,16,0.000,0.000,,,,,"
    "NVIDIA RTX 5000 Ada Generation Laptop GPU (0),1,,14,"
    "\"vernier::bench::demo::<unnamed>::saxpyKernel(float, const float *, float *,"
    " unsigned long)\"\n"
    "1498799148,500528,156,,,,,,,,,,4.194,8376.025,Device,Pinned,"
    "NVIDIA RTX 5000 Ada Generation Laptop GPU (0),1,,14,[CUDA memcpy Device-to-Host]\n"
    "1501710988,502896,158,,,,,,,,,,4.194,8338.276,Pinned,Device,"
    "NVIDIA RTX 5000 Ada Generation Laptop GPU (0),1,,14,[CUDA memcpy Host-to-Device]\n"
    "1502215708,507345,159,,,,,,,,,,4.194,8266.973,Pinned,Device,"
    "NVIDIA RTX 5000 Ada Generation Laptop GPU (0),1,,14,[CUDA memcpy Host-to-Device]\n"
    "1502812848,6592,162,4096,1,1,256,1,1,16,0.000,0.000,,,,,"
    "NVIDIA RTX 5000 Ada Generation Laptop GPU (0),1,,14,"
    "\"vernier::bench::demo::<unnamed>::saxpyKernel(float, const float *, float *,"
    " unsigned long)\"\n"
    "1502833457,504112,165,,,,,,,,,,4.194,8317.305,Device,Pinned,"
    "NVIDIA RTX 5000 Ada Generation Laptop GPU (0),1,,14,[CUDA memcpy Device-to-Host]\n"};

/// nsys 2025.3.2, on the Jetson AGX Thor reference rig.
constexpr NsysReport NSYS_2025_3_2 = {
    "Nsys2025_3_2",
    "Start (ns),End (ns),Duration (ns),DurChild (ns),DurNonChild (ns),Name,PID,TID,Lvl,NumChild,"
    "RangeId,ParentId,RangeStack,NameTree\n"
    "269081842,281085795,12003953,0,12003953,:NvtxAnnotation.G1,126760,126760,0,0,1,,:1,"
    ":NvtxAnnotation.G1\n"
    "290594221,291296610,702389,0,702389,:copy_in,126760,126760,0,0,2,,:2,:copy_in\n"
    "291297499,291356202,58703,0,58703,:kernel,126760,126760,0,0,3,,:3,:kernel\n"
    "291357017,291604712,247695,0,247695,:copy_out,126760,126760,0,0,4,,:4,:copy_out\n"
    "291609175,292598591,989416,959417,29999,:NvtxAnnotation.G1Phases,126760,126760,0,3,5,,:5,"
    ":NvtxAnnotation.G1Phases\n"
    "291616601,292275110,658509,0,658509,:copy_in,126760,126760,1,0,6,5,:5:6,--:copy_in\n"
    "292275776,292327517,51741,0,51741,:kernel,126760,126760,1,0,7,5,:5:7,--:kernel\n"
    "292328202,292577369,249167,0,249167,:copy_out,126760,126760,1,0,8,5,:5:8,--:copy_out\n",
    "Start (ns),Duration (ns),CorrId,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Reg/Trd,StcSMem (MB),"
    "DymSMem (MB),Bytes (MB),Throughput (MB/s),SrcMemKd,DstMemKd,Device,Ctx,GreenCtx,Strm,Name\n"
    "268525307,21952,123,,,,,,,,,,4.194,191063.130,Pinned,Device,NVIDIA Thor (0),1,,13,"
    "[CUDA memcpy Host-to-Device]\n"
    "268551195,24864,124,,,,,,,,,,4.194,168686.518,Pinned,Device,NVIDIA Thor (0),1,,13,"
    "[CUDA memcpy Host-to-Device]\n"
    "268682427,40672,128,4096,1,1,256,1,1,16,0.000,0.000,,,,,NVIDIA Thor (0),1,,13,"
    "\"vernier::bench::demo::<unnamed>::saxpyKernel(float, const float *, float *,"
    " unsigned long)\"\n"
    "268725979,15424,130,,,,,,,,,,4.194,271933.506,Device,Pinned,NVIDIA Thor (0),1,,13,"
    "[CUDA memcpy Device-to-Host]\n"
    "269720859,15840,132,,,,,,,,,,4.194,264790.606,Pinned,Device,NVIDIA Thor (0),1,,13,"
    "[CUDA memcpy Host-to-Device]\n"
    "269738363,15552,133,,,,,,,,,,4.194,269693.747,Pinned,Device,NVIDIA Thor (0),1,,13,"
    "[CUDA memcpy Host-to-Device]\n"
    "269756795,22528,135,4096,1,1,256,1,1,16,0.000,0.000,,,,,NVIDIA Thor (0),1,,13,"
    "\"vernier::bench::demo::<unnamed>::saxpyKernel(float, const float *, float *,"
    " unsigned long)\"\n"
    "269781723,15744,137,,,,,,,,,,4.194,266405.413,Device,Pinned,NVIDIA Thor (0),1,,13,"
    "[CUDA memcpy Device-to-Host]\n"
    "291233307,19648,149,,,,,,,,,,4.194,213469.102,Pinned,Device,NVIDIA Thor (0),1,,14,"
    "[CUDA memcpy Host-to-Device]\n"
    "291254971,18528,150,,,,,,,,,,4.194,226374.975,Pinned,Device,NVIDIA Thor (0),1,,14,"
    "[CUDA memcpy Host-to-Device]\n"
    "291318875,22624,153,4096,1,1,256,1,1,16,0.000,0.000,,,,,NVIDIA Thor (0),1,,14,"
    "\"vernier::bench::demo::<unnamed>::saxpyKernel(float, const float *, float *,"
    " unsigned long)\"\n"
    "291368155,15456,156,,,,,,,,,,4.194,271367.274,Device,Pinned,NVIDIA Thor (0),1,,14,"
    "[CUDA memcpy Device-to-Host]\n"
    "292221371,18944,158,,,,,,,,,,4.194,221404.725,Pinned,Device,NVIDIA Thor (0),1,,14,"
    "[CUDA memcpy Host-to-Device]\n"
    "292242107,15680,159,,,,,,,,,,4.194,267491.738,Pinned,Device,NVIDIA Thor (0),1,,14,"
    "[CUDA memcpy Host-to-Device]\n"
    "292289723,22624,162,4096,1,1,256,1,1,16,0.000,0.000,,,,,NVIDIA Thor (0),1,,14,"
    "\"vernier::bench::demo::<unnamed>::saxpyKernel(float, const float *, float *,"
    " unsigned long)\"\n"
    "292337563,15392,165,,,,,,,,,,4.194,272495.542,Device,Pinned,NVIDIA Thor (0),1,,14,"
    "[CUDA memcpy Device-to-Host]\n"};

/// A trace as the demo lays it out: G1's warmup call, then its range around
/// its measured calls; G1Phases's warmup phases, then its range around its
/// measured calls' phases, each phase around its own GPU work.
struct Trace {
  std::vector<NvtxRange> ranges;
  std::vector<GpuOp> ops;
};

Trace demoTrace(int measuredCalls, int warmupCalls) {
  Trace t;
  long long at = 1000000;
  long long nextId = 1;
  const auto work = [&](GpuOp::Kind kind) {
    t.ops.push_back({kind, at + 10, at + 60});
    at += 100;
  };
  const auto g1Call = [&] {
    work(GpuOp::Kind::TO_DEVICE);
    work(GpuOp::Kind::TO_DEVICE);
    work(GpuOp::Kind::KERNEL);
    work(GpuOp::Kind::TO_HOST);
  };
  const auto phasedCall = [&](int level, long long parent) {
    for (const char* name : phase::PHASES) {
      NvtxRange range;
      range.start = at;
      range.name = name;
      range.level = level;
      range.id = nextId++;
      range.parent = parent;
      if (range.name == phase::COPY_IN) {
        work(GpuOp::Kind::TO_DEVICE);
        work(GpuOp::Kind::TO_DEVICE);
      } else if (range.name == phase::KERNEL) {
        work(GpuOp::Kind::KERNEL);
      } else {
        work(GpuOp::Kind::TO_HOST);
      }
      range.end = at;
      at += 5;
      t.ranges.push_back(range);
    }
  };

  for (int w = 0; w < warmupCalls; ++w) {
    g1Call();
  }
  NvtxRange g1;
  g1.start = at;
  g1.name = UNANNOTATED;
  g1.id = nextId++;
  for (int c = 0; c < measuredCalls; ++c) {
    g1Call();
  }
  g1.end = at;
  at += 1000;
  t.ranges.push_back(g1);

  for (int w = 0; w < warmupCalls; ++w) {
    phasedCall(0, -1);
  }
  NvtxRange phased;
  phased.start = at;
  phased.name = ANNOTATED;
  phased.id = nextId++;
  at += 5;
  for (int c = 0; c < measuredCalls; ++c) {
    phasedCall(1, phased.id);
  }
  phased.end = at;
  t.ranges.push_back(phased);
  std::sort(t.ranges.begin(), t.ranges.end(),
            [](const NvtxRange& a, const NvtxRange& b) { return a.start < b.start; });
  return t;
}

/// The range named @p name that is the @p nth such (0 first), in start order.
NvtxRange& nthRange(Trace& t, const std::string& name, int nth) {
  for (NvtxRange& range : t.ranges) {
    if (range.name == name && nth-- == 0) {
      return range;
    }
  }
  throw std::runtime_error("the trace has no such range: " + name);
}

/// True when one of @p problems mentions @p text.
bool mentions(const std::vector<std::string>& problems, const std::string& text) {
  return std::any_of(problems.begin(), problems.end(),
                     [&](const std::string& p) { return p.find(text) != std::string::npos; });
}

} // namespace

/** @test A row's quoted fields split on the commas between them, not inside them */
TEST(NsysCsvTest, SplitsQuotedFieldsWithCommasInside) {
  const std::vector<std::string> FIELDS = splitCsvRow("1,2,\"a, b\",\"say \"\"hi\"\"\",,last\r");

  ASSERT_EQ(FIELDS.size(), 6U);
  EXPECT_EQ(FIELDS[2], "a, b");
  EXPECT_EQ(FIELDS[3], "say \"hi\"");
  EXPECT_EQ(FIELDS[4], "");
  EXPECT_EQ(FIELDS[5], "last");
}

/** @test A report without a column the check reads is an error that names the column */
TEST(NsysCsvTest, MissingColumnIsAnError) {
  std::string error;
  const std::vector<NvtxRange> RANGES =
      readRanges("Start (ns),End (ns),Name,Lvl,RangeId\n1,2,:x,0,1\n", error);

  EXPECT_TRUE(RANGES.empty());
  EXPECT_NE(error.find("'ParentId'"), std::string::npos) << error;

  error.clear();
  EXPECT_TRUE(readGpuOps("", error).empty());
  EXPECT_FALSE(error.empty());
}

/// The tests below, once for each nsys version's report.
class NsysReportTest : public ::testing::TestWithParam<NsysReport> {};

/** @test The push/pop trace reads by column name, names without their domain, no parent as -1 */
TEST_P(NsysReportTest, ReadsThePushPopTrace) {
  std::string error;
  const std::vector<NvtxRange> RANGES = readRanges(GetParam().pushPop, error);
  const auto DURATIONS = readColumns(GetParam().pushPop, {"Duration (ns)"}, error);

  ASSERT_TRUE(error.empty()) << error;
  ASSERT_EQ(RANGES.size(), 8U);
  ASSERT_EQ(DURATIONS.size(), RANGES.size());
  EXPECT_EQ(RANGES[0].name, UNANNOTATED);
  EXPECT_EQ(RANGES[0].level, 0);
  EXPECT_EQ(RANGES[0].parent, -1);
  EXPECT_EQ(RANGES[4].name, ANNOTATED);
  EXPECT_EQ(RANGES[4].level, 0);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(RANGES[1 + i].name, phase::PHASES[i]) << "the warmup's phases, outside";
    EXPECT_EQ(RANGES[1 + i].parent, -1);
    EXPECT_EQ(RANGES[5 + i].name, phase::PHASES[i]) << "the measured call's, inside";
    EXPECT_EQ(RANGES[5 + i].level, 1);
    EXPECT_EQ(RANGES[5 + i].parent, RANGES[4].id);
  }
  for (std::size_t i = 0; i < RANGES.size(); ++i) {
    EXPECT_EQ(std::to_string(RANGES[i].end - RANGES[i].start), DURATIONS[i][0])
        << "row " << i << ": Start and End read from the wrong columns";
  }
}

/** @test The GPU trace reads kernels and copies by name, the quoted kernel name included */
TEST_P(NsysReportTest, ReadsTheGpuTrace) {
  std::string error;
  const std::vector<GpuOp> OPS = readGpuOps(GetParam().gpuTrace, error);

  ASSERT_TRUE(error.empty()) << error;
  ASSERT_EQ(OPS.size(), 4 * REPORT_CALLS);
  const GpuOp::Kind CALL[] = {GpuOp::Kind::TO_DEVICE, GpuOp::Kind::TO_DEVICE, GpuOp::Kind::KERNEL,
                              GpuOp::Kind::TO_HOST};
  for (std::size_t i = 0; i < OPS.size(); ++i) {
    EXPECT_EQ(OPS[i].kind, CALL[i % 4]) << "operation " << i;
  }
}

/** @test A real report passes the assessment, with one measured and one warmup call */
TEST_P(NsysReportTest, PassesTheAssessment) {
  std::string error;
  const std::vector<NvtxRange> RANGES = readRanges(GetParam().pushPop, error);
  const std::vector<GpuOp> OPS = readGpuOps(GetParam().gpuTrace, error);
  ASSERT_TRUE(error.empty()) << error;

  const std::vector<std::string> PROBLEMS = assessRanges(RANGES, OPS, 1, 1);

  EXPECT_TRUE(PROBLEMS.empty()) << PROBLEMS.front();
}

INSTANTIATE_TEST_SUITE_P(BothNsysVersions, NsysReportTest,
                         ::testing::Values(NSYS_2026_3_1, NSYS_2025_3_2),
                         [](const ::testing::TestParamInfo<NsysReport>& info) {
                           return std::string(info.param.version);
                         });

/** @test The trace the demo lays out passes, with the counts the check's flags give */
TEST(NvtxTraceTest, AcceptsTheDemosLayout) {
  const Trace T = demoTrace(MEASURED_CALLS, WARMUP_CALLS);

  const std::vector<std::string> PROBLEMS =
      assessRanges(T.ranges, T.ops, MEASURED_CALLS, WARMUP_CALLS);

  EXPECT_TRUE(PROBLEMS.empty()) << PROBLEMS.front();
}

/** @test A call without its kernel range is reported */
TEST(NvtxTraceTest, ReportsAMissingPhase) {
  Trace t = demoTrace(MEASURED_CALLS, WARMUP_CALLS);
  const NvtxRange GONE = nthRange(t, phase::KERNEL, WARMUP_CALLS + 2);
  t.ranges.erase(std::remove_if(t.ranges.begin(), t.ranges.end(),
                                [&](const NvtxRange& r) { return r.id == GONE.id; }),
                 t.ranges.end());

  const std::vector<std::string> PROBLEMS =
      assessRanges(t.ranges, t.ops, MEASURED_CALLS, WARMUP_CALLS);

  EXPECT_TRUE(mentions(PROBLEMS, "holds 23 ranges")) << PROBLEMS.size();
}

/** @test A call whose copy_in and copy_out are swapped is reported */
TEST(NvtxTraceTest, ReportsSwappedPhases) {
  Trace t = demoTrace(MEASURED_CALLS, WARMUP_CALLS);
  NvtxRange& in = nthRange(t, phase::COPY_IN, WARMUP_CALLS + 1);
  NvtxRange& out = nthRange(t, phase::COPY_OUT, WARMUP_CALLS + 1);
  in.name = phase::COPY_OUT;
  out.name = phase::COPY_IN;

  const std::vector<std::string> PROBLEMS =
      assessRanges(t.ranges, t.ops, MEASURED_CALLS, WARMUP_CALLS);

  EXPECT_TRUE(mentions(PROBLEMS, "expected copy_in here")) << PROBLEMS.size();
}

/** @test Phases outside every test's range beyond the warmup's are reported */
TEST(NvtxTraceTest, ReportsPhasesOutsideTheTestRange) {
  Trace t = demoTrace(MEASURED_CALLS, WARMUP_CALLS);
  NvtxRange extra = nthRange(t, phase::KERNEL, 0);
  extra.start = t.ranges.back().end + 1000;
  extra.end = extra.start + 100;
  extra.id = 9999;
  t.ranges.push_back(extra);

  const std::vector<std::string> PROBLEMS =
      assessRanges(t.ranges, t.ops, MEASURED_CALLS, WARMUP_CALLS);

  EXPECT_TRUE(mentions(PROBLEMS, "outside every test's range")) << PROBLEMS.size();
}

/** @test A kernel that ends after its range closed is reported */
TEST(NvtxTraceTest, ReportsAKernelOutsideItsRange) {
  Trace t = demoTrace(MEASURED_CALLS, WARMUP_CALLS);
  const NvtxRange RANGE = nthRange(t, phase::KERNEL, WARMUP_CALLS + 3);
  for (GpuOp& op : t.ops) {
    if (op.kind == GpuOp::Kind::KERNEL && op.start >= RANGE.start && op.end <= RANGE.end) {
      op.end = RANGE.end + 20;
    }
  }

  const std::vector<std::string> PROBLEMS =
      assessRanges(t.ranges, t.ops, MEASURED_CALLS, WARMUP_CALLS);

  EXPECT_TRUE(mentions(PROBLEMS, "call 3, range kernel holds 0 kernel(s)")) << PROBLEMS.size();
}

/** @test A range inside the unannotated test's range is reported */
TEST(NvtxTraceTest, ReportsARangeInsideTheUnannotatedTest) {
  Trace t = demoTrace(MEASURED_CALLS, WARMUP_CALLS);
  NvtxRange extra = nthRange(t, UNANNOTATED, 0);
  extra.parent = extra.id;
  extra.id = 9999;
  extra.level = 1;
  extra.name = phase::KERNEL;
  t.ranges.push_back(extra);

  const std::vector<std::string> PROBLEMS =
      assessRanges(t.ranges, t.ops, MEASURED_CALLS, WARMUP_CALLS);

  EXPECT_TRUE(mentions(PROBLEMS, "where it should hold none")) << PROBLEMS.size();
}

/** @test A test range still open when the next test's opens is reported */
TEST(NvtxTraceTest, ReportsATestRangeLeftOpen) {
  Trace t = demoTrace(MEASURED_CALLS, WARMUP_CALLS);
  NvtxRange& g1 = nthRange(t, UNANNOTATED, 0);
  NvtxRange& phased = nthRange(t, ANNOTATED, 0);
  g1.end = phased.end + 1000;
  phased.level = 1;
  phased.parent = g1.id;

  const std::vector<std::string> PROBLEMS =
      assessRanges(t.ranges, t.ops, MEASURED_CALLS, WARMUP_CALLS);

  EXPECT_TRUE(mentions(PROBLEMS, "the range before it was still open")) << PROBLEMS.size();
}
