#ifndef VERNIER_DEMO_05_NVTXANNOTATION_CHECK_HPP
#define VERNIER_DEMO_05_NVTXANNOTATION_CHECK_HPP
/**
 * @file 05_NvtxAnnotation_Check.hpp
 * @brief The reading and the judgement behind demo 05's check: nsys's push/pop
 *        and GPU traces read by column name, what the walkthrough says they
 *        must show, and what the check does with a capture.
 *
 * NvtxRanges.RecordedByNsightSystems runs demo 05 under nsys through
 * captureUnderNsys(), which judgeCapture() decides on, and holds the report it
 * reads with readRanges() and readGpuOps() to assessRanges(). The process
 * plumbing is walkthrough 15's (12_MemcheckProfiler_Check.hpp); the range
 * names are the demo's own (05_NvtxAnnotation_Phases.hpp).
 *
 * Test support for the check (05_NvtxAnnotation_Ranges_uTest.cpp, built where
 * the GPU demos are) and for its tests, which need neither nsys nor a device
 * (05_NvtxAnnotation_RangesReport_uTest.cpp, built in every configuration);
 * not part of the demo.
 */

#include "src/bench/demo/cpu/utst/12_MemcheckProfiler_Check.hpp"
#include "src/bench/demo/gpu/05_NvtxAnnotation_Phases.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <exception>
#include <filesystem>
#include <iterator>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace vernier {
namespace bench {
namespace demo {
namespace nvtx_check {

namespace fs = std::filesystem;

/* ----------------------------- Constants ----------------------------- */

/// The two tests, as their per-test ranges are named.
inline constexpr const char* UNANNOTATED = "NvtxAnnotation.G1";
inline constexpr const char* ANNOTATED = "NvtxAnnotation.G1Phases";

/// The run: a few calls per test, all traced, after one warmup call.
inline constexpr int CYCLES = 4;
inline constexpr int REPEATS = 2;
inline constexpr int WARMUP_CALLS = 1;
inline constexpr int MEASURED_CALLS = CYCLES * REPEATS;

/// How the GPU trace names the work: the example's kernel, the copy directions.
inline constexpr const char* KERNEL_NAME = "saxpyKernel";
inline constexpr const char* TO_DEVICE = "Host-to-Device";
inline constexpr const char* TO_HOST = "Device-to-Host";

/// At most this many problems are listed; the first ones name the cause.
inline constexpr std::size_t MAX_PROBLEMS = 12;

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
    } else if (CH != '\r') {
      field += CH;
    }
  }
  fields.push_back(field);
  return fields;
}

/// A whole number as nsys prints it; false when the text is not one.
inline bool parseInteger(const std::string& text, long long& value) {
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
inline std::vector<std::vector<std::string>>
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
inline std::string withoutDomain(const std::string& name) {
  const std::size_t COLON = name.find(':');
  return COLON == std::string::npos ? name : name.substr(COLON + 1);
}

/// The ranges of an nvtx_pushpop_trace CSV; @p error says why it cannot be read.
inline std::vector<NvtxRange> readRanges(const std::string& csv, std::string& error) {
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
inline std::vector<GpuOp> readGpuOps(const std::string& csv, std::string& error) {
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

inline Contents contentsOf(const std::vector<GpuOp>& ops, long long start, long long end) {
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

inline std::string describe(const Contents& c) {
  return std::to_string(c.kernels) + " kernel(s), " + std::to_string(c.toDevice) +
         " copy(ies) to the device, " + std::to_string(c.toHost) + " copy(ies) back, " +
         std::to_string(c.other) + " other";
}

inline bool isPhase(const std::string& name) {
  return std::find_if(std::begin(nvtx_annotation::PHASES), std::end(nvtx_annotation::PHASES),
                      [&](const char* p) { return name == p; }) !=
         std::end(nvtx_annotation::PHASES);
}

/// What phase @p name holds, per call: two copies in, one kernel, one copy back.
inline bool holdsItsWork(const std::string& name, const Contents& c) {
  if (name == nvtx_annotation::COPY_IN) {
    return c.toDevice == 2 && c.kernels == 0 && c.toHost == 0 && c.other == 0;
  }
  if (name == nvtx_annotation::KERNEL) {
    return c.kernels == 1 && c.toDevice == 0 && c.toHost == 0 && c.other == 0;
  }
  return c.toHost == 1 && c.kernels == 0 && c.toDevice == 0 && c.other == 0;
}

/**
 * @brief What the trace must show, as the problems it does not: empty when it
 *        shows everything the walkthrough says.
 */
inline std::vector<std::string> assessRanges(std::vector<NvtxRange> ranges,
                                             const std::vector<GpuOp>& ops, int measuredCalls,
                                             int warmupCalls) {
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
    if (range.name != nvtx_annotation::PHASES[i % 3]) {
      problems.push_back(WHERE + ": expected " + std::string(nvtx_annotation::PHASES[i % 3]) +
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
      if (outside[i]->name != nvtx_annotation::PHASES[i % 3] || outside[i]->end > phased.start) {
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
inline std::pair<long long, long long> kernelMargins(const std::vector<NvtxRange>& ranges,
                                                     const std::vector<GpuOp>& ops) {
  long long before = -1;
  long long after = -1;
  for (const NvtxRange& range : ranges) {
    if (range.name != nvtx_annotation::KERNEL) {
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

/// A run's temporary directory, made under the system's: removed when the
/// test passes or skips, kept, and named, when it fails.
class RunDirectory {
public:
  RunDirectory() {
    std::string pattern = (fs::temp_directory_path() / "vernier-demo-gpu05-XXXXXX").string();
    if (::mkdtemp(pattern.data()) != nullptr) {
      dir_ = pattern;
    }
  }
  ~RunDirectory() {
    if (dir_.empty()) {
      return;
    }
    if (::testing::Test::HasFailure()) {
      std::printf("the run's files are kept in %s\n", dir_.c_str());
      return;
    }
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }
  RunDirectory(const RunDirectory&) = delete;
  RunDirectory& operator=(const RunDirectory&) = delete;

  /// False when the directory could not be made.
  [[nodiscard]] bool made() const { return !dir_.empty(); }
  [[nodiscard]] const fs::path& path() const { return dir_; }
  [[nodiscard]] fs::path operator/(const char* name) const { return dir_ / name; }

private:
  fs::path dir_;
};

/// The demo's arguments, in the order the check and the walkthrough give them.
/// Started by hand under nsys, the backend makes an empty folder per test
/// (<Suite.Case>.nsight); @p folders keeps them in the check's own directory.
inline std::vector<std::string> demoArgs(const std::string& demo, const fs::path& folders) {
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

/* ----------------------------- The Capture ----------------------------- */

/// The first line of the note nsys prints under a directory it could not
/// create; the note goes on to name TMPDIR.
inline constexpr const char* TMPDIR_NOTE =
    "NOTE: If you are using a system that does not allow writing to \"/tmp\" or";

/**
 * @brief nsys's own lines, as it printed them, when it could not create its
 *        temporary files and stopped without starting the program; empty for
 *        any other output.
 *
 * Two forms: "Failed to create directory "<dir>": <reason>" followed, after a
 * blank line, by the note nsys prints under it, which points to TMPDIR, and
 * "Failed to create temporary output file". nsys 2026.3.1 printed the first,
 * and exited 1 without starting the program, when the directory it keeps in
 * the temporary directory (nvidia/) was another user's and not writable, or
 * was a file; and the second when its own directory there (nsys-<user>/) was
 * another user's.
 * Not recognised: its warning that the temporary directory is short of space,
 * after which it goes on, and an exception it printed that names no cause. A
 * skip quotes these lines, so what it rests on is nsys's text, not the check's.
 */
inline std::string temporaryFilesRefused(const std::string& output) {
  std::vector<std::string> lines;
  std::istringstream in(output);
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    lines.push_back(line);
  }
  for (std::size_t i = 0; i < lines.size(); ++i) {
    if (lines[i] == "Failed to create temporary output file") {
      return lines[i];
    }
    if (lines[i].rfind("Failed to create directory \"", 0) != 0) {
      continue;
    }
    std::size_t note = i + 1;
    while (note < lines.size() && lines[note].empty()) {
      ++note;
    }
    if (note == lines.size() || lines[note].rfind(TMPDIR_NOTE, 0) != 0) {
      continue;
    }
    std::string quoted = lines[i];
    for (std::size_t j = i + 1; j < lines.size() && j <= note + 2; ++j) {
      quoted += "\n" + lines[j];
      if (lines[j].find("set a different location.") != std::string::npos) {
        break;
      }
    }
    return quoted;
  }
  return "";
}

/// What the check does with its capture: read the report, skip, or fail.
struct CaptureVerdict {
  enum class Action { READ, SKIP, FAIL };
  Action action = Action::FAIL;
  std::string message; ///< A skip's quotation of nsys, or a failure's account of the run
};

/// How a failure names an action.
inline void PrintTo(CaptureVerdict::Action action, std::ostream* out) {
  switch (action) {
  case CaptureVerdict::Action::READ:
    *out << "READ";
    break;
  case CaptureVerdict::Action::SKIP:
    *out << "SKIP";
    break;
  case CaptureVerdict::Action::FAIL:
    *out << "FAIL";
    break;
  }
}

/**
 * @brief The check's verdict on a capture, from how nsys ended (@p end), what
 *        nsys and the demo printed (@p output) and whether the report exists.
 *
 * The report is read only when nsys exited 0, the demo's two tests passed
 * under it and the report exists. Short of that, the check skips only when
 * the demo's tests never started and nsys, exiting with a non-zero status,
 * gave temporaryFilesRefused()'s reason, which the skip quotes. Anything else
 * fails: an exit or a signal with no such reason, no output, tests that did
 * not pass, no report. The failure says how nsys ended, what is missing and
 * where the run's files are, @p dir, which the check keeps when it fails.
 */
inline CaptureVerdict judgeCapture(const memcheck_check::ChildExit& end, const std::string& output,
                                   bool reportWritten, const fs::path& dir) {
  const bool STARTED = memcheck_check::testsStarted(output);
  const bool PASSED = output.find("[  PASSED  ] 2 tests.") != std::string::npos;
  if (memcheck_check::exitedWith(end, 0) && STARTED && PASSED && reportWritten) {
    return {CaptureVerdict::Action::READ, ""};
  }
  if (!STARTED && end.how == memcheck_check::ChildExit::How::Exited && end.code != 0) {
    const std::string REFUSED = temporaryFilesRefused(output);
    if (!REFUSED.empty()) {
      return {CaptureVerdict::Action::SKIP,
              "nsys could not create its temporary files and stopped before the demo's tests "
              "started (nsys " +
                  memcheck_check::describe(end) + "). It printed:\n" + REFUSED};
    }
  }
  std::string message =
      "the capture under nsys did not complete: nsys " + memcheck_check::describe(end);
  if (!STARTED) {
    message += "; the demo's tests never started";
  } else if (!PASSED) {
    message += "; the demo's two tests did not both pass";
  }
  if (!reportWritten) {
    message += "; nsys wrote no report";
  }
  message += ". The run's files are kept in " + dir.string() + ". ";
  message += output.empty() ? std::string("The run printed nothing.\n")
                            : "The run printed:\n" + memcheck_check::lastLines(output, 40);
  return {CaptureVerdict::Action::FAIL, message};
}

/**
 * @brief Captures the demo's two tests under @p nsys, the tool and any
 *        arguments before nsys's own ("nsys" for the real one), and returns
 *        judgeCapture()'s verdict on the run.
 *
 * CUDA and NVTX are traced without CPU sampling; the report is
 * <dir>/capture.nsys-rep, what nsys and the demo print goes to
 * <dir>/program.txt, and the demo gets demoArgs()'s arguments.
 */
inline CaptureVerdict captureUnderNsys(const std::vector<std::string>& nsys,
                                       const std::string& demo, const fs::path& dir) {
  std::vector<std::string> args = nsys;
  const std::vector<std::string> PROFILE = {"profile",
                                            "-o",
                                            (dir / "capture").string(),
                                            "-t",
                                            "cuda,nvtx",
                                            "--sample=none",
                                            "--cpuctxsw=none",
                                            "--force-overwrite",
                                            "true"};
  args.insert(args.end(), PROFILE.begin(), PROFILE.end());
  const std::vector<std::string> DEMO_ARGS = demoArgs(demo, dir / "folders");
  args.insert(args.end(), DEMO_ARGS.begin(), DEMO_ARGS.end());
  const memcheck_check::ChildExit END = memcheck_check::runLogged(args, dir / "program.txt");
  return judgeCapture(END, memcheck_check::readText(dir / "program.txt"),
                      fs::exists(dir / "capture.nsys-rep"), dir);
}

} // namespace nvtx_check
} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_05_NVTXANNOTATION_CHECK_HPP
