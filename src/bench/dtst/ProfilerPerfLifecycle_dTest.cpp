/**
 * @file ProfilerPerfLifecycle_dTest.cpp
 * @brief Component tests for the perf backend's run: its start, its stop and
 * the output it leaves.
 *
 * A fake perf (fixtures/readiness/fake_perf.sh) stands in for the tool, and
 * FAKE_PERF_MODE selects how the run's perf behaves. Each test builds the
 * profiler from a plan that names the fake, runs one measured window and
 * reads what the run recorded as failed and what perf left in the folder; the
 * check's tests ask checkPerfRequest about the same fake.
 */

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfRegistry.hpp"
#include "src/bench/inc/Profiler.hpp"
#include "src/bench/inc/ProfilerPerf.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/utst/ReadinessFixtures.hpp"
#include "src/bench/utst/StderrCapture.hpp"

#include <gtest/gtest.h>

#include <csignal>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using vernier::bench::checkPerfRequest;
using vernier::bench::CHILD_STATUS_REMEDY;
using vernier::bench::parsePerfMode;
using vernier::bench::PerfConfig;
using vernier::bench::PerfPlan;
using vernier::bench::PerfStatProfiler;
using vernier::bench::ProfileFailure;
using vernier::bench::ProfilerRegistry;
using vernier::bench::ReadinessCause;
using vernier::bench::ReadinessRequest;
using vernier::bench::ReadinessResult;
using vernier::bench::ReadinessScope;
using vernier::bench::ReadinessStage;
using vernier::bench::test::ChildReaping;
using vernier::bench::test::FakeToolDir;
using vernier::bench::test::ScopedEnv;
using vernier::bench::test::StderrCapture;

/** @brief True once process @p pid has ended: a zombie or gone from /proc. */
bool ended(pid_t pid) {
  std::ifstream stat("/proc/" + std::to_string(pid) + "/stat");
  std::string line;
  if (!std::getline(stat, line)) {
    return true;
  }
  const std::size_t CLOSE = line.rfind(')');
  return CLOSE != std::string::npos && line.compare(CLOSE + 1, 3, " Z ") == 0;
}

std::string readText(const std::string& path) {
  std::ifstream in(path);
  std::stringstream text;
  text << in.rdbuf();
  return text.str();
}

/** @brief What a measured window throws in the abort test, with a mark to recognise it by. */
struct WindowFailed {
  int mark = 0;
};

// perf.data record types: the kernel's (linux/perf_event.h) and perf's own.
constexpr std::uint32_t LOST = 2;
constexpr std::uint32_t COMM = 3;
constexpr std::uint32_t SAMPLE = 9;
constexpr std::uint32_t MMAP2 = 10;
constexpr std::uint32_t AUX = 11;
constexpr std::uint32_t LOST_SAMPLES = 13;
constexpr std::uint32_t FINISHED_ROUND = 68;
constexpr std::uint32_t AUXTRACE = 71;
constexpr std::uint32_t COMPRESSED = 81;

/** @brief The size of each record recordBytes() writes: its 8-byte header and 8 bytes. */
constexpr std::uint64_t PERF_RECORD_BYTES = 16;

/** @brief Append @p value to @p bytes as @p width little-endian bytes. */
void putLittleEndian(std::string& bytes, std::uint64_t value, int width) {
  for (int i = 0; i < width; ++i) {
    bytes.push_back(static_cast<char>((value >> (8 * i)) & 0xFFU));
  }
}

/** @brief One record of each of @p types, as a data section or a stream holds them. */
std::string recordBytes(const std::vector<std::uint32_t>& types) {
  std::string bytes;
  for (const std::uint32_t TYPE : types) {
    putLittleEndian(bytes, TYPE, 4);
    putLittleEndian(bytes, 0, 2); // misc
    putLittleEndian(bytes, PERF_RECORD_BYTES, 2);
    putLittleEndian(bytes, 0, 8); // the record's body
  }
  return bytes;
}

/**
 * @brief The bytes of a perf data file as perf writes one on a little-endian
 * machine, starting with @p magic: the 104-byte header (no attributes), then a
 * data section of recordBytes() of @p types, whose size the header gives as
 * @p dataSize (by default the records' own). With @p dirFormat, the header
 * file of the directory perf record --threads writes: the DIR_FORMAT feature
 * (bit 24) set, and after the data section the feature sections' table (one
 * offset and size) and the section, which holds that version.
 */
std::string perfDataBytes(const std::vector<std::uint32_t>& types, const char* magic,
                          std::uint64_t dataSize, std::uint64_t dirFormat = 0) {
  constexpr std::uint64_t HEADER_BYTES = 104;
  constexpr std::size_t DIR_FORMAT_BYTE = 72 + 3; // feature bit 24
  std::string bytes;
  bytes.append(magic, 8);
  putLittleEndian(bytes, HEADER_BYTES, 8); // the header's size
  putLittleEndian(bytes, 0, 8);            // the size of an attribute
  putLittleEndian(bytes, HEADER_BYTES, 8); // the attributes: offset, size
  putLittleEndian(bytes, 0, 8);
  putLittleEndian(bytes, HEADER_BYTES, 8); // the data section: offset, size
  putLittleEndian(bytes, dataSize, 8);
  bytes.resize(HEADER_BYTES, '\0'); // no event types; feature bits below
  if (dirFormat != 0) {
    bytes[DIR_FORMAT_BYTE] = 0x01;
  }
  bytes += recordBytes(types);
  if (dirFormat != 0) {
    putLittleEndian(bytes, bytes.size() + 16, 8); // the section's offset and size
    putLittleEndian(bytes, 8, 8);
    putLittleEndian(bytes, dirFormat, 8);
  }
  return bytes;
}

/** @brief A field of a byte image: @p value written little-endian over 8 bytes at @p at. */
struct Field {
  std::size_t at;
  std::uint64_t value;
};

/**
 * @brief @p size bytes as given, independent of the reader under test:
 * @p magic at byte 0 (when the size holds it), each of @p fields written
 * little-endian at its offset, every other byte zero.
 */
std::string bytesWith(std::size_t size, const std::vector<Field>& fields,
                      const char* magic = "PERFILE2") {
  std::string bytes(size, '\0');
  bytes.replace(0, std::min<std::size_t>(8, size), magic, std::min<std::size_t>(8, size));
  for (const Field& F : fields) {
    for (std::size_t i = 0; i < 8 && F.at + i < size; ++i) {
      bytes[F.at + i] = static_cast<char>((F.value >> (8 * i)) & 0xFFU);
    }
  }
  return bytes;
}

/** @brief Feature bit 24 (DIR_FORMAT) in the first word of the feature bits, byte 72. */
constexpr std::uint64_t DIR_FORMAT_BIT = std::uint64_t{1} << 24;

/** @brief One measured window under a fake perf in a private folder. */
class PerfLifecycleTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(dir_.ok());
    perf_ = dir_.install("fake_perf.sh", "perf");
    ProfilerRegistry::instance().resetFailures();
  }

  void TearDown() override { ProfilerRegistry::instance().resetFailures(); }

  /**
   * @brief What a window left: the failures recorded, the folder, how long the
   * start took, and what the backend printed.
   */
  struct Window {
    std::vector<ProfileFailure> failures;
    std::string folder;
    std::chrono::steady_clock::duration startTook{};
    std::string err;
  };

  /**
   * @brief Run one window with the fake in @p mode and @p args; when
   * @p afterExit, the stop waits for the fake to report its own exit first.
   * @p answersPing is what the check found: the launch then waits for perf's
   * answer on its control fifo, else the fixed grace.
   */
  Window window(const std::string& mode, const std::string& args = "", bool afterExit = false,
                bool answersPing = true) {
    auto plan = std::make_shared<PerfPlan>();
    plan->perf = perf_;
    plan->mode = parsePerfMode(args);
    plan->answersPing = answersPing;
    PerfConfig cfg;
    cfg.profileTool = "perf";
    cfg.profileArgs = args;
    cfg.artifactRoot = dir_.path();
    ScopedEnv fakeMode("FAKE_PERF_MODE", mode);
    ScopedEnv log("FAKE_LOG", dir_.logPath());
    StderrCapture quiet;
    Window out;
    {
      PerfStatProfiler profiler(cfg, "Perf.Life", plan);
      out.folder = profiler.artifactDir();
      const auto START = std::chrono::steady_clock::now();
      profiler.beforeMeasure();
      out.startTook = std::chrono::steady_clock::now() - START;
      if (afterExit) {
        // The fake logs its exit just before it exits: wait for the line,
        // then for the process to have ended (a zombie until the stop reaps
        // it), so that no timing is assumed.
        for (int i = 0; i < 500 && dir_.logLines("perf exited").empty(); ++i) {
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        const std::vector<std::string> EXITED = dir_.logLines("perf exited");
        EXPECT_FALSE(EXITED.empty()) << dir_.log();
        if (!EXITED.empty()) {
          const pid_t FAKE = std::stoi(EXITED.front().substr(EXITED.front().find('=') + 1));
          for (int i = 0; i < 500 && !ended(FAKE); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
          }
          EXPECT_TRUE(ended(FAKE)) << "perf " << FAKE << " logged its exit but still runs";
        }
      }
      profiler.afterMeasure(vernier::bench::Stats{});
    }
    out.failures = ProfilerRegistry::instance().failures();
    out.err = quiet.text();
    return out;
  }

  /**
   * @brief Write perfDataBytes() for @p types to @p name in the private folder
   * and return its path, for the fake to write as its data file
   * (FAKE_PERF_DATA); @p dataSize 0 gives the records' own size.
   */
  std::string writePerfData(const std::string& name, const std::vector<std::uint32_t>& types,
                            const char* magic = "PERFILE2", std::uint64_t dataSize = 0) const {
    const std::string PATH = dir_.path() + "/" + name;
    std::ofstream(PATH, std::ios::binary)
        << perfDataBytes(types, magic, dataSize != 0 ? dataSize : types.size() * PERF_RECORD_BYTES);
    return PATH;
  }

  /**
   * @brief Write to @p name in the private folder a directory as perf record
   * --threads writes its capture, and return its path: the header file "data"
   * (perfDataBytes() of @p header, recording directory format @p dirFormat)
   * and a stream "data.<n>" of recordBytes() for each of @p streams.
   */
  std::string writePerfDirectory(const std::string& name, const std::vector<std::uint32_t>& header,
                                 const std::vector<std::vector<std::uint32_t>>& streams,
                                 std::uint64_t dirFormat = 1) const {
    const std::string PATH = dir_.path() + "/" + name;
    std::filesystem::create_directories(PATH);
    std::ofstream(PATH + "/data", std::ios::binary)
        << perfDataBytes(header, "PERFILE2", header.size() * PERF_RECORD_BYTES, dirFormat);
    for (std::size_t i = 0; i < streams.size(); ++i) {
      std::ofstream(PATH + "/data." + std::to_string(i), std::ios::binary)
          << recordBytes(streams[i]);
    }
    return PATH;
  }

  /** @brief Write @p bytes to @p name in the private folder and return its path. */
  std::string writeBytes(const std::string& name, const std::string& bytes) const {
    const std::string PATH = dir_.path() + "/" + name;
    std::ofstream(PATH, std::ios::binary) << bytes;
    return PATH;
  }

  /**
   * @brief Write to @p name in the private folder a directory holding the
   * header file "data" of @p header and a stream "data.<n>" of each of
   * @p streams, all as given, and return its path.
   */
  std::string writeRawDirectory(const std::string& name, const std::string& header,
                                const std::vector<std::string>& streams) const {
    const std::string PATH = dir_.path() + "/" + name;
    std::filesystem::create_directories(PATH);
    std::ofstream(PATH + "/data", std::ios::binary) << header;
    for (std::size_t i = 0; i < streams.size(); ++i) {
      std::ofstream(PATH + "/data." + std::to_string(i), std::ios::binary) << streams[i];
    }
    return PATH;
  }

  /** @brief The check's decision about the fake in @p mode, for @p args. */
  ReadinessResult check(const std::string& mode, const std::string& args = "") const {
    ReadinessRequest request;
    request.backend = "perf";
    request.profileArgs = args;
    request.scope = ReadinessScope::PREFLIGHT;
    return checkPerfRequest(request, dir_.context({{"FAKE_PERF_MODE", mode}}, 0));
  }

  /** @brief True when no process of a "pid=" in the fake's log is still alive. */
  bool fakesGone() const {
    for (const std::string& line : dir_.logLines("perf ")) {
      const std::size_t AT = line.rfind(" pid=");
      if (AT != std::string::npos &&
          ::kill(static_cast<pid_t>(std::atol(line.c_str() + AT + 5)), 0) == 0) {
        return false;
      }
    }
    return true;
  }

  FakeToolDir dir_;
  std::string perf_;
};

} // namespace

/** @test A perf stopped by SIGINT leaves its counts, and nothing fails. */
TEST_F(PerfLifecycleTest, CleanStopKeepsCounts) {
  const Window W = window("ok");
  EXPECT_TRUE(W.failures.empty()) << W.failures.front().result.report.message;
  EXPECT_NE(readText(W.folder + "/stat.txt").find("Performance counter stats"), std::string::npos);
}

/** @test The stop returns once perf has written, not after a fixed wait. */
TEST_F(PerfLifecycleTest, StopWaitsForPerfToWrite) {
  const Window W = window("slow-stop");
  EXPECT_TRUE(W.failures.empty()) << W.failures.front().result.report.message;
  EXPECT_NE(readText(W.folder + "/stat.txt").find("Performance counter stats"), std::string::npos)
      << "the hook returned before perf wrote its counts";
}

/** @test A perf that ends before the measured phase fails the capture with its words. */
TEST_F(PerfLifecycleTest, EarlyExitFails) {
  const Window W = window("exit-early");
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_EQ(W.failures[0].backend, "perf");
  EXPECT_EQ(W.failures[0].test, "Perf.Life");
  EXPECT_EQ(W.failures[0].result.stage, ReadinessStage::COLLECTION);
  EXPECT_EQ(W.failures[0].result.report.message,
            "unusable: perf ended (exit status 1) before the measured phase: perf: Error: failed "
            "to open counters: No such process");
}

/** @test A perf that ends during the measured phase fails its completion. */
TEST_F(PerfLifecycleTest, EndDuringTheWindowFails) {
  const Window W = window("exit-soon", "", /*afterExit=*/true);
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_EQ(W.failures[0].result.stage, ReadinessStage::COMPLETION);
  EXPECT_EQ(W.failures[0].result.report.message,
            "completion: unusable: perf ended (exit status 3) during the measured phase, so its "
            "output covers part of it at most: perf: Error: the target process exited");
}

/** @test A perf that ignores SIGINT is stopped by SIGTERM, and its output does not count. */
TEST_F(PerfLifecycleTest, SigtermStopFails) {
  const Window W = window("ignore-int");
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_EQ(W.failures[0].result.stage, ReadinessStage::COMPLETION);
  EXPECT_EQ(W.failures[0].result.report.message,
            "completion: unusable: perf did not finish writing within 5 s of SIGINT and was "
            "stopped by SIGTERM, so its output may be incomplete");
}

/** @test A perf that ignores SIGINT and SIGTERM is killed, and its output does not count. */
TEST_F(PerfLifecycleTest, HangIsKilledAndFails) {
  const Window W = window("hang");
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_NE(W.failures[0].result.report.message.find("was stopped by SIGKILL"), std::string::npos)
      << W.failures[0].result.report.message;
}

/** @test stat.txt holding an error message instead of counts fails (a zero count would not). */
TEST_F(PerfLifecycleTest, ErrorTextIsNotACount) {
  const Window W = window("error-text");
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_EQ(W.failures[0].result.stage, ReadinessStage::COMPLETION);
  EXPECT_EQ(
      W.failures[0].result.report.message,
      "completion: unusable: " + W.folder +
          "/stat.txt holds no counts: perf: Error: the fake perf could not read its counters");
}

/**
 * @test A perf that ends after SIGINT otherwise than a perf that has written
 * its report ends fails the capture, whatever it wrote: here a count, then
 * "Error: final read failed", and exit status 42.
 */
TEST_F(PerfLifecycleTest, OtherEndAfterSigintFails) {
  const Window W = window("error-exit");
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_EQ(W.failures[0].result.stage, ReadinessStage::COMPLETION);
  EXPECT_EQ(W.failures[0].result.report.message,
            "completion: unusable: perf ended (exit status 42) after SIGINT, not as a perf that "
            "has written its report ends (by SIGINT, or exit status 130 or 0), so its output does "
            "not count: perf: 0.101234567 seconds time elapsed | Error: final read failed");
}

/**
 * @test A perf whose end this process could not observe is never taken for
 * one that ended cleanly. With SIGCHLD ignored, or with SA_NOCLDWAIT, the
 * system discards perf's status: a perf stopped by SIGINT fails its
 * completion, and one that ends before the measured phase fails its
 * collection, each saying that its end could not be observed and why, with
 * how to let it be read. With SIGCHLD at its default action the same perfs
 * keep their counts, and fail with their exit status (CleanStopKeepsCounts,
 * EarlyExitFails).
 */
TEST_F(PerfLifecycleTest, EndNotObservedIsNoCleanEnd) {
  using Policy = ChildReaping::Policy;
  const std::vector<std::pair<Policy, std::string>> POLICIES{
      {Policy::IGNORED,
       "this process ignores SIGCHLD, so the system discards its children's exit statuses"},
      {Policy::NO_CHILD_WAIT, "this process sets SA_NOCLDWAIT for SIGCHLD, so the system "
                              "discards its children's exit statuses"}};
  for (const auto& [POLICY, WHY] : POLICIES) {
    Window stopped;
    Window early;
    {
      const ChildReaping REAPING(POLICY);
      ASSERT_TRUE(REAPING.ok());
      stopped = window("ok");
      ProfilerRegistry::instance().resetFailures();
      early = window("exit-early");
    }
    ASSERT_EQ(stopped.failures.size(), 1U) << WHY;
    EXPECT_EQ(stopped.failures[0].result.stage, ReadinessStage::COMPLETION) << WHY;
    const std::string STOPPED = stopped.failures[0].result.report.message;
    EXPECT_EQ(STOPPED.rfind("completion: unusable: perf ended (an end this process could not "
                            "observe: " +
                                WHY +
                                ") after SIGINT, so nothing shows that it wrote its report, and "
                                "its output does not count: perf: ",
                            0),
              0U)
        << STOPPED;
    EXPECT_EQ(stopped.failures[0].result.report.hint, CHILD_STATUS_REMEDY) << WHY;
    ASSERT_EQ(early.failures.size(), 1U) << WHY;
    EXPECT_EQ(early.failures[0].result.stage, ReadinessStage::COLLECTION) << WHY;
    EXPECT_EQ(early.failures[0].result.report.message,
              "unusable: perf ended (an end this process could not observe: " + WHY +
                  ") before the measured phase: perf: Error: failed to open counters: No such "
                  "process");
    EXPECT_EQ(early.failures[0].result.report.hint, CHILD_STATUS_REMEDY) << WHY;
    ProfilerRegistry::instance().resetFailures();
  }
}

/** @test The control: perf ending by SIGINT itself, as it does after its report, passes. */
TEST_F(PerfLifecycleTest, EndBySigintKeepsCounts) {
  const Window W = window("raise-int");
  EXPECT_TRUE(W.failures.empty()) << W.failures.front().result.report.message;
  EXPECT_NE(readText(W.folder + "/stat.txt").find("66,055"), std::string::npos);
}

/** @test stat's heading alone, with perf's usual exit status 130, is no count. */
TEST_F(PerfLifecycleTest, HeadingAloneIsNoCount) {
  const Window W = window("heading-only");
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_EQ(W.failures[0].result.stage, ReadinessStage::COMPLETION);
  const std::string MESSAGE = W.failures[0].result.report.message;
  EXPECT_EQ(MESSAGE.rfind("completion: unusable: " + W.folder +
                              "/stat.txt holds no counts: perf: Performance counter stats for "
                              "process id '",
                          0),
            0U)
      << MESSAGE;
}

/**
 * @test Events perf marks <not supported> or <not counted> are named when
 * none is counted, apart from missing data; one counted event among them
 * passes, as on a CPU without one of the counters.
 */
TEST_F(PerfLifecycleTest, UnavailableEventsAreNamed) {
  const Window NONE = window("none-counted");
  ASSERT_EQ(NONE.failures.size(), 1U);
  EXPECT_EQ(NONE.failures[0].result.report.message,
            "completion: unsupported: " + NONE.folder +
                "/stat.txt holds no count: perf counted none of its events here (cpu-cycles:u "
                "<not supported>, task-clock <not counted>)");
  ProfilerRegistry::instance().resetFailures();
  const Window SOME = window("some-unsupported");
  EXPECT_TRUE(SOME.failures.empty()) << SOME.failures.front().result.report.message;
}

/** @test An error perf prints after its counts fails them, even with exit status 130. */
TEST_F(PerfLifecycleTest, ErrorAfterTheCountsFails) {
  const Window W = window("error-after");
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_EQ(W.failures[0].result.report.message,
            "completion: unusable: " + W.folder +
                "/stat.txt holds an error from perf after its counts, which therefore do not "
                "count: perf: 0.101234567 seconds time elapsed | Error: final read failed");
}

/**
 * @test A measured window that ends by an exception, run through the harness
 * as a benchmark runs it, in stat mode and in record mode: the exception
 * reaches the caller as it was thrown, nothing is published, and by then
 * perf has ended, judged by no one: no failure is recorded, and what it wrote
 * is set aside (stat.aborted.txt, perf.aborted.data), with a line that says
 * where. The case then measures again, and that window's output is perf's
 * own, its row published and nothing failed.
 */
TEST_F(PerfLifecycleTest, AbortStopsPerf) {
  const char* INHERITED = std::getenv("PATH");
  const ScopedEnv PATH_ENV("PATH", dir_.path() + ":" +
                                       (INHERITED != nullptr ? INHERITED : "/usr/bin:/bin"));
  const ScopedEnv MODE("FAKE_PERF_MODE", "ok");
  const ScopedEnv LOG("FAKE_LOG", dir_.logPath());
  struct Mode {
    std::string test;
    std::string args;
    std::string output;
    std::string aside;
  };
  for (const Mode& M :
       {Mode{"Perf.Aborted", "", "stat.txt", "stat.aborted.txt"},
        Mode{"Perf.AbortedRecord", "record -g", "perf.data", "perf.aborted.data"}}) {
    PerfConfig cfg;
    cfg.profileTool = "perf";
    cfg.profileArgs = M.args;
    cfg.artifactRoot = dir_.path();
    ProfilerRegistry::instance().resetFailures();
    (void)vernier::bench::PerfRegistry::instance().takeAll();
    StderrCapture err;
    std::string folder;
    {
      vernier::bench::PerfCase pc = vernier::bench::makePerfCaseWithProfiler(M.test, cfg);
      try {
        pc.measured([] { throw WindowFailed{9}; });
        ADD_FAILURE() << M.test << ": the window's exception did not reach the caller";
      } catch (const WindowFailed& failed) {
        EXPECT_EQ(failed.mark, 9) << M.test << ": another exception reached the caller";
      }
      folder = dir_.path() + "/" + M.test + ".perf";
      // While the case lives, as a benchmark that caught the exception keeps it.
      EXPECT_TRUE(fakesGone()) << M.test << ": perf still runs after the window ended\n"
                               << dir_.log();
      EXPECT_FALSE(std::filesystem::exists(folder + "/" + M.output))
          << M.test << ": the aborted window's output is left as the case's";
      EXPECT_FALSE(readText(folder + "/" + M.aside).empty()) << M.test;
      EXPECT_FALSE(vernier::bench::PerfRegistry::instance().take().has_value())
          << M.test << ": the aborted window published a row";
      const std::vector<ProfileFailure> ABORTED = ProfilerRegistry::instance().failures();
      EXPECT_TRUE(ABORTED.empty()) << M.test << ": " << ABORTED.front().result.report.message;

      pc.measured([] {});
    }
    EXPECT_FALSE(readText(folder + "/" + M.output).empty())
        << M.test << ": the window measured after the aborted one left no output";
    EXPECT_TRUE(vernier::bench::PerfRegistry::instance().take().has_value())
        << M.test << ": the window measured after the aborted one published no row";
    const std::vector<ProfileFailure> FAILED = ProfilerRegistry::instance().failures();
    EXPECT_TRUE(FAILED.empty()) << M.test << ": " << FAILED.front().result.report.message;
    EXPECT_NE(err.text().find("[perf] the measured window ended by an exception: perf was "
                              "stopped; its output was set aside as " +
                              folder + "/" + M.aside + "\n"),
              std::string::npos)
        << M.test << ":\n"
        << err.text();
  }
}

/** @test Record mode's data file, confirmed by perf, passes; without it the capture fails. */
TEST_F(PerfLifecycleTest, RecordNeedsItsConfirmedData) {
  const Window GOOD = window("ok", "record -g");
  EXPECT_TRUE(GOOD.failures.empty()) << GOOD.failures.front().result.report.message;
  EXPECT_EQ(GOOD.err.find("caveat"), std::string::npos) << GOOD.err;
  std::error_code ec;
  EXPECT_GT(std::filesystem::file_size(GOOD.folder + "/perf.data", ec), 0U);
  ProfilerRegistry::instance().resetFailures();
  std::filesystem::remove(GOOD.folder + "/perf.data", ec);
  const Window BAD = window("error-text", "record -g");
  ASSERT_EQ(BAD.failures.size(), 1U);
  EXPECT_EQ(BAD.failures[0].result.report.message,
            "completion: missing: " + BAD.folder +
                "/perf.data was not written: perf: Error: the fake perf could not read its "
                "counters");
}

/**
 * @test A record whose data file perf wrote but never confirmed (no "Captured
 * and wrote" line in its error output) is not counted complete: one failure,
 * at the completion stage, naming the file, the missing confirmation and
 * what perf said instead.
 */
TEST_F(PerfLifecycleTest, RecordWithoutConfirmationFails) {
  const Window W = window("unconfirmed", "record -g");
  std::error_code ec;
  EXPECT_GT(std::filesystem::file_size(W.folder + "/perf.data", ec), 0U)
      << "the stand-in wrote no data file to leave unconfirmed";
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_EQ(W.failures[0].result.stage, ReadinessStage::COMPLETION);
  EXPECT_EQ(W.failures[0].result.report.message,
            "completion: unusable: " + W.folder +
                "/perf.data was not confirmed written (no \"Captured and wrote\" in " + W.folder +
                "/record.err.txt): perf: [ perf record: Woken up 1 times to write data ]");
}

/**
 * @test A record perf confirmed whose data section holds no sample, as perf
 * leaves a window too short for its sampling period (its synthesized task
 * and map records, then the end of its first round), is complete but empty:
 * no failure, and one caveat that names the file and how to get samples:
 * perf c2c, which takes no arguments, is not told to sample more often.
 */
TEST_F(PerfLifecycleTest, RecordOfNoSampleIsACaveat) {
  ScopedEnv data("FAKE_PERF_DATA", writePerfData("no_sample.data", {COMM, MMAP2, FINISHED_ROUND}));
  const std::string SAID =
      " holds no sample: perf confirmed it written, and none of its records is "
      "a sample. A measured window that runs on the CPU for less than perf's "
      "sampling period records none: lengthen it (--target-time 250ms sizes "
      "its cycles)";
  const Window W = window("ok", "record -g");
  EXPECT_TRUE(W.failures.empty()) << W.failures.front().result.report.message;
  EXPECT_NE(W.err.find("[perf] caveat: " + W.folder + "/perf.data" + SAID +
                       ", or sample more often (-F in --profile-args).\n"),
            std::string::npos)
      << W.err;
  ProfilerRegistry::instance().resetFailures();
  const Window C2C = window("ok", "c2c");
  EXPECT_TRUE(C2C.failures.empty()) << C2C.failures.front().result.report.message;
  EXPECT_NE(C2C.err.find("[perf] caveat: " + C2C.folder + "/perf.c2c.data" + SAID + ".\n"),
            std::string::npos)
      << C2C.err;
}

/**
 * @test No caveat for a confirmed record not shown to hold no sample: one
 * that holds a sample (which perf record -B confirms without a count), AUX
 * area data, compressed records, lost events or a record type the reader does
 * not know, a file of the other byte order, and a data section that ends
 * inside a record. None of them fails.
 */
TEST_F(PerfLifecycleTest, RecordNotShownEmptyHasNoCaveat) {
  struct Case {
    const char* what;
    std::string file;
  };
  const std::vector<Case> CASES = {
      {"a sample", writePerfData("sample.data", {COMM, SAMPLE, FINISHED_ROUND})},
      {"AUX area data", writePerfData("aux.data", {COMM, AUX})},
      {"AUX trace data", writePerfData("auxtrace.data", {COMM, AUXTRACE})},
      {"compressed records", writePerfData("compressed.data", {COMM, COMPRESSED})},
      {"lost events", writePerfData("lost.data", {COMM, LOST})},
      {"lost samples", writePerfData("lost_samples.data", {COMM, LOST_SAMPLES})},
      {"a type the reader does not know", writePerfData("unknown.data", {COMM, 99})},
      {"the other byte order", writePerfData("swapped.data", {COMM}, "2ELIFREP")},
      {"a data section that ends inside a record",
       writePerfData("cut.data", {COMM, FINISHED_ROUND}, "PERFILE2", PERF_RECORD_BYTES + 8)},
  };
  for (const Case& C : CASES) {
    ProfilerRegistry::instance().resetFailures();
    ScopedEnv data("FAKE_PERF_DATA", C.file);
    const Window W = window("ok", "record -g");
    EXPECT_TRUE(W.failures.empty()) << C.what << ": " << W.failures.front().result.report.message;
    EXPECT_EQ(W.err.find("caveat"), std::string::npos) << C.what << ":\n" << W.err;
  }
}

/**
 * @test A record perf writes as a directory (--threads among the arguments),
 * confirmed, is a capture as a file is, its samples in the per-CPU streams: no
 * failure, no caveat, and the directory left as perf wrote it.
 */
TEST_F(PerfLifecycleTest, RecordIntoADirectoryIsACapture) {
  ScopedEnv data("FAKE_PERF_DATA", writePerfDirectory("threads_samples", {COMM, FINISHED_ROUND},
                                                      {{}, {MMAP2, SAMPLE, SAMPLE}}));
  const Window W = window("ok", "record --threads -g");
  EXPECT_TRUE(W.failures.empty()) << W.failures.front().result.report.message;
  EXPECT_EQ(W.err.find("caveat"), std::string::npos) << W.err;
  EXPECT_TRUE(std::filesystem::is_directory(W.folder + "/perf.data"));
  EXPECT_TRUE(std::filesystem::is_regular_file(W.folder + "/perf.data/data.1"));
}

/**
 * @test A directory record whose header file and streams hold no sample, as
 * perf leaves a window too short for its sampling period, is complete but
 * empty: no failure, and the caveat names the directory.
 */
TEST_F(PerfLifecycleTest, RecordIntoADirectoryOfNoSampleIsACaveat) {
  ScopedEnv data("FAKE_PERF_DATA", writePerfDirectory("threads_none", {COMM, FINISHED_ROUND},
                                                      {{}, {MMAP2, COMM}, {}}));
  const Window W = window("ok", "record --threads -g");
  EXPECT_TRUE(W.failures.empty()) << W.failures.front().result.report.message;
  EXPECT_NE(W.err.find("[perf] caveat: " + W.folder +
                       "/perf.data holds no sample: perf confirmed it written, and none of its "
                       "records is a sample."),
            std::string::npos)
      << W.err;
}

/**
 * @test No caveat for a directory record not shown to hold no sample: its
 * header file records no directory format, or another than 1; a stream holds
 * a type the reader does not know, or ends inside a record; the directory
 * holds a file perf does not write. None of them fails.
 */
TEST_F(PerfLifecycleTest, RecordIntoADirectoryNotShownEmptyHasNoCaveat) {
  struct Case {
    const char* what;
    std::string dir;
  };
  std::vector<Case> cases = {
      {"no directory format", writePerfDirectory("no_format", {COMM}, {{MMAP2}}, 0)},
      {"directory format 2", writePerfDirectory("format_2", {COMM}, {{MMAP2}}, 2)},
      {"a type the reader does not know", writePerfDirectory("unknown", {COMM}, {{MMAP2, 99}})},
  };
  const std::string CUT = writePerfDirectory("cut_stream", {COMM}, {{MMAP2}});
  std::ofstream(CUT + "/data.0", std::ios::binary | std::ios::app)
      << recordBytes({COMM}).substr(0, 12);
  cases.push_back({"a stream that ends inside a record", CUT});
  const std::string EXTRA = writePerfDirectory("extra_file", {COMM}, {{MMAP2}});
  std::ofstream(EXTRA + "/data.0.copy", std::ios::binary) << recordBytes({COMM});
  cases.push_back({"a file perf does not write, records and all", EXTRA});
  for (const Case& C : cases) {
    ProfilerRegistry::instance().resetFailures();
    ScopedEnv data("FAKE_PERF_DATA", C.dir);
    const Window W = window("ok", "record --threads -g");
    EXPECT_TRUE(W.failures.empty()) << C.what << ": " << W.failures.front().result.report.message;
    EXPECT_EQ(W.err.find("caveat"), std::string::npos) << C.what << ":\n" << W.err;
  }
}

/**
 * @test A directory record without its header file is no capture: one
 * failure, at the completion stage, naming the directory, the mode that
 * writes one and the header file it lacks.
 */
TEST_F(PerfLifecycleTest, RecordIntoADirectoryWithoutItsHeaderFails) {
  const std::string NO_HEADER = dir_.path() + "/threads_no_header";
  std::filesystem::create_directories(NO_HEADER);
  std::ofstream(NO_HEADER + "/data.0", std::ios::binary) << recordBytes({MMAP2, SAMPLE});
  ScopedEnv data("FAKE_PERF_DATA", NO_HEADER);
  const Window W = window("ok", "record --threads -g");
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_EQ(W.failures[0].result.stage, ReadinessStage::COMPLETION);
  EXPECT_EQ(W.failures[0].result.report.message,
            "completion: missing: " + W.folder +
                "/perf.data is a directory, as perf record --threads writes its capture, but its "
                "header file " +
                W.folder +
                "/perf.data/data was not written: perf: [ perf record: Woken up 1 times to write "
                "data ] | [ perf record: Captured and wrote 0.001 MB " +
                W.folder + "/perf.data ]");
}

/**
 * @test A measured window that throws while perf records into a directory
 * sets the directory aside whole, and the next such window's replaces it:
 * each time perf.data is gone, perf.aborted.data is a directory holding the
 * header file, and the line names it; nothing fails.
 */
TEST_F(PerfLifecycleTest, AbortSetsADirectoryAsideWhole) {
  const char* INHERITED = std::getenv("PATH");
  const ScopedEnv PATH_ENV("PATH", dir_.path() + ":" +
                                       (INHERITED != nullptr ? INHERITED : "/usr/bin:/bin"));
  const ScopedEnv MODE("FAKE_PERF_MODE", "ok");
  const ScopedEnv LOG("FAKE_LOG", dir_.logPath());
  PerfConfig cfg;
  cfg.profileTool = "perf";
  cfg.profileArgs = "record --threads -g";
  cfg.artifactRoot = dir_.path();
  ProfilerRegistry::instance().resetFailures();
  (void)vernier::bench::PerfRegistry::instance().takeAll();
  const std::string FOLDER = dir_.path() + "/Perf.AbortedThreads.perf";
  StderrCapture err;
  {
    vernier::bench::PerfCase pc =
        vernier::bench::makePerfCaseWithProfiler("Perf.AbortedThreads", cfg);
    for (int mark = 1; mark <= 2; ++mark) {
      try {
        pc.measured([mark] { throw WindowFailed{mark}; });
        ADD_FAILURE() << "window " << mark << ": the exception did not reach the caller";
      } catch (const WindowFailed& failed) {
        EXPECT_EQ(failed.mark, mark) << "another exception reached the caller";
      }
      EXPECT_FALSE(std::filesystem::exists(FOLDER + "/perf.data")) << "window " << mark;
      EXPECT_TRUE(std::filesystem::is_regular_file(FOLDER + "/perf.aborted.data/data"))
          << "window " << mark;
    }
  }
  const std::string LINE =
      "[perf] the measured window ended by an exception: perf was stopped; its output was set "
      "aside as " +
      FOLDER + "/perf.aborted.data\n";
  const std::string TEXT = err.text();
  const std::size_t FIRST = TEXT.find(LINE);
  ASSERT_NE(FIRST, std::string::npos) << TEXT;
  EXPECT_NE(TEXT.find(LINE, FIRST + LINE.size()), std::string::npos)
      << "the second window's directory was not set aside:\n"
      << TEXT;
  const std::vector<ProfileFailure> FAILED = ProfilerRegistry::instance().failures();
  EXPECT_TRUE(FAILED.empty()) << FAILED.front().result.report.message;
}

/* ----------------------------- Data Extents ----------------------------- */

/**
 * @test A data file whose header or data section does not lie within the file
 * is not shown to hold no sample, whatever its records: no caveat, and no
 * failure (perf confirmed it). The bytes are written as given: a 56-byte file
 * whose header declares 104 bytes and its data at 104; a 104-byte header
 * whose empty data section lies at 256, past the end; a 56-byte file whose
 * header declares 104 bytes and keeps its empty data section within the file;
 * a 96-byte file whose header declares 104 bytes; a header declaring fewer
 * bytes than perf writes; a data section starting inside the header; and the
 * data section's offset and size wrapping past 2^64, each way.
 */
TEST_F(PerfLifecycleTest, RecordWhoseExtentsDoNotFitHasNoCaveat) {
  const std::string COMM_RECORD = recordBytes({COMM});
  struct Case {
    const char* what;
    std::string bytes;
  };
  const std::vector<Case> CASES = {
      {"56 bytes, a 104-byte header declared, data at 104", bytesWith(56, {{8, 104}, {40, 104}})},
      {"a 104-byte header, its empty data section at 256", bytesWith(104, {{8, 104}, {40, 256}})},
      {"56 bytes, a 104-byte header declared, data at 56", bytesWith(56, {{8, 104}, {40, 56}})},
      {"96 bytes, a 104-byte header declared, data at 96", bytesWith(96, {{8, 104}, {40, 96}})},
      {"a 64-byte header declared", bytesWith(104, {{8, 64}, {40, 104}})},
      {"a data section starting inside the header", bytesWith(104, {{8, 104}, {40, 56}})},
      {"data offset 104, size 2^64 - 8",
       bytesWith(104, {{8, 104}, {40, 104}, {48, ~std::uint64_t{7}}}) + COMM_RECORD},
      {"data offset 2^64 - 8, size 16",
       bytesWith(104, {{8, 104}, {40, ~std::uint64_t{7}}, {48, 16}}) + COMM_RECORD},
  };
  for (const Case& C : CASES) {
    ProfilerRegistry::instance().resetFailures();
    ScopedEnv data("FAKE_PERF_DATA", writeBytes("extent.data", C.bytes));
    const Window W = window("ok", "record -g");
    EXPECT_TRUE(W.failures.empty()) << C.what << ": " << W.failures.front().result.report.message;
    EXPECT_EQ(W.err.find("caveat"), std::string::npos) << C.what << ":\n" << W.err;
  }
}

/**
 * @test The same reading's positive controls, written as given: a whole
 * 104-byte header with an empty data section is shown to hold no sample (the
 * caveat); one whose data section holds a sample record is not.
 */
TEST_F(PerfLifecycleTest, RecordWhoseExtentsFitIsReadThrough) {
  ScopedEnv empty("FAKE_PERF_DATA",
                  writeBytes("empty.data", bytesWith(104, {{8, 104}, {40, 104}})));
  const Window EMPTY = window("ok", "record -g");
  EXPECT_TRUE(EMPTY.failures.empty()) << EMPTY.failures.front().result.report.message;
  EXPECT_NE(EMPTY.err.find("[perf] caveat: " + EMPTY.folder + "/perf.data holds no sample"),
            std::string::npos)
      << EMPTY.err;
  ProfilerRegistry::instance().resetFailures();
  ScopedEnv sample("FAKE_PERF_DATA",
                   writeBytes("sample.data", bytesWith(104, {{8, 104}, {40, 104}, {48, 16}}) +
                                                 recordBytes({SAMPLE})));
  const Window SAMPLED = window("ok", "record -g");
  EXPECT_TRUE(SAMPLED.failures.empty()) << SAMPLED.failures.front().result.report.message;
  EXPECT_EQ(SAMPLED.err.find("caveat"), std::string::npos) << SAMPLED.err;
}

/**
 * @test The directory's header file is held to the same extents, and its
 * feature table and DIR_FORMAT section must lie within it too; a stream
 * shorter than its own record header says is read through to no conclusion.
 * Written as given, each with an empty stream unless named: the three files of
 * RecordWhoseExtentsDoNotFitHasNoCaveat's first three cases as the header
 * file; a 96-byte header file declaring 104 bytes whose table and section
 * still read; the feature table past the end; the DIR_FORMAT section past the
 * end, and at an offset near 2^64; a stream of 12 bytes whose record says 16.
 * None gets the caveat; none fails.
 */
TEST_F(PerfLifecycleTest, DirectoryWhoseExtentsDoNotFitHasNoCaveat) {
  const std::vector<Field> FORMAT_1 = {{8, 104},   {40, 104}, {72, DIR_FORMAT_BIT},
                                       {104, 120}, {112, 8},  {120, 1}};
  struct Case {
    const char* what;
    std::string header;
    std::string stream;
  };
  const std::vector<Case> CASES = {
      {"a 56-byte header file, a 104-byte header declared, data at 104",
       bytesWith(56, {{8, 104}, {40, 104}, {72, DIR_FORMAT_BIT}}), ""},
      {"a 104-byte header file, its empty data section at 256",
       bytesWith(104, {{8, 104}, {40, 256}, {72, DIR_FORMAT_BIT}}), ""},
      {"a 56-byte header file, a 104-byte header declared, data at 56",
       bytesWith(56, {{8, 104}, {40, 56}, {72, DIR_FORMAT_BIT}}), ""},
      {"a 96-byte header file declaring 104 bytes, its table and section readable",
       bytesWith(96, {{8, 104}, {40, 80}, {56, 1}, {72, DIR_FORMAT_BIT}, {80, 56}, {88, 8}}), ""},
      {"the feature table past the end",
       bytesWith(104, {{8, 104}, {40, 104}, {72, DIR_FORMAT_BIT}}), ""},
      {"the DIR_FORMAT section past the end",
       bytesWith(124, {{8, 104}, {40, 104}, {72, DIR_FORMAT_BIT}, {104, 120}, {112, 8}}), ""},
      {"the DIR_FORMAT section at 2^64 - 4",
       bytesWith(128,
                 {{8, 104}, {40, 104}, {72, DIR_FORMAT_BIT}, {104, ~std::uint64_t{3}}, {112, 8}}),
       ""},
      {"a 12-byte stream whose record says 16", bytesWith(128, FORMAT_1),
       recordBytes({COMM}).substr(0, 12)},
  };
  for (const Case& C : CASES) {
    ProfilerRegistry::instance().resetFailures();
    const std::string DIR = writeRawDirectory("extent_dir", C.header, {C.stream});
    ScopedEnv data("FAKE_PERF_DATA", DIR);
    const Window W = window("ok", "record --threads -g");
    std::filesystem::remove_all(DIR);
    EXPECT_TRUE(W.failures.empty()) << C.what << ": " << W.failures.front().result.report.message;
    EXPECT_EQ(W.err.find("caveat"), std::string::npos) << C.what << ":\n" << W.err;
  }
}

/**
 * @test The directory's positive controls, written as given: a 128-byte
 * header file (a whole header, an empty data section, the feature table's one
 * entry and the DIR_FORMAT section recording 1) and an empty stream is shown
 * to hold no sample; the same with a sample record in the stream is not.
 */
TEST_F(PerfLifecycleTest, DirectoryWhoseExtentsFitIsReadThrough) {
  const std::string HEADER =
      bytesWith(128, {{8, 104}, {40, 104}, {72, DIR_FORMAT_BIT}, {104, 120}, {112, 8}, {120, 1}});
  ScopedEnv empty("FAKE_PERF_DATA", writeRawDirectory("empty_dir", HEADER, {""}));
  const Window EMPTY = window("ok", "record --threads -g");
  EXPECT_TRUE(EMPTY.failures.empty()) << EMPTY.failures.front().result.report.message;
  EXPECT_NE(EMPTY.err.find("[perf] caveat: " + EMPTY.folder + "/perf.data holds no sample"),
            std::string::npos)
      << EMPTY.err;
  ProfilerRegistry::instance().resetFailures();
  ScopedEnv sample("FAKE_PERF_DATA",
                   writeRawDirectory("sample_dir", HEADER, {recordBytes({SAMPLE})}));
  const Window SAMPLED = window("ok", "record --threads -g");
  EXPECT_TRUE(SAMPLED.failures.empty()) << SAMPLED.failures.front().result.report.message;
  EXPECT_EQ(SAMPLED.err.find("caveat"), std::string::npos) << SAMPLED.err;
}

/* ----------------------------- Start Handshake ----------------------------- */

/**
 * @test The measured phase starts once perf answers on its control fifo: with
 * a perf that answers 2 s after it starts, beforeMeasure() returns after at
 * least 2 s (a lower bound, which load cannot pass falsely), and nothing fails.
 */
TEST_F(PerfLifecycleTest, MeasuredPhaseWaitsForTheAck) {
  const Window W = window("slow-ack");
  EXPECT_TRUE(W.failures.empty()) << W.failures.front().result.report.message;
  EXPECT_GE(W.startTook, std::chrono::milliseconds(2000));
  EXPECT_EQ(dir_.logLines("perf " + perf_ + " stat -e " + vernier::bench::PERF_STAT_EVENTS +
                          " -p " + std::to_string(::getpid()) + " --control fifo:")
                .size(),
            1U)
      << dir_.log();
}

/**
 * @test A perf that never answers on its control fifo fails the request at
 * the collection stage once the wait's bound has passed, and is stopped.
 */
TEST_F(PerfLifecycleTest, NoAckFailsTheRequest) {
  const Window W = window("no-ack");
  ASSERT_EQ(W.failures.size(), 1U);
  EXPECT_EQ(W.failures[0].result.stage, ReadinessStage::COLLECTION);
  EXPECT_EQ(W.failures[0].result.report.message,
            "unusable: perf did not answer on its --control fifo within 5 s, so it was not known "
            "to be counting; it was stopped, and this case is not profiled");
  EXPECT_GE(W.startTook, std::chrono::milliseconds(PerfStatProfiler::PERF_ACK_WAIT_MS));
  EXPECT_TRUE(fakesGone()) << dir_.log();
}

/**
 * @test A perf the check found not answering is started without --control,
 * after the fixed grace, and its capture is kept as before.
 */
TEST_F(PerfLifecycleTest, NoControlKeepsTheFixedWait) {
  const Window W = window("ok", "", /*afterExit=*/false, /*answersPing=*/false);
  EXPECT_TRUE(W.failures.empty()) << W.failures.front().result.report.message;
  EXPECT_GE(W.startTook, std::chrono::milliseconds(PerfStatProfiler::PERF_START_GRACE_MS));
  const auto LAUNCH = dir_.logLines("perf " + perf_ + " stat -e ");
  ASSERT_EQ(LAUNCH.size(), 1U) << dir_.log();
  EXPECT_EQ(LAUNCH[0].find("--control"), std::string::npos) << LAUNCH[0];
  EXPECT_NE(readText(W.folder + "/stat.txt").find("Performance counter stats"), std::string::npos);
}

/**
 * @test Without the handshake (a perf the check found not answering, and
 * perf mem) a perf that ends at its start fails the capture before the
 * measured phase, at the collection stage, with its words: the start's grace
 * sees it gone (the fake ends within milliseconds; the grace is 200 ms).
 */
TEST_F(PerfLifecycleTest, EarlyExitWithoutTheHandshakeFails) {
  const std::string SAID = "unusable: perf ended (exit status 1) before the measured phase: perf: "
                           "Error: failed to open counters: No such process";
  const Window SILENT = window("exit-early", "", /*afterExit=*/false, /*answersPing=*/false);
  ASSERT_EQ(SILENT.failures.size(), 1U);
  EXPECT_EQ(SILENT.failures[0].result.stage, ReadinessStage::COLLECTION);
  EXPECT_EQ(SILENT.failures[0].result.report.message, SAID);
  ProfilerRegistry::instance().resetFailures();
  const Window MEM = window("exit-early", "mem");
  ASSERT_EQ(MEM.failures.size(), 1U);
  EXPECT_EQ(MEM.failures[0].result.stage, ReadinessStage::COLLECTION);
  EXPECT_EQ(MEM.failures[0].result.report.message, SAID);
  EXPECT_EQ(dir_.log().find("--control"), std::string::npos) << dir_.log();
}

/** @test perf mem keeps the fixed grace, also with a perf that answers. */
TEST_F(PerfLifecycleTest, MemKeepsTheFixedStart) {
  const Window W = window("ok", "mem");
  EXPECT_TRUE(W.failures.empty()) << W.failures.front().result.report.message;
  const auto LAUNCH = dir_.logLines("perf " + perf_ + " mem record ");
  ASSERT_EQ(LAUNCH.size(), 1U) << dir_.log();
  EXPECT_EQ(LAUNCH[0].find("--control"), std::string::npos) << LAUNCH[0];
}

/**
 * @test The check's access probe finds whether perf answers a ping: one that
 * does is ready and planned for the wait; one that does not answer, or does
 * not take --control (the probe then runs again without it), is a caveat
 * naming the fixed start, and its plan keeps the fixed grace.
 */
TEST_F(PerfLifecycleTest, CheckLearnsWhetherPerfAnswers) {
  const ReadinessResult ANSWERS = check("ok");
  EXPECT_EQ(ANSWERS.cause, ReadinessCause::READY) << ANSWERS.report.message;
  const auto ANSWERS_PLAN = std::dynamic_pointer_cast<const PerfPlan>(ANSWERS.plan);
  ASSERT_NE(ANSWERS_PLAN, nullptr);
  EXPECT_TRUE(ANSWERS_PLAN->answersPing);

  const ReadinessResult SILENT = check("no-ack");
  EXPECT_EQ(SILENT.cause, ReadinessCause::CAVEAT);
  EXPECT_EQ(SILENT.report.message,
            "perf stat counts this process, but " + perf_ +
                " did not answer a ping on its --control fifo, so the measured phase starts "
                "after a fixed 200 ms instead of once perf is counting");
  const auto SILENT_PLAN = std::dynamic_pointer_cast<const PerfPlan>(SILENT.plan);
  ASSERT_NE(SILENT_PLAN, nullptr);
  EXPECT_FALSE(SILENT_PLAN->answersPing);

  const std::string PROBE = "perf " + perf_ + " stat -x, -e " + vernier::bench::PERF_STAT_EVENTS +
                            " -p " + std::to_string(::getpid()) + " --timeout 100";
  const std::size_t PROBES_BEFORE = dir_.logLines(PROBE).size();
  const ReadinessResult OLD = check("no-control");
  EXPECT_EQ(OLD.cause, ReadinessCause::CAVEAT);
  EXPECT_EQ(OLD.report.message,
            "perf stat counts this process, but " + perf_ +
                " does not take --control, so the measured phase starts after a fixed 200 ms "
                "instead of once perf is counting");
  const auto OLD_PLAN = std::dynamic_pointer_cast<const PerfPlan>(OLD.plan);
  ASSERT_NE(OLD_PLAN, nullptr);
  EXPECT_FALSE(OLD_PLAN->answersPing);
  const auto PROBES = dir_.logLines(PROBE);
  ASSERT_EQ(PROBES.size(), PROBES_BEFORE + 2) << dir_.log();
  EXPECT_NE(PROBES[PROBES_BEFORE].find(" --control fifo:"), std::string::npos) << PROBES.back();
  EXPECT_EQ(PROBES.back().find("--control"), std::string::npos) << PROBES.back();

  EXPECT_EQ(check("no-ack", "record -g").report.message,
            "unverified: perf stat counts this process; perf record itself is not probed before "
            "the run; " +
                perf_ +
                " did not answer a ping on its --control fifo, so the measured phase starts after "
                "a fixed 200 ms");
  EXPECT_EQ(check("ok", "mem").report.message,
            "unverified: perf stat counts this process; perf mem itself is not probed before the "
            "run, and the measured phase starts after a fixed 200 ms");
}
