/**
 * @file 13_OffCpuProfiler_uTest.cpp
 * @brief The checks behind walkthrough 16: both of demo 13's versions reach the
 *        same total, and an off-CPU capture of the demo finds its threads
 *        waiting in the coarse lock, and not in the version that shares
 *        nothing.
 *
 * Not part of demo 13, whose source shows only what it teaches. The totals
 * checks run each version through contentionRun(), as the demo does, with a
 * fixed number of threads, calls and repeats, and require every call's joined
 * length in the total, which the machine's load cannot change. The traced
 * check runs the demo binary (BenchDemo_13_OffCpuProfiler, whose path the build
 * passes in) under --profile offcpu with BENCH_SUDO=1, as the walkthrough
 * does, and reads the capture each case left; 09_BpftraceProfiler_Check.hpp
 * holds the traced run and the reading of its lines. What decides the traced
 * check's findings and its skips is checked in every run, on captures and
 * notices from the reference rig. ctest runs them under the demo label, the
 * traced one also under the bpftrace label.
 *
 * Usage:
 *   @code{.sh}
 *   ctest --test-dir build -L demo
 *   ./build/bin/tests/TestDemoOffCpu      # the same, by hand
 *   @endcode
 */

#include "src/bench/demo/cpu/13_OffCpuProfiler_Totals.hpp"

#include "src/bench/demo/cpu/utst/09_BpftraceProfiler_Check.hpp"
#include "src/bench/demo/cpu/utst/12_MemcheckProfiler_Check.hpp"
#include "src/bench/demo/examples/join/inc/Join.hpp"
#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfHarness.hpp"
#include "src/bench/inc/ProfilerEnv.hpp"
#include "src/bench/inc/ProfilerOffCpu.hpp"

#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <filesystem>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>

namespace demo = vernier::bench::demo;
namespace check = vernier::bench::demo::bpftrace_check;
namespace vg = vernier::bench::demo::memcheck_check;
namespace fs = std::filesystem;

using vernier::bench::PerfCase;
using vernier::bench::PerfConfig;
using vernier::bench::demo::offcpu_demo::addToThreadTotal;
using vernier::bench::demo::offcpu_demo::addUnderCoarseLock;
using vernier::bench::demo::offcpu_demo::finishedThreadsTotal;
using vernier::bench::demo::offcpu_demo::PART_COUNT;
using vernier::bench::demo::offcpu_demo::PART_SEED;
using vernier::bench::demo::offcpu_demo::SharedTotal;

/* ----------------------------- Constants ----------------------------- */

namespace {

/// The run each version gets: threads, calls per thread and repeats.
constexpr int THREADS = 4;
constexpr int CALLS_PER_THREAD = 25;
constexpr int REPEATS = 3;

/// A configuration under which contentionRun() makes exactly
/// THREADS x CALLS_PER_THREAD x REPEATS calls.
PerfConfig fixedRun() {
  PerfConfig cfg;
  cfg.threads = THREADS;
  cfg.cycles = CALLS_PER_THREAD;
  cfg.repeats = REPEATS;
  return cfg;
}

/// Every call's joined length, added up.
std::size_t everyCallsLength(const std::vector<std::string>& parts) {
  return static_cast<std::size_t>(THREADS) * CALLS_PER_THREAD * REPEATS * demo::joinedSize(parts);
}

} // namespace

/* ----------------------------- API Tests ----------------------------- */

/** @test The coarse-lock version's shared total holds every call's joined length */
TEST(OffCpuTotalsTest, CoarseLockCountsEveryCall) {
  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  SharedTotal total;

  PerfCase perf{"OffCpuTotalsTest.CoarseLock", fixedRun()};
  perf.contentionRun([&] { addUnderCoarseLock(total, PARTS); }, "coarse_lock");

  EXPECT_EQ(total.value, everyCallsLength(PARTS));
}

/** @test Every no-sharing thread has handed its total over when contentionRun returns */
TEST(OffCpuTotalsTest, NoSharingCountsEveryCallOnceItsThreadsEnd) {
  const auto PARTS = demo::makeParts(PART_COUNT, PART_SEED);
  finishedThreadsTotal = 0;

  PerfCase perf{"OffCpuTotalsTest.NoSharing", fixedRun()};
  perf.contentionRun([&] { addToThreadTotal(PARTS); }, "no_sharing");

  EXPECT_EQ(finishedThreadsTotal.load(), everyCallsLength(PARTS));
}

/* ----------------------------- Reading a Capture ----------------------------- */

namespace {

/// One @offcpu_blocks entry: its frames, innermost first, and its count.
struct OffCpuStack {
  std::vector<std::string> frames;
  long count = 0;
};

/// What an off-CPU capture's output (offcpu.txt) holds.
struct OffCpuDump {
  int armedLines = 0;               ///< "offcpu armed " lines
  int disarmedLines = 0;            ///< "offcpu disarmed " lines
  bool armedBeforeDisarmed = false; ///< The first arm line comes before any disarm line
  bool disarmedState = false;       ///< "@armed: 2": the maps as the stop left them
  int startLines = 0;               ///< "@start[" lines: starts left behind
  long recorded = -1;               ///< The "@recorded: " value; -1 without the line
  std::vector<OffCpuStack> stacks;  ///< The @offcpu_blocks entries
  std::vector<long> threads;        ///< The @offcpu_ns keys: the threads that slept
};

/// @p text without the spaces at its start.
std::string trimmedStart(const std::string& text) {
  const std::size_t FIRST = text.find_first_not_of(" \t");
  return FIRST == std::string::npos ? std::string{} : text.substr(FIRST);
}

/// Read the output an off-CPU capture left: its two acknowledgement lines and
/// the maps bpftrace printed at the stop.
OffCpuDump readDump(const std::string& text) {
  OffCpuDump dump;
  bool inStack = false;
  std::istringstream lines(text);
  std::string line;
  while (std::getline(lines, line)) {
    if (inStack) {
      if (line.rfind(", ", 0) == 0) {
        const std::size_t CLOSE = line.rfind("]: ");
        dump.stacks.back().count =
            CLOSE == std::string::npos ? 0 : std::strtol(line.c_str() + CLOSE + 3, nullptr, 10);
        inStack = false;
      } else {
        dump.stacks.back().frames.push_back(trimmedStart(line));
      }
    } else if (line.rfind("offcpu armed ", 0) == 0) {
      dump.armedBeforeDisarmed =
          dump.armedLines == 0 ? dump.disarmedLines == 0 : dump.armedBeforeDisarmed;
      ++dump.armedLines;
    } else if (line.rfind("offcpu disarmed ", 0) == 0) {
      ++dump.disarmedLines;
    } else if (line == "@armed: 2") {
      dump.disarmedState = true;
    } else if (line.rfind("@start[", 0) == 0) {
      ++dump.startLines;
    } else if (line.rfind("@recorded: ", 0) == 0) {
      dump.recorded = std::strtol(line.c_str() + 11, nullptr, 10);
    } else if (line.rfind("@offcpu_blocks[", 0) == 0) {
      dump.stacks.emplace_back();
      inStack = true;
    } else if (line.rfind("@offcpu_ns[", 0) == 0) {
      dump.threads.push_back(std::strtol(line.c_str() + 11, nullptr, 10));
    }
  }
  return dump;
}

/// The sleeping switch-outs the capture's stacks count.
long switchOuts(const OffCpuDump& dump) {
  long total = 0;
  for (const OffCpuStack& stack : dump.stacks) {
    total += stack.count;
  }
  return total;
}

/// The sleeping switch-outs of the stacks that hold a frame naming @p name.
long switchOutsThrough(const OffCpuDump& dump, const std::string& name) {
  long total = 0;
  for (const OffCpuStack& stack : dump.stacks) {
    for (const std::string& frame : stack.frames) {
      if (frame.find(name) != std::string::npos) {
        total += stack.count;
        break;
      }
    }
  }
  return total;
}

/**
 * @brief The sleeping switch-outs of the threads asleep in a mutex that demo
 *        13's addUnderCoarseLock took: stacks with a pthread_mutex_lock frame
 *        and, below it (called from), addUnderCoarseLock's.
 */
long lockWaitsUnderTheCoarseLock(const OffCpuDump& dump) {
  long total = 0;
  for (const OffCpuStack& stack : dump.stacks) {
    bool inLock = false;
    for (const std::string& frame : stack.frames) {
      if (frame.find("pthread_mutex_lock") != std::string::npos) {
        inLock = true;
      } else if (inLock && frame.find("offcpu_demo::addUnderCoarseLock") != std::string::npos) {
        total += stack.count;
        break;
      }
    }
  }
  return total;
}

/// The @offcpu_blocks entries of @p text, as bpftrace printed them.
std::string stacksOf(const std::string& text) {
  std::string stacks;
  bool inStack = false;
  std::istringstream lines(text);
  std::string line;
  while (std::getline(lines, line)) {
    inStack = inStack || line.rfind("@offcpu_blocks[", 0) == 0;
    if (inStack) {
      stacks += line + "\n";
      inStack = line.rfind(", ", 0) != 0;
    }
  }
  return stacks;
}

// Captures and notices from real runs on the reference rig (Raspberry Pi 4,
// bpftrace 0.23.2), with frames whose argument or template lists run past 80
// columns shortened to "(...)" and "<...>".

/// Demo 13's CoarseLock capture: 3 threads x 1,000 calls x 5 repeats.
constexpr const char* RIG_COARSE_LOCK_DUMP =
    "Attaching 2 probes...\n"
    "offcpu armed 85324 85344\n"
    "offcpu disarmed 85324 85324 10111\n"
    "\n"
    "\n"
    "@armed: 2\n"
    "@offcpu_blocks[\n"
    "        0x7fb34cef6c\n"
    "        __syscall_cancel+16\n"
    "        _IO_new_file_underflow+320\n"
    "        __GI__IO_default_uflow+56\n"
    "        _IO_getline_info+212\n"
    "        _IO_fgets+168\n"
    "        vernier::bench::captureGitHash[abi:cxx11]()+88\n"
    "        vernier::bench::PerfCase::contentionRun(...)+6656\n"
    "        OffCpu_CoarseLock_Test::TestBody()+1576\n"
    "        void testing::internal::HandleExceptionsInMethodIfSupported<...>(...)+72\n"
    "        testing::Test::Run()+236\n"
    "        testing::TestInfo::Run()+376\n"
    "        testing::TestSuite::Run()+700\n"
    "        testing::internal::UnitTestImpl::RunAllTests()+912\n"
    "        testing::UnitTest::Run()+104\n"
    "        main+128\n"
    "        __libc_start_call_main+124\n"
    "        __libc_start_main@GLIBC_2.17+156\n"
    "        _start+48\n"
    ", BenchDemo_13_Of]: 1\n"
    "@offcpu_blocks[\n"
    "        __clone3+52\n"
    "        popen+100\n"
    "        vernier::bench::captureGitHash[abi:cxx11]()+64\n"
    "        vernier::bench::PerfCase::contentionRun(...)+6656\n"
    "        OffCpu_CoarseLock_Test::TestBody()+1576\n"
    "        void testing::internal::HandleExceptionsInMethodIfSupported<...>(...)+72\n"
    "        testing::Test::Run()+236\n"
    "        testing::TestInfo::Run()+376\n"
    "        testing::TestSuite::Run()+700\n"
    "        testing::internal::UnitTestImpl::RunAllTests()+912\n"
    "        testing::UnitTest::Run()+104\n"
    "        main+128\n"
    "        __libc_start_call_main+124\n"
    "        __libc_start_main@GLIBC_2.17+156\n"
    "        _start+48\n"
    ", BenchDemo_13_Of]: 1\n"
    "@offcpu_blocks[\n"
    "        0x7fb34cef6c\n"
    "        __GI___futex_abstimed_wait_cancelable64+72\n"
    "        __pthread_clockjoin_ex+420\n"
    "        std::thread::join()+36\n"
    "        vernier::bench::PerfCase::contentionRun(...)+840\n"
    "        OffCpu_CoarseLock_Test::TestBody()+1576\n"
    "        void testing::internal::HandleExceptionsInMethodIfSupported<...>(...)+72\n"
    "        testing::Test::Run()+236\n"
    "        testing::TestInfo::Run()+376\n"
    "        testing::TestSuite::Run()+700\n"
    "        testing::internal::UnitTestImpl::RunAllTests()+912\n"
    "        testing::UnitTest::Run()+104\n"
    "        main+128\n"
    "        __libc_start_call_main+124\n"
    "        __libc_start_main@GLIBC_2.17+156\n"
    "        _start+48\n"
    ", BenchDemo_13_Of]: 9\n"
    "@offcpu_blocks[\n"
    "        __lll_lock_wait+80\n"
    "        __pthread_mutex_lock+264\n"
    "        vernier::bench::demo::offcpu_demo::addUnderCoarseLock(...)+24\n"
    "        std::thread::_State_impl<...>::_M_run(...)+104\n"
    "        0x7fb37cb4e0\n"
    "        start_thread+920\n"
    "        thread_start+12\n"
    ", BenchDemo_13_Of]: 10103\n"
    "@offcpu_ns[85349]: 2858726\n"
    "@offcpu_ns[85358]: 17952228\n"
    "@offcpu_ns[85356]: 18246216\n"
    "@offcpu_ns[85362]: 18282181\n"
    "@offcpu_ns[85353]: 18916858\n"
    "@offcpu_ns[85351]: 24590128\n"
    "@offcpu_ns[85361]: 33765527\n"
    "@offcpu_ns[85359]: 34797809\n"
    "@offcpu_ns[85355]: 34936154\n"
    "@offcpu_ns[85354]: 35045335\n"
    "@offcpu_ns[85350]: 36788698\n"
    "@offcpu_ns[85324]: 322742249\n"
    "@recorded: 10111\n"
    "\n";

/// Demo 13's NoSharing capture, the same run: one worker's wait is in the C
/// library's own lock, taken when a thread first registers a thread_local
/// destructor, not in the demo's.
constexpr const char* RIG_NO_SHARING_DUMP =
    "Attaching 2 probes...\n"
    "offcpu armed 85324 85381\n"
    "offcpu disarmed 85324 85324 10\n"
    "\n"
    "\n"
    "@armed: 2\n"
    "@offcpu_blocks[\n"
    "        __lll_lock_wait+80\n"
    "        __pthread_mutex_lock+368\n"
    "        __cxa_thread_atexit_impl+124\n"
    "        __cxa_thread_atexit+16\n"
    "        std::_Function_handler<...>::_M_invoke(...)+40\n"
    "        std::thread::_State_impl<...>::_M_run(...)+104\n"
    "        0x7fb37cb4e0\n"
    "        start_thread+920\n"
    "        thread_start+12\n"
    ", BenchDemo_13_Of]: 1\n"
    "@offcpu_blocks[\n"
    "        0x7fb34cef6c\n"
    "        __GI___futex_abstimed_wait_cancelable64+72\n"
    "        __pthread_clockjoin_ex+420\n"
    "        std::thread::join()+36\n"
    "        vernier::bench::PerfCase::contentionRun(...)+840\n"
    "        OffCpu_NoSharing_Test::TestBody()+1580\n"
    "        void testing::internal::HandleExceptionsInMethodIfSupported<...>(...)+72\n"
    "        testing::Test::Run()+236\n"
    "        testing::TestInfo::Run()+376\n"
    "        testing::TestSuite::Run()+700\n"
    "        testing::internal::UnitTestImpl::RunAllTests()+912\n"
    "        testing::UnitTest::Run()+104\n"
    "        main+128\n"
    "        __libc_start_call_main+124\n"
    "        __libc_start_main@GLIBC_2.17+156\n"
    "        _start+48\n"
    ", BenchDemo_13_Of]: 9\n"
    "@offcpu_ns[85386]: 31259\n"
    "@offcpu_ns[85324]: 102879711\n"
    "@recorded: 10\n"
    "\n";

/// A run in a PID namespace of its own: the decision refused the request.
constexpr const char* RIG_NAMESPACE_RUN =
    "[ RUN      ] ReadinessFixture.First\n"
    "\n"
    "[FAIL] Profiler 'offcpu': unsupported: the off-CPU script ran through sudo -n "
    "(BENCH_SUDO=1) but did not acknowledge its arm probe within 4000 ms; this process runs in "
    "PID namespace pid:[4026532563], not in the initial one (pid:[4026531836]), and offcpu "
    "traces only from the host's PID view\n"
    "   Run the benchmark on the host, or in a container started with --pid=host.\n"
    "   Falling back to no-op (measurements will proceed without profiling).\n";

/// A run without BENCH_SUDO: bpftrace's own refusal, quoted by the decision.
constexpr const char* RIG_DENIED_RUN =
    "[FAIL] Profiler 'offcpu': denied: the off-CPU script could not attach as the current "
    "user: ERROR: bpftrace currently only supports running as the root user.\n"
    "   Set BENCH_SUDO=1 with a scoped sudoers grant for /usr/bin/bpftrace and /usr/bin/kill, "
    "run with CAP_BPF and CAP_PERFMON, or run as root.\n"
    "   Falling back to no-op (measurements will proceed without profiling).\n";

/// A capture whose tracer ended before the stop: a failure of the capture,
/// not a machine that cannot trace.
constexpr const char* RIG_EARLY_END_RUN =
    "[offcpu] unusable: the tracer ended by itself before the stop (exit status 0); it misses "
    "the end of the measured region; the capture in "
    "/tmp/captures/ThreadScaling.MutexContention.offcpu is incomplete\n"
    "[offcpu] The script ends by itself when this process's main thread exits; anything else "
    "is bpftrace failing: see offcpu.err.txt there.\n";

} // namespace

/**
 * @test The coarse-lock capture from the rig: both acknowledgements, the maps
 * the stop left, and its threads' waits in the lock under addUnderCoarseLock.
 */
TEST(OffCpuCaptureReadingTest, FindsTheLockWaitInTheRigsCoarseLockCapture) {
  const OffCpuDump DUMP = readDump(RIG_COARSE_LOCK_DUMP);
  EXPECT_EQ(DUMP.armedLines, 1);
  EXPECT_EQ(DUMP.disarmedLines, 1);
  EXPECT_TRUE(DUMP.armedBeforeDisarmed);
  EXPECT_TRUE(DUMP.disarmedState);
  EXPECT_EQ(DUMP.startLines, 0);
  EXPECT_EQ(DUMP.recorded, 10111);
  ASSERT_EQ(DUMP.stacks.size(), 4U);
  EXPECT_EQ(switchOuts(DUMP), 10114);
  EXPECT_EQ(lockWaitsUnderTheCoarseLock(DUMP), 10103);
  EXPECT_EQ(switchOutsThrough(DUMP, "OffCpu_CoarseLock_Test::TestBody"), 11)
      << "the main thread's own waits, git describe's and join's";
  EXPECT_EQ(DUMP.threads.size(), 12U);
  const std::vector<long> ARMED = check::acknowledgement(RIG_COARSE_LOCK_DUMP, "offcpu", "armed");
  const std::vector<long> DISARMED =
      check::acknowledgement(RIG_COARSE_LOCK_DUMP, "offcpu", "disarmed");
  EXPECT_EQ(ARMED, (std::vector<long>{85324, 85344}));
  EXPECT_EQ(DISARMED, (std::vector<long>{85324, 85324, 10111}));
}

/**
 * @test The no-sharing capture from the rig holds a thread asleep in a mutex,
 * the C library's own, and no wait under addUnderCoarseLock: the reading
 * counts only the lock the demo takes.
 */
TEST(OffCpuCaptureReadingTest, FindsNoCoarseLockWaitInTheRigsNoSharingCapture) {
  const OffCpuDump DUMP = readDump(RIG_NO_SHARING_DUMP);
  EXPECT_TRUE(DUMP.disarmedState);
  EXPECT_EQ(DUMP.startLines, 0);
  EXPECT_EQ(switchOuts(DUMP), 10);
  EXPECT_EQ(switchOutsThrough(DUMP, "pthread_mutex_lock"), 1);
  EXPECT_EQ(lockWaitsUnderTheCoarseLock(DUMP), 0);
  EXPECT_EQ(switchOutsThrough(DUMP, "OffCpu_NoSharing_Test::TestBody"), 9);
}

/**
 * @test The traced check skips only where the run says it could not trace
 * here, in the decision's refusal for want of a privilege or of the host's PID
 * view; a capture that failed is not such a line, and fails the check.
 */
TEST(OffCpuCaptureReadingTest, SkipsOnlyWhereTheRunCouldNotTrace) {
  EXPECT_NE(check::cannotTraceHere(RIG_NAMESPACE_RUN, "offcpu")
                .find("unsupported: the off-CPU script ran through sudo -n"),
            std::string::npos);
  EXPECT_NE(check::cannotTraceHere(RIG_DENIED_RUN, "offcpu")
                .find("ERROR: bpftrace currently only supports running as the root user."),
            std::string::npos);
  EXPECT_EQ(check::cannotTraceHere(RIG_EARLY_END_RUN, "offcpu"), "");
}

/* ----------------------------- Traced Check ----------------------------- */

namespace {

/// The demo binary the build passes in.
constexpr const char* DEMO_BINARY = VERNIER_DEMO_13_BINARY;

/// The traced runs' fixed size: threads, calls per thread and repeats.
std::vector<std::string> tracedSize() {
  return {"--threads", "3", "--cycles", "200", "--repeats", "3"};
}

/// The target of this process's /proc/self/ns/pid link ("" if unreadable).
std::string ownPidNamespaceLink() {
  std::error_code ec;
  return fs::read_symlink("/proc/self/ns/pid", ec).string();
}

/// The demo binary's path, or "" when it is missing.
std::string demoPath() {
  std::error_code ec;
  const fs::path PATH = fs::canonical(DEMO_BINARY, ec);
  return ec ? std::string{} : PATH.string();
}

/// A new directory under the temporary directory, or "" when none could be made.
fs::path scratchDir() {
  std::error_code ec;
  std::string pattern = (fs::temp_directory_path(ec) / "offcpu_demo_check_XXXXXX").string();
  if (ec || ::mkdtemp(pattern.data()) == nullptr) {
    return {};
  }
  return pattern;
}

/// One case of the demo under --profile offcpu, and what its capture holds.
struct TracedCase {
  check::TracedRun run;
  std::string outcome; ///< The run's "[offcpu] " lines
  std::string output;  ///< The capture's offcpu.txt
  OffCpuDump dump;
};

TracedCase traceCase(const std::string& demoBinary, const std::string& testName,
                     const fs::path& dir) {
  TracedCase traced;
  traced.run = check::runTraced(demoBinary, testName, {"offcpu"}, tracedSize(), dir);
  traced.outcome = check::backendLines(traced.run.output, "offcpu");
  traced.output = vg::readText(check::captureFolder(traced.run, testName, "offcpu") / "offcpu.txt");
  traced.dump = readDump(traced.output);
  return traced;
}

/// The checks every capture of the demo must pass, whatever it recorded.
void expectAWholeCapture(const TracedCase& traced, const std::string& testName) {
  SCOPED_TRACE(testName);
  const fs::path OUTPUT = check::captureFolder(traced.run, testName, "offcpu") / "offcpu.txt";
  EXPECT_TRUE(vg::exitedWith(traced.run.end, 0)) << "the demo " << vg::describe(traced.run.end);
  EXPECT_EQ(check::linesWith(traced.outcome, "[offcpu] "), 1) << "one outcome line per capture:\n"
                                                              << traced.outcome;
  const bool WRITTEN =
      traced.outcome.rfind("[offcpu] stacks written to " + OUTPUT.string() + " (", 0) == 0;
  const bool ZERO =
      traced.outcome.rfind("[offcpu] no thread of this process went to sleep", 0) == 0;
  EXPECT_TRUE(WRITTEN || ZERO) << "the capture is not a verified one:\n" << traced.outcome;
  const std::vector<long> ARMED = check::acknowledgement(traced.output, "offcpu", "armed");
  const std::vector<long> DISARMED = check::acknowledgement(traced.output, "offcpu", "disarmed");
  ASSERT_EQ(ARMED.size(), 2U) << "no arm acknowledgement:\n" << traced.output;
  ASSERT_EQ(DISARMED.size(), 3U) << "no stop acknowledgement:\n" << traced.output;
  EXPECT_EQ(traced.dump.armedLines, 1) << traced.output;
  EXPECT_EQ(traced.dump.disarmedLines, 1) << traced.output;
  EXPECT_TRUE(traced.dump.armedBeforeDisarmed) << traced.output;
  EXPECT_EQ(DISARMED[0], ARMED[0]) << "the stop was acknowledged for another process";
  EXPECT_EQ(DISARMED[1], ARMED[0])
      << "the stop was acknowledged from a thread other than the demo's main thread, which "
         "ends the measured repeats";
  EXPECT_NE(ARMED[1], ARMED[0]) << "the arm was acknowledged from the main thread";
  EXPECT_TRUE(traced.dump.disarmedState) << "no \"@armed: 2\": the maps the stop left are missing";
  EXPECT_EQ(traced.dump.startLines, 0) << "starts were left behind:\n" << traced.output;
  EXPECT_EQ(DISARMED[2], traced.dump.recorded < 0 ? 0 : traced.dump.recorded)
      << "the stop's count and the dump's @recorded differ";
  EXPECT_LE(traced.dump.recorded, switchOuts(traced.dump));
}

} // namespace

/**
 * @test Demo 13 under the offcpu backend, as the walkthrough runs it: each
 * case's capture is verified from its arm to its stop, both lines naming the
 * demo, and whole; the coarse-lock case's threads wait in the lock under
 * addUnderCoarseLock, and the no-sharing case's do not. How long anything
 * waited, how often, and which workers slept at all are scheduling, described
 * on the page, and not checked.
 *
 * Skipped where this process runs in a PID namespace other than the initial
 * one (offcpu traces only from the host's view), without bpftrace, and without
 * root or sudo, each known before anything runs; where the run says offcpu
 * could not trace here, for want of a privilege, of a kernel probe or of the
 * host's PID view, quoting the run's line, which ends with sudo's or
 * bpftrace's own words; and, after the captures are checked, where their
 * stacks do not reach the demo's test body, as on a build whose C library
 * keeps no frame pointers: bpftrace unwinds user stacks through them, so the
 * lock cannot be seen from there. The skip then quotes the stacks.
 */
TEST(OffCpu, CapturesTheLockWait) {
  const std::string NOTE = vernier::bench::offCpuPidNamespaceNote(ownPidNamespaceLink());
  if (!NOTE.empty()) {
    GTEST_SKIP() << NOTE << ": run it natively, or in a container started with --pid=host";
  }
  if (!vernier::bench::profiler_env::isOnPath("bpftrace")) {
    GTEST_SKIP() << "bpftrace is not installed; this test captures the demo's waits with it";
  }
  if (::geteuid() != 0 && !vernier::bench::profiler_env::isOnPath("sudo")) {
    GTEST_SKIP() << "sudo is not installed, and this test runs bpftrace through sudo -n "
                    "(BENCH_SUDO=1) when it is not root";
  }
  const std::string DEMO = demoPath();
  ASSERT_FALSE(DEMO.empty()) << "the demo binary is missing: " << DEMO_BINARY;
  const fs::path DIR = scratchDir();
  ASSERT_FALSE(DIR.empty()) << "cannot create a temporary directory";

  const TracedCase COARSE = traceCase(DEMO, "OffCpu.CoarseLock", DIR / "CoarseLock");
  const std::string CANNOT = check::cannotTraceHere(COARSE.run.output, "offcpu");
  if (!CANNOT.empty()) {
    std::error_code ec;
    fs::remove_all(DIR, ec);
    GTEST_SKIP() << "offcpu could not trace here. The run printed:\n" << CANNOT;
  }
  ASSERT_TRUE(vg::testPassed(COARSE.run.output, "OffCpu.CoarseLock") &&
              vg::oneTestPassed(COARSE.run.output))
      << "CoarseLock did not run to its end under offcpu: the demo " << vg::describe(COARSE.run.end)
      << " (its files are in " << DIR << "). It printed:\n"
      << vg::lastLines(COARSE.run.output, 40);
  const TracedCase SHARED_NOTHING = traceCase(DEMO, "OffCpu.NoSharing", DIR / "NoSharing");
  ASSERT_TRUE(vg::testPassed(SHARED_NOTHING.run.output, "OffCpu.NoSharing") &&
              vg::oneTestPassed(SHARED_NOTHING.run.output))
      << "NoSharing did not run to its end under offcpu: the demo "
      << vg::describe(SHARED_NOTHING.run.end) << ". It printed:\n"
      << vg::lastLines(SHARED_NOTHING.run.output, 40);

  expectAWholeCapture(COARSE, "OffCpu.CoarseLock");
  expectAWholeCapture(SHARED_NOTHING, "OffCpu.NoSharing");
  EXPECT_EQ(COARSE.outcome.rfind("[offcpu] stacks written to ", 0), 0U)
      << "no thread of the coarse-lock case slept:\n"
      << COARSE.outcome;
  std::printf("[OffCpu.CapturesTheLockWait]  CoarseLock: %ld sleeping switch-outs, %ld in the "
              "lock under addUnderCoarseLock; NoSharing: %ld, %ld there\n",
              switchOuts(COARSE.dump), lockWaitsUnderTheCoarseLock(COARSE.dump),
              switchOuts(SHARED_NOTHING.dump), lockWaitsUnderTheCoarseLock(SHARED_NOTHING.dump));

  if (!HasFailure() && switchOutsThrough(COARSE.dump, "OffCpu_CoarseLock_Test::TestBody") == 0) {
    std::error_code ec;
    fs::remove_all(DIR, ec);
    GTEST_SKIP() << "both captures are whole, but bpftrace could not unwind this build's user "
                    "stacks to the demo's test body, where the main thread's own waits are, so "
                    "the lock wait cannot be seen from here. CoarseLock's stacks:\n"
                 << stacksOf(COARSE.output);
  }
  EXPECT_GT(lockWaitsUnderTheCoarseLock(COARSE.dump), 0)
      << "no thread of the coarse-lock case waited in the lock under addUnderCoarseLock:\n"
      << stacksOf(COARSE.output);
  EXPECT_EQ(lockWaitsUnderTheCoarseLock(SHARED_NOTHING.dump), 0)
      << "a thread of the no-sharing case waited in the coarse lock:\n"
      << stacksOf(SHARED_NOTHING.output);

  if (HasFailure()) {
    std::printf("the traced runs' files are kept in %s\n", DIR.c_str());
    return;
  }
  std::error_code ec;
  fs::remove_all(DIR, ec);
}
