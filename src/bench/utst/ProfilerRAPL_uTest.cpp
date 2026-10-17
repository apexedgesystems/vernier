/**
 * @file ProfilerRAPL_uTest.cpp
 * @brief Unit tests for the RAPL backend's readiness decision.
 *
 * The decision reads /proc/cpuinfo and the msr device. checkRaplRequestFrom()
 * points it at files a test makes: a cpuinfo text, and a regular file standing
 * in for /dev/cpu/0/msr, with each register's 8 bytes at its number as offset,
 * as the msr driver serves them. A regular file cannot fail with EIO or EPERM,
 * so raplAccessRefusal(), which gives each errno its cause, is tested over its
 * table on its own. The tests of the run, the direct constructor and the
 * factory decide on this machine's real files and expect a refusal: they skip
 * where this process can read the energy counter. The capture's tests give a
 * profiler a plan naming such a file, and rewrite the counter in it between
 * the hooks.
 */

#include "src/bench/inc/PerfConfig.hpp"
#include "src/bench/inc/PerfStats.hpp"
#include "src/bench/inc/Profiler.hpp"
#include "src/bench/inc/ProfilerRAPL.hpp"
#include "src/bench/inc/ProfilerReadiness.hpp"
#include "src/bench/inc/ProfilerRegistry.hpp"
#include "src/bench/utst/ReadinessFixtures.hpp"
#include "src/bench/utst/StderrCapture.hpp"

#include <gtest/gtest.h>

#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

namespace {

using vernier::bench::checkRaplRequestFrom;
using vernier::bench::EnvReport;
using vernier::bench::makeRAPLProfiler;
using vernier::bench::PerfConfig;
using vernier::bench::ProfileFailure;
using vernier::bench::ProfilerRegistry;
using vernier::bench::raplAccessRefusal;
using vernier::bench::RaplPlan;
using vernier::bench::RAPLProfiler;
using vernier::bench::RaplSources;
using vernier::bench::ReadinessCause;
using vernier::bench::ReadinessRequest;
using vernier::bench::ReadinessResult;
using vernier::bench::ReadinessScope;
using vernier::bench::ReadinessStage;
using vernier::bench::Stats;
using vernier::bench::test::FakeToolDir;
using vernier::bench::test::StderrCapture;

/* ----------------------------- Fixtures ----------------------------- */

/// The first CPU in the /proc/cpuinfo of the Intel laptop walkthrough 08 was written on.
constexpr const char* INTEL_CPUINFO = "processor\t: 0\n"
                                      "vendor_id\t: GenuineIntel\n"
                                      "cpu family\t: 6\n"
                                      "model\t\t: 170\n"
                                      "model name\t: Intel(R) Core(TM) Ultra 7 165H\n"
                                      "stepping\t: 4\n"
                                      "\n"
                                      "processor\t: 1\n"
                                      "vendor_id\t: GenuineIntel\n";

/// A Raspberry Pi 4's first CPU: Linux on ARM lists no vendor_id.
constexpr const char* ARM_CPUINFO = "processor\t: 0\n"
                                    "BogoMIPS\t: 108.00\n"
                                    "Features\t: fp asimd evtstrm crc32 cpuid\n"
                                    "CPU implementer\t: 0x41\n"
                                    "CPU architecture: 8\n"
                                    "CPU variant\t: 0x0\n"
                                    "CPU part\t: 0xd08\n"
                                    "CPU revision\t: 3\n"
                                    "\n";

/// An AMD CPU's first lines.
constexpr const char* AMD_CPUINFO = "processor\t: 0\n"
                                    "vendor_id\t: AuthenticAMD\n"
                                    "cpu family\t: 25\n"
                                    "model\t\t: 33\n"
                                    "\n";

constexpr std::uint32_t MSR_POWER_UNIT = 0x606;
constexpr std::uint32_t MSR_PKG_ENERGY = 0x611;

/// MSR 0x606 as Intel CPUs report it: power unit 3, energy-status unit 14
/// (bits 12:8, so one count is 2^-14 J) and time unit 10.
constexpr std::uint64_t POWER_UNIT = 0x000A0E03;

/// A device file long enough for both registers.
constexpr std::size_t DEVICE_SIZE = MSR_PKG_ENERGY + 8;

/**
 * @brief Write a regular file standing in for an msr device: each register's
 * 8 bytes (little-endian, as the driver returns them) at its number as
 * offset, in a file @p size bytes long, zeros elsewhere.
 */
bool writeDevice(const std::string& path, const std::map<std::uint32_t, std::uint64_t>& registers,
                 std::size_t size) {
  std::vector<unsigned char> bytes(size, 0);
  for (const auto& [MSR, VALUE] : registers) {
    for (std::size_t i = 0; i < 8 && MSR + i < size; ++i) {
      bytes[MSR + i] = static_cast<unsigned char>((VALUE >> (8 * i)) & 0xFF);
    }
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(size));
  return static_cast<bool>(out);
}

/** @brief A rapl request as the command line states it. */
ReadinessRequest raplRequest(const std::string& profileArgs = "", bool analyze = false,
                             ReadinessScope scope = ReadinessScope::RUNTIME) {
  ReadinessRequest request;
  request.backend = "rapl";
  request.profileArgs = profileArgs;
  request.analyze = analyze;
  request.scope = scope;
  return request;
}

/** @brief The cause word a result's message starts with. */
bool startsWith(const std::string& text, const std::string& prefix) {
  return text.compare(0, prefix.size(), prefix) == 0;
}

/**
 * @brief Decisions against a cpuinfo text and a device file of this test's
 * own, in a private directory.
 */
class RaplCheckTest : public ::testing::Test {
protected:
  void SetUp() override { ASSERT_TRUE(dir_.ok()); }

  /** @brief The sources: @p cpuinfo as the CPU description, the directory's msr file. */
  RaplSources sources(const std::string& cpuinfo = INTEL_CPUINFO) const {
    RaplSources out;
    out.cpuinfo = dir_.writeFile("cpuinfo", cpuinfo);
    out.device = dir_.path() + "/msr";
    return out;
  }

  /** @brief The directory's msr file, holding both registers. */
  void writeReadableDevice(std::uint64_t energy = 0x12345678) const {
    ASSERT_TRUE(writeDevice(dir_.path() + "/msr",
                            {{MSR_POWER_UNIT, POWER_UNIT}, {MSR_PKG_ENERGY, energy}}, DEVICE_SIZE));
  }

  FakeToolDir dir_;
};

} // namespace

/* ----------------------------- The Check ----------------------------- */

/** @test Both registers read: ready, with the unit decoded from MSR 0x606 and the CPU named. */
TEST_F(RaplCheckTest, ReadsBothRegistersAndDecodesTheUnit) {
  writeReadableDevice();
  const RaplSources SOURCES = sources();

  const ReadinessResult RESULT = checkRaplRequestFrom(raplRequest(), SOURCES);

  EXPECT_EQ(RESULT.cause, ReadinessCause::READY) << RESULT.report.message;
  EXPECT_EQ(RESULT.report.status, EnvReport::Status::Ok);
  EXPECT_EQ(RESULT.report.message, "read package 0's energy counter (MSR 0x611) through " +
                                       SOURCES.device +
                                       "; one count is 61.035 uJ (2^-14 J, from MSR 0x606)");
  const auto PLAN = std::dynamic_pointer_cast<const RaplPlan>(RESULT.plan);
  ASSERT_NE(PLAN, nullptr) << "a ready decision carries a RaplPlan";
  EXPECT_EQ(PLAN->device, SOURCES.device);
  EXPECT_EQ(PLAN->energyUnitJoules, 1.0 / 16384.0);
  EXPECT_EQ(PLAN->cpuModel, "Intel(R) Core(TM) Ultra 7 165H");
  EXPECT_EQ(PLAN->cpuFamily, 6);
  EXPECT_EQ(PLAN->cpuModelNumber, 170);
}

/** @test A mode word is refused before anything is read: rapl takes none. */
TEST_F(RaplCheckTest, ModeWordIsAConfigurationError) {
  const ReadinessResult RESULT = checkRaplRequestFrom(raplRequest("dram"), sources());

  EXPECT_EQ(RESULT.cause, ReadinessCause::CONFIGURATION);
  EXPECT_EQ(RESULT.report.message, "configuration: 'dram' is not a mode of rapl, which takes none");
  EXPECT_FALSE(RESULT.collectionReady());
}

/** @test A CPU whose cpuinfo names no vendor (ARM) is unsupported, with no remedy. */
TEST_F(RaplCheckTest, ArmCpuIsUnsupported) {
  writeReadableDevice();
  const RaplSources SOURCES = sources(ARM_CPUINFO);

  const ReadinessResult RESULT = checkRaplRequestFrom(raplRequest(), SOURCES);

  EXPECT_EQ(RESULT.cause, ReadinessCause::UNSUPPORTED);
  EXPECT_EQ(RESULT.report.message, "unsupported: rapl reads Intel's RAPL energy registers, and " +
                                       SOURCES.cpuinfo + " names no CPU vendor here");
  EXPECT_EQ(RESULT.report.hint, "") << "no remedy makes an ARM CPU read Intel's registers";
  EXPECT_FALSE(RESULT.collectionReady());
}

/** @test A CPU of another vendor is unsupported, naming the vendor. */
TEST_F(RaplCheckTest, OtherVendorIsUnsupported) {
  writeReadableDevice();

  const ReadinessResult RESULT = checkRaplRequestFrom(raplRequest(), sources(AMD_CPUINFO));

  EXPECT_EQ(RESULT.cause, ReadinessCause::UNSUPPORTED);
  EXPECT_EQ(RESULT.report.message,
            "unsupported: rapl reads Intel's RAPL energy registers, and this CPU's vendor is "
            "AuthenticAMD");
}

/** @test No device: missing, with the driver and the container's device as the remedy. */
TEST_F(RaplCheckTest, MissingDeviceIsMissing) {
  const RaplSources SOURCES = sources();

  const ReadinessResult RESULT = checkRaplRequestFrom(raplRequest(), SOURCES);

  EXPECT_EQ(RESULT.cause, ReadinessCause::MISSING);
  EXPECT_EQ(RESULT.report.message, "missing: cannot open " + SOURCES.device +
                                       ": No such file or directory; the msr driver provides it");
  EXPECT_NE(RESULT.report.hint.find("sudo modprobe msr"), std::string::npos) << RESULT.report.hint;
  EXPECT_NE(RESULT.report.hint.find("--device /dev/cpu/0/msr"), std::string::npos)
      << RESULT.report.hint;
}

/** @test A device this user may not open: denied, with both requirements as the remedy. */
TEST_F(RaplCheckTest, UnreadableDeviceIsDenied) {
  if (::geteuid() == 0) {
    GTEST_SKIP() << "root opens a file whatever its mode, so this refusal cannot happen here";
  }
  writeReadableDevice();
  const RaplSources SOURCES = sources();
  ASSERT_EQ(::chmod(SOURCES.device.c_str(), 0), 0) << std::strerror(errno);

  const ReadinessResult RESULT = checkRaplRequestFrom(raplRequest(), SOURCES);

  EXPECT_EQ(RESULT.cause, ReadinessCause::DENIED);
  EXPECT_EQ(RESULT.report.message, "denied: cannot open " + SOURCES.device + ": Permission denied");
  for (const char* PART : {"root", "CAP_SYS_RAWIO", "read access to /dev/cpu/0/msr",
                           "does not raise its privileges"}) {
    EXPECT_NE(RESULT.report.hint.find(PART), std::string::npos)
        << "the remedy does not say '" << PART << "': " << RESULT.report.hint;
  }
}

/** @test A device that ends before the power-unit register: a short read, unusable. */
TEST_F(RaplCheckTest, ShortDeviceIsUnusable) {
  ASSERT_TRUE(writeDevice(dir_.path() + "/msr", {}, 8));
  const RaplSources SOURCES = sources();

  const ReadinessResult RESULT = checkRaplRequestFrom(raplRequest(), SOURCES);

  EXPECT_EQ(RESULT.cause, ReadinessCause::UNUSABLE);
  EXPECT_EQ(RESULT.report.message,
            "unusable: read MSR 0x606 from " + SOURCES.device + " returned 0 of 8 bytes");
}

/** @test The check reads the energy counter it will use, not only the unit register. */
TEST_F(RaplCheckTest, EnergyCounterIsRead) {
  ASSERT_TRUE(
      writeDevice(dir_.path() + "/msr", {{MSR_POWER_UNIT, POWER_UNIT}}, MSR_POWER_UNIT + 8));
  const RaplSources SOURCES = sources();

  const ReadinessResult RESULT = checkRaplRequestFrom(raplRequest(), SOURCES);

  EXPECT_EQ(RESULT.cause, ReadinessCause::UNUSABLE);
  EXPECT_EQ(RESULT.report.message,
            "unusable: read MSR 0x611 from " + SOURCES.device + " returned 0 of 8 bytes");
}

/** @test A directory where the device should be: its read fails, unusable, with the error. */
TEST_F(RaplCheckTest, DirectoryDeviceIsUnusable) {
  const RaplSources SOURCES = sources();
  dir_.makeDirectory("msr");

  const ReadinessResult RESULT = checkRaplRequestFrom(raplRequest(), SOURCES);

  EXPECT_EQ(RESULT.cause, ReadinessCause::UNUSABLE);
  EXPECT_EQ(RESULT.report.message,
            "unusable: cannot read MSR 0x606 from " + SOURCES.device + ": Is a directory");
}

/** @test A cpuinfo that cannot be read: unusable, with the error. */
TEST_F(RaplCheckTest, UnreadableCpuinfoIsUnusable) {
  RaplSources sources;
  sources.cpuinfo = dir_.path() + "/no-cpuinfo";
  sources.device = dir_.path() + "/msr";

  const ReadinessResult RESULT = checkRaplRequestFrom(raplRequest(), sources);

  EXPECT_EQ(RESULT.cause, ReadinessCause::UNUSABLE);
  EXPECT_EQ(RESULT.report.message,
            "unusable: cannot read " + sources.cpuinfo + ": No such file or directory");
}

/**
 * @test --profile-analyze on a request that can collect: an analysis-stage
 * error that keeps the plan, so the capture runs and the run fails at its end.
 */
TEST_F(RaplCheckTest, AnalysisFailsAtItsStageAndStillCollects) {
  writeReadableDevice();

  const ReadinessResult RESULT = checkRaplRequestFrom(raplRequest("", /*analyze=*/true), sources());

  EXPECT_EQ(RESULT.report.status, EnvReport::Status::Error);
  EXPECT_EQ(RESULT.stage, ReadinessStage::ANALYSIS);
  EXPECT_EQ(RESULT.cause, ReadinessCause::UNSUPPORTED);
  EXPECT_TRUE(RESULT.collectionReady());
  EXPECT_TRUE(startsWith(RESULT.report.message,
                         "analysis: unsupported: --profile-analyze: rapl has no automatic "
                         "analysis"))
      << RESULT.report.message;
  EXPECT_NE(std::dynamic_pointer_cast<const RaplPlan>(RESULT.plan), nullptr)
      << "the capture keeps its plan";
}

/** @test --profile-analyze on a request that cannot collect keeps the collection's refusal. */
TEST_F(RaplCheckTest, AnalysisKeepsACollectionRefusal) {
  const ReadinessResult RESULT = checkRaplRequestFrom(raplRequest("", /*analyze=*/true), sources());

  EXPECT_EQ(RESULT.cause, ReadinessCause::MISSING);
  EXPECT_EQ(RESULT.stage, ReadinessStage::COLLECTION);
}

/* ----------------------------- Causes ----------------------------- */

/** @test Each errno of an open or a read has its cause, and the message quotes it. */
TEST(RaplAccessRefusalTest, EachErrnoHasItsCause) {
  const std::string OPEN = "open /dev/cpu/0/msr";
  const std::string READ = "read MSR 0x611 from /dev/cpu/0/msr";
  struct Case {
    std::string step;
    int err;
    ReadinessCause cause;
    std::string prefix;
  };
  const std::vector<Case> CASES = {
      {OPEN, EACCES, ReadinessCause::DENIED, "denied: "},
      {OPEN, EPERM, ReadinessCause::DENIED, "denied: "},
      {OPEN, ENOENT, ReadinessCause::MISSING, "missing: "},
      {OPEN, ENODEV, ReadinessCause::MISSING, "missing: "},
      {OPEN, EIO, ReadinessCause::UNSUPPORTED, "unsupported: "},
      {READ, EIO, ReadinessCause::UNSUPPORTED, "unsupported: "},
      {OPEN, ENXIO, ReadinessCause::UNUSABLE, "unusable: "},
      {OPEN, EBUSY, ReadinessCause::UNUSABLE, "unusable: "},
      {READ, EINVAL, ReadinessCause::UNUSABLE, "unusable: "},
      {READ, EFAULT, ReadinessCause::UNUSABLE, "unusable: "},
      {READ, EISDIR, ReadinessCause::UNUSABLE, "unusable: "},
  };
  for (const Case& C : CASES) {
    const ReadinessResult RESULT = raplAccessRefusal(C.step, C.err);
    EXPECT_EQ(RESULT.cause, C.cause) << C.step << ": " << std::strerror(C.err);
    EXPECT_EQ(RESULT.report.status, EnvReport::Status::Error);
    EXPECT_TRUE(startsWith(RESULT.report.message,
                           C.prefix + "cannot " + C.step + ": " + std::strerror(C.err)))
        << RESULT.report.message;
  }
  EXPECT_NE(raplAccessRefusal(OPEN, EIO).report.message.find("no model-specific registers"),
            std::string::npos);
  EXPECT_NE(raplAccessRefusal(READ, EIO).report.message.find("does not implement that register"),
            std::string::npos);
}

/* ----------------------------- The Run on This Machine ----------------------------- */

namespace {

/**
 * @brief The registry's decisions and failures, cleared around each test; a
 * private output root for anything a profiler might create.
 */
class RaplRunTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(dir_.ok());
    ProfilerRegistry::instance().resetFailures();
    ProfilerRegistry::instance().resetReadiness();
    here_ = ProfilerRegistry::instance().checkRequest(raplRequest());
    if (here_.collectionReady()) {
      GTEST_SKIP() << "this process can read the energy counter here (" << here_.report.message
                   << "), so the refusal path cannot be checked";
    }
  }

  void TearDown() override {
    ProfilerRegistry::instance().resetFailures();
    ProfilerRegistry::instance().resetReadiness();
  }

  /** @brief A rapl request writing under this test's directory. */
  PerfConfig config() const {
    PerfConfig cfg;
    cfg.profileTool = "rapl";
    cfg.artifactRoot = dir_.path();
    cfg.profileTestTimeoutSecs = 0;
    return cfg;
  }

  /** @brief Everything the profiler left in the output root. */
  std::vector<std::string> outputRoot() const {
    std::vector<std::string> out;
    std::error_code ec;
    for (const auto& ENTRY : std::filesystem::directory_iterator(dir_.path(), ec)) {
      out.push_back(ENTRY.path().filename().string());
    }
    return out;
  }

  /** @brief The failures the run recorded, as "<backend>|<test>|<message>". */
  static std::vector<std::string> failures() {
    std::vector<std::string> out;
    for (const ProfileFailure& F : ProfilerRegistry::instance().failures()) {
      out.push_back(F.backend + "|" + F.test + "|" + F.result.report.message);
    }
    return out;
  }

  FakeToolDir dir_;
  ReadinessResult here_;
};

} // namespace

/** @test The registry decides rapl requests with RAPL's own check: a mode word is refused. */
TEST_F(RaplRunTest, RegistryRefusesAModeWord) {
  const ReadinessResult RESULT = ProfilerRegistry::instance().checkRequest(
      raplRequest("dram", false, ReadinessScope::PREFLIGHT));

  EXPECT_EQ(RESULT.cause, ReadinessCause::CONFIGURATION) << RESULT.report.message;
  EXPECT_EQ(RESULT.report.status, EnvReport::Status::Error);
}

/** @test Where the counter cannot be read, the refusal names why: denied, missing or unsupported.
 */
TEST_F(RaplRunTest, RefusalHereNamesItsCause) {
  const std::string& MESSAGE = here_.report.message;

  EXPECT_EQ(here_.report.status, EnvReport::Status::Error);
  EXPECT_EQ(here_.stage, ReadinessStage::COLLECTION);
  EXPECT_TRUE(startsWith(MESSAGE, "denied: ") || startsWith(MESSAGE, "missing: ") ||
              startsWith(MESSAGE, "unsupported: "))
      << "a refusal on a machine without access names its cause: " << MESSAGE;
}

/**
 * @test The run prints the doctor's row as its notice, records the refusal
 * once and creates no folder.
 */
TEST_F(RaplRunTest, RunNoticeIsTheDoctorRow) {
  StderrCapture capture;
  const auto PROFILER = ProfilerRegistry::instance().make("rapl", config(), "RaplRunTest.Notice");
  const std::string PRINTED = capture.text();

  ASSERT_NE(PROFILER, nullptr);
  EXPECT_EQ(PROFILER->artifactDir(), "");
  EXPECT_NE(PRINTED.find("[FAIL] Profiler 'rapl': " + here_.report.message), std::string::npos)
      << PRINTED;
  EXPECT_EQ(failures(), std::vector<std::string>{"rapl||" + here_.report.message});
  EXPECT_EQ(outputRoot(), std::vector<std::string>{});
}

/** @test The direct constructor reports the refusal as the registry does and creates no folder. */
TEST_F(RaplRunTest, DirectConstructorRecordsTheRefusal) {
  StderrCapture capture;
  const RAPLProfiler PROFILER(config(), "RaplRunTest.Direct");
  const std::string PRINTED = capture.text();

  EXPECT_EQ(PROFILER.artifactDir(), "");
  EXPECT_NE(PRINTED.find("[FAIL] Profiler 'rapl': " + here_.report.message), std::string::npos)
      << PRINTED;
  EXPECT_EQ(failures(), std::vector<std::string>{"rapl||" + here_.report.message});
  EXPECT_EQ(outputRoot(), std::vector<std::string>{});
}

/** @test The factory reports the refusal as the registry does, returns nothing and creates no
 * folder. */
TEST_F(RaplRunTest, FactoryRecordsTheRefusal) {
  StderrCapture capture;
  const auto PROFILER = makeRAPLProfiler(config(), "RaplRunTest.Factory");
  (void)capture.text();

  EXPECT_EQ(PROFILER, nullptr);
  EXPECT_EQ(failures(), std::vector<std::string>{"rapl||" + here_.report.message});
  EXPECT_EQ(outputRoot(), std::vector<std::string>{});
}

/* ----------------------------- The Capture ----------------------------- */

namespace {

/**
 * @brief A profiler built from a plan that names this test's device file,
 * whose counter a test sets before each hook; its report goes under this
 * test's directory.
 */
class RaplCaptureTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(dir_.ok());
    ProfilerRegistry::instance().resetFailures();
  }

  void TearDown() override { ProfilerRegistry::instance().resetFailures(); }

  /** @brief The plan the check would give for this test's device: one count is 2^-14 J. */
  std::shared_ptr<const RaplPlan> plan() const {
    auto out = std::make_shared<RaplPlan>();
    out->device = device();
    out->energyUnitJoules = 1.0 / 16384.0;
    out->cpuModel = "Intel(R) Core(TM) Ultra 7 165H";
    out->cpuFamily = 6;
    out->cpuModelNumber = 170;
    return out;
  }

  /** @brief A rapl request writing under @p root (this test's out/ by default). */
  PerfConfig config(const std::string& root = "") const {
    PerfConfig cfg;
    cfg.profileTool = "rapl";
    cfg.artifactRoot = root.empty() ? dir_.path() + "/out" : root;
    return cfg;
  }

  std::string device() const { return dir_.path() + "/msr"; }

  /** @brief The device file, its package counter holding @p counter. */
  void setCounter(std::uint64_t counter) const {
    ASSERT_TRUE(writeDevice(device(), {{MSR_POWER_UNIT, POWER_UNIT}, {MSR_PKG_ENERGY, counter}},
                            DEVICE_SIZE));
  }

  /** @brief The case's report; "" when there is none. */
  std::string report() const {
    std::ifstream in(dir_.path() + "/out/RaplCaptureTest.Case.rapl/energy.txt");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  }

  /** @brief A window's statistics with @p medianUs as its median call time. */
  static Stats window(double medianUs) {
    Stats stats;
    stats.median = medianUs;
    return stats;
  }

  /** @brief The failures the run recorded, as "<test>|<stage>|<message>". */
  static std::vector<std::string> failures() {
    std::vector<std::string> out;
    for (const ProfileFailure& F : ProfilerRegistry::instance().failures()) {
      out.push_back(F.test + "|" + std::to_string(static_cast<int>(F.result.stage)) + "|" +
                    F.result.report.message);
    }
    return out;
  }

  FakeToolDir dir_;
};

/** @brief How many times @p text holds @p part. */
int count(const std::string& text, const std::string& part) {
  int n = 0;
  for (std::size_t at = text.find(part); at != std::string::npos; at = text.find(part, at + 1)) {
    ++n;
  }
  return n;
}

/** @brief The number after "<field>: " in @p text's first line holding it; -1 when none. */
double field(const std::string& text, const std::string& name) {
  const std::size_t AT = text.find("\n" + name + ": ");
  return AT == std::string::npos ? -1.0 : std::strtod(text.c_str() + AT + name.size() + 3, nullptr);
}

const std::string COLLECTION = std::to_string(static_cast<int>(ReadinessStage::COLLECTION));
const std::string COMPLETION = std::to_string(static_cast<int>(ReadinessStage::COMPLETION));

} // namespace

/** @test A start reading of zero is a reading: the window is measured from it. */
TEST_F(RaplCaptureTest, ZeroStartIsMeasured) {
  RAPLProfiler profiler(config(), "RaplCaptureTest.Case", plan());
  setCounter(0);
  profiler.beforeMeasure();
  setCounter(16384);
  testing::internal::CaptureStdout();
  profiler.afterMeasure(window(10.0));
  (void)testing::internal::GetCapturedStdout();

  const std::string REPORT = report();
  EXPECT_NE(REPORT.find("\nStart counter: 0\n"), std::string::npos) << REPORT;
  EXPECT_NE(REPORT.find("\nDelta counter: 16384\n"), std::string::npos) << REPORT;
  EXPECT_NE(REPORT.find("\nEnergy consumed: 1.000000 J\n"), std::string::npos) << REPORT;
  EXPECT_EQ(failures(), std::vector<std::string>{});
}

/** @test The 32-bit counter wrapping once in the window gives the right delta. */
TEST_F(RaplCaptureTest, OneWrapIsTheDelta) {
  RAPLProfiler profiler(config(), "RaplCaptureTest.Case", plan());
  setCounter(0xABCD0000FFFFFF00ULL); // reserved upper bits set: not the counter's
  profiler.beforeMeasure();
  setCounter(0x0000000000000100ULL);
  testing::internal::CaptureStdout();
  profiler.afterMeasure(window(10.0));
  (void)testing::internal::GetCapturedStdout();

  const std::string REPORT = report();
  EXPECT_NE(REPORT.find("\nStart counter: 4294967040\n"), std::string::npos) << REPORT;
  EXPECT_NE(REPORT.find("\nEnd counter: 256\n"), std::string::npos) << REPORT;
  EXPECT_NE(REPORT.find("\nDelta counter: 512\n"), std::string::npos) << REPORT;
}

/** @test A failed start reading fails the run with its cause, and the window has no report. */
TEST_F(RaplCaptureTest, StartReadFailureFailsTheRun) {
  RAPLProfiler profiler(config(), "RaplCaptureTest.Case", plan());
  ASSERT_TRUE(writeDevice(device(), {{MSR_POWER_UNIT, POWER_UNIT}}, MSR_POWER_UNIT + 8));
  StderrCapture capture;
  profiler.beforeMeasure();
  setCounter(16384);
  testing::internal::CaptureStdout();
  profiler.afterMeasure(window(10.0));
  const std::string PRINTED = testing::internal::GetCapturedStdout();
  (void)capture.text();

  EXPECT_EQ(failures(), std::vector<std::string>{"RaplCaptureTest.Case|" + COLLECTION +
                                                 "|unusable: window 1's start: read MSR 0x611 "
                                                 "from " +
                                                 device() + " returned 0 of 8 bytes"});
  EXPECT_EQ(report(), "") << "a window without a start has no section";
  EXPECT_EQ(PRINTED.find("=== Energy"), std::string::npos) << PRINTED;
}

/** @test A failed end reading fails the run with its cause, and the window has no report. */
TEST_F(RaplCaptureTest, EndReadFailureFailsTheRun) {
  RAPLProfiler profiler(config(), "RaplCaptureTest.Case", plan());
  setCounter(100);
  profiler.beforeMeasure();
  ASSERT_TRUE(std::filesystem::remove(device()));
  StderrCapture capture;
  testing::internal::CaptureStdout();
  profiler.afterMeasure(window(10.0));
  const std::string PRINTED = testing::internal::GetCapturedStdout();
  (void)capture.text();

  EXPECT_EQ(failures(),
            std::vector<std::string>{"RaplCaptureTest.Case|" + COLLECTION +
                                     "|missing: window 1's end: cannot open " + device() +
                                     ": No such file or directory; the msr driver "
                                     "provides it"});
  EXPECT_EQ(report(), "");
  EXPECT_EQ(PRINTED.find("=== Energy"), std::string::npos) << PRINTED;
}

/**
 * @test A report that cannot be written (the output root is a regular file)
 * fails the run at the completion stage, naming the file; no summary is
 * printed for it.
 */
TEST_F(RaplCaptureTest, UnwritableReportFailsTheRun) {
  const std::string ROOT = dir_.writeFile("not-a-folder", "a regular file\n");
  RAPLProfiler profiler(config(ROOT), "RaplCaptureTest.Case", plan());
  setCounter(100);
  profiler.beforeMeasure();
  setCounter(200);
  StderrCapture capture;
  testing::internal::CaptureStdout();
  profiler.afterMeasure(window(10.0));
  const std::string PRINTED = testing::internal::GetCapturedStdout();
  (void)capture.text();

  EXPECT_EQ(failures(),
            std::vector<std::string>{"RaplCaptureTest.Case|" + COMPLETION +
                                     "|completion: unusable: window 1's energy report: cannot "
                                     "open " +
                                     ROOT +
                                     "/RaplCaptureTest.Case.rapl/energy.txt for writing: Not a "
                                     "directory"});
  EXPECT_EQ(PRINTED.find("=== Energy"), std::string::npos)
      << "a summary announced a report that was not written:\n"
      << PRINTED;
}

/** @test The summary comes after the report is written, names it, and gives no hints. */
TEST_F(RaplCaptureTest, SummaryNamesTheWrittenReport) {
  RAPLProfiler profiler(config(), "RaplCaptureTest.Case", plan());
  setCounter(100);
  profiler.beforeMeasure();
  setCounter(16484);
  testing::internal::CaptureStdout();
  profiler.afterMeasure(window(10.0));
  const std::string PRINTED = testing::internal::GetCapturedStdout();

  EXPECT_NE(PRINTED.find("Report:              " + dir_.path() +
                         "/out/RaplCaptureTest.Case.rapl/energy.txt\n"),
            std::string::npos)
      << PRINTED;
  EXPECT_NE(report(), "");
  EXPECT_EQ(PRINTED.find("Hint"), std::string::npos) << PRINTED;
  EXPECT_EQ(PRINTED.find("DRAM"), std::string::npos) << PRINTED;
  EXPECT_EQ(failures(), std::vector<std::string>{});
}

/** @test A window over which the counter did not advance is reported as such, not as a failure. */
TEST_F(RaplCaptureTest, ZeroDeltaIsStated) {
  RAPLProfiler profiler(config(), "RaplCaptureTest.Case", plan());
  setCounter(777);
  profiler.beforeMeasure();
  testing::internal::CaptureStdout();
  profiler.afterMeasure(window(10.0));
  const std::string PRINTED = testing::internal::GetCapturedStdout();

  const std::string REPORT = report();
  EXPECT_NE(REPORT.find("\nDelta counter: 0\n"), std::string::npos) << REPORT;
  EXPECT_NE(REPORT.find("\nNote: the counter did not advance during this "), std::string::npos)
      << REPORT;
  EXPECT_NE(PRINTED.find("did not advance"), std::string::npos) << PRINTED;
  EXPECT_EQ(failures(), std::vector<std::string>{});
}

/** @test Each measured window is a numbered section of one report, in order. */
TEST_F(RaplCaptureTest, EachWindowHasItsSection) {
  RAPLProfiler profiler(config(), "RaplCaptureTest.Case", plan());
  testing::internal::CaptureStdout();
  setCounter(1000);
  profiler.beforeMeasure();
  setCounter(2000);
  profiler.afterMeasure(window(10.0));
  setCounter(5000);
  profiler.beforeMeasure();
  setCounter(9000);
  profiler.afterMeasure(window(20.0));
  (void)testing::internal::GetCapturedStdout();

  const std::string REPORT = report();
  EXPECT_EQ(count(REPORT, "=== RAPL Energy Measurement ==="), 1) << REPORT;
  const std::size_t FIRST = REPORT.find("\n--- Window 1 ---\n");
  const std::size_t SECOND = REPORT.find("\n--- Window 2 ---\n");
  ASSERT_NE(FIRST, std::string::npos) << REPORT;
  ASSERT_NE(SECOND, std::string::npos) << REPORT;
  EXPECT_LT(FIRST, SECOND);
  EXPECT_NE(REPORT.find("\nDelta counter: 1000\n", FIRST), std::string::npos) << REPORT;
  EXPECT_NE(REPORT.find("\nStart counter: 5000\n", SECOND), std::string::npos) << REPORT;
  EXPECT_NE(REPORT.find("\nDelta counter: 4000\n", SECOND), std::string::npos) << REPORT;
}

/**
 * @test Each window starts from its own reading: after a window ended by an
 * exception (no after hook), a start that cannot be read does not fall back
 * on the aborted window's.
 */
TEST_F(RaplCaptureTest, StartIsTheWindowsOwn) {
  RAPLProfiler profiler(config(), "RaplCaptureTest.Case", plan());
  setCounter(100);
  profiler.beforeMeasure();
  profiler.abortMeasure();
  ASSERT_TRUE(writeDevice(device(), {{MSR_POWER_UNIT, POWER_UNIT}}, MSR_POWER_UNIT + 8));
  StderrCapture capture;
  profiler.beforeMeasure();
  setCounter(300);
  testing::internal::CaptureStdout();
  profiler.afterMeasure(window(10.0));
  (void)testing::internal::GetCapturedStdout();
  (void)capture.text();

  EXPECT_EQ(report(), "") << "the window was measured from the aborted window's start";
  EXPECT_EQ(failures().size(), 1U);
}

/**
 * @test The report states its scope, the counter's wrap and the per-call
 * definition, and its numbers follow them: energy is the delta in units,
 * power the energy over the duration, energy per call the power times the
 * median call time.
 */
TEST_F(RaplCaptureTest, ReportFollowsItsDefinitions) {
  RAPLProfiler profiler(config(), "RaplCaptureTest.Case", plan());
  setCounter(1000);
  profiler.beforeMeasure();
  setCounter(1000 + 3 * 16384);
  testing::internal::CaptureStdout();
  profiler.afterMeasure(window(250.0));
  (void)testing::internal::GetCapturedStdout();

  const std::string REPORT = report();
  EXPECT_NE(REPORT.find("\nScope: package 0, read through " + device() + " (MSR 0x611): "),
            std::string::npos)
      << REPORT;
  EXPECT_NE(REPORT.find("\nCounter: 32 bits, wrapping after 262144.000 J at this unit; "),
            std::string::npos)
      << REPORT;
  EXPECT_NE(REPORT.find("\nEnergy per operation: the window's average package power x its "
                        "median time per call\n"),
            std::string::npos)
      << REPORT;
  EXPECT_EQ(REPORT.find("DRAM"), std::string::npos) << REPORT;

  const std::size_t WINDOW = REPORT.find("\n--- Window 1 ---\n");
  ASSERT_NE(WINDOW, std::string::npos) << REPORT;
  const std::string SECTION = REPORT.substr(WINDOW);
  const double ENERGY = field(SECTION, "Energy consumed");
  const double DURATION = field(SECTION, "Duration");
  const double POWER = field(SECTION, "Average power");
  const double PER_CALL = field(SECTION, "Energy per operation");
  EXPECT_EQ(ENERGY, 3.0);
  EXPECT_EQ(field(SECTION, "Median time per call"), 250.0);
  EXPECT_EQ(field(SECTION, "Energy unit"), 61.0352);
  ASSERT_GT(DURATION, 0.0) << REPORT;
  EXPECT_NEAR(POWER, ENERGY / DURATION, 1e-6 + POWER * 1e-6) << REPORT;
  EXPECT_NEAR(PER_CALL, POWER * 250.0 * 1e-3, 1e-6 + PER_CALL * 1e-5) << REPORT;
}
