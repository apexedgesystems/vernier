/**
 * @file PerfConfig_uTest.cpp
 * @brief Unit tests for vernier::bench::PerfConfig and parsePerfFlags().
 *
 * Tests default values, flag parsing, and quick mode behavior.
 */

#include "src/bench/inc/PerfConfig.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

using vernier::bench::parsePerfFlags;
using vernier::bench::PerfConfig;

namespace {

/** @brief Helper to create argc/argv from strings. */
class ArgvBuilder {
public:
  explicit ArgvBuilder(std::initializer_list<const char*> args) {
    for (const char* arg : args) {
      storage_.emplace_back(arg);
    }
    for (auto& s : storage_) {
      argv_.push_back(const_cast<char*>(s.c_str()));
    }
    argc_ = static_cast<int>(argv_.size());
  }

  int* argc() { return &argc_; }
  char** argv() { return argv_.data(); }

private:
  std::vector<std::string> storage_;
  std::vector<char*> argv_;
  int argc_ = 0;
};

} // namespace

/* ----------------------------- Default Values Tests ----------------------------- */

/** @test Default config has expected values. */
TEST(PerfConfigTest, DefaultValues) {
  const PerfConfig CFG;

  EXPECT_EQ(CFG.cycles, 10000);
  EXPECT_EQ(CFG.repeats, 10);
  EXPECT_EQ(CFG.warmup, 1);
  EXPECT_EQ(CFG.threads, 1);
  EXPECT_EQ(CFG.msgBytes, 64);
  EXPECT_FALSE(CFG.console);
  EXPECT_FALSE(CFG.nonBlocking);
  EXPECT_EQ(CFG.minLevel, "INFO");
  EXPECT_FALSE(CFG.csv.has_value());
  EXPECT_TRUE(CFG.profileTool.empty());
  EXPECT_TRUE(CFG.profileArgs.empty());
  EXPECT_TRUE(CFG.bpfScripts.empty());
  EXPECT_TRUE(CFG.artifactRoot.empty());
  EXPECT_FALSE(CFG.quickMode);
}

/* ----------------------------- Flag Parsing Tests ----------------------------- */

/** @test Parse --cycles flag. */
TEST(PerfConfigTest, ParseCyclesFlag) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--cycles", "5000"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  EXPECT_EQ(cfg.cycles, 5000);
}

/** @test Parse --repeats flag. */
TEST(PerfConfigTest, ParseRepeatsFlag) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--repeats", "20"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  EXPECT_EQ(cfg.repeats, 20);
}

/** @test Parse --warmup flag. */
TEST(PerfConfigTest, ParseWarmupFlag) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--warmup", "3"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  EXPECT_EQ(cfg.warmup, 3);
}

/** @test Parse --threads flag. */
TEST(PerfConfigTest, ParseThreadsFlag) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--threads", "8"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  EXPECT_EQ(cfg.threads, 8);
}

/** @test Parse --msg-bytes flag. */
TEST(PerfConfigTest, ParseMsgBytesFlag) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--msg-bytes", "1024"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  EXPECT_EQ(cfg.msgBytes, 1024);
}

/** @test Parse --console flag. */
TEST(PerfConfigTest, ParseConsoleFlag) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--console"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  EXPECT_TRUE(cfg.console);
}

/** @test Parse --nonblocking flag. */
TEST(PerfConfigTest, ParseNonblockingFlag) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--nonblocking"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  EXPECT_TRUE(cfg.nonBlocking);
}

/** @test Parse --min-level flag. */
TEST(PerfConfigTest, ParseMinLevelFlag) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--min-level", "DEBUG"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  EXPECT_EQ(cfg.minLevel, "DEBUG");
}

/** @test Parse --csv flag. */
TEST(PerfConfigTest, ParseCsvFlag) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--csv", "/tmp/results.csv"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  ASSERT_TRUE(cfg.csv.has_value());
  EXPECT_EQ(cfg.csv.value(), "/tmp/results.csv");
}

/** @test Parse --profile flag. */
TEST(PerfConfigTest, ParseProfileFlag) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--profile", "perf"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  EXPECT_EQ(cfg.profileTool, "perf");
}

/** @test Parse --profile-args flag. */
TEST(PerfConfigTest, ParseProfileArgsFlag) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--profile-args", "record -g"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  EXPECT_EQ(cfg.profileArgs, "record -g");
}

/** @test Parse --artifact-root flag. */
TEST(PerfConfigTest, ParseArtifactRootFlag) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--artifact-root", "/tmp/artifacts"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  EXPECT_EQ(cfg.artifactRoot, "/tmp/artifacts");
}

/* ----------------------------- List Parsing Tests ----------------------------- */

/** @test Parse --bpf with comma-separated list. */
TEST(PerfConfigTest, ParseBpfListFlag) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--bpf", "offcpu,syslat,bio"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  ASSERT_EQ(cfg.bpfScripts.size(), 3U);
  EXPECT_EQ(cfg.bpfScripts[0], "offcpu");
  EXPECT_EQ(cfg.bpfScripts[1], "syslat");
  EXPECT_EQ(cfg.bpfScripts[2], "bio");
}

/** @test Parse --bpf with single item. */
TEST(PerfConfigTest, ParseBpfSingleItem) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--bpf", "offcpu"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  ASSERT_EQ(cfg.bpfScripts.size(), 1U);
  EXPECT_EQ(cfg.bpfScripts[0], "offcpu");
}

/** @test Parse --bpf trims whitespace. */
TEST(PerfConfigTest, ParseBpfTrimsWhitespace) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--bpf", " offcpu , syslat "};

  parsePerfFlags(cfg, args.argc(), args.argv());

  ASSERT_EQ(cfg.bpfScripts.size(), 2U);
  EXPECT_EQ(cfg.bpfScripts[0], "offcpu");
  EXPECT_EQ(cfg.bpfScripts[1], "syslat");
}

/* ----------------------------- Quick Mode Tests ----------------------------- */

/** @test Quick mode applies lighter defaults. */
TEST(PerfConfigTest, QuickModeAppliesLighterDefaults) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--quick"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  EXPECT_TRUE(cfg.quickMode);
  EXPECT_EQ(cfg.cycles, 5000); // Reduced from 10000
  EXPECT_EQ(cfg.repeats, 5);   // Reduced from 10
  EXPECT_EQ(cfg.warmup, 2);    // Changed from 1
}

/** @test Quick mode doesn't override explicit values. */
TEST(PerfConfigTest, QuickModeDoesNotOverrideExplicit) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--cycles", "2000", "--quick"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  EXPECT_TRUE(cfg.quickMode);
  EXPECT_EQ(cfg.cycles, 2000); // Explicit value preserved
  EXPECT_EQ(cfg.repeats, 5);   // Quick mode applied (not explicitly set)
}

/* ----------------------------- Pass-through Tests ----------------------------- */

/** @test Unknown flags are passed through (argc updated). */
TEST(PerfConfigTest, UnknownFlagsPassedThrough) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--gtest_filter=*Test*", "--cycles", "1000", "--unknown"};

  const int ORIGINAL_ARGC = *args.argc();
  parsePerfFlags(cfg, args.argc(), args.argv());

  // --cycles and 1000 consumed, 3 args remain: prog, --gtest_filter, --unknown
  EXPECT_EQ(*args.argc(), 3);
  EXPECT_LT(*args.argc(), ORIGINAL_ARGC);
}

/* ----------------------------- Minimum Value Tests ----------------------------- */

/** @test Cycles minimum is 1. */
TEST(PerfConfigTest, CyclesMinimumIsOne) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--cycles", "0"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  EXPECT_EQ(cfg.cycles, 1);
}

/** @test Repeats minimum is 1. */
TEST(PerfConfigTest, RepeatsMinimumIsOne) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--repeats", "-5"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  EXPECT_EQ(cfg.repeats, 1);
}

/** @test Warmup minimum is 0. */
TEST(PerfConfigTest, WarmupMinimumIsZero) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--warmup", "-1"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  EXPECT_EQ(cfg.warmup, 0);
}

/** @test Threads minimum is 1. */
TEST(PerfConfigTest, ThreadsMinimumIsOne) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--threads", "0"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  EXPECT_EQ(cfg.threads, 1);
}

/** @test MsgBytes minimum is 1. */
TEST(PerfConfigTest, MsgBytesMinimumIsOne) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--msg-bytes", "0"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  EXPECT_EQ(cfg.msgBytes, 1);
}

/* ----------------------------- Multiple Flags Tests ----------------------------- */

/** @test Multiple flags parsed together. */
TEST(PerfConfigTest, MultipleFlagsParsedTogether) {
  PerfConfig cfg;
  ArgvBuilder args{"prog", "--cycles", "500",     "--repeats", "3",     "--threads",
                   "2",    "--csv",    "out.csv", "--profile", "gperf", "--console"};

  parsePerfFlags(cfg, args.argc(), args.argv());

  EXPECT_EQ(cfg.cycles, 500);
  EXPECT_EQ(cfg.repeats, 3);
  EXPECT_EQ(cfg.threads, 2);
  ASSERT_TRUE(cfg.csv.has_value());
  EXPECT_EQ(cfg.csv.value(), "out.csv");
  EXPECT_EQ(cfg.profileTool, "gperf");
  EXPECT_TRUE(cfg.console);
}
