/**
 * @file 05_NvtxAnnotation_RangesReport_uTest.cpp
 * @brief What demo 05's check reads and decides, held to its cases without
 *        nsys, CUDA or a device: its reading of nsys's CSV, its assessment of
 *        the trace and its judgement of a capture.
 *
 * The check, NvtxRanges.RecordedByNsightSystems
 * (05_NvtxAnnotation_Ranges_uTest.cpp), needs a GPU build, nsys and a device.
 * What it reads and decides with is its private support
 * (05_NvtxAnnotation_Check.hpp), and these tests run that code in every
 * build: two real reports of the demo, from nsys 2025.3.2 and 2026.3.1
 * (NsysReportTest), quoted fields and named columns (NsysCsvTest), traces laid
 * out as the demo's calls lay them out (NvtxTraceTest), and stand-ins for nsys
 * run the way the check runs nsys (NsysCaptureTest). ctest runs them under the
 * demo and nsight labels.
 *
 * Usage:
 *   @code{.sh}
 *   ctest --test-dir build -L nsight
 *   ./build/bin/tests/TestDemoNvtxRangesReport      # these alone, by hand
 *   @endcode
 */

#include "src/bench/demo/gpu/utst/05_NvtxAnnotation_Check.hpp"

#include "src/bench/demo/gpu/05_NvtxAnnotation_Phases.hpp"

#include <gtest/gtest.h>

#include <cstddef>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace phase = vernier::bench::demo::nvtx_annotation;
namespace fs = std::filesystem;

using vernier::bench::demo::nvtx_check::ANNOTATED;
using vernier::bench::demo::nvtx_check::assessRanges;
using vernier::bench::demo::nvtx_check::captureUnderNsys;
using vernier::bench::demo::nvtx_check::CaptureVerdict;
using vernier::bench::demo::nvtx_check::GpuOp;
using vernier::bench::demo::nvtx_check::MEASURED_CALLS;
using vernier::bench::demo::nvtx_check::NvtxRange;
using vernier::bench::demo::nvtx_check::readColumns;
using vernier::bench::demo::nvtx_check::readGpuOps;
using vernier::bench::demo::nvtx_check::readRanges;
using vernier::bench::demo::nvtx_check::RunDirectory;
using vernier::bench::demo::nvtx_check::splitCsvRow;
using vernier::bench::demo::nvtx_check::UNANNOTATED;
using vernier::bench::demo::nvtx_check::WARMUP_CALLS;

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

/* ----------------------------- Capture Controls ----------------------------- */

// What RecordedByNsightSystems does with a capture that does not end in a
// report, checked with stand-ins for nsys: shell scripts run the way the check
// runs nsys (captureUnderNsys()), given nsys's arguments, in which "$3" is the
// report's path without its extension. They need neither nsys nor a device.

namespace {

/// GoogleTest's banner and its line for two passed tests.
constexpr const char* TESTS_PASSED = R"(echo '[==========] Running 2 tests from 1 test suite.'
echo '[  PASSED  ] 2 tests.'
)";

/// The lines nsys 2026.3.1 printed for a program that ended before its tests
/// started, and the report it wrote all the same.
constexpr const char* NSYS_GENERATED = R"(echo 'Collecting data...'
echo "Generating '/tmp/nsys-runner/nsys-report-7f78.qdstrm'"
echo 'Generated:'
printf '\t%s\n' "$3.nsys-rep"
: > "$3.nsys-rep"
)";

/// nsys 2026.3.1's words, blank line included, when /tmp/nvidia was another
/// user's and not writable: what temporaryFilesRefused() recognises.
constexpr const char* DIRECTORY_REFUSED =
    "Failed to create directory \"/tmp/nvidia/nsight_systems\": Permission denied\n"
    "\n"
    "NOTE: If you are using a system that does not allow writing to \"/tmp\" or\n"
    "where the \"/tmp\" directory has limited storage you can use the TMPDIR environment\n"
    "variable to set a different location.";

/// The check's verdict on a capture by a stand-in for nsys that runs
/// @p script with /bin/sh, in @p dir.
CaptureVerdict standInVerdict(const std::string& script, const RunDirectory& dir) {
  const fs::path STAND_IN = dir / "nsys.sh";
  {
    std::ofstream out(STAND_IN);
    out << script;
  }
  return captureUnderNsys({"/bin/sh", STAND_IN.string()}, "BenchDemo_Gpu_05_NvtxAnnotation",
                          dir.path());
}

/// A shell script that prints each line of @p text as it is, then exits with
/// @p status.
std::string printsThenExits(const std::string& text, int status) {
  std::string script;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    script += "echo '" + line + "'\n";
  }
  return script + "exit " + std::to_string(status) + "\n";
}

} // namespace

/** @test A run that started both tests, passed them and wrote its report is read */
TEST(NsysCaptureTest, ACompleteRunIsRead) {
  const RunDirectory DIR;
  ASSERT_TRUE(DIR.made());

  const CaptureVerdict VERDICT =
      standInVerdict(std::string(TESTS_PASSED) + ": > \"$3.nsys-rep\"\n", DIR);

  EXPECT_EQ(VERDICT.action, CaptureVerdict::Action::READ) << VERDICT.message;
}

/** @test nsys exiting 0 with nothing printed and no report fails, naming the run's files */
TEST(NsysCaptureTest, ExitZeroWithoutOutputFails) {
  const RunDirectory DIR;
  ASSERT_TRUE(DIR.made());

  const CaptureVerdict VERDICT = standInVerdict("exit 0\n", DIR);

  EXPECT_EQ(VERDICT.action, CaptureVerdict::Action::FAIL);
  EXPECT_NE(VERDICT.message.find("nsys exited with status 0; the demo's tests never started; "
                                 "nsys wrote no report"),
            std::string::npos)
      << VERDICT.message;
  EXPECT_NE(VERDICT.message.find("The run printed nothing."), std::string::npos) << VERDICT.message;
  EXPECT_NE(VERDICT.message.find("kept in " + DIR.path().string()), std::string::npos)
      << VERDICT.message;
}

/** @test An exit before the tests start that nsys gives no reason for fails, quoting the run */
TEST(NsysCaptureTest, AnUnexplainedExitFails) {
  const RunDirectory DIR;
  ASSERT_TRUE(DIR.made());

  const CaptureVerdict VERDICT = standInVerdict(std::string(NSYS_GENERATED) + "exit 3\n", DIR);

  EXPECT_EQ(VERDICT.action, CaptureVerdict::Action::FAIL);
  EXPECT_NE(VERDICT.message.find("nsys exited with status 3; the demo's tests never started."),
            std::string::npos)
      << VERDICT.message;
  EXPECT_NE(VERDICT.message.find("The run printed:\nCollecting data...\nGenerating "
                                 "'/tmp/nsys-runner/nsys-report-7f78.qdstrm'\nGenerated:\n"),
            std::string::npos)
      << VERDICT.message;
  EXPECT_NE(VERDICT.message.find("kept in " + DIR.path().string()), std::string::npos)
      << VERDICT.message;
}

/** @test A program killed by SIGSEGV before its tests start fails: nsys exits 139 (128 + 11) */
TEST(NsysCaptureTest, AProgramKilledBySignalFails) {
  const RunDirectory DIR;
  ASSERT_TRUE(DIR.made());

  const CaptureVerdict VERDICT = standInVerdict(std::string(NSYS_GENERATED) + "exit 139\n", DIR);

  EXPECT_EQ(VERDICT.action, CaptureVerdict::Action::FAIL);
  EXPECT_NE(VERDICT.message.find("nsys exited with status 139; the demo's tests never started."),
            std::string::npos)
      << VERDICT.message;
  EXPECT_NE(VERDICT.message.find("kept in " + DIR.path().string()), std::string::npos)
      << VERDICT.message;
}

/** @test nsys killed by a signal fails (SIGKILL, which leaves no core dump behind) */
TEST(NsysCaptureTest, NsysKilledBySignalFails) {
  const RunDirectory DIR;
  ASSERT_TRUE(DIR.made());

  const CaptureVerdict VERDICT = standInVerdict("echo 'Collecting data...'\nkill -KILL $$\n", DIR);

  EXPECT_EQ(VERDICT.action, CaptureVerdict::Action::FAIL);
  EXPECT_NE(VERDICT.message.find("nsys was killed by signal 9"), std::string::npos)
      << VERDICT.message;
  EXPECT_NE(VERDICT.message.find("kept in " + DIR.path().string()), std::string::npos)
      << VERDICT.message;
}

/** @test nsys refusing its temporary directory before the tests start skips, quoting nsys */
TEST(NsysCaptureTest, TemporaryDirectoryRefusedSkips) {
  const RunDirectory DIR;
  ASSERT_TRUE(DIR.made());

  const CaptureVerdict VERDICT = standInVerdict(printsThenExits(DIRECTORY_REFUSED, 1), DIR);

  EXPECT_EQ(VERDICT.action, CaptureVerdict::Action::SKIP) << VERDICT.message;
  EXPECT_NE(VERDICT.message.find(DIRECTORY_REFUSED), std::string::npos) << VERDICT.message;
}

/** @test nsys refusing its temporary output file before the tests start skips, quoting nsys */
TEST(NsysCaptureTest, TemporaryFileRefusedSkips) {
  const RunDirectory DIR;
  ASSERT_TRUE(DIR.made());

  const CaptureVerdict VERDICT =
      standInVerdict(printsThenExits("Failed to create temporary output file", 1), DIR);

  EXPECT_EQ(VERDICT.action, CaptureVerdict::Action::SKIP) << VERDICT.message;
  EXPECT_NE(VERDICT.message.find("Failed to create temporary output file"), std::string::npos)
      << VERDICT.message;
}

/** @test nsys's words for its temporary files do not excuse a run whose tests started */
TEST(NsysCaptureTest, ARefusalAfterTheTestsStartedFails) {
  const RunDirectory DIR;
  ASSERT_TRUE(DIR.made());

  const CaptureVerdict VERDICT = standInVerdict(
      printsThenExits(
          "[==========] Running 2 tests from 1 test suite.\nFailed to create temporary output file",
          1),
      DIR);

  EXPECT_EQ(VERDICT.action, CaptureVerdict::Action::FAIL);
  EXPECT_NE(VERDICT.message.find("the demo's two tests did not both pass"), std::string::npos)
      << VERDICT.message;
}

/** @test A run that passed both tests but left no report fails */
TEST(NsysCaptureTest, AMissingReportFails) {
  const RunDirectory DIR;
  ASSERT_TRUE(DIR.made());

  const CaptureVerdict VERDICT = standInVerdict(TESTS_PASSED, DIR);

  EXPECT_EQ(VERDICT.action, CaptureVerdict::Action::FAIL);
  EXPECT_NE(VERDICT.message.find("nsys exited with status 0; nsys wrote no report."),
            std::string::npos)
      << VERDICT.message;
}
