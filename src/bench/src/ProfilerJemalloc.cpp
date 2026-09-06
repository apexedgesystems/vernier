/**
 * @file ProfilerJemalloc.cpp
 * @brief jemalloc heap-profiler backend implementation.
 *
 * Wrap-externally pattern via LD_PRELOAD + MALLOC_CONF; the backend stays
 * passive otherwise and prints the precise invocation when not preloaded.
 */

#include "src/bench/inc/ProfilerJemalloc.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#include "src/bench/inc/ProfilerRegistry.hpp"

namespace vernier {
namespace bench {

namespace {

bool isJeprofOnPath() { return std::system("command -v jeprof >/dev/null 2>&1") == 0; }

enum class JemallocPreload { Missing, NoProf, ProfCapable };

/// One subprocess answers both questions through the real mechanism: ld.so
/// resolves LD_PRELOAD for /bin/true exactly as it will for a wrapped
/// benchmark, by soname, through the full loader search. Nothing weaker is
/// equivalent -- path lists miss multiarch dirs, and dlopen fails where
/// preload succeeds (jemalloc's initial-exec TLS can exhaust the static TLS
/// surplus mid-process: "cannot allocate memory in static TLS block",
/// observed on aarch64). /bin/true keeps the child trivial; a shell builtin
/// would never exec, so the preload would not engage.
///
/// Verdicts, from the child's stderr:
///  - "cannot be preloaded"  -> the loader cannot resolve it: Missing.
///  - "Invalid conf pair"    -> loads, but built without --enable-prof
///                              (Debian/Ubuntu distro packages): NoProf.
///  - silence                -> loads and accepts prof:true: ProfCapable.
JemallocPreload probeJemallocPreload() {
  std::FILE* pipe = popen("MALLOC_CONF=prof:true LD_PRELOAD=libjemalloc.so.2 /bin/true 2>&1", "r");
  if (!pipe)
    return JemallocPreload::Missing;
  std::string out;
  char buf[512];
  while (std::fgets(buf, sizeof(buf), pipe))
    out += buf;
  pclose(pipe);
  if (out.find("cannot be preloaded") != std::string::npos)
    return JemallocPreload::Missing;
  if (out.find("Invalid conf pair") != std::string::npos)
    return JemallocPreload::NoProf;
  return JemallocPreload::ProfCapable;
}

/// jemalloc is active when libjemalloc shows up in /proc/self/maps OR when
/// LD_PRELOAD names it explicitly. Either is sufficient as a "wrapping
/// detected" signal.
bool detectUnderJemalloc() {
  const char* preload = std::getenv("LD_PRELOAD");
  if (preload && std::strstr(preload, "jemalloc"))
    return true;
  std::FILE* fp = std::fopen("/proc/self/maps", "r");
  if (!fp)
    return false;
  char line[512];
  bool found = false;
  while (std::fgets(line, sizeof(line), fp)) {
    if (std::strstr(line, "libjemalloc")) {
      found = true;
      break;
    }
  }
  std::fclose(fp);
  return found;
}

} // namespace

/* ----------------------------- JemallocProfiler ----------------------------- */

JemallocProfiler::JemallocProfiler(const PerfConfig& cfg, std::string testName)
    : cfg_(cfg), testName_(std::move(testName)) {
  runningUnderJemalloc_ = detectUnderJemalloc();
  artifactDir_ = cfg_.artifactRoot.empty() ? "./" + testName_ + ".jemalloc"
                                           : cfg_.artifactRoot + "/" + testName_ + ".jemalloc";
  std::error_code ec;
  std::filesystem::create_directories(artifactDir_, ec);
  (void)ec;
}

void JemallocProfiler::beforeMeasure() {
  if (runningUnderJemalloc_) {
    // Report the actual dump location: when a wrapper (bench run) set
    // prof_prefix via MALLOC_CONF, samples land there, not in this
    // test's own artifact dir.
    std::string prefix = artifactDir_ + "/jeprof";
    if (const char* mc = std::getenv("MALLOC_CONF")) {
      if (const char* p = std::strstr(mc, "prof_prefix:")) {
        const char* start = p + 12;
        const char* end = std::strchr(start, ',');
        prefix.assign(start, end ? static_cast<std::size_t>(end - start) : std::strlen(start));
      }
    }
    std::fprintf(stderr,
                 "[jemalloc] wrapping detected; heap samples written to %s.*.heap at\n"
                 "[jemalloc] process exit.\n",
                 prefix.c_str());
    return;
  }
  std::fprintf(
      stderr,
      "\n[jemalloc] NOT running under jemalloc; this measurement will execute\n"
      "[jemalloc] normally but no heap samples are collected. To collect:\n"
      "[jemalloc]   LD_PRELOAD=$(ldconfig -p | awk '/libjemalloc.so / {print $4; exit}') \\\n"
      "[jemalloc]   MALLOC_CONF=prof:true,prof_final:true,prof_prefix:%s/jeprof \\\n"
      "[jemalloc]       <this-binary> --profile jemalloc [...]\n"
      "[jemalloc] Then: jeprof --text <this-binary> %s/jeprof.*.heap | head -20\n\n",
      artifactDir_.c_str(), artifactDir_.c_str());
}

void JemallocProfiler::afterMeasure(const Stats& /*s*/) {
  // jemalloc dumps the final prof heap at exit when MALLOC_CONF carries
  // prof:true,prof_final:true (the recipe both the hint and bench run use).
}

/* ----------------------------- Env check ----------------------------- */

EnvReport checkJemallocEnvironment() {
  const JemallocPreload PRELOAD = probeJemallocPreload();
  const bool LIB = PRELOAD != JemallocPreload::Missing;
  const bool TOOL = isJeprofOnPath();
  if (!LIB && !TOOL) {
    return EnvReport{EnvReport::Status::Error,
                     "jemalloc not detected (no libjemalloc.so, no jeprof)",
                     "apt install libjemalloc2 libjemalloc-dev (or your distro's equivalent)."};
  }
  if (!LIB) {
    return EnvReport{EnvReport::Status::Warning, "jeprof present but libjemalloc.so missing",
                     "apt install libjemalloc2 so LD_PRELOAD can find the runtime."};
  }
  if (!TOOL) {
    return EnvReport{EnvReport::Status::Warning, "libjemalloc.so present but jeprof missing",
                     "apt install libjemalloc-dev (ships jeprof for analysis)."};
  }
  if (PRELOAD == JemallocPreload::NoProf) {
    return EnvReport{EnvReport::Status::Warning,
                     "libjemalloc present but built without profiling (prof:true rejected)",
                     "Debian/Ubuntu ship jemalloc without --enable-prof, so no heap profile can "
                     "be written. Build jemalloc from source with --enable-prof, or use the "
                     "heaptrack backend."};
  }
  return EnvReport{EnvReport::Status::Ok, "libjemalloc + jeprof available (profiling build)", ""};
}

/* --------------------------------- API --------------------------------- */

std::unique_ptr<Profiler> makeJemallocProfiler(const PerfConfig& cfg, const std::string& testName) {
  // No build-time dependency; backend is always constructible. The runtime
  // env check is what tells the user whether libjemalloc is actually usable.
  return std::make_unique<JemallocProfiler>(cfg, testName);
}

} // namespace bench
} // namespace vernier

VERNIER_REGISTER_PROFILER_BACKEND(
    "jemalloc", ::vernier::bench::makeJemallocProfiler, ::vernier::bench::checkJemallocEnvironment,
    "apt install libjemalloc2 libjemalloc-dev (preloaded via LD_PRELOAD).")
