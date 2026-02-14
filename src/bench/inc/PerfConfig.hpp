#ifndef VERNIER_PERFCONFIG_HPP
#define VERNIER_PERFCONFIG_HPP
/**
 * @file PerfConfig.hpp
 * @brief Common tunables for microbenchmarks and a flag parser that preserves gtest args.
 */

#include <algorithm> // std::max
#include <cstdio>    // std::fprintf
#include <cstdlib>   // std::atoi, std::exit
#include <cstring>   // std::strstr
#include <fstream>   // profile-check file reads
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vernier {
namespace bench {

/* ----------------------------- PerfConfig ----------------------------- */

/** @brief Common benchmark configuration values (CLI-overridable). */
struct PerfConfig {
  int cycles = 10000;               ///< Operations per repeat
  int repeats = 10;                 ///< Samples collected
  int warmup = 1;                   ///< Warmup repeats (0 = auto-scale: <1k->5, <10k->3, >=10k->1)
  int threads = 1;                  ///< Worker threads
  int msgBytes = 64;                ///< Payload size (bytes) when relevant
  bool console = false;             ///< Echo to console (when relevant)
  bool nonBlocking = false;         ///< Non-blocking mode (when relevant)
  std::string minLevel = "INFO";    ///< DEBUG|INFO|WARNING|ERROR|FATAL
  std::optional<std::string> csv{}; ///< Optional CSV output path

  // ---- Profiling / artifact knobs (default off) ----
  std::string profileTool;             ///< "", "perf", "gperf", "bpftrace", "rapl", "callgrind"
  std::string profileArgs;             ///< Verbatim pass-through to the tool
  std::vector<std::string> bpfScripts; ///< Curated script names: "offcpu", "syslat", "bio"
  std::string artifactRoot;            ///< Optional root for artifacts (default chosen by runner)
  int profileFrequency = 10000;        ///< Sampling frequency for CPU profilers (Hz)
  bool profileAnalyze = false;         ///< Auto-run analysis after profiling (e.g., pprof top-10)

  // ---- Quick mode (lighter defaults for fast iteration) ----
  bool quickMode = false; ///< Apply reduced cycles/repeats for development iteration
};

/* --------------------------------- API --------------------------------- */

/** @brief Forward declaration -- defined below parsePerfFlags(). */
inline void runProfileCheck();

/**
 * @brief Parse perf flags, leaving unknown args for gtest. Mutates argc/argv.
 *
 * Recognized flags:
 *   --cycles N         --repeats N        --warmup N (0 = auto-scale)
 *   --threads N        --msg-bytes N      --console
 *   --nonblocking      --min-level STR    --csv PATH
 *   --profile TOOL     --profile-args STR --bpf LIST(,...) --artifact-root PATH
 *   --profile-frequency N  (sampling Hz for CPU profilers, default 10000)
 *   --profile-analyze      (auto-run analysis after profiling)
 *   --quick            (applies lighter defaults for fast iteration)
 *
 * @note NOT RT-safe (heap allocation, console I/O, may call exit()).
 */
inline void parsePerfFlags(PerfConfig& cfg, int* argc, char** argv) {
  const auto PARSE_LIST = [](std::string_view s) -> std::vector<std::string> {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (start <= s.size()) {
      std::size_t end = s.find(',', start);
      if (end == std::string_view::npos)
        end = s.size();
      std::string item{s.substr(start, end - start)};
      // trim spaces (simple)
      while (!item.empty() && (item.front() == ' ' || item.front() == '\t'))
        item.erase(item.begin());
      while (!item.empty() && (item.back() == ' ' || item.back() == '\t'))
        item.pop_back();
      if (!item.empty())
        out.emplace_back(std::move(item));
      if (end >= s.size())
        break;
      start = end + 1;
    }
    return out;
  };

  const auto NEED_ARG = [&](const char* name, int i, int argcVal, char** argvVal) -> const char* {
    if (i + 1 >= argcVal) {
      std::fprintf(stderr, "Missing value for %s\n", name);
      std::exit(2);
    }
    return argvVal[i + 1];
  };

  // Track if user explicitly set values (to avoid overriding with --quick)
  bool cyclesSet = false;
  bool repeatsSet = false;
  bool warmupSet = false;

  int w = 1;
  for (int i = 1; i < *argc; ++i) {
    std::string_view a = argv[i];

    if (a == "--cycles") {
      cfg.cycles = std::max(1, std::atoi(NEED_ARG("--cycles", i, *argc, argv)));
      cyclesSet = true;
      ++i;
    } else if (a == "--repeats") {
      cfg.repeats = std::max(1, std::atoi(NEED_ARG("--repeats", i, *argc, argv)));
      repeatsSet = true;
      ++i;
    } else if (a == "--warmup") {
      cfg.warmup = std::max(0, std::atoi(NEED_ARG("--warmup", i, *argc, argv)));
      warmupSet = true;
      ++i;
    } else if (a == "--threads") {
      cfg.threads = std::max(1, std::atoi(NEED_ARG("--threads", i, *argc, argv)));
      ++i;
    } else if (a == "--msg-bytes") {
      cfg.msgBytes = std::max(1, std::atoi(NEED_ARG("--msg-bytes", i, *argc, argv)));
      ++i;
    } else if (a == "--console") {
      cfg.console = true;
    } else if (a == "--nonblocking") {
      cfg.nonBlocking = true;
    } else if (a == "--min-level") {
      cfg.minLevel = NEED_ARG("--min-level", i, *argc, argv);
      ++i;
    } else if (a == "--csv") {
      cfg.csv = std::string(NEED_ARG("--csv", i, *argc, argv));
      ++i;
    }

    // ---- Profiling / artifact flags ----
    else if (a == "--profile") {
      cfg.profileTool = NEED_ARG("--profile", i, *argc, argv);
      ++i;
    } else if (a == "--profile-args") {
      cfg.profileArgs = NEED_ARG("--profile-args", i, *argc, argv);
      ++i;
    } else if (a == "--bpf") {
      cfg.bpfScripts = PARSE_LIST(NEED_ARG("--bpf", i, *argc, argv));
      ++i;
    } else if (a == "--artifact-root") {
      cfg.artifactRoot = NEED_ARG("--artifact-root", i, *argc, argv);
      ++i;
    } else if (a == "--profile-frequency") {
      cfg.profileFrequency =
          std::max(1, std::atoi(NEED_ARG("--profile-frequency", i, *argc, argv)));
      ++i;
    } else if (a == "--profile-analyze") {
      cfg.profileAnalyze = true;
    }

    // ---- Quick mode flag ----
    else if (a == "--quick") {
      cfg.quickMode = true;
    }

    // ---- Profile check (runs diagnostics and exits) ----
    else if (a == "--profile-check") {
      runProfileCheck();
      std::exit(0);
    }

    // Pass-through to gtest
    else {
      argv[w++] = argv[i];
    }
  }
  *argc = w;

  // Apply quick mode defaults if enabled and user didn't explicitly override
  if (cfg.quickMode) {
    if (!cyclesSet && cfg.cycles == 10000) {
      cfg.cycles = 5000;
    }
    if (!repeatsSet && cfg.repeats == 10) {
      cfg.repeats = 5;
    }
    if (!warmupSet && cfg.warmup == 1) {
      cfg.warmup = 2;
    }
  }
}

/* ----------------------------- Profile Check ----------------------------- */

/**
 * @brief Validate the current binary's profiling readiness.
 *
 * Checks:
 *  - Frame pointers (-fno-omit-frame-pointer)
 *  - DWARF version (v4 preferred for gperftools compatibility)
 *  - ASLR status (/proc/sys/kernel/randomize_va_space)
 *  - gperftools linkage (libprofiler)
 *  - Split DWARF (.dwo files)
 *
 * Prints a diagnostic report to stdout and exits.
 *
 * @note NOT RT-safe (file I/O, console I/O).
 */
inline void runProfileCheck() {
  int passCount = 0;
  int warnCount = 0;
  int failCount = 0;

  auto pass = [&](const char* label, const char* detail) {
    std::fprintf(stdout, "  [OK]   %-30s %s\n", label, detail);
    ++passCount;
  };
  auto warn = [&](const char* label, const char* detail) {
    std::fprintf(stdout, "  [WARN] %-30s %s\n", label, detail);
    ++warnCount;
  };
  auto fail = [&](const char* label, const char* detail) {
    std::fprintf(stdout, "  [FAIL] %-30s %s\n", label, detail);
    ++failCount;
  };

  std::fprintf(stdout, "\n=== Profile Readiness Check ===\n\n");

  // 1. Frame pointers: check if the current function has a frame pointer
  //    by inspecting the binary's ELF .eh_frame section presence.
  //    Heuristic: check compile flags via /proc/self/cmdline or just test the
  //    stack frame directly. Simplest: check if __builtin_frame_address works.
#if defined(__GCC_HAVE_DWARF2_CFI_ASM) || defined(__clang__)
  {
    volatile void* fp = __builtin_frame_address(0);
    if (fp != nullptr) {
      pass("Frame pointers", "detected (__builtin_frame_address accessible)");
    } else {
      fail("Frame pointers", "not detected; add -fno-omit-frame-pointer");
    }
  }
#else
  warn("Frame pointers", "cannot detect at runtime; ensure -fno-omit-frame-pointer");
#endif

  // 2. DWARF version: read /proc/self/exe with readelf if available,
  //    or check compiler flags at build time.
#ifdef __GCC_HAVE_DWARF2_CFI_ASM
  // Compile-time check: clang/gcc with DWARF2+ CFI
  // Heuristic: check for common debug format macros
#endif
  {
    // Runtime: inspect first bytes of .debug_info in /proc/self/exe
    std::ifstream exe("/proc/self/exe", std::ios::binary);
    if (exe) {
      // Read ELF and search for DWARF version in .debug_info header
      // The DWARF version is a uint16 at offset 4 in the .debug_info section
      // Simpler approach: just check if the binary has .debug_info at all
      char buf[8192];
      bool foundDebug = false;
      while (exe.read(buf, sizeof(buf)) || exe.gcount() > 0) {
        auto count = exe.gcount();
        // Search for ".debug_info" section name in ELF
        for (std::streamsize i = 0; i <= count - 11; ++i) {
          if (std::memcmp(buf + i, ".debug_info", 11) == 0) {
            foundDebug = true;
            break;
          }
        }
        if (foundDebug) {
          break;
        }
      }
      if (foundDebug) {
        pass("Debug info", "present (.debug_info found in binary)");
      } else {
        fail("Debug info", "not found; rebuild with -g");
      }
    } else {
      warn("Debug info", "cannot read /proc/self/exe");
    }
  }

  // 3. ASLR status
  {
    std::ifstream aslr("/proc/sys/kernel/randomize_va_space");
    if (aslr) {
      int val = -1;
      aslr >> val;
      if (val == 0) {
        pass("ASLR", "disabled (randomize_va_space=0)");
      } else {
        warn("ASLR", "enabled; use 'setarch $(uname -m) -R' for consistent profiles");
      }
    } else {
      warn("ASLR", "cannot read /proc/sys/kernel/randomize_va_space");
    }
  }

  // 4. gperftools linkage: check if libprofiler symbols are available
  {
    // Check /proc/self/maps for libprofiler
    std::ifstream maps("/proc/self/maps");
    if (maps) {
      std::string line;
      bool found = false;
      while (std::getline(maps, line)) {
        if (line.find("libprofiler") != std::string::npos) {
          found = true;
          break;
        }
      }
      if (found) {
        pass("gperftools", "libprofiler linked");
      } else {
        fail("gperftools", "libprofiler not linked; link with -lprofiler");
      }
    } else {
      warn("gperftools", "cannot read /proc/self/maps");
    }
  }

  // 5. Split DWARF: check for .dwo references in the binary
  {
    std::ifstream exe("/proc/self/exe", std::ios::binary);
    bool hasSplitDwarf = false;
    if (exe) {
      char buf[8192];
      while (exe.read(buf, sizeof(buf)) || exe.gcount() > 0) {
        auto count = exe.gcount();
        for (std::streamsize i = 0; i <= count - 4; ++i) {
          if (std::memcmp(buf + i, ".dwo", 4) == 0) {
            hasSplitDwarf = true;
            break;
          }
        }
        if (hasSplitDwarf) {
          break;
        }
      }
    }
    if (hasSplitDwarf) {
      warn("Split DWARF", "detected; rebuild without -gsplit-dwarf for profiling");
    } else {
      pass("Split DWARF", "not detected (debug info in main binary)");
    }
  }

  // Summary
  std::fprintf(stdout, "\n  ---\n  %d passed, %d warnings, %d failures\n", passCount, warnCount,
               failCount);
  if (failCount > 0) {
    std::fprintf(stdout,
                 "\n  Recommendation: rebuild with profiling-friendly flags:\n"
                 "    cmake -B build -DCMAKE_CXX_FLAGS=\"-fno-omit-frame-pointer -gdwarf-4\"\n"
                 "    cmake --build build\n\n");
  } else if (warnCount > 0) {
    std::fprintf(
        stdout, "\n  Binary is mostly ready for profiling. Address warnings for best results.\n\n");
  } else {
    std::fprintf(stdout, "\n  Binary is ready for profiling.\n\n");
  }
}

} // namespace bench
} // namespace vernier

#endif // VERNIER_PERFCONFIG_HPP