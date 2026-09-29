/**
 * @file BpfScripts_uTest.cpp
 * @brief Unit tests for the bundled bpftrace scripts' text.
 *
 * Notes:
 *  - The scripts run only under bpftrace with privileges these tests do not
 *    have, so they read each script as text and check the probes the backend
 *    relies on; what bpftrace does with them is checked where it can trace.
 *  - The scripts are read from the source tree (VERNIER_BUNDLED_BPF_DIR), where
 *    the backend looks them up.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

/** @brief Every bundled script's text, by file name without ".bt". */
std::map<std::string, std::string> bundledScripts() {
  std::map<std::string, std::string> scripts;
  for (const auto& entry : std::filesystem::directory_iterator(VERNIER_BUNDLED_BPF_DIR)) {
    if (entry.path().extension() != ".bt") {
      continue;
    }
    std::ifstream in(entry.path());
    std::stringstream text;
    text << in.rdbuf();
    scripts[entry.path().stem().string()] = text.str();
  }
  return scripts;
}

/** @brief The lines of @p text that start with @p prefix. */
std::vector<std::string> linesStartingWith(const std::string& text, const std::string& prefix) {
  std::vector<std::string> lines;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind(prefix, 0) == 0) {
      lines.push_back(line);
    }
  }
  return lines;
}

} // namespace

/* ----------------------------- API Tests ----------------------------- */

/** @test Holds the four scripts the guide documents */
TEST(BundledBpfScriptsTest, BundledSetIsTheDocumentedOne) {
  const auto SCRIPTS = bundledScripts();
  for (const char* name : {"cpu_migrations", "fsync_latency", "wakeup_latency", "write_latency"}) {
    EXPECT_EQ(SCRIPTS.count(name), 1U) << name;
  }
}

/** @test Ends every script only when the traced process's main thread exits */
TEST(BundledBpfScriptsTest, ExitProbeMatchesTheMainThreadOnly) {
  const auto SCRIPTS = bundledScripts();
  ASSERT_GE(SCRIPTS.size(), 4U);
  for (const auto& [NAME, TEXT] : SCRIPTS) {
    const auto PROBES = linesStartingWith(TEXT, "tracepoint:sched:sched_process_exit");
    ASSERT_EQ(PROBES.size(), 1U) << NAME;
    EXPECT_EQ(PROBES[0], "tracepoint:sched:sched_process_exit /tid == {{PID}}/ {") << NAME;
  }
}

/** @test Records wakeup latency in the waking thread, on sched_waking */
TEST(BundledBpfScriptsTest, WakeupLatencyRecordsInTheWaker) {
  const auto SCRIPTS = bundledScripts();
  ASSERT_EQ(SCRIPTS.count("wakeup_latency"), 1U);
  const std::string& text = SCRIPTS.at("wakeup_latency");
  const auto WAKING = linesStartingWith(text, "tracepoint:sched:sched_waking");
  ASSERT_EQ(WAKING.size(), 1U) << text;
  EXPECT_EQ(WAKING[0], "tracepoint:sched:sched_waking /args->pid == {{PID}} || pid == {{PID}}/ {");
  EXPECT_TRUE(linesStartingWith(text, "tracepoint:sched:sched_wakeup ").empty()) << text;
}
