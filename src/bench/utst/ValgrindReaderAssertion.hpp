#ifndef VERNIER_VALGRINDREADERASSERTION_HPP
#define VERNIER_VALGRINDREADERASSERTION_HPP
/**
 * @file ValgrindReaderAssertion.hpp
 * @brief Test helper: valgrind's own line when an assertion in its ELF
 * debug-information reader stopped it before the program started.
 *
 * valgrind 3.18.1 fails an assertion at readelf.c:2478 while it reads a GCC
 * 11.4 Debug binary that mold linked, before the program's first instruction,
 * and is killed by SIGSEGV. A check that runs a program under valgrind may
 * skip on that line, quoting it, only with the evidence that the program never
 * ran. This is the one reading of that evidence, whether valgrind writes to a
 * log file of its own or shares a stream with the program;
 * ValgrindReaderAssertionCli.cpp gives it to checks written in CMake.
 */

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace vernier {
namespace bench {
namespace test {

/* ----------------------------- Reading valgrind's Text ----------------------------- */

/// True when @p text ends with @p suffix.
inline bool endsWith(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// A log line without its "==pid== " prefix; the line itself when it has none.
inline std::string payload(const std::string& line) {
  if (line.size() > 2 && line[0] == '=' && line[1] == '=') {
    const std::size_t END = line.find("== ", 2);
    if (END != std::string::npos) {
      return line.substr(END + 3);
    }
    if (line.compare(line.size() - 2, 2, "==") == 0) {
      return "";
    }
  }
  return line;
}

/* ----------------------------- The Reader's Assertion ----------------------------- */

/// The assertion, on either side of the source line number valgrind prints.
inline constexpr const char* READER_ASSERTION_PREFIX = "valgrind: m_debuginfo/readelf.c:";
inline constexpr const char* READER_ASSERTION_SUFFIX =
    " (vgModuleLocal_read_elf_debug_info): Assertion 'di->bss_svma + di->bss_size == svma' "
    "failed.";

/// How valgrind's opening lines end before it reads the program: with
/// "Parent PID: <n>" when it writes to a log file of its own, and, for
/// callgrind writing to the stream it shares with the program, with this
/// notice.
inline constexpr const char* OPENING_END_IN_A_LOG_FILE = "Parent PID: ";
inline constexpr const char* OPENING_END_CALLGRIND_ON_THE_STREAM =
    "For interactive control, run 'callgrind_control -h'.";

/**
 * @brief valgrind's own line when that assertion stopped it before the
 *        program started, as it printed it; empty for any other outcome.
 * @param killedBySignal  valgrind was killed by a signal.
 * @param output  What the program wrote, where valgrind's text is a log file
 *                of its own; empty where the two share one stream.
 * @param text    valgrind's text: its log file, or the stream it shares with
 *                the program.
 *
 * Taken only with the evidence that the program never ran, as valgrind 3.18.1
 * ended when it did not: the program wrote nothing (@p output is empty); in
 * @p text the line stands alone, with only blank lines between it and
 * valgrind's opening lines, which end with "Parent PID:" in a log file and
 * with callgrind's notice on the shared stream, and only blank lines after it;
 * and valgrind was killed by a signal. On the shared stream anything the
 * program wrote stands between or after them. Once the program has run,
 * valgrind prints its crash report after the same line and exits with status
 * 1: not taken, nor is the program's own assertion, another of valgrind's, or
 * the same text anywhere else. A skip quotes the line, so what it rests on is
 * valgrind's text, not the check's.
 */
inline std::string readerAssertionBeforeStart(bool killedBySignal, const std::string& output,
                                              const std::string& text) {
  if (!output.empty() || !killedBySignal) {
    return "";
  }
  std::vector<std::string> lines;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    lines.push_back(line);
  }
  // The text's last line that is not blank is the assertion...
  std::size_t at = lines.size();
  while (at > 0 && payload(lines[at - 1]).empty()) {
    --at;
  }
  if (at == 0) {
    return "";
  }
  const std::string& ASSERTION = lines[at - 1];
  const std::string PREFIX = READER_ASSERTION_PREFIX;
  const std::string SUFFIX = READER_ASSERTION_SUFFIX;
  if (ASSERTION.size() <= PREFIX.size() + SUFFIX.size() || ASSERTION.rfind(PREFIX, 0) != 0 ||
      !endsWith(ASSERTION, SUFFIX) ||
      ASSERTION.substr(PREFIX.size(), ASSERTION.size() - PREFIX.size() - SUFFIX.size())
              .find_first_not_of("0123456789") != std::string::npos) {
    return "";
  }
  // ...and the last one before it ends valgrind's opening lines
  std::size_t before = at - 1;
  while (before > 0 && payload(lines[before - 1]).empty()) {
    --before;
  }
  if (before == 0 || lines[before - 1].rfind("==", 0) != 0) {
    return "";
  }
  const std::string OPENING_END = payload(lines[before - 1]);
  if (OPENING_END.rfind(OPENING_END_IN_A_LOG_FILE, 0) != 0 &&
      OPENING_END != OPENING_END_CALLGRIND_ON_THE_STREAM) {
    return "";
  }
  return ASSERTION;
}

} // namespace test
} // namespace bench
} // namespace vernier

#endif // VERNIER_VALGRINDREADERASSERTION_HPP
