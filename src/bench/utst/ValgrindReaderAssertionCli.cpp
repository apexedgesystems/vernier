/**
 * @file ValgrindReaderAssertionCli.cpp
 * @brief Test helper program: ValgrindReaderAssertion.hpp's reading, for
 * checks written in CMake.
 *
 * Prints valgrind's own line, as it printed it, when an assertion in its ELF
 * debug-information reader stopped it before the program started, with the
 * same evidence a C++ check asks of it.
 *
 * Usage:
 *   @code{.sh}
 *   ValgrindReaderAssertionCli <signal|exit> <program output file|-> <valgrind text file>
 *   @endcode
 * "signal" when valgrind was killed by a signal, "exit" when it exited; "-"
 * when the program shared valgrind's stream, so its output is in valgrind's
 * text. Exits 0 having printed the line, 1 when the evidence does not hold, 2
 * on a usage error or a file it cannot read.
 */

#include "src/bench/utst/ValgrindReaderAssertion.hpp"

#include <cstdio>

#include <fstream>
#include <iterator>
#include <string>

namespace {

/* ----------------------------- File Helpers ----------------------------- */

/// A whole file as text; false when it cannot be read.
bool readText(const char* path, std::string& text) {
  std::ifstream in(path);
  if (!in) {
    return false;
  }
  text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return true;
}

} // namespace

/* ----------------------------- Main ----------------------------- */

int main(int argc, char** argv) {
  const std::string ENDED = argc == 4 ? argv[1] : "";
  if (ENDED != "signal" && ENDED != "exit") {
    std::fputs("usage: ValgrindReaderAssertionCli <signal|exit> <program output file|-> "
               "<valgrind text file>\n",
               stderr);
    return 2;
  }
  std::string output;
  std::string text;
  if ((std::string(argv[2]) != "-" && !readText(argv[2], output)) || !readText(argv[3], text)) {
    std::fprintf(stderr, "ValgrindReaderAssertionCli: cannot read %s or %s\n", argv[2], argv[3]);
    return 2;
  }
  const std::string LINE =
      vernier::bench::test::readerAssertionBeforeStart(ENDED == "signal", output, text);
  if (LINE.empty()) {
    return 1;
  }
  std::printf("%s\n", LINE.c_str());
  return 0;
}
