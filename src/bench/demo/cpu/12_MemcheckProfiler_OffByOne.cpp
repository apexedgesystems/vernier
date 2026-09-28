/**
 * @file 12_MemcheckProfiler_OffByOne.cpp
 * @brief The wrong join, in a source of its own so it can be compiled with
 *        debug information: memcheck's report then names its lines.
 */

#include "src/bench/demo/cpu/12_MemcheckProfiler_OffByOne.hpp"

#include <cstring>

namespace vernier {
namespace bench {
namespace demo {
namespace memcheck_demo {

/* ----------------------------- API ----------------------------- */

std::string joinOffByOne(const std::vector<std::string>& parts, char sep) {
  std::size_t total = 0;
  for (const std::string& part : parts) {
    total += part.size() + 1;
  }

  // Room for the characters, none for the terminator: the bug
  char* buf = new char[total];
  char* at = buf;
  for (const std::string& part : parts) {
    std::memcpy(at, part.data(), part.size());
    at += part.size();
    *at++ = sep;
  }
  *at = '\0'; // One byte past the end of buf

  // Reads up to the terminator, one byte past the end again; the answer
  // comes out right, so no test of the result can tell
  std::string out(buf);
  delete[] buf;
  return out;
}

} // namespace memcheck_demo
} // namespace demo
} // namespace bench
} // namespace vernier
