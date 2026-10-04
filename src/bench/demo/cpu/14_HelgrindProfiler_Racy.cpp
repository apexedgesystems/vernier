/**
 * @file 14_HelgrindProfiler_Racy.cpp
 * @brief The racy addition, in a source of its own so it can be compiled with
 *        debug information: helgrind's report then names its line.
 */

#include "src/bench/demo/cpu/14_HelgrindProfiler_Racy.hpp"

#include "src/bench/demo/examples/join/inc/Join.hpp"

namespace vernier {
namespace bench {
namespace demo {
namespace helgrind_demo {

/* ----------------------------- API ----------------------------- */

void addJoinedLength(std::size_t& total, const std::vector<std::string>& parts, char sep) {
  // Read the total, add, write it back, with no lock: two threads can read
  // the same total and each write back its own sum, and one addition is lost
  total += joinV1(parts, sep).size();
}

} // namespace helgrind_demo
} // namespace demo
} // namespace bench
} // namespace vernier
