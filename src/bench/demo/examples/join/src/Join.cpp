/**
 * @file Join.cpp
 * @brief Both versions of join, and the input generator.
 */

#include "src/bench/demo/examples/join/inc/Join.hpp"

#include <random>

namespace vernier {
namespace bench {
namespace demo {

/* ----------------------------- Constants ----------------------------- */

constexpr int MIN_PART_LENGTH = 3;
constexpr int MAX_PART_LENGTH = 10;

/* ----------------------------- API ----------------------------- */

std::string joinV0(const std::vector<std::string>& parts, char sep) {
  std::string out;
  for (const std::string& part : parts) {
    // Two temporaries per part, and a copy of everything joined so far
    out = out + part + sep;
  }
  return out;
}

std::string joinV1(const std::vector<std::string>& parts, char sep) {
  std::size_t total = 0;
  for (const std::string& part : parts) {
    total += part.size() + 1;
  }

  std::string out;
  out.reserve(total); // One allocation, the final size
  for (const std::string& part : parts) {
    out += part;
    out += sep;
  }
  return out;
}

std::vector<std::string> makeParts(std::size_t count, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> length(MIN_PART_LENGTH, MAX_PART_LENGTH);
  std::uniform_int_distribution<int> letter('a', 'z');

  std::vector<std::string> parts(count);
  for (std::string& part : parts) {
    part.resize(static_cast<std::size_t>(length(rng)));
    for (char& c : part) {
      c = static_cast<char>(letter(rng));
    }
  }
  return parts;
}

} // namespace demo
} // namespace bench
} // namespace vernier
