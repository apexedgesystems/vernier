#ifndef VERNIER_DEMO_EXAMPLES_JOIN_HPP
#define VERNIER_DEMO_EXAMPLES_JOIN_HPP
/**
 * @file Join.hpp
 * @brief Join a list of strings with a separator, two ways.
 *
 * The shared CPU example for the walkthroughs. Both versions return the same
 * string and differ only in how they build it, so every tool has the same
 * answer to explain: a sampler finds the time in memmove and malloc, an
 * instruction counter finds the line, a heap profiler counts the allocations.
 */

#include <cstddef>

#include <string>
#include <vector>

namespace vernier {
namespace bench {
namespace demo {

/* ----------------------------- API ----------------------------- */

/**
 * @brief Join parts, appending @p sep after each part (the one-liner).
 * @return The joined string.
 * @note NOT RT-safe: every part builds two temporaries and copies everything
 *       joined so far.
 */
[[nodiscard]] std::string joinV0(const std::vector<std::string>& parts, char sep);

/**
 * @brief Join parts, appending @p sep after each part (measure, reserve,
 *        append in place).
 * @return The joined string, identical to joinV0's for the same arguments.
 * @note NOT RT-safe: allocates once, for the final size.
 */
[[nodiscard]] std::string joinV1(const std::vector<std::string>& parts, char sep);

/**
 * @brief Deterministic input: @p count words of 3 to 10 lowercase letters.
 * @param count Number of words.
 * @param seed Seed for the generator; the same seed gives the same words.
 * @note NOT RT-safe: allocates.
 */
[[nodiscard]] std::vector<std::string> makeParts(std::size_t count, unsigned seed);

} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_EXAMPLES_JOIN_HPP
