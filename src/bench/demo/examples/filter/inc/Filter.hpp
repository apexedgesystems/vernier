#ifndef VERNIER_DEMO_EXAMPLES_FILTER_HPP
#define VERNIER_DEMO_EXAMPLES_FILTER_HPP
/**
 * @file Filter.hpp
 * @brief Copy the values above a threshold, two ways: with a branch per
 *        value, and without one.
 *
 * The shared example for the walkthroughs that read hardware counters. Both
 * versions leave the same values, in the same order, in the output, and
 * differ only in how they decide what to keep. filterBranchy() tests each
 * value and stores it only when it passes, so the processor has to predict
 * the outcome of every test; filterBranchless() stores every value and
 * advances the output cursor by the outcome instead. The store in
 * filterBranchy() is conditional, and a compiler may not add a store the
 * source does not make, which leaves it fewer ways to remove the branch than
 * a conditional sum gives it: an optimizer can turn a conditional sum into a
 * conditional select, and the two versions become the same machine code.
 * The branch survived optimization in every build the walkthrough reports
 * on; the unit tests fail in a build where it does not.
 *
 * Both versions are noinline, as the join example's are: a profile of an
 * optimized build attributes its samples to the function the instructions
 * belong to, and a version inlined into its caller would vanish from it.
 */

#include <cstddef>

#include <span>
#include <vector>

namespace vernier {
namespace bench {
namespace demo {

/* ----------------------------- API ----------------------------- */

/**
 * @brief Copy the values above @p threshold into @p out, in order, testing
 *        each value with a branch.
 * @param out Room for at least values.size() elements; only the first
 *            returned count are written.
 * @return How many values were kept.
 * @note RT-safe: no allocation, one pass over the input.
 */
[[nodiscard, gnu::noinline]] std::size_t filterBranchy(std::span<const double> values,
                                                       double threshold, std::span<double> out);

/**
 * @brief The same result as filterBranchy(), without a branch on the data:
 *        every value is stored, and the cursor advances only past the ones
 *        that pass.
 * @param out Room for at least values.size() elements. The elements at and
 *            past the returned count are scratch: each holds the last value
 *            stored there, kept or not.
 * @return How many values were kept, as filterBranchy() returns.
 * @note RT-safe: no allocation, one pass over the input.
 */
[[nodiscard, gnu::noinline]] std::size_t filterBranchless(std::span<const double> values,
                                                          double threshold, std::span<double> out);

/**
 * @brief Deterministic input: @p count values uniform in [0, 1), in the
 *        order the generator produced them.
 * @param seed Seed for the generator; the same seed gives the same values.
 * @note NOT RT-safe: allocates.
 */
[[nodiscard]] std::vector<double> makeValues(std::size_t count, unsigned seed);

/**
 * @brief The values makeValues() gives for the same arguments, ascending.
 * @note NOT RT-safe: allocates.
 */
[[nodiscard]] std::vector<double> makeSortedValues(std::size_t count, unsigned seed);

} // namespace demo
} // namespace bench
} // namespace vernier

#endif // VERNIER_DEMO_EXAMPLES_FILTER_HPP
