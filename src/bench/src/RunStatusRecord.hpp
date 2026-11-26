#ifndef VERNIER_RUNSTATUSRECORD_HPP
#define VERNIER_RUNSTATUSRECORD_HPP
/**
 * @file RunStatusRecord.hpp
 * @brief The record of a run's exit status that libbench keeps for bench run
 *        when the tool wrapping the benchmark does not pass that status
 *        through (heaptrack 1.3's script exits with its trace compressor's).
 *
 * bench run asks for it by setting TOKEN_VARIABLE, beside
 * VERNIER_EXTERNAL_WRAP_DIR, to a token naming the run. libbench takes the
 * token from its environment as it loads, so a program the benchmark starts
 * does not record for it, and record() writes "<token> <status>\n" into
 * RECORD_NAME in that folder: ProfilerRegistry::finishRun() as a run ends, the
 * ABI check before it ends the process. Without the variable, nothing is
 * written.
 *
 * Internal to libbench; not installed.
 */

namespace vernier {
namespace bench {
namespace run_status {

/// The variable that asks for the record; its value is the run's token.
constexpr const char* TOKEN_VARIABLE = "VERNIER_EXTERNAL_WRAP_STATUS";

/// The record's name in the wrap's folder.
constexpr const char* RECORD_NAME = "exit_status.txt";

/**
 * @brief Record @p status, the status the process is about to exit with, when
 * bench run asked for the record and this is the process its token was given
 * to; otherwise do nothing. A record that cannot be written is reported on
 * stderr, and bench run then finds none.
 * @note Allocates nothing once libbench has loaded: the ABI check calls it
 * after a mismatch, when no object shared with the benchmark can be trusted.
 */
__attribute__((visibility("hidden"))) void record(int status) noexcept;

} // namespace run_status
} // namespace bench
} // namespace vernier

#endif // VERNIER_RUNSTATUSRECORD_HPP
