#ifndef VERNIER_HEAPTRACKSTATUSTRAP_HPP
#define VERNIER_HEAPTRACKSTATUSTRAP_HPP
/**
 * @file HeaptrackStatusTrap.hpp
 * @brief The trap bash runs heaptrack's script with when heaptrack does not
 *        pass the exit status of the program it runs through (heaptrack 1.3's
 *        script exits with its trace compressor's): given as BASH_ENV, it
 *        writes the status of the program the script starts into the file
 *        STATUS_VARIABLE names. The doctor's heaptrack row checks it with
 *        three programs (ProfilerHeaptrack.cpp); bench run gives bash the same
 *        text before each such run (tools/rust, runner.rs), and both are held
 *        to tools/rust/tests/fixtures/heaptrack_status.bash.
 *
 * Internal to libbench; not installed.
 */

namespace vernier {
namespace bench {
namespace heaptrack_status {

/// The variable that names the file the trap writes the status into.
constexpr const char* STATUS_VARIABLE = "VERNIER_HEAPTRACK_STATUS";

/// The trap, as bash reads it from BASH_ENV.
constexpr const char* TRAP =
    R"TRAP(# bench run and bench doctor give bash this file as BASH_ENV when they run
# heaptrack's script under it: after the command that starts the program
# ("$client" "$@"), the trap writes that program's exit status, as the shell
# that waited on it read it, into the file VERNIER_HEAPTRACK_STATUS names.
__vernier_status_file=$VERNIER_HEAPTRACK_STATUS
unset BASH_ENV VERNIER_HEAPTRACK_STATUS
__vernier_status_next=0
__vernier_status_trap() {
  local status=$?
  if [ "$__vernier_status_next" = 1 ]; then
    printf '%s\n' "$status" >"$__vernier_status_file"
    __vernier_status_next=0
  fi
  # The command as heaptrack's script writes it, not expanded.
  # shellcheck disable=SC2016
  case $BASH_COMMAND in
  *'"$client" "$@"'*) __vernier_status_next=1 ;;
  esac
}
trap __vernier_status_trap DEBUG
)TRAP";

} // namespace heaptrack_status
} // namespace bench
} // namespace vernier

#endif // VERNIER_HEAPTRACKSTATUSTRAP_HPP
