# bench run and bench doctor give bash this file as BASH_ENV when they run
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
