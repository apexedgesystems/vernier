#!/bin/sh
# Fake heaptrack for the readiness tests.
#
# Records the path it was run as and its arguments in FAKE_LOG, then acts on
# FAKE_HEAPTRACK_MODE:
#   ok            writes its trace to the -o path with .zst appended, runs the
#                 program after the -o value and exits with its status, as
#                 heaptrack 1.5 does
#   gz            the same with .gz, as a heaptrack built with zlib only does
#   drops-status  the same as ok, then exits 0 whatever the program returned,
#                 as heaptrack 1.3's script does; it starts the program with the
#                 command that script starts it with, "$client" "$@"
#   drops-status-unkeyed
#                 the same as drops-status, but the program is started by
#                 another command, "$program" "$@", so a shell trap keyed to
#                 the script's command does not see it
#   drops-status-inverted
#                 the same as drops-status, but the program is started as
#                 ! "$client" "$@", so the status a shell reads after it is
#                 the inverse of the program's
#   empty         exits 0 and writes nothing
#   broken        fails at start with a message, as when its libraries are missing

PATH=/usr/bin:/bin
export PATH
if [ -n "${FAKE_LOG:-}" ]; then
  printf 'heaptrack %s %s\n' "$0" "$*" >>"$FAKE_LOG"
fi
mode=${FAKE_HEAPTRACK_MODE:-ok}
output=""
previous=""
for arg in "$@"; do
  if [ "$previous" = "-o" ]; then
    output=$arg
  fi
  previous=$arg
done
case "$mode" in
broken)
  echo "heaptrack: cannot find libheaptrack_preload.so" >&2
  exit 1
  ;;
empty)
  exit 0
  ;;
gz)
  echo "fake heaptrack trace" >"$output.gz"
  ;;
*)
  echo "fake heaptrack trace" >"$output.zst"
  ;;
esac
if [ "$1" = "-o" ]; then
  shift 2
fi
client="$1"
shift
case "$mode" in
drops-status-unkeyed)
  program=$client
  "$program" "$@"
  exit 0
  ;;
drops-status-inverted)
  ! "$client" "$@"
  exit 0
  ;;
esac
"$client" "$@"
status=$?
if [ "$mode" = "drops-status" ]; then
  exit 0
fi
exit "$status"
