#!/bin/sh
# Fake bpftrace for the readiness tests.
#
# `--version` prints a version, or fails in mode version-fails. Any other
# invocation is an attach (`-q [-f json] <script>` or `-e <script> [pid]`),
# which FAKE_BPFTRACE_MODE selects:
#   ok           run until signalled, or until the program's own
#                interval:s:N / interval:ms:N probe ends it (at most 30 s)
#   ignore-int   like ok, but ignore SIGINT
#   ignore-int-run
#                like ok for a program with an interval probe (a readiness
#                probe copy); ignore SIGINT for one without (a run's launch)
#   eperm        exit 1 at once with bpftrace's message for a non-root user
#   unsupported  exit 1 at once with bpftrace's message for a missing tracepoint
#   broken       exit 1 at once with a message of no known kind
# Every invocation is recorded in FAKE_LOG with the fake's pid.

PATH=/usr/bin:/bin
export PATH
if [ -n "${FAKE_LOG:-}" ]; then
  printf 'bpftrace %s pid=%s\n' "$*" "$$" >>"$FAKE_LOG"
fi
mode=${FAKE_BPFTRACE_MODE:-ok}

if [ "${1:-}" = "--version" ]; then
  if [ "$mode" = "version-fails" ]; then
    echo "bpftrace: error while loading shared libraries: libbcc.so.0: cannot open shared object file" >&2
    exit 127
  fi
  echo "bpftrace v0.20.2"
  exit 0
fi

program=""
while [ $# -gt 0 ]; do
  case "$1" in
  -e)
    program=$2
    shift 2
    ;;
  -f)
    shift 2
    ;;
  -*)
    shift
    ;;
  *)
    if [ -z "$program" ] && [ -f "$1" ]; then
      program=$(cat "$1")
    fi
    shift
    ;;
  esac
done

case "$mode" in
eperm)
  echo "ERROR: bpftrace currently only supports running as the root user." >&2
  exit 1
  ;;
unsupported)
  echo "stdin:1:1-36: ERROR: tracepoint not found: syscalls:sys_enter_write" >&2
  exit 1
  ;;
broken)
  echo "fake bpftrace: the program could not be loaded" >&2
  exit 1
  ;;
esac

# Run as long as the program would by itself.
limit=30
seconds=$(printf '%s\n' "$program" | sed -n 's/.*interval:s:\([0-9][0-9]*\).*/\1/p' | head -n 1)
millis=$(printf '%s\n' "$program" | sed -n 's/.*interval:ms:\([0-9][0-9]*\).*/\1/p' | head -n 1)
if [ -n "$seconds" ]; then
  limit=$seconds
elif [ -n "$millis" ]; then
  limit=$(printf '%d.%03d' $((millis / 1000)) $((millis % 1000)))
fi
if [ "$mode" = "ignore-int" ]; then
  trap '' INT
fi
if [ "$mode" = "ignore-int-run" ] && [ -z "$seconds" ] && [ -z "$millis" ]; then
  trap '' INT
fi
exec sleep "$limit"
