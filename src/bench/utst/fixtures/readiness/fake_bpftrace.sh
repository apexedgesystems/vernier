#!/bin/sh
# Fake bpftrace for the readiness tests.
#
# `--version` prints a version, or fails in mode version-fails. Any other
# invocation is an attach (`-q [-f json] <script>` or `-e <program> [pid]`),
# which FAKE_BPFTRACE_MODE selects:
#   ok           run until signalled; a program with an interval:s:N or
#                interval:ms:N probe (a readiness probe's self-exit) ends by
#                itself then; one without that exits on sched_process_exit
#                ends when its target pid does (at most 30 s otherwise)
#   ignore-int   like ok, but ignore SIGINT
#   ignore-target
#                like ok, but a program never ends with its target (a pid
#                bpftrace does not see, as across a pid namespace)
#   slow-attach  like ok, but take FAKE_ATTACH_S seconds (default 3) to
#                attach before an interval starts counting
#   ignore-exit  like ok, but a program never ends by itself (30 s at most):
#                a tracer that ignores its own exit()
#   ignore-int-run
#                like ok for a readiness probe; ignore SIGINT for a run's
#                launch: a program without an interval probe
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
inline=no
target=""
while [ $# -gt 0 ]; do
  case "$1" in
  -e)
    program=$2
    inline=yes
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
    elif [ "$inline" = yes ]; then
      target=$1
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
if [ "$mode" = "ignore-exit" ]; then
  limit=30
fi

# A run's launch, as opposed to a readiness probe: a program without the
# probe's self-exit.
run_launch=no
if [ -z "$seconds" ] && [ -z "$millis" ]; then
  run_launch=yes
fi
if [ "$mode" = "ignore-int" ]; then
  trap '' INT
fi
if [ "$mode" = "ignore-int-run" ] && [ "$run_launch" = yes ]; then
  trap '' INT
fi
if [ "$mode" = "slow-attach" ]; then
  sleep "${FAKE_ATTACH_S:-3}"
fi

# bpftrace's exit() on the target's sched_process_exit: a launch ends with
# its target. A probe's self-exit ends it first, whatever its target does.
if [ "$run_launch" = yes ] && [ -n "$target" ] && [ "$mode" != "ignore-target" ] &&
  printf '%s\n' "$program" | grep -q 'sched_process_exit'; then
  exec tail -s 0.1 -f /dev/null --pid="$target"
fi
exec sleep "$limit"
