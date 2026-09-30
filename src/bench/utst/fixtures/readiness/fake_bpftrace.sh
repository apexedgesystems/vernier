#!/bin/sh
# Fake bpftrace for the readiness tests.
#
# `--version` prints a version, or fails in mode version-fails. Any other
# invocation is an attach (`-q [-f json] <script>` or `[-B <mode>] -e
# <program> [pid]`), which FAKE_BPFTRACE_MODE selects:
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
# A program that prints "<label> armed %d %d" (the off-CPU script) runs the
# armed capture window, below, under the same modes.
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
  -f | -B)
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
# The same bound in centiseconds, for the armed window's clock.
limit_cs=3000
if [ -n "$seconds" ] && [ "$mode" != "ignore-exit" ]; then
  limit_cs=$((seconds * 100))
elif [ -n "$millis" ] && [ "$mode" != "ignore-exit" ]; then
  limit_cs=$(((millis + 9) / 10))
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

# The armed capture window of a program that prints "<label> armed %d %d":
# the fake watches the target's threads, as the script's sched_switch probe
# would, prints "<label> armed <target> <tid>" once a thread named
# vernier-arm other than the main thread exists, then "<label> disarmed
# <target> <tid> <n>" once one named vernier-stop does. On SIGINT or SIGTERM,
# at its self-exit, or when a launch's target ends, it prints its maps and
# exits 0, as bpftrace does. FAKE_BPFTRACE_OUTPUT names the maps it prints
# (default "@armed: 2" alone), FAKE_OFFCPU_RECORDED the count its disarm line
# gives (default: the sum of the @offcpu_blocks counts in those maps).
# FAKE_OFFCPU changes one step:
#   no-arm         never acknowledges the arm
#   wrong-arm      acknowledges it for the main thread, not the arm thread
#   end-after-arm  prints its maps and exits 0 right after the arm
#   no-disarm      never acknowledges the stop
#   remove-output  removes its output file once it has printed its maps
#   bad-exit       exits 3 instead of 0 once it has printed its maps
label=$(printf '%s\n' "$program" | sed -n 's/.*printf("\([a-z]*\) armed %d %d.*/\1/p' | head -n 1)
if [ -n "$label" ]; then
  window=${FAKE_OFFCPU:-}
  recorded=${FAKE_OFFCPU_RECORDED:-}
  if [ -z "$recorded" ]; then
    recorded=0
    if [ -n "${FAKE_BPFTRACE_OUTPUT:-}" ]; then
      recorded=$(awk '/^, .*\]: [0-9]+$/ { n += $NF } END { print n + 0 }' "$FAKE_BPFTRACE_OUTPUT")
    fi
  fi
  finish() {
    if [ -n "${FAKE_BPFTRACE_OUTPUT:-}" ]; then
      cat "$FAKE_BPFTRACE_OUTPUT"
    else
      printf '\n\n@armed: 2\n'
    fi
    if [ "$window" = remove-output ]; then
      rm -f "$(readlink "/proc/$$/fd/1")"
    fi
    if [ "$window" = bad-exit ]; then
      exit 3
    fi
    exit 0
  }
  if [ "$mode" != ignore-int ] && { [ "$mode" != ignore-int-run ] || [ "$run_launch" != yes ]; }; then
    trap finish INT
  fi
  trap finish TERM
  # The loop below starts a short sleep at a time. A stop through sudo
  # signals the only child of the process it started when there is exactly
  # one (sudo's monitor keeps bpftrace as its only child); two idle children,
  # which end with this process, keep the fake from ever having exactly one,
  # so the stop signals the fake itself, as it signals bpftrace.
  tail -s 0.1 -f /dev/null --pid=$$ &
  tail -s 0.1 -f /dev/null --pid=$$ &
  echo "Attaching 2 probes..."
  # Centiseconds since boot, from /proc/uptime: the self-exit's clock.
  read -r up _ </proc/uptime
  deadline=$((${up%.*}${up#*.} + limit_cs))
  state=arming
  while :; do
    if [ "$run_launch" = yes ] && [ -n "$target" ] && [ "$mode" != ignore-target ] &&
      [ ! -d "/proc/$target" ]; then
      finish
    fi
    read -r up _ </proc/uptime
    if [ "${up%.*}${up#*.}" -ge "$deadline" ]; then
      finish
    fi
    for task in /proc/"$target"/task/*; do
      name=
      { IFS= read -r name <"$task/comm"; } 2>/dev/null
      tid=${task##*/}
      if [ "$state" = arming ] && [ "$name" = vernier-arm ] && [ "$tid" != "$target" ] &&
        [ "$window" != no-arm ]; then
        if [ "$window" = wrong-arm ]; then
          tid=$target
        fi
        printf '%s armed %s %s\n' "$label" "$target" "$tid"
        state=armed
        if [ "$window" = end-after-arm ]; then
          finish
        fi
      elif [ "$state" = armed ] && [ "$name" = vernier-stop ] && [ "$window" != no-disarm ]; then
        printf '%s disarmed %s %s %s\n' "$label" "$target" "$tid" "$recorded"
        state=disarmed
      fi
    done
    sleep 0.01
  done
fi

# bpftrace's exit() on the target's sched_process_exit: a launch ends with
# its target. A probe's self-exit ends it first, whatever its target does.
if [ "$run_launch" = yes ] && [ -n "$target" ] && [ "$mode" != "ignore-target" ] &&
  printf '%s\n' "$program" | grep -q 'sched_process_exit'; then
  exec tail -s 0.1 -f /dev/null --pid="$target"
fi
exec sleep "$limit"
