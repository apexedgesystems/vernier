#!/bin/sh
# Fake perf for the readiness tests.
#
# Records the path it was run as and its arguments in FAKE_LOG, then acts on
# FAKE_PERF_MODE:
#   ok           --version works; `stat ... --timeout N` prints counts; any
#                other invocation (a run's launch) runs until SIGINT and then
#                writes what perf writes: stat's counts to stderr, or record's
#                data file (-o) and its "Captured and wrote" line. Given
#                `--control fifo:<ctl>,<ack>`, it reads the ping waiting in
#                <ctl> and answers "ack" in <ack>, as perf does once it counts
#   broken       --version fails the way a wrapper without the kernel's build does
#   denied       stat fails with the kernel's counter-access message
#   unsupported  stat counts, but one event is <not supported>
#   no-control   --control is an unknown option, as for a perf without it
#   no-ack       takes --control and never answers on it
# and, for a run's launch only:
#   slow-ack     answers the ping 2 s after it starts
#   exit-early   fails at once with an error message
#   exit-soon    exits (status 3) half a second after it answers, logging
#                "perf exited pid=<pid>" to FAKE_LOG first
#   slow-stop    writes its output 2 s after SIGINT
#   ignore-int   ignores SIGINT; SIGTERM ends it without output
#   hang         ignores SIGINT and SIGTERM (SIGKILL ends it)
#   error-text   answers SIGINT with an error message instead of counts
#   error-exit   answers SIGINT with a count, then "Error: final read
#                failed", and exits 42
#   error-after  the same, but exits 130 as a finished perf does
#   heading-only answers SIGINT with stat's heading and no count
#   none-counted answers SIGINT with stat's heading and every event
#                <not supported> or <not counted>
#   raise-int    answers SIGINT with the counts, then ends by SIGINT itself,
#                as perf does, instead of exiting 130
#   some-unsupported  answers SIGINT with the counts and one event
#                <not supported>, as perf on the Pi reports branches:u

PATH=/usr/bin:/bin
export PATH
if [ -n "${FAKE_LOG:-}" ] && [ -z "${FAKE_PERF_RESTORED:-}" ]; then
  printf 'perf %s %s pid=%s\n' "$0" "$*" "$$" >>"$FAKE_LOG"
fi
mode=${FAKE_PERF_MODE:-ok}

if [ "${1:-}" = "--version" ]; then
  if [ "$mode" = "broken" ]; then
    echo "WARNING: perf not found for kernel 6.8.0-138" >&2
    echo "  You may need to install the following packages for this specific kernel:" >&2
    echo "    linux-tools-6.8.0-138-generic" >&2
    exit 2
  fi
  echo "perf version 6.8.12"
  exit 0
fi

probe=no
ctl=""
ack=""
prev=""
for arg in "$@"; do
  if [ "$arg" = "--timeout" ]; then
    probe=yes
  fi
  if [ "$prev" = "--control" ]; then
    spec=${arg#fifo:}
    ctl=${spec%%,*}
    ack=${spec#*,}
  fi
  prev=$arg
done

if [ -n "$ctl" ] && [ "$mode" = "no-control" ]; then
  echo "  Error: unknown option \`control'" >&2
  echo "" >&2
  echo " Usage: perf stat [<options>] [<command>]" >&2
  exit 129
fi

# Read the ping waiting in the control fifo and answer it, as perf does from
# its main loop once its counters are on. Both fifos are opened read-write,
# so neither open waits.
answer() {
  if [ -n "$ctl" ] && [ "$mode" != "no-ack" ]; then
    exec 5<>"$ctl" 6<>"$ack"
    if read -r cmd <&5 && [ "$cmd" = "ping" ]; then
      printf 'ack\n' >&6
    fi
  fi
}

if [ "${1:-}" = "stat" ] && [ "$probe" = "yes" ]; then
  case "$mode" in
  denied)
    echo "Error:" >&2
    echo "Access to performance monitoring and observability operations is limited." >&2
    echo "Consider adjusting /proc/sys/kernel/perf_event_paranoid setting to open" >&2
    exit 255
    ;;
  unsupported)
    answer
    printf '%s\n' "66055,,cpu-cycles,60563,100.00,," "51605,,instructions,57709,100.00,," \
      "9805,,branches,55840,100.00,," "355,,branch-misses,52953,100.00,," \
      "<not supported>,,cache-misses,0,100.00,," >&2
    exit 0
    ;;
  *)
    answer
    printf '%s\n' "66055,,cpu-cycles,60563,100.00,," "51605,,instructions,57709,100.00,," \
      "9805,,branches,55840,100.00,," "355,,branch-misses,52953,100.00,," \
      "670,,cache-misses,49528,100.00,," >&2
    exit 0
    ;;
  esac
fi

# A run's launch. perf handles SIGINT even when it starts with SIGINT ignored
# (a shell ignores it for a command it starts in the background), and a shell
# cannot trap a signal ignored on entry: start again with the default, once.
if [ -z "${FAKE_PERF_RESTORED:-}" ] && env --default-signal=INT true 2>/dev/null; then
  FAKE_PERF_RESTORED=1 exec env --default-signal=INT /bin/sh "$0" "$@"
fi
if [ "$mode" = "exit-early" ]; then
  echo "Error: failed to open counters: No such process" >&2
  exit 1
fi
out=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "-o" ]; then
    out=$arg
  fi
  prev=$arg
done

finish() {
  case "$mode" in
  error-text)
    echo "Error: the fake perf could not read its counters" >&2
    ;;
  heading-only)
    echo "" >&2
    echo " Performance counter stats for process id '$$':" >&2
    echo "" >&2
    ;;
  none-counted)
    echo "" >&2
    echo " Performance counter stats for process id '$$':" >&2
    echo "" >&2
    echo "     <not supported>      cpu-cycles:u" >&2
    echo "     <not counted> msec   task-clock" >&2
    echo "" >&2
    echo "       0.101234567 seconds time elapsed" >&2
    ;;
  *)
    if [ -n "$out" ]; then
      printf 'fake perf data\n' >"$out"
      echo "[ perf record: Woken up 1 times to write data ]" >&2
      echo "[ perf record: Captured and wrote 0.001 MB $out ]" >&2
    else
      echo "" >&2
      echo " Performance counter stats for process id '$$':" >&2
      echo "" >&2
      echo "            66,055      cpu-cycles:u" >&2
      echo "            51,605      instructions:u      #    0.78  insn per cycle" >&2
      echo "             9,805      branches:u                                  (79.98%)" >&2
      if [ "$mode" = "some-unsupported" ]; then
        echo "     <not supported>      cache-misses:u" >&2
      fi
      echo "" >&2
      echo "       0.101234567 seconds time elapsed" >&2
      if [ "$mode" = "error-exit" ] || [ "$mode" = "error-after" ]; then
        echo "Error: final read failed" >&2
      fi
    fi
    ;;
  esac
}

on_int() {
  kill "$sleeper" 2>/dev/null
  if [ "$mode" = "slow-stop" ]; then
    sleep 2
  fi
  finish
  if [ "$mode" = "error-exit" ]; then
    exit 42
  fi
  if [ "$mode" = "raise-int" ]; then
    trap - INT
    kill -INT $$
  fi
  exit 130
}

# The signals are handled before the answer: the measured phase, and so the
# stop, may follow it at once.
case "$mode" in
hang) trap '' INT TERM ;;
ignore-int)
  trap '' INT
  trap 'kill "$sleeper" 2>/dev/null; exit 143' TERM
  ;;
*) trap on_int INT ;;
esac
if [ "$mode" = "slow-ack" ]; then
  sleep 2
fi
answer
if [ "$mode" = "exit-soon" ]; then
  sleep 0.5
  echo "Error: the target process exited" >&2
  if [ -n "${FAKE_LOG:-}" ]; then
    printf 'perf exited pid=%s\n' "$$" >>"$FAKE_LOG"
  fi
  exit 3
fi
# Wait in one-second steps: a trapped signal interrupts `wait` at once, and a
# SIGKILL leaves no sleep behind for more than a second.
while :; do
  sleep 1 &
  sleeper=$!
  wait "$sleeper"
done
