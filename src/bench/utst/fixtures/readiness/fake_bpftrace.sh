#!/bin/sh
# Fake bpftrace for the readiness tests.
#
# `--version` prints a version, or fails in mode version-fails. Any other
# invocation is an attach (`-q [-B none] [-f json] <script>` or
# `-e <program> [pid]`), which FAKE_BPFTRACE_MODE selects:
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
#   late-unsupported
#                like unsupported, but FAKE_FAIL_AFTER_S seconds (default 1.5)
#                after the start: a tracer still compiling when the start
#                grace ends. bpftrace sets its SIGINT handler only once it has
#                compiled, so SIGINT ends it before then, or does nothing with
#                FAKE_INT=ignore (SIGINT ignored by its parent, as in a
#                background job of a script)
#   end-early    exit 0 FAKE_END_AFTER_S seconds (default 0.3) after the start,
#                printing nothing: a script whose own exit() comes before the
#                check stops it
#   silent-status
#                like ok, but exit 3 on SIGINT with nothing on stderr
#   int-error    like ok, but on SIGINT print unsupported's message and exit 1
#   no-sched-switch
#                like unsupported, for the sched_switch tracepoint
#   broken       exit 1 at once with a message of no known kind
#
# A program with a probe "interval:ms:N { printf("<line>\n"); }" (a
# readiness probe's attach line) prints <line> once attached, as bpftrace
# prints its programs' output only once it has attached every probe. A
# tracer still starting, or attaching (slow-attach), ends at once on SIGINT,
# as bpftrace does before it sets its handler; once attached it ends on
# SIGINT with status 0.
#
# A program that holds the capture window the backends append (a printf of
# "<label> armed %d %d" in a program filtered on "pid == <target>", whose arm
# and stop name their threads as "tid == <id> && comm == ...") is answered as
# bpftrace answers it, once attached: when thread <arm id> of the target is
# named vernier-arm, the fake prints "<label> armed <target> <arm id>"; after
# that, when thread <stop id> is named vernier-stop, "<label> disarmed
# <target> <stop id>". On SIGINT it then prints three empty lines and
# FAKE_BPFTRACE_EXIT (default "@c: 1"), as bpftrace prints its maps, and exits
# 0. A program that binds no thread ids is answered for the first thread of
# the target, other than its main thread, named vernier-arm, and the first
# named vernier-stop (its main thread's id where a mode prints a stop line
# before any thread has that name), and a stop line whose printf takes a count
# too gets a count of 0. FAKE_WINDOW changes the answer, for every
# tracer, or, with FAKE_WINDOW_FOR set, only for one whose script path
# contains that text:
#   end-before-arm exit 0 at once, acknowledging nothing
#   no-arm         never acknowledge the arm
#   wrong-arm      acknowledge it with the main thread's id for the thread's
#   wrong-pid      acknowledge it with a pid one above the target's
#   disarm-first   print a stop line, then acknowledge the arm
#   end-after-arm  exit 0 right after acknowledging the arm
#   early-disarm   print the stop line right after the arm line, before the
#                  stop's thread takes its name, and never again
#   no-disarm      never acknowledge the stop
#   wrong-disarm   acknowledge the stop with a thread id one above the thread's
#   wrong-disarm-pid
#                  acknowledge the stop with a pid one above the target's
#   marker-only    on SIGINT, print the three empty lines only: no data
#   slow-drain     on SIGINT, wait FAKE_DRAIN_S seconds (default 1) first
#   remove-output  on SIGINT, delete the output file, then exit 0
#   empty-output   on SIGINT, empty the output file, then exit 0
#   replace-output on SIGINT, write the output file anew with FAKE_BPFTRACE_EXIT
#                  alone, then exit 0
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
script_path=""
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
      script_path=$1
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
no-sched-switch)
  echo "stdin:5:1-30: ERROR: tracepoint not found: sched:sched_switch" >&2
  exit 1
  ;;
broken)
  echo "fake bpftrace: the program could not be loaded" >&2
  exit 1
  ;;
esac

# Sleep $1 seconds in short naps: a shell takes a signal only once the
# command it runs ends, and a tracer still starting ends on SIGINT at once.
nap() {
  naps=$(awk -v s="$1" 'BEGIN { printf "%d", s * 20 }')
  while [ "$naps" -gt 0 ]; do
    sleep 0.05
    naps=$((naps - 1))
  done
}

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
int_ignored=no
if [ "$mode" = "ignore-int" ]; then
  trap '' INT
  int_ignored=yes
fi
if [ "$mode" = "ignore-int-run" ] && [ "$run_launch" = yes ]; then
  trap '' INT
  int_ignored=yes
fi

# The capture window: its label, the process it watches and the threads that
# arm and stop it.
label=$(printf '%s\n' "$program" | sed -n 's/.*printf("\([A-Za-z0-9_-]*\) armed %d %d.*/\1/p' | head -n 1)
watched=$(printf '%s\n' "$program" | sed -n 's/.*if (pid == \([0-9][0-9]*\) && (args->prev_state.*/\1/p' | head -n 1)
arm_tid=$(printf '%s\n' "$program" | sed -n 's/.*if (tid == \([0-9][0-9]*\) && comm == "vernier-arm").*/\1/p' | head -n 1)
stop_tid=$(printf '%s\n' "$program" | sed -n 's/.*if (tid == \([0-9][0-9]*\) && comm == "vernier-stop").*/\1/p' | head -n 1)
has_window=yes
if [ -z "$label" ] || [ -z "$watched" ]; then
  has_window=no
fi

# A readiness probe's attach line.
attach_line=$(printf '%s\n' "$program" | sed -n 's/^interval:ms:[0-9][0-9]* { printf("\([^"\\]*\)\\n"); }$/\1/p' | head -n 1)

if [ "$has_window" = no ] && [ "$run_launch" = yes ]; then
  if [ "$mode" = "slow-attach" ]; then
    sleep "${FAKE_ATTACH_S:-3}"
  fi
  # bpftrace's exit() on the target's sched_process_exit: a launch ends with
  # its target.
  if [ -n "$target" ] && [ "$mode" != "ignore-target" ] &&
    printf '%s\n' "$program" | grep -q 'sched_process_exit'; then
    exec tail -s 0.1 -f /dev/null --pid="$target"
  fi
  exec sleep "$limit"
fi

# On the sudo route the backend signals the only child of the process it
# started, when there is exactly one (sudo's monitor keeps the tool as its
# only child). This fake is sudo and tool in one process, and its naps and
# its loop start short-lived children; two children that live as long as it
# does keep that choice on the fake itself. They hold none of its output.
tail -s 0.1 -f /dev/null --pid=$$ >/dev/null 2>&1 &
tail -s 0.1 -f /dev/null --pid=$$ >/dev/null 2>&1 &

if [ "$mode" = "late-unsupported" ]; then
  if [ "${FAKE_INT:-}" = ignore ]; then
    trap '' INT
  fi
  nap "${FAKE_FAIL_AFTER_S:-1.5}"
  echo "stdin:1:1-36: ERROR: tracepoint not found: syscalls:sys_enter_write" >&2
  exit 1
fi
if [ "$mode" = "end-early" ]; then
  nap "${FAKE_END_AFTER_S:-0.3}"
  exit 0
fi
if [ "$mode" = "slow-attach" ]; then
  nap "${FAKE_ATTACH_S:-3}"
fi

# True when thread $1 of the watched process is named $2.
thread_is_named() {
  name=""
  read -r name 2>/dev/null <"/proc/$watched/task/$1/comm"
  [ "$name" = "$2" ]
}

# The thread of the watched process named $1, other than its main thread when
# $2 is "worker"; empty when there is none. For programs that bind no ids.
thread_named() {
  for task in /proc/"$watched"/task/*; do
    tid=${task##*/}
    if [ "$2" = worker ] && [ "$tid" = "$watched" ]; then
      continue
    fi
    name=""
    read -r name 2>/dev/null <"$task/comm"
    if [ "$name" = "$1" ]; then
      printf '%s\n' "$tid"
      return
    fi
  done
}

# The thread that arms ($1 arm) or stops ($1 stop) the window now; empty when
# none does yet.
window_thread() {
  if [ "$1" = arm ]; then
    if [ -n "$arm_tid" ]; then
      thread_is_named "$arm_tid" vernier-arm && printf '%s\n' "$arm_tid"
    else
      thread_named vernier-arm worker
    fi
  elif [ -n "$stop_tid" ]; then
    thread_is_named "$stop_tid" vernier-stop && printf '%s\n' "$stop_tid"
  else
    thread_named vernier-stop any
  fi
}
early_stop_tid=${stop_tid:-$watched}

# The stop line as the program prints it: with a count when its printf has one.
stop_count=""
if printf '%s\n' "$program" | grep -q 'disarmed %d %d %d'; then
  stop_count=" 0"
fi

# The output file, for the modes that remove or empty it: only a regular
# file, never a device such as /dev/null.
output=$(readlink /proc/$$/fd/1)
if [ ! -f "$output" ]; then
  output=""
fi
window=${FAKE_WINDOW:-}
if [ -n "${FAKE_WINDOW_FOR:-}" ]; then
  case "$script_path" in
  *"$FAKE_WINDOW_FOR"*) ;;
  *) window="" ;;
  esac
fi
on_interrupt() {
  case "$mode" in
  silent-status) exit 3 ;;
  int-error)
    echo "stdin:1:1-36: ERROR: tracepoint not found: syscalls:sys_enter_write" >&2
    exit 1
    ;;
  esac
  case "$window" in
  remove-output) [ -n "$output" ] && rm -f "$output" ;;
  empty-output) [ -n "$output" ] && : >"$output" ;;
  replace-output) [ -n "$output" ] && printf '%s\n' "${FAKE_BPFTRACE_EXIT:-@c: 1}" >"$output" ;;
  marker-only) printf '\n\n\n' ;;
  *)
    if [ "$window" = slow-drain ]; then
      sleep "${FAKE_DRAIN_S:-1}"
    fi
    printf '\n\n\n%s\n\n' "${FAKE_BPFTRACE_EXIT:-@c: 1}"
    ;;
  esac
  exit 0
}
if [ "$int_ignored" = no ]; then
  trap on_interrupt INT
fi

if [ "$window" = end-before-arm ]; then
  exit 0
fi
if [ -n "$attach_line" ]; then
  printf '%s\n' "$attach_line"
fi
limit_ms=$(printf '%s\n' "$limit" | awk '{ printf "%d", $1 * 1000 }')
started=$(date +%s%N)
armed=no
disarmed=no
while :; do
  if [ "$has_window" = no ]; then
    : # nothing to acknowledge: a probe of a script the backend adds no window to
  elif [ "$armed" = no ] && [ "$window" != no-arm ]; then
    tid=$(window_thread arm)
    if [ -n "$tid" ]; then
      pid_shown=$watched
      tid_shown=$tid
      case "$window" in
      wrong-arm) tid_shown=$watched ;;
      wrong-pid) pid_shown=$((watched + 1)) ;;
      disarm-first) printf '%s disarmed %s %s%s\n' "$label" "$watched" "$early_stop_tid" "$stop_count" ;;
      esac
      printf '%s armed %s %s\n' "$label" "$pid_shown" "$tid_shown"
      armed=yes
      if [ "$window" = end-after-arm ]; then
        exit 0
      fi
      if [ "$window" = early-disarm ]; then
        printf '%s disarmed %s %s%s\n' "$label" "$watched" "$early_stop_tid" "$stop_count"
        disarmed=yes
      fi
    fi
  elif [ "$armed" = yes ] && [ "$disarmed" = no ] && [ "$window" != no-disarm ]; then
    tid=$(window_thread stop)
    if [ -n "$tid" ]; then
      pid_shown=$watched
      tid_shown=$tid
      case "$window" in
      wrong-disarm) tid_shown=$((tid + 1)) ;;
      wrong-disarm-pid) pid_shown=$((watched + 1)) ;;
      esac
      printf '%s disarmed %s %s%s\n' "$label" "$pid_shown" "$tid_shown" "$stop_count"
      disarmed=yes
    fi
  fi
  now=$(date +%s%N)
  if [ $(((now - started) / 1000000)) -ge "$limit_ms" ]; then
    exit 0
  fi
  sleep 0.01
done
