#!/bin/sh
# Generic fake helper for the readiness tests; the first argument selects
# what it does:
#   exit N       print its arguments, exit with status N
#   env          print its environment
#   flood        print more output than a probe keeps
#   run          run until signalled (at most 30 s)
#   ignore-int   like run, but ignore SIGINT
#   fail TEXT    print TEXT on stderr, exit 3
#   both         print one line on stdout and one on stderr
#   parent       run one child and wait for it (a monitor with one child)
#   parents      run two children and wait for them (an ambiguous monitor)
#   orphan KIND [STATUS|stay]
#                start one child and leave it behind: KIND quiet sleeps
#                holding stdout, detached sleeps with its output closed,
#                writer floods stdout, escaped floods stdout from a session
#                of its own (outside the caller's process group); then exit
#                with STATUS (default 0), or with "stay" run until signalled
# Every start is recorded in FAKE_LOG with the helper's pid.

if [ -n "${FAKE_LOG:-}" ]; then
  printf 'helper %s pid=%s\n' "$*" "$$" >>"$FAKE_LOG"
fi
PATH=/usr/bin:/bin
export PATH
mode=${1:-exit}
case "$mode" in
exit)
  printf 'args: %s\n' "$*"
  exit "${2:-0}"
  ;;
env)
  env
  ;;
flood)
  yes "flood line ..................................................." | head -c 400000
  ;;
run)
  exec sleep 30
  ;;
ignore-int)
  trap '' INT
  exec sleep 30
  ;;
fail)
  echo "${2:-fake helper failed}" >&2
  exit 3
  ;;
both)
  echo "to stdout"
  echo "to stderr" >&2
  ;;
parent)
  sleep 30 &
  if [ -n "${FAKE_LOG:-}" ]; then
    printf 'helper child pid=%s\n' "$!" >>"$FAKE_LOG"
  fi
  wait
  ;;
parents)
  sleep 5 &
  first=$!
  sleep 5 &
  if [ -n "${FAKE_LOG:-}" ]; then
    printf 'helper child pid=%s\nhelper child pid=%s\n' "$first" "$!" >>"$FAKE_LOG"
  fi
  wait
  ;;
orphan)
  case "${2:-quiet}" in
  detached) sleep 30 >/dev/null 2>&1 & ;;
  writer) cat /dev/zero & ;;
  escaped) setsid cat /dev/zero & ;;
  *) sleep 30 & ;;
  esac
  if [ -n "${FAKE_LOG:-}" ]; then
    printf 'helper child pid=%s\n' "$!" >>"$FAKE_LOG"
  fi
  case "${2:-quiet}" in
  writer | escaped) sleep 0.3 ;; # the writer fills the pipe before this program ends
  esac
  if [ "${3:-0}" = "stay" ]; then
    exec sleep 30
  fi
  exit "${3:-0}"
  ;;
*)
  echo "fake helper: unknown mode $mode" >&2
  exit 2
  ;;
esac
