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
*)
  echo "fake helper: unknown mode $mode" >&2
  exit 2
  ;;
esac
