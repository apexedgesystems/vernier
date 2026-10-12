#!/bin/sh
# Fake valgrind for the readiness tests.
#
# Records the path it was run as and its arguments in FAKE_LOG, then acts on
# FAKE_VALGRIND_MODE:
#   ok      --version works, as does starting any tool: exits 0 without running
#           the program. Like valgrind, a tool started without its output
#           option writes its default output file into the working directory
#           (massif.out.<pid> or callgrind.out.<pid>)
#   broken  --version works; starting a tool fails with valgrind's message, as
#           when the tool is not installed

PATH=/usr/bin:/bin
export PATH
if [ -n "${FAKE_LOG:-}" ]; then
  printf 'valgrind %s %s\n' "$0" "$*" >>"$FAKE_LOG"
fi
mode=${FAKE_VALGRIND_MODE:-ok}

if [ "${1:-}" = "--version" ]; then
  echo "valgrind-3.22.0"
  exit 0
fi

tool=memcheck
output=""
for arg in "$@"; do
  case "$arg" in
  --tool=*) tool=${arg#--tool=} ;;
  --log-file=* | --massif-out-file=* | --callgrind-out-file=*) output=${arg#*=} ;;
  esac
done

if [ "$mode" = "broken" ]; then
  echo "valgrind: failed to start tool '$tool' for platform 'amd64-linux': No such file or directory" >&2
  exit 1
fi
if [ -z "$output" ]; then
  case "$tool" in
  massif | callgrind) : >"$tool.out.$$" ;;
  esac
fi
exit 0
