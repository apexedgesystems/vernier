#!/bin/sh
# Fake heaptrack for the readiness tests.
#
# Records the path it was run as and its arguments in FAKE_LOG, then acts on
# FAKE_HEAPTRACK_MODE:
#   ok      writes its trace to the -o path with .zst appended, and exits 0
#   gz      the same with .gz, as a heaptrack built with zlib only does
#   empty   exits 0 and writes nothing
#   broken  fails at start with a message, as when its libraries are missing

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
exit 0
