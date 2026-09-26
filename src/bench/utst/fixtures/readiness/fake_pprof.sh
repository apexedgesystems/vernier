#!/bin/sh
# Fake gperftools analyzer (installed as google-pprof or pprof) for the
# readiness tests. Records the path it was run as and its arguments in
# FAKE_LOG, then acts on FAKE_PPROF_MODE:
#   ok               prints 30 numbered report lines on stdout and a note on stderr
#   empty            prints only the note, as google-pprof does for a profile
#                    without samples
#   fail             every invocation, --help included, prints an error and exits 1
#   fail-on-profile  --help works; reading a profile fails as in mode fail

PATH=/usr/bin:/bin
export PATH
if [ -n "${FAKE_LOG:-}" ]; then
  printf 'pprof %s %s\n' "$0" "$*" >>"$FAKE_LOG"
fi
mode=${FAKE_PPROF_MODE:-ok}
if [ "$mode" = "fail-on-profile" ]; then
  if [ "${1:-}" = "--help" ]; then
    echo "usage: pprof [options] <program> <profile>"
    exit 0
  fi
  mode=fail
fi
if [ "$mode" = "fail" ]; then
  echo "fake pprof: cannot read profile" >&2
  exit 1
fi
echo "fake pprof: using local file" >&2
if [ "$mode" = "empty" ]; then
  exit 0
fi
i=1
while [ "$i" -le 30 ]; do
  echo "fake pprof line $i"
  i=$((i + 1))
done
