#!/bin/sh
# Fake gperftools analyzer (installed as google-pprof or pprof) for the
# readiness tests. Records the path it was run as and its arguments in
# FAKE_LOG, then acts on FAKE_PPROF_MODE:
#   ok     prints 30 numbered report lines on stdout and a note on stderr
#   empty  prints only the note, as google-pprof does for a profile without samples
#   fail   prints an error on stderr and exits 1

PATH=/usr/bin:/bin
export PATH
if [ -n "${FAKE_LOG:-}" ]; then
  printf 'pprof %s %s\n' "$0" "$*" >>"$FAKE_LOG"
fi
if [ "${FAKE_PPROF_MODE:-ok}" = "fail" ]; then
  echo "fake pprof: cannot read profile" >&2
  exit 1
fi
echo "fake pprof: using local file" >&2
if [ "${FAKE_PPROF_MODE:-ok}" = "empty" ]; then
  exit 0
fi
i=1
while [ "$i" -le 30 ]; do
  echo "fake pprof line $i"
  i=$((i + 1))
done
