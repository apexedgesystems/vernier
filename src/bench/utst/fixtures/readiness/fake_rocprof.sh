#!/bin/sh
# Fake rocprof for the readiness tests: records the path it was run as and its
# arguments in FAKE_LOG, and exits 0. The rocprof check only resolves the
# tool; a log line means something started it.

PATH=/usr/bin:/bin
export PATH
if [ -n "${FAKE_LOG:-}" ]; then
  printf 'rocprof %s %s\n' "$0" "$*" >>"$FAKE_LOG"
fi
exit 0
