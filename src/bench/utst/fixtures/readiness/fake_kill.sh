#!/bin/sh
# Fake kill for the readiness tests: records its arguments in FAKE_LOG, then
# delivers the signal with the system kill.

if [ -n "${FAKE_LOG:-}" ]; then
  printf 'kill %s\n' "$*" >>"$FAKE_LOG"
fi
PATH=/usr/bin:/bin
export PATH
exec kill "$@"
