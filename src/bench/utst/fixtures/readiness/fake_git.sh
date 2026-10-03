#!/bin/sh
# Fake git for the readiness tests: records its arguments in FAKE_LOG, then
# answers `describe` (the run's metadata asks it) with a fixed name.

if [ -n "${FAKE_LOG:-}" ]; then
  printf 'git %s\n' "$*" >>"$FAKE_LOG"
fi
case "${1:-}" in
describe) echo "v0.0.0-fake" ;;
*) exit 1 ;;
esac
