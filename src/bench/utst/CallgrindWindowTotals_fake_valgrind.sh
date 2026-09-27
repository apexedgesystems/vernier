#!/bin/sh
# Stand-in for valgrind in CallgrindWindowTotals_test.cmake: writes the profile
# the wrap names with all three of the probe's phases in it and the "totals:"
# line FAKE_CALLGRIND_TOTALS asks for (valid, zero, missing or malformed), and
# prints GoogleTest's banner as a probe that ran would. Nothing is run.

if [ "${1:-}" = "--version" ]; then
  echo "valgrind-stand-in"
  exit 0
fi

out=""
for arg in "$@"; do
  case "$arg" in
  --callgrind-out-file=*) out="${arg#--callgrind-out-file=}" ;;
  esac
done
if [ -z "$out" ]; then
  echo "stand-in valgrind: no --callgrind-out-file" >&2
  exit 2
fi

{
  echo "events: Ir"
  for phase in workBeforeWindow workInsideWindow workAfterWindow; do
    printf 'fn=(1) %s(unsigned int)\n10 100\n' "$phase"
  done
  case "${FAKE_CALLGRIND_TOTALS:-}" in
  valid) echo "totals: 300" ;;
  zero) echo "totals: 0" ;;
  malformed) echo "totals: not_a_number" ;;
  missing) ;;
  *)
    echo "stand-in valgrind: FAKE_CALLGRIND_TOTALS is '${FAKE_CALLGRIND_TOTALS:-}'" >&2
    exit 2
    ;;
  esac
} >"$out"

echo "[==========] Running 1 test from 1 test suite."
echo "[  PASSED  ] 1 test."
