#!/bin/sh
# Stand-in for valgrind in CallgrindWindowReaderAssertion_test.cmake: prints
# callgrind's opening lines and its reader's assertion as valgrind 3.18.1 does
# on a GCC 11.4 Debug binary that mold linked, then ends as
# FAKE_READER_ASSERTION asks: stopped_before_start (killed by SIGSEGV),
# program_wrote_first (a line of the program's before the assertion, killed
# by SIGSEGV) or exit_1 (exits with status 1). Nothing is run.

if [ "${1:-}" = "--version" ]; then
  echo "valgrind-stand-in"
  exit 0
fi

case "${FAKE_READER_ASSERTION:-}" in
stopped_before_start | program_wrote_first | exit_1) ;;
*)
  echo "stand-in valgrind: FAKE_READER_ASSERTION is '${FAKE_READER_ASSERTION:-}'" >&2
  exit 2
  ;;
esac

{
  echo "==7== Callgrind, a call-graph generating cache profiler"
  echo "==7== Copyright (C) 2002-2017, and GNU GPL'd, by Josef Weidendorfer et al."
  echo "==7== Using Valgrind-3.18.1 and LibVEX; rerun with -h for copyright info"
  echo "==7== Command: /stand-in/CallgrindWindowProbe --profile callgrind"
  echo "==7== "
  echo "==7== For interactive control, run 'callgrind_control -h'."
  if [ "$FAKE_READER_ASSERTION" = program_wrote_first ]; then
    echo "Running main() from gmock_main.cc"
  fi
  echo
  echo "valgrind: m_debuginfo/readelf.c:2478 (vgModuleLocal_read_elf_debug_info): Assertion" \
    "'di->bss_svma + di->bss_size == svma' failed."
} >&2

if [ "$FAKE_READER_ASSERTION" = exit_1 ]; then
  exit 1
fi
kill -SEGV $$
