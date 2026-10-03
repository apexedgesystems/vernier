#!/bin/sh
# Fake perf for the readiness tests.
#
# Records the path it was run as and its arguments in FAKE_LOG, then acts on
# FAKE_PERF_MODE:
#   ok           --version works; `stat ... --timeout N` prints counts; any
#                other invocation (a run's launch) runs until SIGINT
#   broken       --version fails the way a wrapper without the kernel's build does
#   denied       stat fails with the kernel's counter-access message
#   unsupported  stat counts, but one event is <not supported>

PATH=/usr/bin:/bin
export PATH
if [ -n "${FAKE_LOG:-}" ]; then
  printf 'perf %s %s pid=%s\n' "$0" "$*" "$$" >>"$FAKE_LOG"
fi
mode=${FAKE_PERF_MODE:-ok}

if [ "${1:-}" = "--version" ]; then
  if [ "$mode" = "broken" ]; then
    echo "WARNING: perf not found for kernel 6.8.0-138" >&2
    echo "  You may need to install the following packages for this specific kernel:" >&2
    echo "    linux-tools-6.8.0-138-generic" >&2
    exit 2
  fi
  echo "perf version 6.8.12"
  exit 0
fi

probe=no
for arg in "$@"; do
  if [ "$arg" = "--timeout" ]; then
    probe=yes
  fi
done

if [ "${1:-}" = "stat" ] && [ "$probe" = "yes" ]; then
  case "$mode" in
  denied)
    echo "Error:" >&2
    echo "Access to performance monitoring and observability operations is limited." >&2
    echo "Consider adjusting /proc/sys/kernel/perf_event_paranoid setting to open" >&2
    exit 255
    ;;
  unsupported)
    printf '%s\n' "66055,,cpu-cycles,60563,100.00,," "51605,,instructions,57709,100.00,," \
      "9805,,branches,55840,100.00,," "355,,branch-misses,52953,100.00,," \
      "<not supported>,,cache-misses,0,100.00,," >&2
    exit 0
    ;;
  *)
    printf '%s\n' "66055,,cpu-cycles,60563,100.00,," "51605,,instructions,57709,100.00,," \
      "9805,,branches,55840,100.00,," "355,,branch-misses,52953,100.00,," \
      "670,,cache-misses,49528,100.00,," >&2
    exit 0
    ;;
  esac
fi

# A run's launch: perf handles SIGINT (it prints its counts and exits), even
# though the shell that starts it in the background ignores SIGINT for its
# children; restore the default so the fake stops on SIGINT the same way.
if env --default-signal=INT true 2>/dev/null; then
  exec env --default-signal=INT sleep 30
fi
exec sleep 30
