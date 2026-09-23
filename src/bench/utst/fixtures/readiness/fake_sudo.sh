#!/bin/sh
# Fake sudo for the readiness tests.
#
# Records its arguments in FAKE_LOG, refuses the command lines that
# FAKE_SUDO_DENY names, and runs every other command as the same user.
# FAKE_SUDO_DENY holds patterns separated by '|'; each is matched as a
# substring of the command line after "-n --" (for example "-q|kill -2"). A
# match answers the way `sudo -n` answers without a grant.

if [ -n "${FAKE_LOG:-}" ]; then
  printf 'sudo %s\n' "$*" >>"$FAKE_LOG"
fi
if [ "${1:-}" = "-n" ]; then
  shift
fi
if [ "${1:-}" = "--" ]; then
  shift
fi
command_line="$*"
if [ -n "${FAKE_SUDO_DENY:-}" ]; then
  saved_ifs=$IFS
  IFS='|'
  for pattern in $FAKE_SUDO_DENY; do
    case "$command_line" in
    *"$pattern"*)
      IFS=$saved_ifs
      echo "sudo: a password is required" >&2
      exit 1
      ;;
    esac
  done
  IFS=$saved_ifs
fi
exec "$@"
