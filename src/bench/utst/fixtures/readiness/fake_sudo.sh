#!/bin/sh
# Fake sudo for the readiness tests.
#
# Records its arguments in FAKE_LOG, refuses the command lines that
# FAKE_SUDO_DENY names, and runs every other command as the same user.
# FAKE_SUDO_DENY holds patterns separated by '|'; each is matched as a
# substring of the command line after "-n --" (for example "-q|kill -2"),
# or, when it ends in '$', as the end of that command line (" 1234$"). A
# match answers the way `sudo -n` answers without a grant. FAKE_SUDO_FAIL,
# when set, is printed for every command instead, which then fails: sudo
# itself cannot run (for example its "no new privileges" refusal).

if [ -n "${FAKE_LOG:-}" ]; then
  printf 'sudo %s\n' "$*" >>"$FAKE_LOG"
fi
if [ -n "${FAKE_SUDO_FAIL:-}" ]; then
  printf '%s\n' "$FAKE_SUDO_FAIL" >&2
  exit 1
fi
if [ "${1:-}" = "-n" ]; then
  shift
fi
if [ "${1:-}" = "--" ]; then
  shift
fi
command_line="$*"
refuse() {
  echo "sudo: a password is required" >&2
  exit 1
}
if [ -n "${FAKE_SUDO_DENY:-}" ]; then
  saved_ifs=$IFS
  IFS='|'
  for pattern in $FAKE_SUDO_DENY; do
    case "$pattern" in
    *'$')
      case "$command_line" in
      *"${pattern%?}") refuse ;;
      esac
      ;;
    *)
      case "$command_line" in
      *"$pattern"*) refuse ;;
      esac
      ;;
    esac
  done
  IFS=$saved_ifs
fi
exec "$@"
