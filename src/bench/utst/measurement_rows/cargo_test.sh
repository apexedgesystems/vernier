#!/bin/sh
# CTest's launcher for one ignored test of a cargo test target. It passes only
# when cargo exits 0 and reports that the test passed: cargo also exits 0 when
# its filter matches no test, and a CTest pass expression ignores the exit
# status. Cargo keeps to the lockfile and, where VERNIER_RUST_OFFLINE is not
# empty when the test runs, to the crates already fetched, as the tools build
# does.
# Usage: cargo_test.sh <cargo> <test name> <cargo test option>...

cargo=$1
name=$2
shift 2
out=$("$cargo" test --locked ${VERNIER_RUST_OFFLINE:+--offline} "$@" "$name" -- --ignored --exact 2>&1)
status=$?
printf '%s\n' "$out"
if [ "$status" -ne 0 ]; then
  echo "cargo_test.sh: cargo exited $status"
  exit "$status"
fi
if ! printf '%s\n' "$out" | grep -Fqx "test $name ... ok"; then
  echo "cargo_test.sh: cargo reported no passing test named $name"
  exit 1
fi
