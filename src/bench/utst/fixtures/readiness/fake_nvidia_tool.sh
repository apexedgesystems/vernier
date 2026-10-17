#!/bin/sh
# Fake NVIDIA tool for the readiness tests, installed as nsys, ncu or
# compute-sanitizer: records the name it was run as and its arguments in
# FAKE_LOG, then answers --version as that tool does, or, with
# FAKE_NVIDIA_TOOL_MODE=broken, fails as a tool that cannot load its
# libraries. Any other invocation fails: the checks only ask --version.

name=${0##*/}
if [ -n "${FAKE_LOG:-}" ]; then
  printf '%s %s\n' "$name" "$*" >>"$FAKE_LOG"
fi
if [ "${FAKE_NVIDIA_TOOL_MODE:-ok}" = "broken" ]; then
  echo "$name: error while loading shared libraries: libcuda.so.1: cannot open shared object file" >&2
  exit 127
fi
if [ "${1:-}" != "--version" ]; then
  echo "fake $name: only --version is answered" >&2
  exit 1
fi
case "$name" in
nsys) echo "NVIDIA Nsight Systems version 2026.3.1.157-263138048394v0" ;;
ncu)
  echo "NVIDIA (R) Nsight Compute Command Line Profiler"
  echo "Version 2026.2.0.0 (build 37790515) (public-release)"
  ;;
*)
  echo "NVIDIA (R) Compute Sanitizer"
  echo "Version 2025.4.0.0 (build 36782660) (public-release)"
  ;;
esac
