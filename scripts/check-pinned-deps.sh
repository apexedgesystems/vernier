#!/usr/bin/env bash
# ==============================================================================
# check-pinned-deps.sh - fail if any FetchContent GIT_TAG is not immutable.
#
# The hermetic build bakes each dependency's source at its pinned tag, and the
# dev image only rebuilds when ExternalDependencies.cmake changes. A moving ref
# (a branch, or a mutable/re-pointed tag) would let upstream drift without the
# file changing, so the baked source would go stale with no rebuild signal.
# Require an immutable pin: a version tag (1.2.3 / v1.2.3, optional -rc/+meta) or
# a full 40-char commit SHA.
#
# Usage: check-pinned-deps.sh [ExternalDependencies.cmake]
# ==============================================================================
set -euo pipefail

file="${1:-ExternalDependencies.cmake}"
status=0

while read -r tag; do
  [ -n "$tag" ] || continue
  if [[ "$tag" =~ ^v?[0-9]+(\.[0-9]+)*([-+.][0-9A-Za-z.]+)?$ ]] ||
    [[ "$tag" =~ ^[0-9a-fA-F]{40}$ ]]; then
    continue
  fi
  echo "$file: GIT_TAG '$tag' is not an immutable ref (use a version tag or a full commit SHA)"
  status=1
done < <(grep -E '^[[:space:]]*GIT_TAG[[:space:]]' "$file" | awk '{print $2}')

if [ "$status" -eq 0 ]; then
  echo "ExternalDependencies: all FetchContent GIT_TAGs are immutable pins"
fi
exit "$status"
