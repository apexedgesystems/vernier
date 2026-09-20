#!/usr/bin/env bash
# ==============================================================================
# check-release-assets.sh - resolve and verify the files a release publishes.
#
# The expected set is scripts/release-assets.txt. `resolve` turns it into the
# paths for one version; `verify` reads such a path list and fails unless every
# entry is a regular, non-empty file. The release workflow pipes the resolved
# list through a step output into both `verify` and the publish step, so the
# list that is checked is the list that is uploaded. The expected set is never
# derived from what a build happened to produce: an output that was not built,
# or was built under another name, is reported instead of silently dropped.
#
# Usage:
#   check-release-assets.sh resolve <version> [directory]
#   check-release-assets.sh verify < path-list
#
# Options:
#   resolve       Print one path per line: <directory>/<name> for every name
#                 in the asset list, with {version} replaced
#   verify        Read one path per line from stdin and check each
#   <version>     Project version the assets are named after (e.g. 1.2.3)
#   [directory]   Directory holding the assets (default: output)
#   --help        Show this help
#
# Output:
#   resolve: the path list on stdout.
#   verify:  one line per path on stdout (ok, MISSING, EMPTY or NOT-A-FILE);
#            exit 1 if any path is not ok, exit 2 for a usage error.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
readonly ASSET_LIST="$SCRIPT_DIR/release-assets.txt"

usage() {
  sed -n '2,/^# ==/{ /^#/s/^# \{0,1\}//p }' "$0"
  exit 0
}

err() {
  echo "check-release-assets: $1" >&2
}

# ==============================================================================
# resolve
# ==============================================================================

resolve() {
  local version="${1:-}"
  local dir="${2:-output}"
  local name
  local count=0

  if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    err "'$version' is not a MAJOR.MINOR.PATCH version"
    exit 2
  fi
  if [[ ! -f "$ASSET_LIST" ]]; then
    err "asset list not found: $ASSET_LIST"
    exit 2
  fi

  while IFS= read -r name || [[ -n "$name" ]]; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    echo "$dir/${name//\{version\}/$version}"
    count=$((count + 1))
  done <"$ASSET_LIST"

  if [[ "$count" -eq 0 ]]; then
    err "asset list names no files: $ASSET_LIST"
    exit 2
  fi
}

# ==============================================================================
# verify
# ==============================================================================

verify() {
  local path
  local total=0
  local bad=0

  while IFS= read -r path || [[ -n "$path" ]]; do
    [[ -z "$path" ]] && continue
    total=$((total + 1))
    if [[ ! -e "$path" ]]; then
      echo "MISSING     $path"
    elif [[ ! -f "$path" ]]; then
      echo "NOT-A-FILE  $path"
    elif [[ ! -s "$path" ]]; then
      echo "EMPTY       $path"
    else
      echo "ok          $path"
      continue
    fi
    bad=$((bad + 1))
  done

  if [[ "$total" -eq 0 ]]; then
    err "no paths on stdin; nothing would be published"
    exit 2
  fi
  if [[ "$bad" -gt 0 ]]; then
    err "$bad of $total release assets are not publishable"
    exit 1
  fi
  echo "check-release-assets: all $total release assets present"
}

# ==============================================================================
# Dispatch
# ==============================================================================

case "${1:-}" in
--help | -h) usage ;;
resolve)
  shift
  if [[ $# -lt 1 || $# -gt 2 ]]; then
    err "usage: check-release-assets.sh resolve <version> [directory]"
    exit 2
  fi
  resolve "$@"
  ;;
verify)
  if [[ $# -ne 1 ]]; then
    err "usage: check-release-assets.sh verify < path-list"
    exit 2
  fi
  verify
  ;;
*)
  err "usage: check-release-assets.sh resolve <version> [directory] | verify"
  exit 2
  ;;
esac
