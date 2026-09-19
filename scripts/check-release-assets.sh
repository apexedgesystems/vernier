#!/usr/bin/env bash
# ==============================================================================
# check-release-assets.sh - fail if a release asset is missing.
#
# The release publishes a fixed set of files named after the project version.
# An asset built under another name (a tool whose own version differs from the
# project's) or not built at all leaves a hole that the upload step does not
# notice on its own. This check names every expected file that is absent or
# empty and exits non-zero, so the pipeline stops before anything is published.
#
# The list below is the release contract: keep it identical to the `files:`
# list of the publish step in .github/workflows/release.yml.
#
# Usage:
#   check-release-assets.sh <version> [directory]
#
# Options:
#   <version>     Project version the assets are named after (e.g. 1.2.3)
#   [directory]   Directory holding the assets (default: output)
#   --help        Show this help
#
# Output:
#   One line per asset on stdout ("ok" or "MISSING"); exit 1 if any is missing.
# ==============================================================================
set -euo pipefail

usage() {
  sed -n '2,/^# ==/{ /^#/s/^# \{0,1\}//p }' "$0"
  exit 0
}

case "${1:-}" in
--help | -h) usage ;;
esac

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: check-release-assets.sh <version> [directory]" >&2
  exit 2
fi

version="$1"
dir="${2:-output}"

if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "check-release-assets: '$version' is not a MAJOR.MINOR.PATCH version" >&2
  exit 2
fi
if [[ ! -d "$dir" ]]; then
  echo "check-release-assets: asset directory not found: $dir" >&2
  exit 2
fi

assets=(
  "vernier-${version}-x86_64-linux.tar.gz"
  "vernier-${version}-x86_64-linux-cuda.tar.gz"
  "vernier-${version}-aarch64-jetson.tar.gz"
  "vernier-${version}-aarch64-rpi.tar.gz"
  "vernier-${version}-riscv64-linux.tar.gz"
  "vernier-tools-${version}-x86_64-linux.tar.gz"
  "vernier_py_tools-${version}-py3-none-any.whl"
)

missing=0
for asset in "${assets[@]}"; do
  if [[ -s "$dir/$asset" ]]; then
    echo "ok       $asset"
  else
    echo "MISSING  $asset"
    missing=$((missing + 1))
  fi
done

if [[ "$missing" -gt 0 ]]; then
  echo "check-release-assets: $missing of ${#assets[@]} release assets missing from $dir/" >&2
  exit 1
fi
echo "check-release-assets: all ${#assets[@]} release assets present in $dir/"
