#!/usr/bin/env bash
# ==============================================================================
# bake-external-deps.sh - pre-stage the vernier build's external dependencies
# into the dev image for hermetic, offline release builds.
#
# Reads the FetchContent declarations in ExternalDependencies.cmake (the single
# source of truth for the pinned versions) and clones each dependency at its
# pinned GIT_TAG into <dest>, so CMake FetchContent uses the local copy via
# FETCHCONTENT_SOURCE_DIR_<NAME> -- no GitHub at configure time.
#
# Usage: bake-external-deps.sh <ExternalDependencies.cmake> <dest-dir>
# ==============================================================================
set -euo pipefail

CMAKE_FILE="${1:?usage: bake-external-deps.sh <ExternalDependencies.cmake> <dest-dir>}"
DEST="${2:?usage: bake-external-deps.sh <ExternalDependencies.cmake> <dest-dir>}"
mkdir -p "$DEST"

# Extract (name repo tag) for every fetchcontent_declare() block. The name is
# the first token after the opening line; GIT_REPOSITORY/GIT_TAG follow.
parse_decls() {
  awk '
    # Strip comments first so a "#" line (or an inline "# ... )" with parens)
    # cannot be mistaken for the closing paren of a declare block.
    { sub(/#.*/, "") }
    tolower($0) ~ /fetchcontent_declare\(/ { inblk = 1; name = ""; repo = ""; tag = ""; next }
    inblk && name == "" && $1 != "" && $1 != "SYSTEM" { name = $1 }
    inblk && $1 == "GIT_REPOSITORY" { repo = $2 }
    inblk && $1 == "GIT_TAG" { tag = $2 }
    inblk && $0 ~ /\)/ {
      if (name != "" && repo != "" && tag != "") print name, repo, tag
      inblk = 0
    }
  ' "$CMAKE_FILE"
}

while read -r name repo tag; do
  [ -n "$name" ] || continue
  echo "[bake] ${name} @ ${tag} <- ${repo}"
  git clone --quiet --branch "$tag" --depth 1 "$repo" "$DEST/$name"
  rm -rf "$DEST/$name/.git"
done < <(parse_decls)

echo "[bake] done -> $DEST"
ls -1 "$DEST"
