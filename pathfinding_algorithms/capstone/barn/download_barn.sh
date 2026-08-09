#!/usr/bin/env bash
# Fetch the canonical 300 BARN worlds + reference-path files (MIT license)
# from the official BARN Challenge repo, pinned to a commit for
# reproducibility. Sparse checkout keeps it to ~150 MB (worlds + run.py).
#
# Produces: capstone/barn/raw/  (gitignored)
set -euo pipefail
cd "$(dirname "$0")"

REPO=https://github.com/Daffan/the-barn-challenge.git
SHA=bf5a226f6088ec96bf0d2dbee3253a8ea6119b83  # main as of 2026-07-07

if [ ! -d raw/.git ]; then
  git clone --filter=blob:none --no-checkout "$REPO" raw
fi
git -C raw sparse-checkout set jackal_helper/worlds/BARN
git -C raw checkout --force "$SHA"

WORLD_DIR=raw/jackal_helper/worlds/BARN
PATH_DIR=$(find raw -type d -name path_files | head -1)

WORLDS=$(find "$WORLD_DIR" -maxdepth 1 -name 'world_*.world' | wc -l)
PATHS=$(find "$PATH_DIR" -name 'path_*.npy' | wc -l)
echo "worlds: $WORLDS   path files: $PATHS   (path dir: $PATH_DIR)"
[ "$WORLDS" -eq 300 ] || { echo "ERROR: expected 300 worlds, got $WORLDS" >&2; exit 1; }
[ "$PATHS" -eq 300 ] || { echo "ERROR: expected 300 path files, got $PATHS" >&2; exit 1; }

# Spawn/goal convention — verify against the values hardcoded in our metadata
# (INIT_POSITION [-2.25, 3, 1.57], GOAL_POSITION [0, 10] relative).
echo "--- spawn/goal constants in run.py @ $SHA:"
grep -n 'INIT_POSITION\|GOAL_POSITION' raw/run.py || echo "(run.py not in sparse set — fetching)"
if ! git -C raw cat-file -e "$SHA:run.py" 2>/dev/null; then
  echo "(no run.py at repo root at this SHA)"
else
  git -C raw show "$SHA:run.py" | grep -n 'INIT_POSITION\|GOAL_POSITION' | head -5
fi

echo "download_barn: OK"
