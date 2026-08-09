#!/usr/bin/env python3
"""Derive occupancy grids from the raw BARN world files.

The challenge repo ships no grids and the full-dataset mirror carries no
license, so the grids are rebuilt here by parsing cylinder poses straight out
of each .world (exact: every cylinder sits on the 0.15 m lattice — asserted).

Produces barn/grids/grid_%03d.npy — (30, 64) uint8, indexed [i, j] with
i along +x, j along +y, cell (0, 0) at world (-4.425, 0.075). These are small
and consumed at runtime (subgoal planner, metrics), so they ARE committed.

Usage: .venv/bin/python barn/world_to_grid.py [--raw-dir DIR] [--out-dir DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import grid as G  # noqa: E402

RAW_DEFAULT = Path(__file__).parent / 'raw/jackal_helper/worlds/BARN'
OUT_DEFAULT = Path(__file__).parent / 'grids'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', type=Path, default=RAW_DEFAULT)
    ap.add_argument('--out-dir', type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    worlds = sorted(args.raw_dir.glob('world_*.world'),
                    key=lambda p: int(p.stem.split('_')[1]))
    if not worlds:
        sys.exit(f'no worlds under {args.raw_dir} — run download_barn.sh first')

    shapes, origins, counts = set(), set(), []
    for wf in worlds:
        idx = int(wf.stem.split('_')[1])
        cyl = G.parse_world_cylinders(wf)
        occ, origin = G.cylinders_to_arena_grid(cyl)

        # structural sanity: side walls full, back wall full, top row has an
        # opening (clutter may reach the top edge — 207/300 worlds do — but
        # is never a solid wall; actual reachability is proven per-world by
        # the A* pass in build_metadata.py)
        assert occ[0, :].all() and occ[-1, :].all(), f'{wf.name}: side walls not solid'
        assert occ[:, 0].all(), f'{wf.name}: back wall not solid'
        assert not occ[1:-1, -1].all(), f'{wf.name}: top row fully closed'
        # lossless round-trip back to cylinder centers
        back = G.arena_grid_to_cylinders(occ, origin)
        assert len(back) == len(cyl)
        assert np.allclose(np.sort(back, axis=0), np.sort(cyl, axis=0), atol=1e-9)

        np.save(args.out_dir / f'grid_{idx:03d}.npy', occ)
        shapes.add(occ.shape)
        origins.add((round(origin[0], 6), round(origin[1], 6)))
        counts.append(len(cyl))

    print(f'grids written: {len(worlds)} -> {args.out_dir}')
    print(f'shapes: {sorted(shapes)}   origins: {sorted(origins)}')
    print(f'cylinders per world: min {min(counts)}, max {max(counts)}, '
          f'mean {sum(counts)/len(counts):.1f}')
    if shapes != {(30, 64)} or origins != {G.ARENA_ORIGIN}:
        sys.exit('ERROR: unexpected arena shape/origin — update common/grid.py '
                 'constants before trusting downstream geometry')


if __name__ == '__main__':
    main()
