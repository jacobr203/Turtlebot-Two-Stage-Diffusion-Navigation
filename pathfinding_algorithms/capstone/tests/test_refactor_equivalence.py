"""Differential tests: rewritten helpers must match the originals.

The readability pass renamed variables, extracted constants and
reworded comments across the deployment stack. These tests pin the
pure, testable helpers against a reference copy of the pre-cleanup
source and assert identical output on randomized inputs, so a rename
can never quietly change behaviour.

Set REFERENCE to a checkout of the previous revision to re-run:
    git show <rev>:capstone/deploy/policy/geometry.py > /tmp/ref/geometry.py
"""
import ast
import math
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIVE = HERE.parent / 'deploy' / 'policy'
REFERENCE = Path('/tmp/deploy_PRECLEAN/policy')

OCCUPIED, FREE, UNKNOWN = 100, 0, -1


def load_helpers(path, names, extra_globals=None):
    """Bind selected methods to a bare object, no ROS node needed."""
    src = Path(path).read_text()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    members = [n for n in cls.body
               if (isinstance(n, ast.FunctionDef) and n.name in names)
               or isinstance(n, ast.Assign)]
    module_consts = [n for n in tree.body if isinstance(n, ast.Assign)]
    mod = ast.Module(
        body=module_consts + [ast.ClassDef(name='Helpers', bases=[],
                                           keywords=[], body=members,
                                           decorator_list=[])],
        type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {'np': np, 'math': math, 'time': time,
          'UNKNOWN_CELL': UNKNOWN, 'FREE_CELL': FREE}
    ns.update(extra_globals or {})
    exec(compile(mod, str(path), 'exec'), ns)
    obj = ns['Helpers']()
    obj.get_logger = lambda: types.SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None)
    return obj


def random_grid_msg(rng, size=60, resolution=0.05):
    grid = rng.choice([FREE, OCCUPIED, UNKNOWN], size=(size, size),
                      p=[0.75, 0.15, 0.10]).astype(np.int8)
    info = types.SimpleNamespace(
        resolution=resolution, width=size, height=size,
        origin=types.SimpleNamespace(
            position=types.SimpleNamespace(x=0.0, y=0.0)))
    return types.SimpleNamespace(info=info, data=grid.ravel().tolist()), grid


def test_los_clear_matches_reference(trials=400):
    """Line-of-sight must answer identically after the rename pass."""
    old = load_helpers(REFERENCE / 'geometry.py', {'_los_clear'})
    new = load_helpers(LIVE / 'geometry.py', {'_los_clear'})
    rng = np.random.default_rng(7)
    for _ in range(trials):
        msg, _ = random_grid_msg(rng)
        ax, ay, bx, by = rng.uniform(0.2, 2.5, 4)
        assert (old._los_clear(msg, ax, ay, bx, by)
                == new._los_clear(msg, ax, ay, bx, by))
    return trials


def test_visibility_gain_matches_reference(trials=60):
    """Visibility scoring must return the same counts after the rename."""
    old = load_helpers(REFERENCE / 'viewpoint.py', {'_visibility_gain'})
    new = load_helpers(LIVE / 'viewpoint.py', {'_visibility_gain'})
    rng = np.random.default_rng(11)
    for _ in range(trials):
        msg, grid = random_grid_msg(rng, size=80)
        cands = rng.uniform(0.5, 3.0, size=(rng.integers(1, 12), 2))
        assert np.array_equal(old._visibility_gain(grid, msg.info, cands),
                              new._visibility_gain(grid, msg.info, cands))
    return trials


def test_corridor_memory_matches_reference(trials=300):
    """Corridor accept/reject decisions must be unchanged."""
    old = load_helpers(REFERENCE / 'viewpoint.py',
                       {'_corridor_kill', '_corridor_dead'})
    new = load_helpers(LIVE / 'viewpoint.py',
                       {'_corridor_kill', '_corridor_dead',
                        '_live_corridors'})
    rng = np.random.default_rng(23)
    for _ in range(trials):
        start = tuple(rng.uniform(-4, 4, 2))
        goal = tuple(rng.uniform(-4, 4, 2))
        assert (bool(old._corridor_kill(start, goal))
                == bool(new._corridor_kill(start, goal)))
        probe_start = tuple(rng.uniform(-4, 4, 2))
        probe_goal = tuple(rng.uniform(-4, 4, 2))
        assert (old._corridor_dead(probe_start, probe_goal)
                == new._corridor_dead(probe_start, probe_goal))
    return trials


if __name__ == '__main__':
    if not REFERENCE.exists():
        print(f'reference copy missing at {REFERENCE} — skipping')
        sys.exit(0)
    for fn in (test_los_clear_matches_reference,
               test_visibility_gain_matches_reference,
               test_corridor_memory_matches_reference):
        n = fn()
        print(f'{fn.__name__}: {n} randomized trials identical  PASS')
    print('\nall differential equivalence tests PASS')
