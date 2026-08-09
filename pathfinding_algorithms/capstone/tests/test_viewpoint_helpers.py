"""Unit tests for viewpoint scoring and corridor memory.

Exercises the real methods without constructing a ROS node, by
extracting them from the mixin and binding them to a bare object."""
import math, time, types
import numpy as np
import sys
sys.path.insert(0, '/home/turtlebot/pathfinding_algorithms/capstone')

# --- pull the two helpers off the class without constructing a ROS node
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pn", "/home/turtlebot/pathfinding_algorithms/capstone/deploy/policy/viewpoint.py")

import ast, textwrap
src = open("/home/turtlebot/pathfinding_algorithms/capstone/deploy/policy/viewpoint.py").read()
tree = ast.parse(src)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
want = {'_visibility_gain', '_corridor_kill', '_corridor_dead',
        '_live_corridors'}
fns = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in want]
WANT_CONSTS = ('VIS_RANGE_M', 'VIS_RAYS', 'VANTAGE_MIN_MOVE_M',
               'CORRIDOR_MIN_LEN_M', 'CORRIDOR_COOL_S',
               'CORRIDOR_MATCH_RADIUS_M', 'CORRIDOR_MATCH_ANGLE_RAD',
               'OCCUPIED_THRESHOLD', 'VIS_STEP_FRACTION', 'VIS_MIN_STEPS')
consts = [n for n in cls.body if isinstance(n, ast.Assign)
          and any(getattr(t, 'id', '') in WANT_CONSTS for t in n.targets)]
assert len(consts) == len(WANT_CONSTS), \
    f"harness missed a constant: got {len(consts)} of {len(WANT_CONSTS)}"
mod = ast.Module(body=[ast.ClassDef(name='T', bases=[], keywords=[],
                                    body=consts + fns, decorator_list=[])],
                 type_ignores=[])
ast.fix_missing_locations(mod)
ns = {'np': np, 'math': math, 'time': time,
      'UNKNOWN_CELL': -1, 'FREE_CELL': 0}
exec(compile(mod, '<t>', 'exec'), ns)
T = ns['T']
t = T()
t.get_logger = lambda: types.SimpleNamespace(warning=lambda *a, **k: None,
                                             info=lambda *a, **k: None)

# ---------------- visibility gain ----------------
# 20x20 m map at 0.05 => 400x400. Left half unknown behind a solid wall
# at x=10.0; right half free. A candidate just RIGHT of the wall sees no
# unknown (wall blocks); a candidate in the open with unknown ahead sees lots.
res = 0.05
W = H = 400
grid = np.zeros((H, W), dtype=np.int8)      # free
grid[:, :200] = -1                          # unknown left half
grid[:, 199:201] = 100                      # solid wall on the seam
info = types.SimpleNamespace(
    resolution=res, width=W, height=H,
    origin=types.SimpleNamespace(position=types.SimpleNamespace(x=0.0, y=0.0)))

behind_wall = np.array([[10.5, 10.0]])      # right of wall, wall blocks LOS
g_blocked = t._visibility_gain(grid, info, behind_wall)

# now open a 1 m doorway in the wall at y=10 and re-measure
grid2 = grid.copy()
grid2[190:210, 199:201] = 0
g_door = t._visibility_gain(grid2, info, behind_wall)

print(f"visibility gain behind solid wall : {g_blocked[0]:.0f}")
print(f"visibility gain with 1 m doorway  : {g_door[0]:.0f}")
assert g_blocked[0] == 0, "solid wall must occlude ALL unknown"
assert g_door[0] > 50, "doorway must reveal unknown area"

# two candidates: one facing the doorway, one far down the wall
cands = np.array([[10.5, 10.0], [10.5, 17.0]])
gg = t._visibility_gain(grid2, info, cands)
print(f"doorway-facing {gg[0]:.0f}  vs  far-down-wall {gg[1]:.0f}")
assert gg[0] > gg[1], "the seam that reveals area must outrank the one that doesn't"

# ---------------- corridor memory ----------------
t2 = T()
t2.get_logger = t.get_logger
frm, to = (0.0, 0.0), (4.0, 0.0)
assert not t2._corridor_dead(frm, to), "clean slate must not be dead"
t2._corridor_kill(frm, to)
assert t2._corridor_dead(frm, to), "the exact failed approach must be dead"

# a neighbouring pose behind the SAME blockage must also be rejected
neighbour = (4.0, 0.4)
assert t2._corridor_dead(frm, neighbour), \
    "REGRESSION: 0.4 m neighbour behind the same gap must be dead"

# a genuinely different approach: same target, opposite side
far_side = (0.0, 0.0)
assert not t2._corridor_dead((8.0, 0.0), (4.0, 0.0)), \
    "approach from the far side is a DIFFERENT corridor and must survive"
# a distant unrelated corridor
assert not t2._corridor_dead((0.0, 9.0), (4.0, 9.0)), \
    "unrelated corridor elsewhere must survive"

# cooldown, not blacklist
t3 = T(); t3.get_logger = t.get_logger
t3._corridor_kill(frm, to, cool_s=-1.0)     # already expired
assert not t3._corridor_dead(frm, to), "expired corridor must revive (cooldown)"

print("\nviewpoint helper tests PASS")

# ---------- regressions observed in live testing ----------
t4 = T(); t4.get_logger = t.get_logger
# A very short failed approach must NOT create a corridor: its
# midpoint sits on the robot, so the dead sector would blanket every
# candidate direction at once and leave no viable pose anywhere.
short_from, short_to = (0.0, 0.0), (0.40, 0.0)
assert t4._corridor_kill(short_from, short_to) is False, \
    "REGRESSION: a 0.40 m wedge must not be recorded as a corridor"
assert not t4._corridor_dead(short_from, short_to), \
    "REGRESSION: short failure must not blanket its direction"
# candidates elsewhere in that direction must stay alive
assert not t4._corridor_dead((0.0, 0.0), (2.5, 0.3)), \
    "REGRESSION: a short failure must not kill the wider region"
# 2. a genuine traverse still earns a corridor
assert t4._corridor_kill((0.0, 0.0), (4.0, 0.0)) is True, \
    "a real 4 m traverse must still be recorded"
assert t4._corridor_dead((0.0, 0.0), (4.0, 0.4)), \
    "the 0.4 m neighbour behind the same gap must still be dead"
print("regression tests PASS "
      "(short-wedge guard + real corridor preserved)")
