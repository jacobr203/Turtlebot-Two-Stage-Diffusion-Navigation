"""Colour-task arenas for the real-robot trials.

An arena is the room the colour scavenger task runs in: its boundary
(furniture and walls, kept as the Cartographer map of the built arena),
the thin poles standing in it, and where the coloured squares are mounted
for each placement. Everything is in the START FRAME, the same convention
as common/course.py: origin at the robot's start mark, +x forward, +y
left — the frame odometry and a fresh Cartographer map both have when
bringup starts on that mark.

Arena file (YAML)::

    name: room1
    id: 9101                    # >= 9100 (courses use 9000-9099)
    map: room1_map.yaml         # relative to this file; mapped from the start
    pole_diameter_m: 0.04445
    target_height_m: 0.0762     # targets' physical height; default 0.0508 (2 in)
    lighting:                   # one entry per lighting condition; a colour run
      day:                      # names one, which picks the detector's colour
        profile: real           # profile and the camera controls locked for it
        camera_params: {auto_exposure: 1, gain: 60}
      night:
        profile: real_night
        camera_params: {auto_exposure: 1, gain: 130}
    poles:                      # centres, from `course_tools.py arena-from-map`
      - [1.20, 0.35]            # then spot-checked with a tape measure
    placements:                 # one entry per square arrangement
      P1:
        - {color: purple, x: 2.10, y: -0.80, facing_deg: 90, mount: wall}
        - {color: orange, x: 1.20, y: 0.35, facing_deg: 180, mount: pole,
           pole: p00}

`facing_deg` is the direction the square's face points (its outward
normal), measured like the robot's yaw. A pole-mounted square is curved,
so its apparent width — and with it the detector's distance estimate —
is biased; the mount is recorded so results can be split by it.

The robot never reads this file during a colour run. It describes the
arena for the write-up and is the ground truth for judging finds.
"""

import math
from pathlib import Path

import numpy as np
import yaml

ARENA_ID_MIN = 9100
PROFILES_FILE = (Path(__file__).resolve().parents[1] / 'deploy'
                 / 'color_profiles.yaml')
UNKNOWN_PIXEL = 205 # map_saver's trinary "unknown"
TARGET_HEIGHT_M = 0.0508 # printed 2-inch squares, as in the sim
KNOWN_COLORS = ('light_green', 'purple', 'orange', 'lime_green', 'sky_blue')
MOUNTS = ('wall', 'pole')
# A square counts as "on" its pole when its recorded position is within
# this of the pole centre (the square sits on the pole's surface).
POLE_MOUNT_TOL_M = 0.10


class ArenaError(ValueError):
    """An arena file that cannot describe a colour-task arena."""


def load(path):
    """Read and validate an arena file (stored path is absolute)."""
    path = Path(path).resolve()
    raw = yaml.safe_load(open(path))
    if not isinstance(raw, dict) or 'placements' not in raw:
        raise ArenaError(f'{path}: not an arena (no "placements" key)')
    arena = {
        'path': path,
        'name': str(raw['name']),
        'id': int(raw['id']),
        'map': (path.parent / raw['map']).resolve() if raw.get('map') else None,
        'pole_diameter_m': float(raw['pole_diameter_m']),
        'target_height_m': float(raw.get('target_height_m', TARGET_HEIGHT_M)),
        'lighting': _lighting(raw),
        'poles': np.asarray(raw.get('poles') or [], dtype=np.float64).reshape(-1, 2),
        'placements': {str(k): list(v or []) for k, v in raw['placements'].items()},
    }
    _validate(arena)
    return arena


def _lighting(raw):
    """Lighting profiles; a file with only a top-level camera_params (the
    first format) is one 'day' profile on the default real colours."""
    if raw.get('lighting'):
        return {str(k): {'profile': str(v.get('profile', 'real')),
                         'camera_params': dict(v.get('camera_params') or {})}
                for k, v in raw['lighting'].items()}
    return {'day': {'profile': 'real',
                    'camera_params': dict(raw.get('camera_params') or {})}}


def lighting_config(arena, lighting, placement=None):
    """The profile and camera controls for one lighting condition. Fails
    loudly when the colour profile has not been calibrated for a colour
    the placement hides: under the wrong light a colour never matches, and
    a run would silently find nothing."""
    if lighting not in arena['lighting']:
        raise ArenaError(f"no lighting {lighting!r} in {arena['name']} "
                         f"(have {', '.join(arena['lighting'])})")
    cfg = arena['lighting'][lighting]
    profiles = yaml.safe_load(open(PROFILES_FILE))
    if cfg['profile'] not in profiles:
        raise ArenaError(f"lighting {lighting}: colour profile "
                         f"{cfg['profile']!r} is not in {PROFILES_FILE.name}")
    if placement is not None:
        missing = [c for c in placement_colors(arena, placement)
                   if c not in profiles[cfg['profile']]]
        if missing:
            raise ArenaError(f"lighting {lighting} ({cfg['profile']}) has no "
                             f"calibrated band for {', '.join(missing)}")
    return cfg


def pole_ids(arena):
    return [f'p{k:02d}' for k in range(len(arena['poles']))]


def _validate(a):
    if a['id'] < ARENA_ID_MIN:
        raise ArenaError(f"id {a['id']} < {ARENA_ID_MIN}")
    if a['map'] is not None and not a['map'].exists():
        raise ArenaError(f"map file missing: {a['map']}")
    ids = pole_ids(a)
    for name, squares in a['placements'].items():
        colors = [s.get('color') for s in squares]
        if len(set(colors)) != len(colors):
            raise ArenaError(f'placement {name}: a colour appears twice')
        for s in squares:
            where = f"placement {name}, {s.get('color')}"
            if s.get('color') not in KNOWN_COLORS:
                raise ArenaError(f"{where}: unknown colour (known: "
                                 f"{', '.join(KNOWN_COLORS)})")
            if s.get('mount') not in MOUNTS:
                raise ArenaError(f'{where}: mount must be wall or pole')
            for key in ('x', 'y', 'facing_deg'):
                if not isinstance(s.get(key), (int, float)):
                    raise ArenaError(f'{where}: missing {key}')
            if s['mount'] == 'pole':
                if s.get('pole') not in ids:
                    raise ArenaError(f'{where}: pole-mounted square needs '
                                     f'pole: one of p00..p{len(ids) - 1:02d}')
                px, py = a['poles'][ids.index(s['pole'])]
                if math.hypot(s['x'] - px, s['y'] - py) > POLE_MOUNT_TOL_M:
                    raise ArenaError(f"{where}: square is not on {s['pole']}")
        if 'light_green' in colors and 'lime_green' in colors:
            raise ArenaError(f'placement {name}: light_green and lime_green '
                             'share a hue band on the real camera')


def placement_colors(arena, placement):
    """The colours a placement hides, in file order (the run's --colors)."""
    if placement not in arena['placements']:
        raise ArenaError(f"no placement {placement!r} "
                         f"(have {', '.join(arena['placements'])})")
    return [s['color'] for s in arena['placements'][placement]]


def read_map(map_yaml):
    """Map image as (occupied, free, resolution, origin_xy, height_px).

    Row 0 of the image is the map's top edge (largest y)."""
    map_yaml = Path(map_yaml)
    info = yaml.safe_load(open(map_yaml))
    with open(map_yaml.parent / info['image'], 'rb') as f:
        if f.readline().split()[0] != b'P5':
            raise ArenaError(f'{map_yaml}: only binary P5 pgm maps supported')
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        w, h = map(int, line.split())
        maxval = int(f.readline())
        img = np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w)
    occ = img / maxval if info.get('negate', 0) else (maxval - img) / maxval
    occupied = occ > info.get('occupied_thresh', 0.65)
    free = occ < info.get('free_thresh', 0.25)
    if info.get('mode', 'trinary') == 'trinary':
        # map_saver writes unknown as 205, which the 0.25 free threshold
        # would otherwise read as free floor
        free &= img != UNKNOWN_PIXEL
    return occupied, free, float(info['resolution']), \
        (float(info['origin'][0]), float(info['origin'][1])), h


def read_pgm(path):
    """Raw P5 image as a uint8 array (rows top to bottom)."""
    with open(path, 'rb') as f:
        if f.readline().split()[0] != b'P5':
            raise ArenaError(f'{path}: only binary P5 pgm maps supported')
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        w, h = map(int, line.split())
        f.readline()
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w)


def map_to_start_frame(map_yaml, start_pose, out_yaml):
    """Resample a map into the frame of start_pose (x, y, yaw in the map).

    Cartographer anchors its map at the IMU and may start a degree or so
    off the robot's heading, so shifting the origin alone leaves every
    position rotated about the start. Each output cell takes the nearest
    source cell (at most half a cell of error); cells the source never
    covered are unknown."""
    map_yaml, out_yaml = Path(map_yaml), Path(out_yaml)
    info = yaml.safe_load(open(map_yaml))
    img = read_pgm(map_yaml.parent / info['image'])
    h, w = img.shape
    res = float(info['resolution'])
    ox, oy = float(info['origin'][0]), float(info['origin'][1])
    x0, y0, yaw = start_pose
    c, s = math.cos(yaw), math.sin(yaw)
    corners = np.array([[ox, oy], [ox + w * res, oy], [ox, oy + h * res],
                        [ox + w * res, oy + h * res]]) - (x0, y0)
    rot = corners @ np.array([[c, -s], [s, c]])        # R(-yaw) applied
    lo = np.floor(rot.min(axis=0) / res) * res
    hi = np.ceil(rot.max(axis=0) / res) * res
    ow, oh = int(round((hi[0] - lo[0]) / res)), int(round((hi[1] - lo[1]) / res))
    cols, rows = np.meshgrid(np.arange(ow), np.arange(oh))
    px = lo[0] + (cols + 0.5) * res                     # start frame
    py = lo[1] + (oh - rows - 0.5) * res
    sx = x0 + c * px - s * py                           # back to the map
    sy = y0 + s * px + c * py
    sc = np.floor((sx - ox) / res).astype(int)
    sr = h - 1 - np.floor((sy - oy) / res).astype(int)
    inside = (sc >= 0) & (sc < w) & (sr >= 0) & (sr < h)
    out = np.full((oh, ow), UNKNOWN_PIXEL, dtype=np.uint8)
    out[inside] = img[sr[inside], sc[inside]]
    pgm = out_yaml.with_suffix('.pgm')
    pgm.write_bytes(f'P5\n{ow} {oh}\n255\n'.encode() + out.tobytes())
    info.update({'image': pgm.name,
                 'origin': [float(lo[0]), float(lo[1]), 0.0]})
    out_yaml.write_text(yaml.safe_dump(info))
    return out_yaml


def cells_to_xy(rows, cols, res, origin, height):
    """Map image (row, col) -> start-frame xy of the cell centres."""
    return np.stack([origin[0] + (np.asarray(cols) + 0.5) * res,
                     origin[1] + (height - np.asarray(rows) - 0.5) * res],
                    axis=1)
