"""Colour-task arena files for the real-robot trials (see evaluation/arenas/room1.yaml).

An arena holds the room's map, its poles and the coloured-square placements, all in
the start frame (origin at the start mark, +x forward, +y left).
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
# Recorded because a curved pole-mounted square biases the distance estimate.
MOUNTS = ('wall', 'pole')
# A pole-mounted square must lie within this of its pole's centre.
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
    """Lighting profiles; a lone top-level camera_params is one 'day' profile."""
    if raw.get('lighting'):
        return {str(k): {'profile': str(v.get('profile', 'real')),
                         'camera_params': dict(v.get('camera_params') or {})}
                for k, v in raw['lighting'].items()}
    return {'day': {'profile': 'real',
                    'camera_params': dict(raw.get('camera_params') or {})}}


def lighting_config(arena, lighting, placement=None):
    """Profile and camera controls for one lighting condition.

    Raises if the profile lacks a colour the placement hides: under the wrong
    light that colour never matches and a run would silently find nothing."""
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
        # map_saver's unknown (205) would otherwise pass the free threshold
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

    Cartographer may start a degree or so off the robot's heading, so shifting
    the origin alone would leave every position rotated about the start."""
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
