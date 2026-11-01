"""Seeded run orders and which cells are still owed."""

from pathlib import Path
import random

from .config import (
    COLOUR_HZ, COLOUR_MODELS, COLOUR_ROUNDS, HZ_LIST, MODELS, RERUN,
    RESULTS, ROUNDS)
from .records import read_rows


def make_schedule(course, run_id, rounds=ROUNDS, models=MODELS,
                  hz_list=HZ_LIST):
    """Randomized complete blocks: each round runs every (model, hz) cell
    once, in an order seeded by run id + course."""
    rng = random.Random(f'{run_id}/{course["name"]}')
    rows, order = [], 0
    for rnd in range(rounds):
        cells = [(m, hz) for m in models for hz in hz_list]
        rng.shuffle(cells)
        for m, hz in cells:
            rows.append({'order_idx': order, 'course': course['name'],
                         'world_idx': course['id'], 'model': m, 'hz': hz,
                         'round': rnd})
            order += 1
    return rows


def cell_key(row):
    return (row['model'], int(row['world_idx']), float(row['hz']),
            int(row['round'] if 'round' in row else row['trial']))


def pending_cells(schedule, done_rows):
    """Cells owed a counted verdict: never attempted first, then those whose
    last attempt was aborted/harness_error."""
    last = {}
    for r in done_rows:
        last[cell_key(r)] = r['verdict']
    fresh = [c for c in schedule if cell_key(c) not in last]
    retry = [c for c in schedule if last.get(cell_key(c)) in RERUN]
    return fresh + retry


def next_trial_index(trials_dir):
    """First unused tNNN index, from folders rather than rows: an interrupted
    run's leftover bag folder would make the next recording fail."""
    used = [int(d.name[1:4]) for d in Path(trials_dir).glob('t[0-9][0-9][0-9]_*')
            if d.is_dir()]
    return max(used) + 1 if used else 0


def run_dir(run_id, shakedown):
    return RESULTS / (f'{run_id}_shakedown' if shakedown else run_id)


def load_schedule(course, run_id):
    path = run_dir(run_id, False) / f'schedule_{course["name"]}.csv'
    if not path.exists():
        raise SystemExit(f'no schedule at {path} — run the "schedule" '
                         'command first, before the session')
    rows = read_rows(path)
    for r in rows:
        r['hz'], r['round'] = float(r['hz']), int(r['round'])
        r['world_idx'], r['order_idx'] = int(r['world_idx']), int(r['order_idx'])
    return rows


def shakedown_cells(course, per_model=2, hz=4.0):
    """Rehearsal list (never counted): each model twice at 4 Hz."""
    return [{'order_idx': -1, 'course': course['name'],
             'world_idx': course['id'], 'model': m, 'hz': hz, 'round': k}
            for k in range(per_model) for m in MODELS]


COLOUR_COUNTED = ('success', 'timeout', 'stuck', 'crash')


def make_colour_schedule(arena, placement, run_id, rounds=COLOUR_ROUNDS,
                         lighting='day'):
    """Each round runs both models once in a seeded order. One schedule per
    lighting condition, so a round's two runs share the same light."""
    rng = random.Random(f'{run_id}/{arena["name"]}/{placement}/{lighting}')
    rows = []
    for rnd in range(rounds):
        models = list(COLOUR_MODELS)
        rng.shuffle(models)
        for m in models:
            rows.append({'order_idx': len(rows), 'arena': arena['name'],
                         'world_idx': arena['id'], 'placement': placement,
                         'lighting': lighting, 'model': m, 'hz': COLOUR_HZ,
                         'round': rnd})
    return rows


def colour_key(row):
    return (row['model'], row['placement'], row.get('lighting', ''),
            int(row['round'] if 'round' in row else row['trial']))


def colour_pending(schedule, done_rows):
    """Owed runs in schedule order: never attempted, or last attempt aborted
    or harness fault. A re-run keeps its place, so rounds stay together."""
    last = {colour_key(r): r['verdict'] for r in done_rows}
    return [c for c in schedule
            if colour_key(c) not in last or last[colour_key(c)] in RERUN]


def colour_schedule_path(arena, placement, run_id, lighting):
    return (run_dir(run_id, False)
            / f"schedule_colour_{arena['name']}_{placement}_{lighting}.csv")


def load_colour_schedule(arena, placement, run_id, lighting):
    path = colour_schedule_path(arena, placement, run_id, lighting)
    if not path.exists():
        raise SystemExit(f'no schedule at {path} — run colour-schedule first')
    rows = read_rows(path)
    for r in rows:
        r['hz'], r['round'] = float(r['hz']), int(r['round'])
        r['order_idx'] = int(r['order_idx'])
    return rows


def practice_cells(arena, placement, lighting, models=None, repeat=1):
    """Uncounted shakedown runs for one session: each chosen model `repeat`
    times, in the order given."""
    models = [m.strip() for m in (models or ','.join(COLOUR_MODELS)).split(',')
              if m.strip()]
    unknown = [m for m in models if m not in COLOUR_MODELS]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown} "
                         f"(have {', '.join(COLOUR_MODELS)})")
    return [{'order_idx': -1, 'arena': arena['name'], 'world_idx': arena['id'],
             'placement': placement, 'lighting': lighting, 'model': m,
             'hz': COLOUR_HZ, 'round': k}
            for k in range(repeat) for m in models]
