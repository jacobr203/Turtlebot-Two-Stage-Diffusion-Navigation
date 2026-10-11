#!/usr/bin/env python3
"""Results, note map and path figures for a block of real colour runs.

A block is every run of one placement under one lighting condition in a
run id (one harness session or several). The command

  .venv/bin/python evaluation/session_report.py --run-id colour_v1 \\
      --arena evaluation/arenas/room1.yaml --placement P1 --lighting night

writes results/real/<run-id>/reports/<arena>_<placement>_<lighting>/:

  report.md           per-run table, per-model summary, data checks
  runs.csv            one row per counted run, as in the table
  note_sites.csv      where each note was, measured from the detections
  notes.png/.pdf      every close detection, the sites, the recorded
                      placement: is the arena file still right?
  paths_by_model.png  every counted run, one panel per model
  paths_runs.png      one small map per run, in schedule order
  (+ .pdf of each figure)

Where the notes were is measured, not taken from the arena file: every
detection closer than MAX_DET_M gives a point (robot pose at that moment +
distance along yaw + bearing), and the points are clustered into sites.
The figures draw the notes at the site most runs found them at, and the
checks flag any run that found a note somewhere else (the note was moved,
or a false find) and any site far from the arena file's position.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import arena as A                                 # noqa: E402
from evaluation import real_paths as P                        # noqa: E402

CAPSTONE = Path(__file__).resolve().parents[1]
COUNTED = ('success', 'timeout', 'stuck', 'crash')
CAM_DX = 0.028      # camera ahead of base_footprint (the path's pose)
MAX_DET_M = 0.8     # detections closer than this place a note to ~5 cm
SITE_RADIUS_M = 0.35
SITE_MIN_DETS = 5   # fewer close detections than this: noise, not a site
MOVED_M = 0.30      # a site this far from the arena file is flagged


# ── reading runs ────────────────────────────────────────────────────────────
def _bag_reader(bag, topics):
    import rosbag2_py
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='sqlite3'),
           rosbag2_py.ConverterOptions('', ''))
    r.set_filter(rosbag2_py.StorageFilter(topics=topics))
    return r


def bag_start_s(bag):
    import yaml
    meta = yaml.safe_load(open(Path(bag) / 'metadata.yaml'))
    return meta['rosbag2_bagfile_information']['starting_time'][
        'nanoseconds_since_epoch'] / 1e9


def note_points(trial_dir):
    """Close detections of a run placed in the arena frame:
    list of dict(color, t, d, x, y). Cached in note_dets.csv."""
    trial_dir = Path(trial_dir)
    cache = trial_dir / 'note_dets.csv'
    if cache.exists():
        return [dict(r, t=float(r['t']), d=float(r['d']), x=float(r['x']),
                     y=float(r['y'])) for r in csv.DictReader(open(cache))]
    loaded = P.load_run(trial_dir)
    if loaded is None or not (trial_dir / 'bag').exists():
        return []
    times, xy, ev = loaded
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import String
    dets, first = [], {}
    r = _bag_reader(trial_dir / 'bag', ['/color_targets', '/color_test/status'])
    while r.has_next():
        topic, data, t = r.read_next()
        m = json.loads(deserialize_message(data, String).data)
        if topic == '/color_targets':
            for d in m.get('detections', []):
                dets.append((t / 1e9, d))
                first.setdefault(('seen', d['color']), t / 1e9)
        else:
            for c, s in m.get('states', {}).items():
                if s == 'FOUND':
                    first.setdefault(('found', c), t / 1e9)
    # bag clock -> path clock, from the events both recorded
    offs = [first[(k, c)] - e['t'] for k in ('found', 'seen')
            for c, e in ev.get(k, {}).items() if (k, c) in first]
    out = []
    if offs:
        off = float(np.median(offs))
        yaw = np.unwrap(xy[:, 2])
        for tb, d in dets:
            t = tb - off
            if d['distance_m'] > MAX_DET_M or not times[0] <= t <= times[-1]:
                continue
            a = np.interp(t, times, yaw) + d['bearing_rad']
            rng = d['distance_m'] + CAM_DX
            out.append(dict(color=d['color'], t=round(t, 2),
                            d=d['distance_m'],
                            x=round(np.interp(t, times, xy[:, 0])
                                    + rng * math.cos(a), 3),
                            y=round(np.interp(t, times, xy[:, 1])
                                    + rng * math.sin(a), 3)))
    with open(cache, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['color', 't', 'd', 'x', 'y'])
        w.writeheader()
        w.writerows(out)
    return out


def find_points(points, events):
    """{colour: (x, y)} of each FOUND: the close detections in the two
    seconds before it."""
    out = {}
    for c, e in events.get('found', {}).items():
        near = [(p['x'], p['y']) for p in points
                if p['color'] == c and e['t'] - 2.0 <= p['t'] <= e['t'] + 0.3]
        if near:
            out[c] = tuple(np.median(np.array(near), axis=0))
    return out


def _round(row):
    """The schedule round (stored as 'trial' in colour_trials.csv)."""
    return int(row.get('round') or row['trial'])


def match_rows(rows, trial_dirs):
    """Pair CSV rows with trial folders: same model and round, the bag
    starting within two minutes after the row's start time. Closest pairs
    are taken first, so a refused start logged a minute earlier cannot
    claim the folder of the run that followed it."""
    from datetime import datetime
    cands = []
    for i, row in enumerate(rows):
        t0 = datetime.fromisoformat(row['t_start_wall']).timestamp()
        for d in trial_dirs:
            if not (d / 'bag' / 'metadata.yaml').exists():
                continue
            if not d.name.endswith(f"_{row['model']}_"
                                   f"{float(row['hz']):g}hz_r{_round(row)}"):
                continue
            dt = bag_start_s(d / 'bag') - t0
            if 0 <= dt <= 120:
                cands.append((dt, i, d))
    match, used = {}, set()
    for dt, i, d in sorted(cands, key=lambda c: c[0]):
        if i not in match and d not in used:
            match[i] = d
            used.add(d)
    return [(row, match.get(i)) for i, row in enumerate(rows)]


# ── note sites ──────────────────────────────────────────────────────────────
def cluster(points, radius=SITE_RADIUS_M):
    """Greedy clusters of (x, y): densest first. -> list of index arrays."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    left = np.arange(len(pts))
    out = []
    while len(left):
        d = np.linalg.norm(pts[left][:, None] - pts[left][None], axis=2)
        seed = np.argmax((d < radius).sum(axis=1))
        centre = np.median(pts[left][d[seed] < radius], axis=0)
        member = np.linalg.norm(pts[left] - centre, axis=1) < radius
        out.append(left[member])
        left = left[~member]
    return out


def note_sites(runs, recorded):
    """Sites per colour from the counted runs' close detections.

    runs: list of dict(label, points, finds). recorded: {colour: (x, y)}
    from the arena file. Returns a list of site dicts, each colour's
    sites ordered by how many runs found the note there."""
    sites = []
    for c in sorted({p['color'] for r in runs for p in r['points']}):
        pts, owner = [], []
        for k, r in enumerate(runs):
            for p in r['points']:
                if p['color'] == c:
                    pts.append((p['x'], p['y']))
                    owner.append(k)
        owner = np.array(owner)
        mine = []
        for idx in cluster(pts):
            if len(idx) < SITE_MIN_DETS:
                continue
            centre = np.median(np.array(pts)[idx], axis=0)
            found_by = [runs[k]['label'] for k in range(len(runs))
                        if c in runs[k]['finds'] and np.linalg.norm(
                            np.array(runs[k]['finds'][c]) - centre)
                        < SITE_RADIUS_M]
            rec = recorded.get(c)
            mine.append(dict(
                color=c, x=round(float(centre[0]), 3),
                y=round(float(centre[1]), 3), dets=len(idx),
                runs_seen=len(set(owner[idx])), found_by=found_by,
                from_recorded_m=(round(float(np.linalg.norm(
                    centre - np.array(rec))), 2) if rec else None)))
        mine.sort(key=lambda s: (-len(s['found_by']), -s['dets']))
        for rank, s in enumerate(mine):
            s['site'] = f"{c}:{'ABCDEFGH'[rank]}" if rank < 8 else f'{c}:?'
        sites += mine
    return sites


def main_sites(sites):
    """The site per colour the most runs found the note at."""
    best = {}
    for s in sites:
        if s['found_by'] and s['color'] not in best:
            best[s['color']] = s
    return best


# ── stats ───────────────────────────────────────────────────────────────────
def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def model_summary(table, colors):
    out = {}
    for m in ('two_stage', 'single_stage'):
        rs = [r for r in table if r['model'] == m]
        if not rs:
            continue
        ok = [r for r in rs if r['verdict'] == 'success']
        lo, hi = wilson(len(ok), len(rs))
        out[m] = dict(
            runs=len(rs), success=len(ok), success_ci=(lo, hi),
            verdicts={v: sum(r['verdict'] == v for r in rs)
                      for v in COUNTED if any(r['verdict'] == v for r in rs)},
            median_total_s=(float(np.median([r['total_s'] for r in ok]))
                            if ok else None),
            median_path_m=float(np.median([r['path_m'] for r in rs])),
            path_per_success_m=(float(np.median([r['path_m'] for r in ok]))
                                if ok else None),
            mean_found=float(np.mean([r['found'] for r in rs])),
            collisions=sum(r['collisions'] for r in rs),
            first_find={c: [r[f'{c}_s'] for r in rs if r[f'{c}_s'] is not None]
                        for c in colors})
    return out


def fisher_p(a_ok, a_n, b_ok, b_n):
    from scipy.stats import fisher_exact
    return float(fisher_exact([[a_ok, a_n - a_ok], [b_ok, b_n - b_ok]])[1])


# ── figures ─────────────────────────────────────────────────────────────────
def fig_notes(arena, placement, runs, sites, out_base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    w, h = P._panel_size(arena)
    fig, ax = plt.subplots(figsize=(w + 2.4, h), facecolor=P.SURFACE,
                           layout='constrained')
    P._style(ax)
    P._draw_arena(ax, arena, placement, notes=[])
    for r in runs:
        for p in r['points']:
            ax.plot(p['x'], p['y'], '.', ms=2.5, alpha=0.35, zorder=2,
                    color=P.NOTE_COLOR.get(p['color'], P.INK_2))
    for s in arena['placements'].get(placement, []):
        ax.plot(s['x'], s['y'], 's', ms=10, mfc='none', mew=1.4, zorder=6,
                mec=P.NOTE_COLOR[s['color']])
    for s in sites:
        main = bool(s['found_by'])
        ax.plot(s['x'], s['y'], 's', ms=9 if main else 6, zorder=7,
                color=P.NOTE_COLOR[s['color']], mec=P.INK, mew=0.8)
        # label on the left when another site sits close on the right
        crowded = any(o is not s and 0 < o['x'] - s['x'] < 0.7
                      and abs(o['y'] - s['y']) < 0.2 for o in sites)
        ax.annotate(f"{s['site'].split(':')[1]}  {len(s['found_by'])} found",
                    (s['x'], s['y']), xytext=(-8 if crowded else 8, 6),
                    ha='right' if crowded else 'left',
                    textcoords='offset points', fontsize=7, color=P.INK_2,
                    zorder=8)
    handles = [
        Line2D([], [], ls='', marker='.', ms=6, color=P.INK_2,
               label=f'Close detection (< {MAX_DET_M} m), in its colour'),
        Line2D([], [], ls='', marker='s', ms=8, color=P.INK_2, mec=P.INK,
               label='Measured site (label: runs that found it there)'),
        Line2D([], [], ls='', marker='s', ms=9, mfc='none', mec=P.INK_2,
               mew=1.4, label='Position in the arena file'),
    ]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1.0),
              fontsize=7, frameon=False, labelcolor=P.INK_2)
    ax.set_title(f"Where the notes were: arena {arena['name']}, placement "
                 f'{placement}', loc='left', fontsize=10, color=P.INK)
    P._save(fig, out_base)
    plt.close(fig)


def fig_by_model(arena, placement, table, notes, out_base, lighting):
    runs = [(r['model'], *r['_path']) for r in table if r['_path']]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = [m for m in ('two_stage', 'single_stage')
              if any(r[0] == m for r in runs)]
    w, h = P._panel_size(arena)
    fig, axes = plt.subplots(1, len(models), squeeze=False,
                             figsize=(w * len(models) + 2.2, h + 0.3),
                             sharex=True, sharey=True, facecolor=P.SURFACE,
                             layout='constrained')
    for ax, m in zip(axes[0], models):
        P._style(ax)
        P._draw_arena(ax, arena, placement, label_notes=(ax is axes[0][0]),
                      notes=notes)
        mine = [r for r in runs if r[0] == m]
        for _, times, xy, ev in mine:
            run = times >= 0
            P._draw_path(ax, xy[run] if run.any() else xy, ev,
                         P.MODEL_COLOR[m], alpha=0.55)
        ok = sum(1 for r in table if r['model'] == m
                 and r['verdict'] == 'success')
        ax.set_title(f"{P.MODEL_LABEL[m]}: {ok}/{len(mine)} runs found "
                     'every note', loc='left', fontsize=9, color=P.INK)
    axes[0][-1].legend(handles=P._legend_handles(models), loc='upper left',
                       bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False,
                       labelcolor=P.INK_2)
    fig.suptitle(f"Paths driven, arena {arena['name']}, placement "
                 f'{placement}, {lighting} lighting (notes where the '
                 'detections put them)', x=0.01, ha='left', fontsize=11,
                 color=P.INK)
    P._save(fig, out_base)
    plt.close(fig)


def fig_runs(arena, placement, table, notes, out_base, lighting):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rs = [r for r in table if r['_path']]
    cols = min(4, len(rs))
    rows_n = math.ceil(len(rs) / cols)
    w, h = P._panel_size(arena, width=3.6)
    fig, axes = plt.subplots(rows_n, cols, squeeze=False,
                             figsize=(w * cols, (h + 0.2) * rows_n),
                             sharex=True, sharey=True, facecolor=P.SURFACE,
                             layout='constrained')
    for ax in axes.flat[len(rs):]:
        ax.set_visible(False)
    for ax, r in zip(axes.flat, rs):
        times, xy, ev = r['_path']
        P._style(ax)
        ax.tick_params(labelsize=6)
        ax.xaxis.label.set_fontsize(6)
        ax.yaxis.label.set_fontsize(6)
        P._draw_arena(ax, arena, placement, label_notes=False, notes=notes)
        run = times >= 0
        P._draw_path(ax, xy[run] if run.any() else xy, ev,
                     P.MODEL_COLOR[r['model']])
        flag = ' *' if r['flags'] else ''
        ax.set_title(f"#{r['order_idx']} {P.MODEL_LABEL[r['model']]}, round "
                     f"{r['round']}{flag}\n{r['verdict'].capitalize()}, {r['found']}/"
                     f"{r['n_notes']} found, {r['total_s']:.0f} s, "
                     f"{r['path_m']:.1f} m", loc='left', fontsize=7,
                     color=P.INK)
    fig.legend(handles=P._legend_handles(['two_stage', 'single_stage']),
               loc='outside lower center', ncol=4, fontsize=7, frameon=False,
               labelcolor=P.INK_2)
    fig.suptitle(f"Every counted run, arena {arena['name']}, placement "
                 f'{placement}, {lighting} lighting, in schedule order '
                 '(* = see the data checks)', x=0.01, ha='left',
                 fontsize=10, color=P.INK)
    P._save(fig, out_base)
    plt.close(fig)


# ── report ──────────────────────────────────────────────────────────────────
def _f(v, fmt='{:.1f}', none='–'):
    return none if v is None else fmt.format(v)


def build(args):
    arena = A.load(args.arena)
    colors = A.placement_colors(arena, args.placement)
    recorded = {s['color']: (s['x'], s['y'])
                for s in arena['placements'][args.placement]}
    run_dir = CAPSTONE / 'results' / 'real' / args.run_id
    out = (run_dir / 'reports'
           / f"{arena['name']}_{args.placement}_{args.lighting}")
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(run_dir / 'colour_trials.csv'))
            if r['placement'] == args.placement
            and r['lighting'] == args.lighting
            and (not args.session or r['session_id'] in args.session)]
    dirs = sorted((run_dir / 'trials').glob(
        f"t*_{arena['name']}_{args.placement}_{args.lighting}_*"))
    pairs = match_rows(rows, dirs)

    table, rerun = [], []
    for row, d in pairs:
        if row['verdict'] not in COUNTED:
            rerun.append((row, d))
            continue
        loaded = P.load_run(d) if d else None
        pts = note_points(d) if d else []
        finds = find_points(pts, loaded[2]) if loaded else {}
        num = lambda k: float(row[k]) if row.get(k) else None  # noqa: E731
        table.append(dict(
            order_idx=int(row['order_idx']), model=row['model'],
            round=_round(row), verdict=row['verdict'],
            total_s=num('total_s'), path_m=num('path_m'),
            found=int(row['found_count']), n_notes=len(colors),
            collisions=int(float(row.get('collisions') or 0)),
            false_found=row.get('false_found', ''),
            t_start=row['t_start_wall'], battery_v=num('battery_v_start'),
            notes=row.get('notes', ''), trial=d.name if d else '',
            **{f'{c}_s': num(f'{c}_found_s') for c in colors},
            _path=loaded, _points=pts, _finds=finds, flags=[]))
    table.sort(key=lambda r: r['order_idx'])

    runs = [dict(label=f"#{r['order_idx']}", points=r['_points'],
                 finds=r['_finds']) for r in table]
    sites = note_sites(runs, recorded)
    main = main_sites(sites)
    notes = [dict(color=c, x=s['x'], y=s['y']) for c, s in main.items()]

    # data checks
    checks = []
    for c, s in main.items():
        if s['from_recorded_m'] is not None and s['from_recorded_m'] > MOVED_M:
            checks.append(
                f"{c}: most finds were at ({s['x']:.2f}, {s['y']:.2f}), "
                f"{s['from_recorded_m']:.2f} m from the arena file's "
                f'({recorded[c][0]:.2f}, {recorded[c][1]:.2f}). Update '
                f'placement {args.placement} or check the note was not '
                'moved.')
    for r in table:
        for c, xy in r['_finds'].items():
            s = main.get(c)
            if s and math.hypot(xy[0] - s['x'], xy[1] - s['y']) > SITE_RADIUS_M:
                r['flags'].append(f'{c} found at ({xy[0]:.2f}, {xy[1]:.2f}), '
                                  'not at its main site')
        for c in colors:
            if r[f'{c}_s'] is not None and c not in r['_finds']:
                r['flags'].append(f'{c}: FOUND with no close detection to '
                                  'place it')
        if r['false_found']:
            r['flags'].append(f"operator marked a FALSE find: "
                              f"{r['false_found']}")
        if not r['_path']:
            r['flags'].append('no path recorded')
    for r in table:
        if r['flags']:
            checks.append(f"#{r['order_idx']} {r['model']} round "
                          f"{r['round']} ({r['t_start']}): "
                          + '; '.join(r['flags']))

    summary = model_summary(table, colors)
    p_succ = (fisher_p(summary['two_stage']['success'],
                       summary['two_stage']['runs'],
                       summary['single_stage']['success'],
                       summary['single_stage']['runs'])
              if len(summary) == 2 else None)
    clean = [r for r in table if not r['flags']]
    clean_sum = model_summary(clean, colors) if len(clean) < len(table) else None

    # files
    with open(out / 'runs.csv', 'w', newline='') as f:
        keys = [k for k in table[0] if not k.startswith('_')] if table else []
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in table:
            w.writerow({k: ('; '.join(r[k]) if k == 'flags' else r[k])
                        for k in keys})
    with open(out / 'note_sites.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['site', 'color', 'x', 'y', 'dets',
                                          'runs_seen', 'found_by',
                                          'from_recorded_m'])
        w.writeheader()
        for s in sites:
            w.writerow({**s, 'found_by': ' '.join(s['found_by'])})
    fig_notes(arena, args.placement, runs, sites, out / 'notes')
    fig_by_model(arena, args.placement, table, notes, out / 'paths_by_model',
                 args.lighting)
    fig_runs(arena, args.placement, table, notes, out / 'paths_runs',
             args.lighting)

    md = [f"# Colour runs: {arena['name']} {args.placement}, "
          f'{args.lighting} lighting ({args.run_id})', '',
          f'{len(table)} counted runs, {len(rerun)} re-run attempts '
          f"(aborted / harness_error), sessions "
          f"{', '.join(sorted({r['session_id'] for r in rows}))}.", '',
          '## Per model', '',
          '| model | runs | success | 95% CI | verdicts | median time to '
          'all found (s) | median path (m) | mean notes found | collisions |',
          '|---|---|---|---|---|---|---|---|---|']
    for m, s in summary.items():
        md.append(f"| {P.MODEL_LABEL[m]} | {s['runs']} | {s['success']}/"
                  f"{s['runs']} | {s['success_ci'][0]:.0%}–"
                  f"{s['success_ci'][1]:.0%} | "
                  + ', '.join(f'{k} {v}' for k, v in s['verdicts'].items())
                  + f" | {_f(s['median_total_s'])} | "
                  f"{_f(s['median_path_m'])} | {s['mean_found']:.1f} | "
                  f"{s['collisions']} |")
    if p_succ is not None:
        md += ['', f'Success, two-stage vs single-stage: Fisher exact '
               f'p = {p_succ:.3f} (two-sided).']
    md += ['', 'Time to find each note (s, successful finds only):', '']
    for m, s in summary.items():
        md.append(f"- {P.MODEL_LABEL[m]}: " + '; '.join(
            f"{c} median {_f(np.median(v) if v else None)} (n={len(v)})"
            for c, v in s['first_find'].items()))
    if clean_sum:
        md += ['', 'Excluding flagged runs (see the data checks):', '']
        for m, s in clean_sum.items():
            md.append(f"- {P.MODEL_LABEL[m]}: {s['success']}/{s['runs']} "
                      f"success, median time {_f(s['median_total_s'])} s, "
                      f"median path {_f(s['median_path_m'])} m")
    md += ['', '## Runs (schedule order)', '',
           '| # | model | round | verdict | time (s) | path (m) | found | '
           + ' | '.join(f'{c} at (s)' for c in colors)
           + ' | collisions | battery (V) | started | flags |',
           '|---' * (11 + len(colors)) + '|']
    for r in table:
        md.append(f"| {r['order_idx']} | {r['model']} | {r['round']} | "
                  f"{r['verdict']} | {_f(r['total_s'])} | {_f(r['path_m'], '{:.2f}')} | "
                  f"{r['found']}/{r['n_notes']} | "
                  + ' | '.join(_f(r[f'{c}_s']) for c in colors)
                  + f" | {r['collisions']} | {_f(r['battery_v'], '{:.2f}')} | "
                  f"{r['t_start'][5:16].replace('T', ' ')} | "
                  f"{'yes' if r['flags'] else ''} |")
    md += ['', '## Where the notes were', '',
           '| site | colour | x | y | close detections | runs that saw it | '
           'found there by | from arena file (m) |', '|---' * 8 + '|']
    for s in sites:
        md.append(f"| {s['site']} | {s['color']} | {s['x']:.2f} | "
                  f"{s['y']:.2f} | {s['dets']} | {s['runs_seen']} | "
                  f"{' '.join(s['found_by']) or '–'} | "
                  f"{_f(s['from_recorded_m'], '{:.2f}')} |")
    md += ['', '## Data checks', '']
    md += [f'- {c}' for c in checks] or ['- none']
    if rerun:
        md += ['', '## Re-run attempts (not counted)', '']
        md += [f"- #{r['order_idx']} {r['model']} {r['verdict']} "
               f"({r['t_start_wall']}): {(r.get('notes') or r.get('verdict_reason') or '')[:160]}"
               for r, _ in rerun]
    md += ['', '## Figures', '',
           '- `notes.png`: every close detection and the measured sites',
           '- `paths_by_model.png`: all counted runs, one panel per model',
           '- `paths_runs.png`: one map per run, in schedule order', '']
    (out / 'report.md').write_text('\n'.join(md))
    return out, table, summary, checks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--arena', required=True)
    ap.add_argument('--placement', required=True)
    ap.add_argument('--lighting', required=True)
    ap.add_argument('--session', nargs='*',
                    help='only these session ids (default: the whole block)')
    args = ap.parse_args()
    out, table, summary, checks = build(args)
    print((out / 'report.md').read_text())
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
