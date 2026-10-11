#!/usr/bin/env python3
"""Aggregate matrix results into the proposal's metrics.

Per (model, hz): success / collision / timeout rates, time-to-goal,
path efficiency, inference latency.

Path efficiency = gt_path_len / (optimal_len_ours - success_radius):
a trial succeeds the moment ground truth enters the 1 m goal circle, so the
executed path is compared against the optimal path truncated at that same
boundary — raw optimal_len would flatter every model by ~1 m.

  .venv/bin/python evaluation/aggregate.py --run-id poster_v1
  .venv/bin/python evaluation/aggregate.py --run-id real/bench_v1 \
      --courses evaluation/courses        # real-robot trials

Real-robot runs (--courses) take each course's optimal length from its
layout, keep only counted verdicts (aborted / harness_error rows are
re-run cells, never outcomes), and add a per-course table.
Writes results/summary/<run_id>_summary.csv (+ per-world matrix) and prints
the summary table.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import course as C  # noqa: E402
from common import grid as G  # noqa: E402

CAPSTONE = Path(__file__).resolve().parents[1]


COUNTED = ('success', 'collision', 'timeout', 'crash')


def course_optimal_lengths(courses_dir):
    """world_idx -> optimal route length for every layout in courses_dir."""
    out = {}
    for f in sorted(Path(courses_dir).glob('*.yaml')):
        if C.is_layout_file(f):
            course = C.load(f)
            out[course['id']] = C.optimal_route(course)[1]
    return out


def load_run(run_id, courses_dir=None):
    run_dir = CAPSTONE / 'results' / run_id
    files = sorted(run_dir.glob('trials_w*.csv'))
    if not files:
        sys.exit(f'no trials_w*.csv under {run_dir}')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # a re-run after harness failures may append a second, valid row for the
    # same key — keep the last
    df = df.drop_duplicates(subset=['model', 'world_idx', 'hz', 'trial'],
                            keep='last')
    if courses_dir is None:
        meta = yaml.safe_load(open(CAPSTONE / 'barn' / 'metadata.yaml'))
        opt = {i: w['optimal_len_ours'] for i, w in meta['worlds'].items()}
    else:
        opt = course_optimal_lengths(courses_dir)
        n = len(df)
        df = df[df.verdict.isin(COUNTED)].copy()
        if n > len(df):
            print(f'{n - len(df)} cells whose last attempt was not counted '
                  '(aborted / harness_error) are excluded — re-run them')
    df['optimal_len'] = df['world_idx'].map(opt)
    df['path_eff'] = df['gt_path_len'] / (df['optimal_len'] - G.SUCCESS_RADIUS)
    df.loc[df['verdict'] != 'success', 'path_eff'] = pd.NA
    return df


def summarize(df):
    rows = []
    for (model, hz), g in df.groupby(['model', 'hz']):
        n = len(g)
        succ = g[g.verdict == 'success']
        rows.append({
            'model': model, 'hz': hz, 'n': n,
            'success_%': round(100 * len(succ) / n, 1),
            'collision_%': round(100 * (g.verdict == 'collision').mean(), 1),
            'timeout_%': round(100 * (g.verdict == 'timeout').mean(), 1),
            **({'crash_%': round(100 * (g.verdict == 'crash').mean(), 1)}
               if (df.verdict == 'crash').any() else {}),
            'time_to_goal_s': round(succ.time_to_goal_s.mean(), 1)
            if len(succ) else None,
            'path_eff': round(succ.path_eff.astype(float).mean(), 3)
            if len(succ) else None,
            'lat_sample_ms': round(g.lat_sample_ms.dropna().mean(), 1)
            if g.lat_sample_ms.notna().any() else None,
            'lat_plan_ms': round(g.lat_plan_ms.dropna().mean(), 1)
            if g.lat_plan_ms.notna().any() else None,
        })
    return pd.DataFrame(rows).sort_values(['model', 'hz'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--courses', help='real-robot course layout directory')
    args = ap.parse_args()

    df = load_run(args.run_id, args.courses)
    summary = summarize(df)

    out = CAPSTONE / 'results' / 'summary'
    out.mkdir(parents=True, exist_ok=True)
    stem = args.run_id.replace('/', '_')
    summary.to_csv(out / f'{stem}_summary.csv', index=False)
    per_world = df.pivot_table(index='world_idx', columns=['model', 'hz'],
                               values='verdict',
                               aggfunc=lambda v: (v == 'success').mean())
    per_world.to_csv(out / f'{stem}_per_world.csv')

    print(f'\n== {args.run_id}: {len(df)} trials ==')
    print(summary.to_string(index=False))
    if args.courses:
        per_course = pd.concat(
            [summarize(g).assign(course=g.course.iloc[0])
             for _, g in df.groupby('world_idx')], ignore_index=True)
        per_course.to_csv(out / f'{stem}_per_course.csv', index=False)
        print('\n-- per course --')
        print(per_course.to_string(index=False))
    print(f'\nwrote {out}/{stem}_summary.csv and _per_world.csv')


if __name__ == '__main__':
    main()
