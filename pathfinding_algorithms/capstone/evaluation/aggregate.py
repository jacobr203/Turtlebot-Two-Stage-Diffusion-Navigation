#!/usr/bin/env python3
"""Aggregate matrix results into the proposal's metrics.

Per (model, hz): success / collision / timeout rates, time-to-goal,
path efficiency, inference latency.

Path efficiency = gt_path_len / (optimal_len_ours - success_radius):
a trial succeeds the moment ground truth enters the 1 m goal circle, so the
executed path is compared against the optimal path truncated at that same
boundary — raw optimal_len would flatter every model by ~1 m.

  .venv/bin/python evaluation/aggregate.py --run-id poster_v1
Writes results/summary/<run_id>_summary.csv (+ per-world matrix) and prints
the summary table.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import grid as G  # noqa: E402

CAPSTONE = Path(__file__).resolve().parents[1]


def load_run(run_id):
    run_dir = CAPSTONE / 'results' / run_id
    files = sorted(run_dir.glob('trials_w*.csv'))
    if not files:
        sys.exit(f'no trials_w*.csv under {run_dir}')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # a re-run after harness failures may append a second, valid row for the
    # same key — keep the last
    df = df.drop_duplicates(subset=['model', 'world_idx', 'hz', 'trial'],
                            keep='last')
    meta = yaml.safe_load(open(CAPSTONE / 'barn' / 'metadata.yaml'))
    opt = {i: w['optimal_len_ours'] for i, w in meta['worlds'].items()}
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
    args = ap.parse_args()

    df = load_run(args.run_id)
    summary = summarize(df)

    out = CAPSTONE / 'results' / 'summary'
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / f'{args.run_id}_summary.csv', index=False)
    per_world = df.pivot_table(index='world_idx', columns=['model', 'hz'],
                               values='verdict',
                               aggfunc=lambda v: (v == 'success').mean())
    per_world.to_csv(out / f'{args.run_id}_per_world.csv')

    print(f'\n== {args.run_id}: {len(df)} trials ==')
    print(summary.to_string(index=False))
    print(f'\nwrote {out}/{args.run_id}_summary.csv and _per_world.csv')


if __name__ == '__main__':
    main()
