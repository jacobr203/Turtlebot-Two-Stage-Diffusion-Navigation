#!/usr/bin/env python3
"""Poster/dissertation figures from a matrix run.

  .venv/bin/python evaluation/plots.py --run-id poster_v1

Writes results/summary/figs/<run_id>_*.{png,pdf}:
  outcomes   stacked success/collision/timeout bars per model
  per_world  success-rate heatmap (model x world)
  time       time-to-goal distribution over successful trials
  latency    per-tick inference cost vs the 2/3/4 Hz budgets
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.aggregate import load_run  # noqa: E402

CAPSTONE = Path(__file__).resolve().parents[1]

MODELS = ['two_stage', 'single_stage', 'dwa']
LABELS = {'two_stage': 'Two-stage\ndiffusion',
          'single_stage': 'Single-stage\ndiffusion', 'dwa': 'DWA\nbaseline'}
# Okabe-Ito (colorblind-safe)
MODEL_C = {'two_stage': '#0072B2', 'single_stage': '#E69F00',
           'dwa': '#009E73'}
VERDICT_C = {'success': '#4C9F70', 'collision': '#C4452C',
             'timeout': '#B8A04A'}

plt.rcParams.update({
    'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 13,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})


def _models_in(df):
    return [m for m in MODELS if m in set(df.model)]


def _save(fig, out_dir, run_id, name):
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'{run_id}_{name}.{ext}', dpi=300,
                    bbox_inches='tight')
    plt.close(fig)
    print(f'  {run_id}_{name}.png/.pdf')


def fig_outcomes(df, out_dir, run_id):
    models = _models_in(df)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(models))
    bottom = np.zeros(len(models))
    for verdict in ('success', 'collision', 'timeout'):
        vals = np.array([100 * (df[df.model == m].verdict == verdict).mean()
                         for m in models])
        ax.bar(x, vals, 0.62, bottom=bottom, color=VERDICT_C[verdict],
               label=verdict)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 6:
                ax.text(xi, b + v / 2, f'{v:.0f}%', ha='center',
                        va='center', color='white', fontweight='bold')
        bottom += vals
    ax.set_xticks(x, [LABELS[m] for m in models])
    ax.set_ylabel('trials [%]')
    ax.set_ylim(0, 100)
    n = df.groupby('model').size().max()
    ax.set_title(f'Trial outcomes — {int(df.world_idx.nunique())} unseen '
                 f'BARN worlds, {int(n)} trials/model @ '
                 f'{df.hz.iloc[0]:.0f} Hz')
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), framealpha=0.95)
    _save(fig, out_dir, run_id, 'outcomes')


def fig_per_world(df, out_dir, run_id):
    models = _models_in(df)
    worlds = sorted(df.world_idx.unique())
    grid = np.array([[ (df[(df.model == m) & (df.world_idx == w)]
                        .verdict == 'success').mean()
                       for w in worlds] for m in models])
    fig, ax = plt.subplots(figsize=(0.62 * len(worlds) + 2.2, 2.6))
    im = ax.imshow(grid, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(worlds)), [str(w) for w in worlds])
    ax.set_yticks(range(len(models)),
                  [LABELS[m].replace('\n', ' ') for m in models])
    for i in range(len(models)):
        for j in range(len(worlds)):
            ax.text(j, i, f'{grid[i, j]:.2f}'.rstrip('0').rstrip('.'),
                    ha='center', va='center', fontsize=10,
                    color='black')
    ax.set_xlabel('holdout world')
    ax.set_title('Per-world success rate')
    fig.colorbar(im, ax=ax, shrink=0.85, label='success rate')
    _save(fig, out_dir, run_id, 'per_world')


def fig_time(df, out_dir, run_id):
    models = _models_in(df)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    data, ticks = [], []
    for m in models:
        vals = df[(df.model == m) & (df.verdict == 'success')] \
            .time_to_goal_s.astype(float).values
        data.append(vals)
        ticks.append(f'{LABELS[m]}\n(n={len(vals)})')
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
    for patch, m in zip(bp['boxes'], models):
        patch.set_facecolor(MODEL_C[m])
        patch.set_alpha(0.45)
    for med in bp['medians']:
        med.set_color('black')
    rng = np.random.default_rng(0)
    for i, (vals, m) in enumerate(zip(data, models)):
        ax.scatter(np.full(len(vals), i + 1) + rng.uniform(-0.13, 0.13,
                                                           len(vals)),
                   vals, s=16, color=MODEL_C[m], alpha=0.8, zorder=3)
    ax.set_xticks(range(1, len(models) + 1), ticks)
    ax.set_ylabel('time to goal [s] (successful trials)')
    ax.set_title('Time to goal')
    _save(fig, out_dir, run_id, 'time')


def fig_latency(df, out_dir, run_id):
    models = [m for m in _models_in(df)
              if df[df.model == m].lat_sample_ms.notna().any()]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.arange(len(models))
    samp = [df[df.model == m].lat_sample_ms.dropna().mean() for m in models]
    plan = [df[df.model == m].lat_plan_ms.dropna().mean() for m in models]
    ax.bar(x, samp, 0.5, color='#4C6FA5', label='DDPM-35 sampling')
    ax.bar(x, plan, 0.5, bottom=samp, color='#A5714C', label='A* subgoal')
    for hz, ls in ((4, '-'), (3, '--'), (2, ':')):
        ax.axhline(1000 / hz, color='#888', lw=1.2, ls=ls)
        ax.text(len(models) - 0.42, 1000 / hz + 6, f'{hz} Hz budget',
                fontsize=10, color='#555')
    ax.set_xticks(x, [LABELS[m] for m in models])
    ax.set_ylabel('per-tick inference [ms]')
    ax.set_ylim(0, 540)
    ax.set_title('Inference cost vs control-loop budgets (CPU)')
    ax.legend(loc='upper left', framealpha=0.95)
    _save(fig, out_dir, run_id, 'latency')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    df = load_run(args.run_id)
    out_dir = CAPSTONE / 'results' / 'summary' / 'figs'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'figures for {args.run_id} ({len(df)} trials) -> {out_dir}')
    fig_outcomes(df, out_dir, args.run_id)
    fig_per_world(df, out_dir, args.run_id)
    fig_time(df, out_dir, args.run_id)
    fig_latency(df, out_dir, args.run_id)


if __name__ == '__main__':
    main()
