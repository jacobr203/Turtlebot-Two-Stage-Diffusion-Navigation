#!/usr/bin/env python3
"""Figures for the dissertation draft, from the banked result CSVs.

  .venv/bin/python report/make_figures.py  -> report/figures/*.png

Palette: the validated dataviz reference slots (two-stage blue, single-stage
orange, DWA aqua), the same colours the real-robot path maps use.
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.patches import FancyBboxPatch                  # noqa: E402
import numpy as np                                             # noqa: E402

CAP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAP))
OUT = CAP / 'report' / 'figures'
SUM = CAP / 'results' / 'summary'

SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e2dc'
COL = {'two_stage': '#2a78d6', 'single_stage': '#eb6834', 'dwa': '#1baf7a'}
LABEL = {'two_stage': 'Two-stage diffusion', 'single_stage':
         'Single-stage diffusion', 'dwa': 'DWA'}


def rows(name):
    return list(csv.DictReader(open(SUM / f'{name}.csv')))


def style(ax, ylabel='', xlabel=''):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.png', dpi=220, facecolor=SURFACE,
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def bar_label(ax, x, h, text):
    ax.text(x, h + 1.2, text, ha='center', va='bottom', fontsize=7.5,
            color=INK)


def fig_sim_rates():
    """Success / collision by control rate: v5 diffusion pair + DWA (v3
    matrix; DWA has no learned weights, so its rows are generation-free)."""
    data = {}
    for r in rows('full_v5_two_stage_summary') + rows(
            'full_v5_single_stage_summary'):
        data[(r['model'], float(r['hz']))] = r
    for r in rows('full_v3_summary'):
        if r['model'] == 'dwa':
            data[('dwa', float(r['hz']))] = r
    models = ['two_stage', 'single_stage', 'dwa']
    rates = [2.0, 3.0, 4.0]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), facecolor=SURFACE,
                             sharey=True, layout='constrained')
    w = 0.26
    for ax, key, title in ((axes[0], 'success_%', 'Success rate'),
                           (axes[1], 'collision_%', 'Collision rate')):
        style(ax, '% of 160 trials' if ax is axes[0] else '',
              'control rate')
        for i, m in enumerate(models):
            for j, hz in enumerate(rates):
                v = float(data[(m, hz)][key])
                x = j + (i - 1) * (w + 0.02)
                ax.bar(x, v, w, color=COL[m], edgecolor=SURFACE, lw=1,
                       label=LABEL[m] if j == 0 else None)
                bar_label(ax, x, v, f'{v:.0f}')
        ax.set_xticks(range(3), [f'{h:g} Hz' for h in rates])
        ax.set_ylim(0, 100)
        ax.set_title(title, loc='left', fontsize=10, color=INK)
    axes[1].legend(loc='upper right', fontsize=8, frameon=False,
                   labelcolor=INK_2)
    save(fig, 'sim_rates')


def fig_generations():
    """Two-stage vs single-stage success at 4 Hz across data generations."""
    gens = [('v2', 'full_v3_summary', 'full_v3_summary'),
            ('v4', 'full_v4_policies_summary', 'full_v4_policies_summary'),
            ('v5', 'full_v5_two_stage_summary', 'full_v5_single_stage_summary')]
    vals = {'two_stage': [], 'single_stage': []}
    for _, f2, f1 in gens:
        for m, f in (('two_stage', f2), ('single_stage', f1)):
            r = [x for x in rows(f) if x['model'] == m and x['hz'] == '4.0'][0]
            vals[m].append(float(r['success_%']))
    # v3 policies (multipath-only pool) sit between v2 and v4
    v3 = {m: float([x for x in rows('full_v3_policies_summary')
                    if x['model'] == m and x['hz'] == '4.0'][0]['success_%'])
          for m in ('two_stage', 'single_stage')}
    labels = ['v2', 'v3', 'v4', 'v5']
    series = {'two_stage': [vals['two_stage'][0], v3['two_stage'],
                            vals['two_stage'][1], vals['two_stage'][2]],
              'single_stage': [vals['single_stage'][0], v3['single_stage'],
                               vals['single_stage'][1],
                               vals['single_stage'][2]]}
    fig, ax = plt.subplots(figsize=(6.0, 3.3), facecolor=SURFACE,
                           layout='constrained')
    style(ax, 'success at 4 Hz (%)', 'policy generation (training data)')
    for m, ys in series.items():
        xs = [i for i, y in enumerate(ys) if y is not None]
        yv = [y for y in ys if y is not None]
        ax.plot(xs, yv, color=COL[m], lw=2, marker='o', ms=7,
                mec=SURFACE, mew=1.5, label=LABEL[m])
        for x, y in zip(xs, yv):
            ax.text(x, y + 3, f'{y:.0f}', ha='center', fontsize=7.5,
                    color=INK)
    ax.set_xticks(range(4), labels)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', fontsize=8, frameon=False, labelcolor=INK_2)
    save(fig, 'sim_generations')


def fig_real_outcomes():
    """Real robot: successes per block and model (all counted runs)."""
    from evaluation import real_trial as R
    counted = ('success', 'timeout', 'stuck', 'crash')
    rs = [r for r in R.read_rows(CAP / 'results' / 'real' / 'colour_v1'
                                 / 'colour_trials.csv')
          if r['verdict'] in counted and r['placement'] in ('P1', 'P2', 'P3')]
    blocks = []
    for r in rs:
        b = f"{r['placement']} {r['lighting']}"
        if b not in blocks:
            blocks.append(b)
    blocks.append('All')
    fig, ax = plt.subplots(figsize=(6.4, 3.3), facecolor=SURFACE,
                           layout='constrained')
    style(ax, 'runs that found every note (%)', '')
    w = 0.36
    for i, m in enumerate(('two_stage', 'single_stage')):
        for j, b in enumerate(blocks):
            sel = [r for r in rs if r['model'] == m and (
                b == 'All' or f"{r['placement']} {r['lighting']}" == b)]
            if not sel:
                continue
            k = sum(r['verdict'] == 'success' for r in sel)
            v = 100 * k / len(sel)
            x = j + (i - 0.5) * (w + 0.03)
            ax.bar(x, v, w, color=COL[m], edgecolor=SURFACE, lw=1,
                   label=LABEL[m] if j == 0 else None)
            bar_label(ax, x, v, f'{k}/{len(sel)}')
    ax.set_xticks(range(len(blocks)), blocks)
    ax.set_ylim(0, 112)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2,
              fontsize=8, frameon=False, labelcolor=INK_2)
    save(fig, 'real_outcomes')


def fig_architecture():
    fig, ax = plt.subplots(figsize=(9.0, 3.6), facecolor=SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis('off')

    def box(x, y, w, h, title, body, fill='#ffffff', edge=INK_2):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle='round,pad=0.4,rounding_size=1.2',
                                    fc=fill, ec=edge, lw=1.1))
        ax.text(x + w / 2, y + h - 2.6, title, ha='center', va='top',
                fontsize=8.5, color=INK, weight='bold')
        ax.text(x + w / 2, y + h - 7.0, body, ha='center', va='top',
                fontsize=7.2, color=INK_2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', color=INK_2, lw=1.1))

    box(1, 14, 15, 18, 'Sensors', 'LDS-02 LiDAR\nwheel odometry\nBRIO camera\n(robot, Pi)')
    box(21, 22, 19, 14, 'Stage 1: subgoal', 'A* on the live\nCartographer map\nsubgoal ~1.2 m ahead',
        fill='#eaf2fc', edge=COL['two_stage'])
    box(21, 3, 19, 15, 'Goal selection', 'point goal, or colour\nseek / explore\nlayers (task)')
    box(45, 14, 21, 22, 'Stage 2: diffusion policy',
        'DDPM, T = 35 steps\n1.1 M-param FiLM U-Net\n2-frame LiDAR stack\n+ goal & subgoal\n-> 8 waypoints (~0.4 m)',
        fill='#eaf2fc', edge=COL['two_stage'])
    box(71, 14, 13, 22, 'Tracker', 'pure pursuit\n20 Hz\n\nsafety gate\n(scan check)')
    box(88, 14, 11, 22, 'Robot', '/cmd_vel\nTurtleBot3\nBurger')
    arrow(16, 23, 21, 29)
    arrow(30.5, 18.6, 30.5, 21.6)
    arrow(40.4, 29, 44.6, 27)
    arrow(16, 22, 44.6, 20)
    arrow(66.4, 25, 70.6, 25)
    arrow(84.4, 25, 87.6, 25)
    ax.text(70, 9.5, 'tethered PC (WSL2, CPU inference), 2 / 3 / 4 Hz control tick',
            ha='center', fontsize=7.5, color=INK_2, style='italic')
    ax.text(70, 6.0, 'single-stage ablation: stage 1 removed, policy sees the goal only',
            ha='center', fontsize=7.5, color=INK_2, style='italic')
    save(fig, 'architecture')


if __name__ == '__main__':
    fig_architecture()
    fig_sim_rates()
    fig_generations()
    fig_real_outcomes()
    print('figures ->', OUT)
