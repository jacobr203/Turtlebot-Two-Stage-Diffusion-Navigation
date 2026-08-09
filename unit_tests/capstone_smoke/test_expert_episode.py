"""End-to-end expert episode on world 0: plan, track, record, verdict.

This is the Phase B gate: a success verdict with a schema-valid npz and sane
tracking error means the whole A* + pure-pursuit + recorder pipeline works.
"""

from pathlib import Path

import numpy as np
import pytest

from evaluation.sim_helpers import barn_world
from expert.collect_episode import run_episode

OUT = Path('/tmp/capstone_smoke/ep_test.npz')


@pytest.fixture(scope='module')
def episode_row(harness):
    harness.kill_all()
    harness.start_world(barn_world(0, fast=True), headless=True)
    OUT.unlink(missing_ok=True)
    rng = np.random.default_rng(1234)
    row = run_episode(harness, 0, 0, OUT, rng)
    yield row
    harness.kill_all()


def test_episode_succeeds(episode_row):
    assert episode_row['verdict'] == 'success', episode_row


def test_npz_schema(episode_row):
    with np.load(OUT) as ep:
        T = len(ep['t'])
        assert T >= 100, f'only {T} samples for a ~10 m path at 4 Hz'
        assert ep['scans'].shape == (T, 360)
        assert ep['odom'].shape == (T, 6)
        assert ep['gt_pose'].shape == (T, 3)
        assert ep['goal'].shape == (T, 3)
        assert ep['subgoal'].shape == (T, 3)
        assert ep['cmd'].shape == (T, 2)
        assert np.all(np.diff(ep['t']) > 0), 'sim time not increasing'
        # ranges within sensor limits after resampling
        assert float(ep['scans'].min()) >= 0.12 - 1e-6
        assert float(ep['scans'].max()) <= 3.5 + 1e-6


def test_goal_distance_shrinks(episode_row):
    with np.load(OUT) as ep:
        d = ep['goal'][:, 2]
        assert d[0] > 8.0, 'goal should start ~10 m away'
        assert d[-1] < 1.2, 'episode ended far from the goal'


def test_subgoal_stays_local(episode_row):
    with np.load(OUT) as ep:
        d = ep['subgoal'][:, 2]
        # subgoal rides ~subgoal_dist ahead until the path runs out near the end
        assert np.median(d) < 1.6
        assert d.max() < 2.5


def test_tracking_quality(episode_row):
    assert episode_row.get('cross_track_mean') is not None
    assert episode_row['cross_track_mean'] < 0.08, episode_row
