"""BARN world 0 loads in Gazebo Classic with the bumper burger spawned at the
official start pose: lidar alive, cylinders present, GT + bumper channels up."""

from gazebo_msgs.msg import ContactsState, ModelStates
from sensor_msgs.msg import LaserScan

from evaluation.sim_helpers import contact_obstacles


def test_scan_publishes_360_rays(barn_sim):
    msgs = barn_sim.collect_msgs('/scan', LaserScan, 4.0)
    assert msgs, '/scan silent'
    scan = msgs[-1]
    assert len(scan.ranges) == 360
    assert scan.header.frame_id == 'base_scan'
    finite = [r for r in scan.ranges if 0.12 <= r <= 3.5]
    assert len(finite) > 20, 'lidar sees no cylinders/walls from the spawn'


def test_state_plugin_reports_world(barn_sim):
    msgs = barn_sim.collect_msgs('/gazebo/model_states', ModelStates, 3.0,
                                 best_effort=False)
    assert msgs, '/gazebo/model_states silent — state plugin not injected?'
    names = msgs[-1].name
    assert 'turtlebot3' in names
    n_cyl = sum(1 for n in names if n.startswith('unit_cylinder'))
    assert n_cyl >= 180, f'only {n_cyl} cylinders in world 0'


def test_bumper_quiet_at_spawn(barn_sim):
    msgs = barn_sim.collect_msgs('/bumper_states', ContactsState, 3.0)
    assert msgs, '/bumper_states silent — bumper plugin not loaded?'
    hits = set()
    for m in msgs:
        hits |= contact_obstacles(m)
    assert not hits, f'phantom contacts at spawn: {hits}'
