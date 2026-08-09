"""
Gazebo sim: Nav2 smoke tests.

Verifies the Nav2 stack (turtlebot3_navigation2 with its default
turtlebot3_world map) reaches the active lifecycle state in sim.
"""

import time

import rclpy
from lifecycle_msgs.srv import GetState

LIFECYCLE_NODES = ['amcl', 'controller_server', 'planner_server',
                   'bt_navigator']
ACTIVE_TIMEOUT_S = 90.0


def _lifecycle_state(node, target: str, timeout: float) -> str:
    client = node.create_client(GetState, f'/{target}/get_state')
    if not client.wait_for_service(timeout_sec=timeout):
        return 'service unavailable'
    future = client.call_async(GetState.Request())
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    if future.result() is None:
        return 'no response'
    return future.result().current_state.label


def test_lifecycle_nodes_active(nav2_sim):
    node = rclpy.create_node('_nav2_sim_lifecycle')
    try:
        deadline = time.time() + ACTIVE_TIMEOUT_S
        pending = dict.fromkeys(LIFECYCLE_NODES, 'unknown')
        while pending and time.time() < deadline:
            for name in list(pending):
                state = _lifecycle_state(node, name, timeout=5.0)
                pending[name] = state
                if state == 'active':
                    del pending[name]
            if pending:
                time.sleep(2)
        assert not pending, f'lifecycle nodes not active: {pending}'
    finally:
        node.destroy_node()


def test_map_served(nav2_sim):
    """map_server must publish the static map."""
    from nav_msgs.msg import OccupancyGrid
    from rclpy.qos import (QoSDurabilityPolicy, QoSProfile,
                           QoSReliabilityPolicy)
    qos = QoSProfile(depth=1,
                     reliability=QoSReliabilityPolicy.RELIABLE,
                     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
    node = rclpy.create_node('_nav2_sim_map')
    maps = []
    node.create_subscription(OccupancyGrid, '/map', maps.append, qos)
    deadline = time.time() + 30
    while not maps and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    assert maps, 'map_server never published /map'
    assert maps[0].info.width > 10 and maps[0].info.height > 10
