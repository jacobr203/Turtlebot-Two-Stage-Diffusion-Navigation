"""Deliberate crash: face the side wall and drive until the bumper fires.

Validates the whole collision channel end-to-end (contact sensor -> plugin ->
/bumper_states -> ground filter) exactly as expert collection and evaluation
consume it."""

import time

from gazebo_msgs.msg import ContactsState
from geometry_msgs.msg import Twist

from evaluation.sim_helpers import contact_obstacles


def test_bumper_fires_on_wall_crash(barn_sim):
    h = barn_sim
    # face +x: the right wall is ~2.1 m away, nothing else in between
    h.respawn_robot(-2.25, 3.0, 0.0)
    h.unpause()

    hits = set()
    sub = h.subscribe('/bumper_states', ContactsState,
                      lambda m: hits.update(contact_obstacles(m)))
    cmd_pub = h.publisher('/cmd_vel', Twist)
    fwd = Twist()
    fwd.linear.x = 0.21

    deadline = time.time() + 90
    try:
        while time.time() < deadline and not hits:
            cmd_pub.publish(fwd)
            h.spin(0.3)
    finally:
        cmd_pub.publish(Twist())   # stop
        h.spin(0.3)
        h._node.destroy_subscription(sub)
        h._node.destroy_publisher(cmd_pub)

    assert hits, 'drove into the wall but /bumper_states never reported contact'
    assert all('unit_cylinder' in c for c in hits), \
        f'unexpected contact partners: {hits}'
