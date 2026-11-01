#!/usr/bin/env python3
"""Send the BARN goal to Nav2 as a through-poses route along a centered A* corridor.

A single NavigateToPose goal leaves routing to NavFn, which corner-hugs into pockets
RPP then refuses. Falls back to that single goal if the through-poses server never
appears; a mid-course abort resumes from the nearest waypoint, --retries times.

Exit codes: 0 goal reached, 1 failed/rejected, 3 sim wedge (commands flowing but odom
frozen for 45 s; run.sh rebuilds the sim stack). Test hook: NAV2_ROUTE_FORCE_WEDGE=<path>
fakes one wedge, creating <path> as its marker.

  .venv/bin/python barn/nav2_route_goal.py --grid barn/grids/grid_306.npy
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import grid as G                                  # noqa: E402

STATUS = {4: 'SUCCEEDED', 5: 'CANCELED', 6: 'ABORTED'}


def plan_waypoints(grid_file, start_xy, goal_xy, spacing):
    """Centered A* route as (x, y, yaw) waypoints, `spacing` metres apart."""
    cylinders = G.arena_grid_to_cylinders(np.load(grid_file))
    planner, pts, inflate = G.plan_route(cylinders, tuple(start_xy),
                                         tuple(goal_xy))
    if pts is None:
        raise SystemExit(f'[route] no A* path in {grid_file} — infeasible grid?')
    minc = planner.min_clearance(pts)

    wps_xy = [tuple(map(float, p)) for p in G._resample(pts, spacing)[1:]]
    wps_xy[-1] = tuple(goal_xy)
    wps = []
    prev = start_xy
    for k, p in enumerate(wps_xy):
        # official goal keeps yaw 0 (matches the single-goal behavior)
        yaw = 0.0 if k == len(wps_xy) - 1 else math.atan2(p[1] - prev[1],
                                                          p[0] - prev[0])
        wps.append((p[0], p[1], yaw))
        prev = p
    print(f'[route] A* corridor (centered, inflate {inflate}): {len(wps)} '
          f'waypoints @ {spacing} m, length {G.path_length(pts):.1f} m, '
          f'min clearance {minc:.3f} m', flush=True)
    return wps, planner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--spacing', type=float, default=0.5)
    ap.add_argument('--start', type=float, nargs=2, default=list(G.START_POSE[:2]))
    ap.add_argument('--goal', type=float, nargs=2, default=list(G.GOAL_XY))
    ap.add_argument('--retries', type=int, default=2,
                    help='resume attempts after a mid-course abort')
    ap.add_argument('--settle', type=float, default=8.0,
                    help='wait after the server is up so AMCL takes the initial pose')
    ap.add_argument('--server-timeout', type=float, default=150.0)
    ap.add_argument('--result-timeout', type=float, default=600.0)
    ap.add_argument('--dry-run', action='store_true',
                    help='plan and print the waypoints, send nothing')
    args = ap.parse_args()

    wps, planner = plan_waypoints(args.grid, tuple(args.start),
                                  tuple(args.goal), args.spacing)
    hook = os.environ.get('NAV2_ROUTE_FORCE_WEDGE')
    if hook and not os.path.exists(hook):
        open(hook, 'w').close()
        print('[route] TEST HOOK: simulating a physics wedge (exit 3)',
              flush=True)
        sys.exit(3)
    if args.dry_run:
        cl = planner._nearest_dist([(x, y) for x, y, _ in wps])
        for (x, y, yaw), c in zip(wps, cl):
            print(f'  ({x:6.2f}, {y:5.2f})  yaw {math.degrees(yaw):6.1f}  '
                  f'clearance {c:.3f}')
        return

    import rclpy
    from rclpy.action import ActionClient
    from rclpy.qos import (QoSDurabilityPolicy, QoSProfile,
                           QoSReliabilityPolicy)
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
    from nav_msgs.msg import Odometry
    from nav2_msgs.action import NavigateThroughPoses, NavigateToPose

    rclpy.init()
    n = rclpy.create_node('_run_sh_nav2_route')

    # Sim-wedge detector: a working recovery ladder moves far more than 3 cm in
    # 45 s (backup alone is 0.3 m), so near-zero motion under commands is terminal.
    WEDGE_WINDOW, WEDGE_DISP = 45.0, 0.03
    wedge = {'hist': [], 'cmd_t': 0.0}
    qos_be = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)

    def on_odom(m):
        p = m.pose.pose.position
        wedge['hist'].append((time.time(), p.x, p.y))
        if len(wedge['hist']) > 2000:
            del wedge['hist'][:1000]

    def on_cmd(m):
        if abs(m.linear.x) > 0.01 or abs(m.angular.z) > 0.05:
            wedge['cmd_t'] = time.time()

    n.create_subscription(Odometry, '/odom', on_odom, qos_be)
    n.create_subscription(Twist, '/cmd_vel', on_cmd, 10)

    def wedged():
        now = time.time()
        if now - wedge['cmd_t'] > 10.0:
            return False
        win = [(t, x, y) for t, x, y in wedge['hist'] if now - t <= WEDGE_WINDOW]
        if len(win) < 20 or win[-1][0] - win[0][0] < WEDGE_WINDOW - 5.0:
            return False
        x0, y0 = win[0][1], win[0][2]
        return max(math.hypot(x - x0, y - y0) for _, x, y in win) < WEDGE_DISP

    def pose(x, y, yaw):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x, p.pose.position.y = x, y
        p.pose.orientation.z = math.sin(yaw / 2.0)
        p.pose.orientation.w = math.cos(yaw / 2.0)
        return p

    def amcl_pose(timeout=10.0):
        got = {}
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        sub = n.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose',
            lambda m: got.__setitem__('p', (m.pose.pose.position.x,
                                            m.pose.pose.position.y)), qos)
        t0 = time.time()
        while 'p' not in got and time.time() - t0 < timeout:
            rclpy.spin_once(n, timeout_sec=0.2)
        n.destroy_subscription(sub)
        return got.get('p')

    def send_and_wait(ac, goal, what):
        fut = ac.send_goal_async(goal)
        deadline = time.time() + 15
        while not fut.done() and time.time() < deadline:
            rclpy.spin_once(n, timeout_sec=0.2)
        handle = fut.result() if fut.done() else None
        accepted = bool(handle and handle.accepted)
        print(f'[route] BARN goal ({args.goal[0]}, {args.goal[1]}) sent as '
              f'{what}: accepted={accepted}', flush=True)
        if not accepted:
            return None
        t0 = time.time()
        rfut = handle.get_result_async()
        while not rfut.done() and time.time() - t0 < args.result_timeout:
            rclpy.spin_once(n, timeout_sec=0.5)
            if wedged():
                print('[route] SIM WEDGE: motion commands flowing but '
                      f'odometry frozen for {WEDGE_WINDOW:.0f} s — the robot '
                      'is not executing commands. Exiting 3 '
                      'for run.sh to restart the sim stack.', flush=True)
                handle.cancel_goal_async()
                for _ in range(10):
                    rclpy.spin_once(n, timeout_sec=0.2)
                sys.exit(3)
        if not rfut.done():
            print(f'[route] no result within {args.result_timeout:.0f} s — '
                  'still navigating?', flush=True)
            return None
        status = rfut.result().status
        print(f'[route] navigation result: {STATUS.get(status, status)} '
              f'after {time.time() - t0:.0f} s', flush=True)
        return status

    ac = ActionClient(n, NavigateThroughPoses, '/navigate_through_poses')
    if not ac.wait_for_server(timeout_sec=args.server_timeout):
        # bt_navigator without the through-poses navigator: stock single goal
        print('[route] /navigate_through_poses unavailable — falling back to '
              'a single NavigateToPose goal', flush=True)
        ac = ActionClient(n, NavigateToPose, '/navigate_to_pose')
        if not ac.wait_for_server(timeout_sec=30):
            raise SystemExit('[route] no Nav2 action server — giving up')
        goal = NavigateToPose.Goal()
        goal.pose = pose(args.goal[0], args.goal[1], 0.0)
        time.sleep(args.settle)   # let AMCL take the auto initial pose first
        ok = send_and_wait(ac, goal, 'single goal') == 4
        rclpy.shutdown()
        sys.exit(0 if ok else 1)

    time.sleep(args.settle)   # let AMCL take the auto initial pose first
    route = wps
    status = None
    for attempt in range(args.retries + 1):
        if attempt:
            at = amcl_pose()
            if at is None:
                print('[route] no AMCL pose — cannot resume', flush=True)
                break
            nearest = min(range(len(wps)),
                          key=lambda i: math.hypot(wps[i][0] - at[0],
                                                   wps[i][1] - at[1]))
            route = wps[nearest + 1:] or wps[-1:]
            print(f'[route] retry {attempt}/{args.retries}: resuming from '
                  f'waypoint {nearest + 1}/{len(wps)} '
                  f'(robot at {at[0]:.2f}, {at[1]:.2f})', flush=True)
        goal = NavigateThroughPoses.Goal()
        goal.poses = [pose(*w) for w in route]
        status = send_and_wait(
            ac, goal, f'through-poses route ({len(goal.poses)} waypoints)')
        if status == 4 or status is None:
            break
    rclpy.shutdown()
    sys.exit(0 if status == 4 else 1)


if __name__ == '__main__':
    main()
