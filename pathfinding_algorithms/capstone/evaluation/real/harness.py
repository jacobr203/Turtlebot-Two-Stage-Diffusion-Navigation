"""Starts, resets and stops the ROS stack around a trial."""

import math
import os
import signal
import subprocess
import time

from common.geometry import yaw_from_quaternion

from .config import (
    BRINGUP_TIMEOUT_S, CAPSTONE, CKPTS, LOCAL_KILL_PATTERNS,
    NODE_TIMEOUT_S, ODOM_ZERO_TOL_M, ODOM_ZERO_TOL_RAD, POLICY_THREADS,
    SENSOR_WAIT_S, STOP_V_TOL, STOP_W_TOL, VENV_PY, WS, WS_SETUP, detached)
from .errors import BatteryLow, HarnessError


class RealHarness:
    """Owned rclpy context on the robot's domain plus process control for
    everything launched on the PC. Mirrors SimHarness where they overlap."""

    def __init__(self, domain=0):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from tf2_ros import Buffer, TransformListener

        env = os.environ.copy()
        env['PYTHONPATH'] = ':'.join(
            p for p in env.get('PYTHONPATH', '').split(':')
            if p and not p.rstrip('/').endswith('/tests'))
        env.update({'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp',
                    'CYCLONEDDS_URI': f'file://{WS}/config/cyclone_config.xml',
                    'TURTLEBOT3_MODEL': 'burger', 'LDS_MODEL': 'LDS-02',
                    'ROS_DOMAIN_ID': str(domain)})
        self.env = env
        self.procs = []
        self._rclpy = rclpy
        self._ctx = Context()
        rclpy.init(context=self._ctx, domain_id=domain)
        self.node = rclpy.create_node(f'real_trial_{os.getpid()}',
                                      context=self._ctx)
        self._exec = SingleThreadedExecutor(context=self._ctx)
        self._exec.add_node(self.node)
        self.tf = Buffer()
        self._tf_listener = TransformListener(self.tf, self.node,
                                              spin_thread=False)

    def popen(self, cmd, log_path):
        full = (f'source /opt/ros/humble/setup.bash && '
                f'source {WS_SETUP} && {cmd}')
        proc = subprocess.Popen(
            full, shell=True, executable='/bin/bash',
            stdin=subprocess.DEVNULL, stdout=open(log_path, 'w'),
            stderr=subprocess.STDOUT, env=self.env, start_new_session=True)
        self.procs.append(proc)
        return proc

    @staticmethod
    def stop(proc, grace_s=0.5):
        """SIGINT the process group (nodes zero cmd_vel, bags close), then
        SIGKILL whatever is left after grace_s."""
        if proc is None:
            return
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass

    def kill_local(self):
        for proc in self.procs:
            self.stop(proc, grace_s=0.2)
        self.procs.clear()
        for pat in LOCAL_KILL_PATTERNS:
            subprocess.run(['pkill', '-9', '-f', pat], capture_output=True)
        time.sleep(1.0)

    def spin(self, seconds=0.1):
        deadline = time.monotonic() + seconds
        while True:
            self._exec.spin_once(timeout_sec=0.02)
            if time.monotonic() >= deadline:
                return

    def nodes(self):
        self.spin(0.3)
        return [n for n, _ in self.node.get_node_names_and_namespaces()]

    def wait_for_node(self, name, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if name in self.nodes():
                return True
        return False

    def subscribe(self, topic, msg_type, callback, depth=50):
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        qos = QoSProfile(depth=depth)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        return self.node.create_subscription(msg_type, topic, callback, qos)

    def unsubscribe(self, sub):
        self.node.destroy_subscription(sub)

    def wait_for_message(self, topic, msg_type, timeout_s):
        box = []
        sub = self.subscribe(topic, msg_type, box.append, depth=5)
        try:
            deadline = time.monotonic() + timeout_s
            while not box and time.monotonic() < deadline:
                self.spin(0.1)
        finally:
            self.unsubscribe(sub)
        return box[-1] if box else None

    def publish_goal(self, x, y, frame='odom', n=3):
        from geometry_msgs.msg import PoseStamped
        msg = PoseStamped()
        msg.header.frame_id = frame
        msg.pose.position.x, msg.pose.position.y = float(x), float(y)
        msg.pose.orientation.w = 1.0
        self._publish_matched('/goal_pose', msg, n)

    def zero_wheels(self):
        """Zeros on /cmd_vel once the robot's subscription is matched —
        a publish before discovery completes is silently dropped."""
        from geometry_msgs.msg import Twist
        self._publish_matched('/cmd_vel', Twist(), 10)

    def _publish_matched(self, topic, msg, n):
        pub = self.node.create_publisher(type(msg), topic, 10)
        try:
            deadline = time.monotonic() + 5.0
            while (pub.get_subscription_count() == 0
                   and time.monotonic() < deadline):
                self.spin(0.1)
            for _ in range(n):
                pub.publish(msg)
                self.spin(0.1)
        finally:
            self.node.destroy_publisher(pub)

    def robot_still(self, timeout_s=10.0):
        """True once /odom reports near-zero velocity."""
        from nav_msgs.msg import Odometry
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            m = self.wait_for_message('/odom', Odometry, 2.0)
            if m is None:
                continue
            if (abs(m.twist.twist.linear.x) < STOP_V_TOL
                    and abs(m.twist.twist.angular.z) < STOP_W_TOL):
                return True
        return False

    def lookup(self, parent, child):
        """Latest (x, y, yaw, stamp_ns) of child in parent, or None. The
        buffer keeps old transforms for seconds, so callers compare stamps
        to tell a live pose from a stale one."""
        from rclpy.time import Time
        from tf2_ros import TransformException
        try:
            tf = self.tf.lookup_transform(parent, child, Time())
        except TransformException:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        stamp = tf.header.stamp.sec * 1_000_000_000 + tf.header.stamp.nanosec
        return (t.x, t.y, yaw_from_quaternion(q.x, q.y, q.z, q.w), stamp)

    def close(self):
        self.kill_local()
        self.node.destroy_node()
        self._rclpy.shutdown(context=self._ctx)


def reset_robot(h, robot, min_battery_v, say):
    """Full reset: both machines cleared, fresh bringup, odom verified zero.
    Returns the battery voltage."""
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import BatteryState, LaserScan
    h.kill_local()
    say('killing ROS on the robot...')
    robot.kill_all()
    say('starting bringup (odometry zeroes here)...')
    robot.start_bringup()
    if not h.wait_for_node('turtlebot3_node', BRINGUP_TIMEOUT_S):
        raise HarnessError('bringup did not come up')
    if h.wait_for_message('/scan', LaserScan, SENSOR_WAIT_S) is None:
        raise HarnessError('no /scan after bringup')
    odom = h.wait_for_message('/odom', Odometry, SENSOR_WAIT_S)
    if odom is None:
        raise HarnessError('no /odom after bringup')
    p, q = odom.pose.pose.position, odom.pose.pose.orientation
    yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
    if (math.hypot(p.x, p.y) > ODOM_ZERO_TOL_M
            or abs(yaw) > ODOM_ZERO_TOL_RAD):
        raise HarnessError(f'odometry not zeroed after bringup '
                           f'({p.x:.3f}, {p.y:.3f}, {yaw:.3f})')
    batt = h.wait_for_message('/battery_state', BatteryState, SENSOR_WAIT_S)
    volts = round(batt.voltage, 2) if batt is not None else None
    if volts is not None and volts < min_battery_v:
        raise BatteryLow(f'battery {volts} V < {min_battery_v} V')
    return volts


def start_controller(h, course, cell, trial_dir, deadman_s):
    model, hz = cell['model'], float(cell['hz'])
    gx, gy = course['goal']
    if model == 'dwa':
        proc = h.popen('ros2 run dwa_avoidance dwa_node --ros-args '
                       f'-p control_period:={1.0 / hz}',
                       trial_dir / 'controller.log')
        if not h.wait_for_node('dwa_node', NODE_TIMEOUT_S):
            raise HarnessError('dwa_node did not start')
        h.publish_goal(gx, gy)
        return proc
    args = (f'--ros-args -p use_sim_time:=false '
            f'-p checkpoint:={CKPTS[model]} -p behavior:=goto '
            f'-p control_rate_hz:={hz} -p denoiser:=ddpm '
            f'-p safety_gate:=false -p threads:={POLICY_THREADS} '
            f'-p sensor_deadman_s:={deadman_s} '
            f'-p spawn_x:=0.0 -p spawn_y:=0.0 -p spawn_yaw:=0.0 '
            f'-p goal_x:={gx} -p goal_y:={gy} '
            f'-p latency_log:={trial_dir / "lat.csv"}')
    if model == 'two_stage':
        args += f' -p grid_file:={course["path"]}'
    proc = h.popen(f'{VENV_PY} {CAPSTONE}/deploy/policy_node.py {args}',
                   trial_dir / 'controller.log')
    if not h.wait_for_node('diffusion_policy', NODE_TIMEOUT_S):
        raise HarnessError('policy node did not start')
    return proc


def stop_everything_moving(h, robot, controller, say):
    """Controller down, wheels zeroed, stop verified on /odom; restarts the
    motor controller as a last resort. Returns whether the stop verified."""
    h.stop(controller, grace_s=0.5)
    h.zero_wheels()
    if h.robot_still():
        return True
    say('robot still reports motion — restarting bringup to reset motors')
    robot.run(detached("pkill -9 -f 'turtlebot[3]'", '/dev/null'))
    time.sleep(2)
    robot.start_bringup()
    h.wait_for_node('turtlebot3_node', BRINGUP_TIMEOUT_S)
    h.zero_wheels()
    if h.robot_still():
        return True
    print('\n!!! ROBOT MAY STILL BE MOVING — USE ITS POWER SWITCH !!!\n')
    return False
