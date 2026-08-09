"""Simulation harness for BARN expert collection and evaluation.

Adapted from unit_tests/gazebo_sim/helpers.py (env scrub, sourced launches,
rclpy graph probes) and extended with:

  - per-worker isolation: worker k gets ROS_DOMAIN_ID=10+k (domain 0 is the
    REAL robot's — sharing it merges sim and real topics) and
    GAZEBO_MASTER_URI=http://localhost:{11345+k}, so several gzservers and
    ROS graphs can run side by side (CycloneDDS domains are isolated);
  - an owned rclpy Context on that domain (probes/publishers/service calls
    do not depend on process-wide env or a global rclpy.init);
  - gazebo lifecycle helpers: pause/unpause, delete_entity, service-based
    spawn — the per-trial reset is delete+respawn, which re-anchors the
    diff-drive plugin's odom at the spawn pose like a real power-on.
"""

import math
import os
import signal
import subprocess
import time
from pathlib import Path

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy

CAPSTONE = Path(__file__).resolve().parents[1]
WS_SETUP = CAPSTONE.parent / 'install' / 'setup.bash'
BUMPER_SDF = CAPSTONE / 'models' / 'turtlebot3_burger_bumper' / 'model.sdf'
DEFAULT_URDF = ('/opt/ros/humble/share/turtlebot3_gazebo/urdf/'
                'turtlebot3_burger.urdf')
VENV_PY = CAPSTONE / '.venv' / 'bin' / 'python'

# Process names a harness may leave behind. Used for the per-worker orphan
# sweep in kill_all() — matched against /proc cmdlines, but a process is only
# killed if its environment carries THIS worker's GAZEBO_MASTER_URI. A global
# name-based pkill here is fatal with concurrent workers: each restart_world
# SIGKILLs every other worker's bringup mid-flight (full_v1's first launch
# died exactly this way), and a policy node killed mid-trial leaves the sim
# running to a genuine-looking timeout verdict — silent data corruption.
KILL_NAMES = [
    'gzserver', 'gzclient', 'spawn_entity', 'robot_state_publisher',
    'sim_world.launch.py', 'tracker_node', 'recorder_node',
    'policy_node', 'dwa_node', 'cartographer_node',
]


def stop_proc(proc, sigint_grace=0.5):
    """Stop one popen()-started tree: SIGINT to the process GROUP first
    (rclpy nodes zero their cmd_vel in shutdown handlers), then SIGKILL,
    then reap. Signalling proc.pid alone reaches only the bash wrapper —
    under `ros2 run` the node survives orphaned and keeps publishing
    cmd_vel into the trial that follows (decisions.md #22)."""
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except (ProcessLookupError, PermissionError):
        pass
    time.sleep(sigint_grace)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=2)
    except Exception:
        pass


def barn_world(idx, fast=False):
    """Absolute path of a fixed-up BARN world (fast = unbounded-RTF variant)."""
    sub = 'worlds_fast' if fast else 'worlds'
    return CAPSTONE / 'barn' / sub / f'world_{idx:03d}.world'


def barn_grid(idx):
    return CAPSTONE / 'barn' / 'grids' / f'grid_{idx:03d}.npy'


class SimHarness:
    """One isolated Gazebo + ROS graph, owned processes, owned rclpy context."""

    # Worker k lives on ROS domain 10+k: domain 0 belongs to the REAL robot,
    # and a sim session sharing it merges both robots' topics (each consumer
    # reads a blend of real and simulated /scan + /odom).
    DOMAIN_BASE = 10

    def __init__(self, worker=0, log_dir=None):
        self.worker = int(worker)
        self.domain = self.DOMAIN_BASE + self.worker
        self.log_dir = Path(log_dir) if log_dir else Path('/tmp')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.procs = []

        env = os.environ.copy()
        # run_tests.py prepends ~/unit_tests to PYTHONPATH; launched ROS nodes
        # must not inherit it (test folders shadow ROS packages — bug #016).
        env['PYTHONPATH'] = ':'.join(
            p for p in env.get('PYTHONPATH', '').split(':')
            if p and not p.rstrip('/').endswith('unit_tests'))
        env['RMW_IMPLEMENTATION'] = 'rmw_cyclonedds_cpp'
        env['CYCLONEDDS_URI'] = 'file:///home/turtlebot/cyclone_config.xml'
        env['TURTLEBOT3_MODEL'] = 'burger'
        env['GAZEBO_MODEL_DATABASE_URI'] = ''   # no online-DB stall on WSL2
        env['ROS_DOMAIN_ID'] = str(self.domain)
        env['GAZEBO_MASTER_URI'] = f'http://localhost:{11345 + self.worker}'
        self.env = env

        self._ctx = Context()
        rclpy.init(context=self._ctx, domain_id=self.domain)
        self._node = rclpy.create_node(
            f'sim_harness_w{self.worker}_{os.getpid()}', context=self._ctx)
        self._node.set_parameters([Parameter('use_sim_time', value=True)])
        self._exec = SingleThreadedExecutor(context=self._ctx)
        self._exec.add_node(self._node)

    # ── process management ───────────────────────────────────────────────────
    def popen(self, cmd, log_name):
        """Run a sourced ROS command in this harness's environment.

        start_new_session gives each command its own process group, so
        kill_all can take down the whole launch tree (ros2 launch,
        gzserver, spawn, ...) by group without touching other workers."""
        full = (f'source /opt/ros/humble/setup.bash && '
                f'source {WS_SETUP} && {cmd}')
        proc = subprocess.Popen(
            full, shell=True, executable='/bin/bash',
            stdout=open(self.log_dir / log_name, 'w'),
            stderr=subprocess.STDOUT, env=self.env,
            start_new_session=True)
        self.procs.append(proc)
        return proc

    def start_world(self, world_path, x=-2.25, y=3.0, yaw=1.5708,
                    model_sdf=BUMPER_SDF, urdf_file=DEFAULT_URDF,
                    headless=True, timeout=60):
        """Launch gzserver(+client)+rsp+spawn; blocks until /scan is up."""
        self.popen(
            'ros2 launch dwa_avoidance sim_world.launch.py '
            f'world:={world_path} headless:={"true" if headless else "false"} '
            f'x_pose:={x} y_pose:={y} yaw_pose:={yaw} '
            f'model_sdf:={model_sdf} urdf_file:={urdf_file}',
            f'sim_world_w{self.worker}.log')
        if not self.wait_for_topic('/scan', timeout):
            raise RuntimeError(
                f'/scan never appeared for {world_path} — see '
                f'{self.log_dir}/sim_world_w{self.worker}.log')

    def _worker_orphans(self):
        """PIDs of sim processes that belong to THIS worker: cmdline matches
        a KILL_NAMES entry AND the environment carries this worker's unique
        GAZEBO_MASTER_URI (inherited by everything popen() spawned, even
        after a crashed predecessor orphaned it). The harness's own python
        never carries the URI — self.env is a copy handed to children."""
        token = (f'GAZEBO_MASTER_URI={self.env["GAZEBO_MASTER_URI"]}'
                 .encode() + b'\0')
        names = [n.encode() for n in KILL_NAMES]
        me = os.getpid()
        pids = []
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name) == me:
                continue
            try:
                cmd = (p / 'cmdline').read_bytes()
                if not any(n in cmd for n in names):
                    continue
                if token in (p / 'environ').read_bytes():
                    pids.append(int(p.name))
            except OSError:
                continue   # raced exit, or not ours to read
        return pids

    def kill_all(self, settle=4.0):
        """Kill everything THIS harness started — and this worker's orphans
        from a crashed predecessor — but never another worker's processes."""
        for proc in self.procs:
            try:
                os.killpg(proc.pid, signal.SIGKILL)   # whole launch tree
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)   # reap — else zombies pile up all night
            except Exception:
                pass
        self.procs.clear()
        for pid in self._worker_orphans():
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(settle)   # gzserver sockets/locks need time to clear

    def close(self):
        self.kill_all()
        self._node.destroy_node()
        rclpy.shutdown(context=self._ctx)

    # ── graph probes ─────────────────────────────────────────────────────────
    def spin(self, seconds=0.5):
        deadline = time.time() + seconds
        while time.time() < deadline:
            self._exec.spin_once(timeout_sec=0.1)

    def topics(self):
        self.spin(0.5)
        return {t for t, _ in self._node.get_topic_names_and_types()}

    def nodes(self):
        self.spin(0.5)
        return [n for n, _ in self._node.get_node_names_and_namespaces()]

    def wait_for_topic(self, topic, timeout_s):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if topic in self.topics():
                return True
        return False

    def wait_for_node(self, name, timeout_s=30):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if name in self.nodes():
                return True
        return False

    def sim_time(self):
        """Current sim time [s] (0.0 until a /clock publisher exists)."""
        self.spin(0.2)
        return self._node.get_clock().now().nanoseconds * 1e-9

    # ── messaging ────────────────────────────────────────────────────────────
    class _Latest:
        """Holder updated while the harness spins; see SimHarness.track()."""
        def __init__(self):
            self.msg = None
            self.count = 0

        def _cb(self, msg):
            self.msg = msg
            self.count += 1

    def track(self, topic, msg_type, best_effort=True):
        """Persistent subscription; returns a holder whose .msg/.count update
        on every spin. Destroy with untrack(holder)."""
        holder = self._Latest()
        qos = QoSProfile(depth=10)
        if best_effort:
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
        holder._sub = self._node.create_subscription(
            msg_type, topic, holder._cb, qos)
        return holder

    def untrack(self, holder):
        self._node.destroy_subscription(holder._sub)

    def subscribe(self, topic, msg_type, callback, best_effort=True):
        """Raw subscription on the harness node (callback runs during spins).
        Use when every message matters (e.g. accumulating bumper contacts —
        a latest-only holder could miss a brief contact between spins)."""
        qos = QoSProfile(depth=50)
        if best_effort:
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
        return self._node.create_subscription(msg_type, topic, callback, qos)

    def publisher(self, topic, msg_type, qos=10):
        """Persistent publisher on the harness node."""
        return self._node.create_publisher(msg_type, topic, qos)

    def collect_msgs(self, topic, msg_type, duration_s, best_effort=True):
        """Subscribe for duration_s (wall) and return every message seen."""
        out = []
        qos = QoSProfile(depth=20)
        if best_effort:
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sub = self._node.create_subscription(msg_type, topic, out.append, qos)
        try:
            self.spin(duration_s)
        finally:
            self._node.destroy_subscription(sub)
        return out

    def publish(self, topic, msg, n=1, interval=0.1, qos=10):
        pub = self._node.create_publisher(type(msg), topic, qos)
        try:
            deadline = time.time() + 5.0
            while pub.get_subscription_count() == 0 and time.time() < deadline:
                self.spin(0.1)
            for _ in range(n):
                pub.publish(msg)
                self.spin(interval)
        finally:
            self._node.destroy_publisher(pub)

    def _call(self, srv_type, name, request, timeout_s=10, attempts=4):
        """Service call with per-attempt fresh clients: the harness node
        outlives gzserver restarts, and a client created against stale graph
        state can silently lose its request — recreating the client forces
        fresh DDS discovery."""
        last = 'no attempts'
        for k in range(attempts):
            client = self._node.create_client(srv_type, name)
            try:
                if not client.wait_for_service(timeout_sec=timeout_s / 2):
                    last = f'unavailable (attempt {k + 1})'
                    continue
                fut = client.call_async(request)
                deadline = time.time() + timeout_s
                while not fut.done() and time.time() < deadline:
                    self._exec.spin_once(timeout_sec=0.1)
                if fut.done():
                    return fut.result()
                last = f'no response (attempt {k + 1})'
            finally:
                self._node.destroy_client(client)
        raise RuntimeError(f'service {name}: {last}')

    # ── gazebo lifecycle ─────────────────────────────────────────────────────
    def pause(self):
        from std_srvs.srv import Empty
        self._call(Empty, '/pause_physics', Empty.Request())

    def unpause(self):
        from std_srvs.srv import Empty
        self._call(Empty, '/unpause_physics', Empty.Request())

    def delete_entity(self, name='turtlebot3'):
        from gazebo_msgs.srv import DeleteEntity
        req = DeleteEntity.Request()
        req.name = name
        res = self._call(DeleteEntity, '/delete_entity', req)
        return res.success

    def spawn_robot(self, x, y, yaw, name='turtlebot3', model_sdf=BUMPER_SDF):
        from gazebo_msgs.srv import SpawnEntity
        req = SpawnEntity.Request()
        req.name = name
        req.xml = Path(model_sdf).read_text()
        req.initial_pose.position.x = float(x)
        req.initial_pose.position.y = float(y)
        req.initial_pose.position.z = 0.01
        req.initial_pose.orientation.z = math.sin(yaw / 2.0)
        req.initial_pose.orientation.w = math.cos(yaw / 2.0)
        req.reference_frame = 'world'
        res = self._call(SpawnEntity, '/spawn_entity', req)
        if not res.success:
            raise RuntimeError(f'spawn failed: {res.status_message}')

    def respawn_robot(self, x, y, yaw, name='turtlebot3',
                      model_sdf=BUMPER_SDF):
        """Delete+respawn under paused physics: fresh odom anchored at the
        new spawn pose, no residual velocities."""
        self.pause()
        try:
            self.delete_entity(name)
        except RuntimeError:
            pass   # first spawn of a fresh world: nothing to delete
        time.sleep(0.5)
        self.spawn_robot(x, y, yaw, name=name, model_sdf=model_sdf)


def contact_obstacles(contacts_msg):
    """Names of non-ground things a ContactsState message says we touched."""
    hits = set()
    for state in contacts_msg.states:
        for other in (state.collision1_name, state.collision2_name):
            if 'turtlebot3' in other or 'ground_plane' in other:
                continue
            hits.add(other)
    return hits
