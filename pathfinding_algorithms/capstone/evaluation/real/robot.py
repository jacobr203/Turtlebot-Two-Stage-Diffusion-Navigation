"""SSH access to the robot."""

import os
import subprocess
import time

from .config import (
    ROBOT_ALIVE, ROBOT_BRINGUP, ROBOT_ENV, ROBOT_KILL, detached)
from .errors import HarnessError
from .metrics import offset_from_return

def read_robot_env(path=ROBOT_ENV):
    """KEY=value pairs from robot.env; the shell environment overrides."""
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip().strip('"\'')
    for k in ('ROBOT_IP', 'ROBOT_USER', 'ROBOT_PASS'):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


class RobotLink:
    """SSH to the robot. The password travels in SSHPASS, never argv."""

    def __init__(self, env):
        if (not env.get('ROBOT_IP') or not env.get('ROBOT_USER')
                or env['ROBOT_IP'].startswith('TODO')):
            raise SystemExit('robot.env needs ROBOT_IP and ROBOT_USER '
                             '(copy robot.env.example to robot.env)')
        self.ip = env['ROBOT_IP']
        self.target = f"{env['ROBOT_USER']}@{env['ROBOT_IP']}"
        self.password = env.get('ROBOT_PASS', '')

    def _argv(self, cmd):
        # -n: a remote background command holding the channel's stdin would
        # keep the SSH session open until it exits, forever for bringup
        opts = ['-n', '-o', 'StrictHostKeyChecking=no',
                '-o', 'ConnectTimeout=5']
        if self.password:
            return ['sshpass', '-e', 'ssh', *opts, self.target, cmd]
        return ['ssh', '-o', 'BatchMode=yes', *opts, self.target, cmd]

    def run(self, cmd, timeout=20):
        """(returncode, stdout). A robot that is off or unreachable gives
        (124, '') instead of raising: teardown must always finish."""
        env = dict(os.environ, SSHPASS=self.password)
        try:
            res = subprocess.run(self._argv(cmd), capture_output=True,
                                 stdin=subprocess.DEVNULL, text=True,
                                 timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return 124, ''
        return res.returncode, res.stdout

    def kill_all(self):
        """Kill every ROS process on the robot and confirm none survive."""
        self.run(ROBOT_KILL)
        deadline = time.time() + 12
        while time.time() < deadline:
            time.sleep(1.5)
            _, out = self.run(ROBOT_ALIVE)
            if not out.strip():
                return
        raise HarnessError(f'robot ROS processes survived the kill:\n{out}')

    def start_bringup(self):
        self.run(ROBOT_BRINGUP)

    def start_camera(self, extra_params=''):
        """BRIO driver on the robot, as run.sh's ensure_camera starts it.
        extra_params pass through (e.g. exposure / white balance locks)."""
        self.run(detached(
            'source /opt/ros/humble/setup.bash && '
            'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && '
            'exec ros2 run v4l2_camera v4l2_camera_node --ros-args '
            '-p video_device:=/dev/video0 -p image_size:=[640,480] '
            f'{extra_params}', '/tmp/tbot_camera.log'))

    def clock_offset_ms(self):
        """Robot minus PC wall clock [ms]. The robot reads its clock at the
        very end of the SSH call, so compare with the moment the call
        returns: the midpoint would count connection setup (up to ~1 s)."""
        rc, out = self.run('date +%s.%N')
        t_after = time.time()
        if rc != 0:
            return None
        return round(offset_from_return(float(out.strip()), t_after) * 1000, 1)
