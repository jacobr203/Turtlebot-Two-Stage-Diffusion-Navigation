#!/usr/bin/env bash
# Create capstone/.venv and install the ML stack (CPU torch + requirements.txt).
#
# --system-site-packages keeps the apt-installed ROS 2 Python stack (rclpy,
# sensor_msgs, ...) importable from inside the venv — the deployment and
# recorder nodes need both torch and rclpy in one interpreter.
#
# No sudo required: python3.10-venv (ensurepip) is not installed on this box,
# so we fall back to a pip-installed virtualenv when the stdlib route fails.
# No -u: /opt/ros/humble/setup.bash reads unset vars (AMENT_TRACE_SETUP_FILES).
set -eo pipefail
cd "$(dirname "$0")"

if [ ! -e .venv/bin/python ]; then
  if python3 -c 'import ensurepip' 2>/dev/null; then
    python3 -m venv --system-site-packages .venv
  else
    python3 -m virtualenv --version >/dev/null 2>&1 \
      || pip3 install --user virtualenv
    python3 -m virtualenv --system-site-packages .venv
  fi
fi

./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
./.venv/bin/pip install -r requirements.txt

# Sanity: torch + numpy<2 + rclpy all importable in the same interpreter.
source /opt/ros/humble/setup.bash
./.venv/bin/python - <<'EOF'
import numpy, torch, rclpy  # noqa: F401
assert numpy.__version__.startswith("1."), f"numpy {numpy.__version__} (need <2)"
print(f"venv OK: torch {torch.__version__}, numpy {numpy.__version__}, rclpy importable")
EOF
