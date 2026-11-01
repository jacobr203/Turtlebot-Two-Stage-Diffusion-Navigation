"""Real-robot evaluation: run with evaluation/real_trial.py."""

# flake8: noqa
from .config import COLOUR_MODELS, ROBOT_BRINGUP, ROBOT_KILL, WS, detached
from .metrics import (latency_stats, offset_from_return, percentile,
                      ping_rtts, relative_pose)
from .monitors import ColourMonitor, TrialMonitor
from .records import (add_late_collisions, append_row, read_rows,
                      set_last_field, set_last_notes)
from .robot import RobotLink
from .schedule import (COLOUR_COUNTED, colour_pending, make_colour_schedule,
                       make_schedule, next_trial_index, pending_cells,
                       practice_cells)
from .tasks import ColourTask
