"""Behaviour layers for the diffusion policy node.

The deployment node is split by concern. Each module is a mixin
composed into DiffusionPolicyNode, so `self` is the node itself and no
module owns state privately:

    geometry    frame conversions, line of sight, coverage memory
    viewpoint   where to stand to see a target, corridor memory
    explore     frontier exploration, wander goals, face coverage
    seek        what to chase and how, during the colour search
    safety      speed gate, recovery behaviours, watchdog
    tracking    the control tick, trajectory sampling, pure pursuit

The diffusion model authors every trajectory the robot follows; these
layers decide where to aim it and refuse commands that would collide.
"""

from deploy.policy.geometry import GeometryMixin
from deploy.policy.viewpoint import ViewpointMixin
from deploy.policy.explore import ExploreMixin
from deploy.policy.seek import SeekMixin
from deploy.policy.safety import SafetyMixin
from deploy.policy.tracking import TrackingMixin

__all__ = ['GeometryMixin', 'ViewpointMixin', 'ExploreMixin',
           'SeekMixin', 'SafetyMixin', 'TrackingMixin']
