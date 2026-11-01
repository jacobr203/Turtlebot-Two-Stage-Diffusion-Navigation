# TODO: fields to fill in

Personal details and network addresses are not stored in this repository.
Simulation works without any of this. Real-robot runs need section 1.

## 1. Robot connection (required for real-robot runs)

```bash
cp pathfinding_algorithms/robot.env.example pathfinding_algorithms/robot.env
```

Then replace each `TODO_` value in `robot.env`, which is gitignored:

- [ ] `ROBOT_IP`: the TurtleBot's IP address
- [ ] `ROBOT_USER`: the SSH user on the robot
- [ ] `ROBOT_PASS`: leave empty if you use SSH keys (recommended)
- [ ] `WORKSTATION_IP`: this PC's IP address on the robot's network

`run.sh`, `tools/` and `capstone/evaluation/real_trial.py` refuse to contact
the robot while `ROBOT_IP` is empty or still a `TODO_` value.

## 2. Network interface

- [ ] In `pathfinding_algorithms/config/cyclone_config.xml`, set
      `NetworkInterface name` to the adapter that reaches the robot (`eth0` by
      default).

## 3. Maintainer details

These are placeholders (`TODO: maintainer name`, `maintainer@example.com`):

- [ ] `dwa_avoidance/package.xml` and `dwa_avoidance/setup.py`
- [ ] `dwa_camera_avoidance/package.xml` and `dwa_camera_avoidance/setup.py`
- [ ] `dwa_camera_avoidance/models/turtlebot3_burger_brio/model.config`
- [ ] `capstone/models/turtlebot3_burger_bumper/model.config`
- [ ] `capstone/models/turtlebot3_burger_bumper_brio/model.config`

## 4. Your own environment

- [ ] Recalibrate `capstone/deploy/color_profiles.yaml` for your camera and
      lighting with `dev_tools/color_mask_viz.py`.
- [ ] Map your space and replace `maps/reference/my_map.*` and
      `capstone/evaluation/arenas/room1*`, which describe the original test room.
