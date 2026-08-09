# maps/ — per-run occupancy maps

Every `run.sh lidar` or `run.sh camera` run saves the occupancy grid it
builds here, refreshed every 30 s while the run is live and once more at
shutdown — so each run leaves its final map behind even after a crash.

Folders match the `run.sh` algorithm names:

```
maps/
├── sim/                          ← Gazebo runs
│   ├── lidar/
│   │   └── 2026-07-06_14-32-01_lidar.{pgm,yaml}
│   └── camera/
│       └── 2026-07-06_15-10-44_camera.{pgm,yaml}
└── real/                         ← physical-robot runs
    ├── lidar/
    └── camera/
```

File names are `<start timestamp>_<algorithm>`; one pair per run,
overwritten in place as the run progresses.

`run.sh nav2` does not save here — Nav2 consumes a map rather than
generating one. To navigate on a map from this folder:

```bash
~/pathfinding_algorithms/run.sh nav2 --map ~/pathfinding_algorithms/maps/real/lidar/<stamp>_lidar.yaml
```

Map files are gitignored (session artifacts); this README is tracked.
