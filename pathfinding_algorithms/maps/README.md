# maps/ — per-run occupancy maps

Every `run.sh lidar` or `run.sh camera` run saves the occupancy grid it
builds here, refreshed every 30 s while the run is live and once more at
shutdown — so each run leaves its final map behind even after a crash.

Folders match the `run.sh` algorithm names:

```
maps/
├── sim/                          ← Gazebo runs
│   ├── lidar/
│   │   └── <stamp>_lidar.{pgm,yaml}
│   └── camera/
│       └── <stamp>_camera.{pgm,yaml}
└── real/                         ← physical-robot runs
    ├── lidar/
    └── camera/
```

File names are `<start timestamp>_<algorithm>`; one pair per run,
overwritten in place as the run progresses.

`run.sh nav2` does not save here — Nav2 consumes a map rather than
generating one. To navigate on a map from this folder:

```bash
./run.sh nav2 --map maps/real/lidar/<stamp>_lidar.yaml
```

Per-run maps are gitignored. `reference/` holds the tracked map of the test room.
