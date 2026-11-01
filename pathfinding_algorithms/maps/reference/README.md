# Reference map

A LiDAR map of the original test room, recorded with Cartographer and
`map_saver_cli`. It is measured data, so it cannot be regenerated. Replace it
when working in a different space (see `TODO.md`).

| File | Contents |
|---|---|
| `my_map.pgm` | Occupancy image, 5 cm per pixel |
| `my_map.yaml` | Resolution, origin and thresholds; this is the file to load |

`run.sh nav2` uses this map by default on the real robot, and
`run.sh policy --goto X Y --map maps/reference/my_map.yaml` uses it as the
obstacle grid for the subgoal planner.

To replace it, copy a saved map from `maps/real/<mode>/` over these files and
set the yaml's `image:` line back to `my_map.pgm`. A new map has a new origin,
so any coordinates recorded against the old one stop being valid.
