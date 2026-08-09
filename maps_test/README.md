# maps_test — saved real-robot LiDAR map ("upstairs")

The robot's persistent map of the upstairs room, captured with Cartographer +
`map_saver_cli` in April 2026. This is a **measured** map (real sensor data,
not derivable from anything) — the `.pgm` here is its only copy, which is why
this folder is tracked in git.

| File | What it is |
|---|---|
| `my_map.pgm` | 121×107 grayscale image, 1 px = 5×5 cm. White = free, black = wall/obstacle, gray = unobserved. |
| `my_map.yaml` | Metadata that anchors the image to the map frame: resolution 0.05 m/px, origin (−2.47, −2.44), occupancy thresholds. This is the file consumers load; it references the `.pgm` beside it. |
| `upstairs_lidar.rviz` | Saved RViz layout from the mapping sessions. Not referenced by anything — load manually with `rviz2 -d` if wanted. |

## Who uses it

- `run.sh nav2` (real robot) — default map when `--map` is not given
  (nav2 `map_server` → AMCL localization + global planning).
- `run.sh policy --goto X Y --map ~/maps_test/my_map.yaml` — obstacle grid
  for the two-stage subgoal A* planner.
- Documented in `~/CLAUDE.md` ("Saved LiDAR map").

## Regenerating

Nothing regenerates this automatically. Mapping sessions (`run.sh
lidar|camera|policy`) save timestamped snapshots to
`~/pathfinding_algorithms/maps/<real|sim>/<mode>/` every 30 s; to promote one:

```bash
cp ~/pathfinding_algorithms/maps/real/<mode>/<stamp>_<mode>.pgm  ~/maps_test/my_map.pgm
cp ~/pathfinding_algorithms/maps/real/<mode>/<stamp>_<mode>.yaml ~/maps_test/my_map.yaml
# edit the yaml's `image:` line back to plain `my_map.pgm`
```

**Caution:** a new map has a new origin — every coordinate expressed in the
old map frame (memorized nav2 goals, `--goto` examples) silently changes
meaning. Only replace the map when the room layout has actually changed, and
prefer a full-coverage teleop+Cartographer session over a short wander map.
