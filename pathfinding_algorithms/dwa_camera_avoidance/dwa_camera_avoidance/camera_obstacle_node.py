#!/usr/bin/env python3
"""
camera_obstacle_node — monocular camera obstacle detector for TurtleBot3.

Converts a standard USB camera image (Logitech BRIO) into a virtual LaserScan
that represents obstacles the LiDAR scan plane cannot see:

  Floor obstacles (cables, debris):
    Below the LiDAR scan plane (~0.17 m).  Detected in the lower image zone
    via Canny edges; each edge pixel is back-projected to the floor plane using
    the camera's known height and tilt via Inverse Perspective Mapping (IPM).

  Overhead obstacles (table tops, shelves):
    Above the LiDAR scan plane.  Detected in the upper-mid image zone via
    horizontal Canny edges; depth is estimated by intersecting the pixel ray
    with a configurable obstacle height plane.

The published /scan_camera LaserScan has frame_id = base_scan so Cartographer
can consume it as a second rangefinder alongside /scan_relayed without needing
an extra TF frame.

Coordinate maths
----------------
Camera optical frame (OpenCV convention):
  +z  forward into scene
  +x  right
  +y  down

Robot base frame (ROS REP 103):
  +x  forward
  +y  left
  +z  up

R_opt2robot:               maps a vector from optical frame → robot frame
  optical +z → robot +x   (forward)
  optical +x → robot −y   (right → −left)
  optical +y → robot −z   (down  → −up)

Tilt rotation (nose-down by tilt_rad around optical x-axis):
  Rx(−tilt) = [[1,  0,         0        ],
               [0,  cos(tilt), sin(tilt)],
               [0, −sin(tilt), cos(tilt)]]

Full transform: d_robot = R_opt2robot @ Rx(−tilt) @ d_optical
"""

import math
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image, CompressedImage, CameraInfo, LaserScan

try:
    import cv2
    from cv_bridge import CvBridge
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False


# Rotation: camera optical frame → robot base frame (no tilt)
_R_OPT2ROBOT = np.array([
    [0.0,  0.0,  1.0],   # robot x = optical z
    [-1.0, 0.0,  0.0],   # robot y = −optical x
    [0.0, -1.0,  0.0],   # robot z = −optical y
], dtype=float)


def _tilt_matrix(tilt_rad: float) -> np.ndarray:
    """Rx(−tilt): nose-down rotation around optical x-axis."""
    c, s = math.cos(tilt_rad), math.sin(tilt_rad)
    return np.array([[1.0, 0.0,  0.0],
                     [0.0, c,    s  ],
                     [0.0, -s,   c  ]], dtype=float)


def _pixel_to_robot_dir(u, v, fx, fy, cx, cy, R_cam2robot: np.ndarray) -> np.ndarray:
    """Unit direction vector in robot frame for pixel (u, v)."""
    d_opt = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    d_opt /= np.linalg.norm(d_opt)
    d_rob = R_cam2robot @ d_opt
    return d_rob / np.linalg.norm(d_rob)


def _ray_z_intersect(d_robot: np.ndarray, cam_origin: np.ndarray,
                     z_target: float) -> float | None:
    """
    Distance t along d_robot from cam_origin to the plane z = z_target
    in robot frame.  Returns None if the ray is nearly horizontal (< 1 deg
    from horizontal) or the intersection is behind the camera.
    """
    dz = d_robot[2]
    if abs(dz) < 0.017:   # sin(1°) ≈ 0.017 — ray nearly horizontal
        return None
    t = (z_target - cam_origin[2]) / dz
    if t <= 0.0:
        return None
    return t


class CameraObstacleNode(Node):
    def __init__(self, **node_kwargs):
        super().__init__('camera_obstacle_node', **node_kwargs)

        if not CV_AVAILABLE:
            self.get_logger().error('OpenCV / cv_bridge not found — node disabled.')
            return

        # ── Camera geometry ───────────────────────────────────────────────────
        self.declare_parameter('camera_height',    0.28)   # m above floor
        self.declare_parameter('camera_tilt_deg',  5.0)    # positive = nose down
        self.declare_parameter('camera_x_offset',  0.08)   # m forward of robot centre
        self.declare_parameter('camera_y_offset',  0.0)    # m lateral offset

        # ── Processing resolution (resize all images to this before processing) ─
        self.declare_parameter('process_width',   640)
        self.declare_parameter('process_height',  480)

        # ── Virtual scan geometry ─────────────────────────────────────────────
        self.declare_parameter('virtual_scan_fov_deg',               92.0)
        self.declare_parameter('virtual_scan_angle_increment_deg',    1.0)
        self.declare_parameter('virtual_scan_range_min',  0.05)
        self.declare_parameter('virtual_scan_range_max',  3.0)

        # ── Floor obstacle detection ──────────────────────────────────────────
        self.declare_parameter('floor_zone_fraction',    0.35)   # bottom N% of image
        self.declare_parameter('floor_canny_low',        40)
        self.declare_parameter('floor_canny_high',       120)
        self.declare_parameter('floor_min_height',       0.01)   # m above floor
        self.declare_parameter('floor_max_height',       0.14)   # m — below LiDAR ~0.17m

        # ── Overhead obstacle detection ───────────────────────────────────────
        self.declare_parameter('overhead_zone_top_fraction',    0.10)  # from image top
        self.declare_parameter('overhead_zone_bottom_fraction', 0.55)  # from image top
        self.declare_parameter('overhead_canny_low',   25)
        self.declare_parameter('overhead_canny_high',  80)
        self.declare_parameter('overhead_assume_height', 0.55)  # m — assumed obstacle height
        self.declare_parameter('overhead_min_height',    0.20)  # m above floor
        self.declare_parameter('overhead_max_height',    1.30)  # m above floor

        # ── Misc ──────────────────────────────────────────────────────────────
        self.declare_parameter('min_edge_density',      8)    # min edge pixels per column (higher = fewer false positives)
        self.declare_parameter('min_obstacle_columns',  4)    # obstacle must span this many adjacent columns
        self.declare_parameter('floor_min_range',       0.30) # m — ignore floor detections closer than this (floor texture FP)
        self.declare_parameter('publish_debug',         False)
        self.declare_parameter('enable_overhead_detection', True)

        # ── Floor vertical-continuity filter (bug #017 follow-up) ────────────
        # A real obstacle's base edge has the object's body continuing above
        # it in the image; floor texture (carpet, wood grain, reflections)
        # is an isolated streak. A candidate must have edge pixels spanning
        # at least floor_support_min_height meters above its base — the
        # pixel window is derived per detection from fy and the candidate's
        # distance, so the filter is camera/resolution/range independent.
        # Set floor_support_min_edges: 0 to disable.
        self.declare_parameter('floor_support_min_height', 0.03)  # m
        self.declare_parameter('floor_support_min_edges',  2)     # edge px

        # ── Input topics ──────────────────────────────────────────────────────
        # Defaults match the real robot's v4l2_camera. In Gazebo the camera
        # plugin publishes under /camera/... — override via launch args.
        self.declare_parameter('image_topic',       '/image_raw/compressed')
        self.declare_parameter('camera_info_topic', '/camera_info')

        # ── SLAM-fusion timing (2026-07-08) ───────────────────────────────────
        # scan_min_period: throttle detection/publish so the sparse camera
        # scan stays SLOWER than the LiDAR in Cartographer's matcher.
        # stamp_lag: stamps are minted after image transport + detection —
        # backdate them into the available odom-TF window or Cartographer
        # matches camera scans against extrapolated (wrong) poses.
        self.declare_parameter('scan_min_period', 0.20)   # s (~5 Hz)
        self.declare_parameter('stamp_lag',       0.45)   # s (wall-clock mode)

        # ── State ─────────────────────────────────────────────────────────────
        self._camera_info: CameraInfo | None = None
        self._bridge = CvBridge()
        self._R_cam2robot: np.ndarray | None = None   # cached after first camera_info
        self._last_process_mono = 0.0
        self._stamp_lag_ns = int(
            self.get_parameter('stamp_lag').value * 1e9)

        # ── ROS I/O ───────────────────────────────────────────────────────────
        img_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2,
        )
        # Subscribe to the JPEG-compressed image (~20 KB/frame) instead of raw
        # (~230-920 KB).  Best-effort delivery of large raw frames does not
        # survive the robot's WiFi link; compressed frames fit in few fragments.
        self.create_subscription(CompressedImage,
                                 self.get_parameter('image_topic').value,
                                 self._image_cb, img_qos)
        self.create_subscription(CameraInfo,
                                 self.get_parameter('camera_info_topic').value,
                                 self._info_cb, 10)

        self._scan_pub  = self.create_publisher(LaserScan, '/scan_camera', 10)
        self._last_stamp_ns = 0
        self._last_mono_ns  = 0
        self._use_sim_time = bool(self.get_parameter('use_sim_time').value)
        if self.get_parameter('publish_debug').value:
            self._debug_pub = self.create_publisher(Image, '/camera_obstacles_debug', 2)
        else:
            self._debug_pub = None

        # Publish empty scans at 1 Hz until camera_info arrives so Cartographer's
        # ordered_multi_queue doesn't stall waiting for the second sensor.
        self._warmup_timer = self.create_timer(1.0, self._warmup_scan_cb)

        self.get_logger().info('Camera obstacle node started.')

    def _next_stamp(self):
        """Step-immune wall-clock stamp.

        The WSL2 wall clock is yanked backwards ~1 s every ~30 s by time sync.
        Follow the wall clock while it is ahead, but never let the stamp line
        move backwards: during a backward step, keep advancing at the true rate
        measured by the monotonic clock until the wall clock catches back up.
        Cartographer needs strictly increasing stamps (it drops scans whose
        subdivision times are not increasing, at 100 ns resolution).

        Under sim time the monotonic coast is wrong (wall-monotonic elapsed
        diverges from sim time whenever the real-time factor is not 1), and
        the sim clock never steps backwards — so just follow the node clock
        with a strictly-monotonic guard.
        """
        # Backdate by stamp_lag: this stamp is minted AFTER Pi->PC image
        # transport + decode + detection (~0.3-0.5 s), so an un-lagged stamp
        # sits AHEAD of the odom TF stream and Cartographer must extrapolate
        # ("Lookup would require extrapolation into the future"), matching
        # camera scans against wrong poses — observed to diverge fused SLAM
        # into disconnected map islands. The lag places the stamp inside the
        # available TF window; interpolating backwards is exact.
        wall_ns = self.get_clock().now().nanoseconds
        if self._use_sim_time:
            stamp_ns = max(wall_ns, self._last_stamp_ns + 1)
            self._last_stamp_ns = stamp_ns
            return TimeMsg(sec=stamp_ns // 1_000_000_000,
                           nanosec=stamp_ns % 1_000_000_000)
        wall_ns -= self._stamp_lag_ns
        mono_ns = time.monotonic_ns()
        if self._last_mono_ns == 0:
            stamp_ns = wall_ns
        else:
            elapsed = max(mono_ns - self._last_mono_ns, 1)
            stamp_ns = max(wall_ns, self._last_stamp_ns + elapsed)
        self._last_stamp_ns = stamp_ns
        self._last_mono_ns = mono_ns
        return TimeMsg(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)

    def _warmup_scan_cb(self):
        """Publish an empty scan until camera_info arrives, then stop."""
        if self._camera_info is not None:
            self._warmup_timer.cancel()
            return
        self._scan_pub.publish(self._empty_scan())

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _info_cb(self, msg: CameraInfo):
        if self._camera_info is not None:
            return   # received once is enough
        self._camera_info = msg
        tilt_rad = math.radians(self.get_parameter('camera_tilt_deg').value)
        self._R_cam2robot = _R_OPT2ROBOT @ _tilt_matrix(tilt_rad)
        self.get_logger().info(
            f'Camera info received: {msg.width}×{msg.height}  '
            f'fx={msg.k[0]:.1f} fy={msg.k[4]:.1f}'
        )

    def _image_cb(self, msg: CompressedImage):
        if self._camera_info is None:
            return

        p = self.get_parameter
        # Throttle: SLAM fusion needs the camera scan slower than the LiDAR
        # (~9 Hz) so sparse camera scans can't dominate scan matching, and
        # detection at image rate is wasted CPU for avoidance anyway.
        min_period = p('scan_min_period').value
        now_mono = time.monotonic()
        if now_mono - self._last_process_mono < min_period:
            return
        self._last_process_mono = now_mono

        proc_w = p('process_width').value
        proc_h = p('process_height').value

        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError('cv2.imdecode returned None')
        except Exception as e:
            self.get_logger().warn(f'image decode error: {e}', throttle_duration_sec=5.0)
            return

        orig_h, orig_w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (proc_w, proc_h))

        # Scale intrinsics to processing resolution
        scale_x = proc_w / self._camera_info.width
        scale_y = proc_h / self._camera_info.height
        fx = self._camera_info.k[0] * scale_x
        fy = self._camera_info.k[4] * scale_y
        cx = self._camera_info.k[2] * scale_x
        cy = self._camera_info.k[5] * scale_y

        cam_origin = np.array([
            p('camera_x_offset').value,
            p('camera_y_offset').value,
            p('camera_height').value,
        ])

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        obstacle_points: list[tuple[float, float]] = []  # (robot_x, robot_y)

        # ── Floor obstacle detection ──────────────────────────────────────────
        floor_pts = self._detect_floor_obstacles(
            gray, fx, fy, cx, cy, cam_origin, proc_h)
        obstacle_points.extend(floor_pts)

        # ── Overhead obstacle detection ───────────────────────────────────────
        # The fixed-height back-projection is geometric guesswork; in
        # environments with no true overhead obstacles (e.g. Gazebo worlds)
        # it produces mostly phantoms — disable via this parameter there.
        overhead_pts: list[tuple[float, float]] = []
        if self.get_parameter('enable_overhead_detection').value:
            overhead_pts = self._detect_overhead_obstacles(
                gray, fx, fy, cx, cy, cam_origin, proc_h)
            obstacle_points.extend(overhead_pts)

        # Use PC wall clock for scan stamp so Cartographer's ordered_multi_queue
        # sees consistent timestamps across LiDAR (relayed by scan_relay_node,
        # also wall-clock) and camera scans.  The Pi image timestamp is on a
        # different clock and causes queue ordering failures.
        wall_stamp = self._next_stamp()
        scan = self._build_virtual_scan(obstacle_points, wall_stamp)
        self._scan_pub.publish(scan)

        if self._debug_pub is not None:
            self._publish_debug(bgr, floor_pts, overhead_pts, wall_stamp,
                                proc_h, p('floor_zone_fraction').value,
                                p('overhead_zone_top_fraction').value,
                                p('overhead_zone_bottom_fraction').value)

    # ── Floor obstacle detection ───────────────────────────────────────────────

    def _detect_floor_obstacles(self, gray, fx, fy, cx, cy,
                                cam_origin, proc_h) -> list[tuple[float, float]]:
        p = self.get_parameter
        zone_frac    = p('floor_zone_fraction').value
        canny_low    = p('floor_canny_low').value
        canny_high   = p('floor_canny_high').value
        min_dens     = p('min_edge_density').value
        min_cols     = p('min_obstacle_columns').value
        floor_min_r  = p('floor_min_range').value
        r_max        = p('virtual_scan_range_max').value
        sup_height   = p('floor_support_min_height').value
        sup_min      = p('floor_support_min_edges').value

        zone_start = int(proc_h * (1.0 - zone_frac))
        zone_img   = gray[zone_start:, :]
        if zone_img.shape[0] < 3:
            return []   # zone empty/degenerate (params are dynamic)

        # Blur reduces floor texture / carpet pattern noise before edge detection
        blurred = cv2.GaussianBlur(zone_img, (7, 7), 0)
        edges   = cv2.Canny(blurred, canny_low, canny_high)

        # Collect one candidate point per column
        candidates: list[tuple[int, float, float]] = []  # (col, robot_x, robot_y)
        w = edges.shape[1]

        for col in range(w):
            col_edges = np.nonzero(edges[:, col])[0]
            if len(col_edges) < min_dens:
                continue
            u = col

            # Nearest VALID floor point per column: walk edges bottom-up.
            # - Bottom rows are often the robot's own chassis or floor
            #   texture (dist < floor_min_r) — skip upward past them.
            # - Rows above a valid hit only project farther; once beyond
            #   r_max, stop.
            # Taking the topmost edge instead (old behavior) back-projected
            # tall objects' silhouettes to phantom mid-range obstacles that
            # blocked free space (bug #017).
            col_edge_set = set(int(r) for r in col_edges)
            best = None
            for row_in_zone in reversed(col_edges):
                row_in_zone = int(row_in_zone)
                v = zone_start + row_in_zone
                d_robot = _pixel_to_robot_dir(u, v, fx, fy, cx, cy,
                                              self._R_cam2robot)
                t = _ray_z_intersect(d_robot, cam_origin, z_target=0.0)
                if t is None:
                    continue
                point = cam_origin + t * d_robot
                if point[0] < 0.01:
                    continue
                dist = math.hypot(point[0], point[1])
                if dist < floor_min_r:
                    continue          # own body / texture — try next edge up
                if dist >= r_max:
                    break             # higher rows only project farther
                if sup_min > 0:
                    # Vertical continuity: object body must reach at least
                    # sup_height meters above the base edge; floor texture
                    # doesn't. Window derived from intrinsics + distance →
                    # camera/resolution/range independent. Only the UPPER
                    # half of the window counts, so a thick texture streak
                    # can't satisfy the check with its own pixels.
                    sup_window = max(4, min(60, int(fy * sup_height / dist)))
                    support = sum(1 for dr in range(sup_window // 2,
                                                    sup_window + 1)
                                  if (row_in_zone - dr) in col_edge_set)
                    if support < sup_min:
                        continue      # lone streak — texture, try next edge up
                best = (col, float(point[0]), float(point[1]))
                break

            if best is not None:
                candidates.append(best)

        # Column-span filter: only keep candidates that have at least
        # min_obstacle_columns consecutive active columns around them.
        # Isolated column hits are almost always texture / noise.
        if not candidates or min_cols <= 1:
            return [(x, y) for (_, x, y) in candidates]

        col_set = {c for (c, _, _) in candidates}
        points: list[tuple[float, float]] = []
        for (col, x, y) in candidates:
            span = sum(1 for dc in range(min_cols) if (col + dc) in col_set)
            if span >= min_cols:
                points.append((x, y))

        return points

    # ── Overhead obstacle detection ────────────────────────────────────────────

    def _detect_overhead_obstacles(self, gray, fx, fy, cx, cy,
                                   cam_origin, proc_h) -> list[tuple[float, float]]:
        p = self.get_parameter
        zone_top     = p('overhead_zone_top_fraction').value
        zone_bot     = p('overhead_zone_bottom_fraction').value
        canny_low    = p('overhead_canny_low').value
        canny_high   = p('overhead_canny_high').value
        assume_h     = p('overhead_assume_height').value
        min_height   = p('overhead_min_height').value
        max_height   = p('overhead_max_height').value
        min_dens     = p('min_edge_density').value
        r_min        = p('virtual_scan_range_min').value
        r_max        = p('virtual_scan_range_max').value

        row_start = int(proc_h * zone_top)
        row_end   = int(proc_h * zone_bot)
        if row_end - row_start < 3:
            return []   # zone empty/degenerate (params are dynamic)

        zone_img = gray[row_start:row_end, :]

        # Blur first, then apply horizontal-emphasis kernel for table-bottom edges
        blurred = cv2.GaussianBlur(zone_img, (5, 5), 0)
        kernel = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float32)
        filtered = cv2.filter2D(blurred, -1, kernel)
        filtered = np.clip(np.abs(filtered), 0, 255).astype(np.uint8)
        edges = cv2.Canny(filtered, canny_low, canny_high)

        points: list[tuple[float, float]] = []
        w = edges.shape[1]

        for col in range(w):
            col_edges = np.nonzero(edges[:, col])[0]
            if len(col_edges) < min_dens:
                continue
            # Use bottommost edge in column (lowest edge of overhang = nearest)
            row_in_zone = int(col_edges[-1])
            v = row_start + row_in_zone
            u = col

            d_robot = _pixel_to_robot_dir(u, v, fx, fy, cx, cy, self._R_cam2robot)

            # Intersect with plane at assumed obstacle height
            t = _ray_z_intersect(d_robot, cam_origin, z_target=assume_h)
            if t is None:
                continue

            point = cam_origin + t * d_robot
            if point[0] < 0.01:
                continue

            # Verify the resulting point height is plausible for an overhead obstacle
            # (assume_h IS the height, so just range-check)
            if not (min_height <= assume_h <= max_height):
                continue

            dist = math.hypot(point[0], point[1])
            if r_min < dist < r_max:
                points.append((float(point[0]), float(point[1])))

        return points

    # ── Virtual LaserScan builder ──────────────────────────────────────────────

    def _empty_scan(self) -> LaserScan:
        """Empty scan with no obstacle ranges — used for Cartographer warmup."""
        p = self.get_parameter
        fov_rad = math.radians(p('virtual_scan_fov_deg').value)
        inc_rad = math.radians(p('virtual_scan_angle_increment_deg').value)
        r_min   = p('virtual_scan_range_min').value
        r_max   = p('virtual_scan_range_max').value
        n_rays  = max(1, int(fov_rad / inc_rad) + 1)
        scan = LaserScan()
        scan.header.stamp    = self._next_stamp()
        scan.header.frame_id = 'base_scan'
        scan.angle_min       = -fov_rad / 2.0
        scan.angle_max       =  fov_rad / 2.0
        scan.angle_increment = inc_rad
        scan.range_min       = r_min
        scan.range_max       = r_max
        # inf = "no return" (LaserScan convention); r_max would read as a
        # solid arc of obstacles at max range in viewers/consumers.
        scan.ranges          = [float('inf')] * n_rays
        return scan

    def _build_virtual_scan(self, points: list[tuple[float, float]],
                            stamp) -> LaserScan:
        p = self.get_parameter
        fov_rad   = math.radians(p('virtual_scan_fov_deg').value)
        inc_rad   = math.radians(p('virtual_scan_angle_increment_deg').value)
        r_min     = p('virtual_scan_range_min').value
        r_max     = p('virtual_scan_range_max').value

        angle_min = -fov_rad / 2.0
        angle_max =  fov_rad / 2.0
        n_rays    = max(1, int((angle_max - angle_min) / inc_rad) + 1)

        # no-detection columns are "no return" (inf), not obstacles at r_max
        ranges = np.full(n_rays, np.inf, dtype=float)

        for (ox, oy) in points:
            bearing = math.atan2(oy, ox)   # robot frame: x=fwd, y=left
            if not (angle_min <= bearing <= angle_max):
                continue
            idx = int((bearing - angle_min) / inc_rad)
            if 0 <= idx < n_rays:
                dist = math.hypot(ox, oy)
                if dist < ranges[idx]:
                    ranges[idx] = dist

        scan = LaserScan()
        scan.header.stamp    = stamp
        scan.header.frame_id = 'base_scan'   # same TF frame as LiDAR
        scan.angle_min       = angle_min
        scan.angle_max       = angle_max
        scan.angle_increment = inc_rad
        scan.time_increment  = 0.0
        scan.scan_time       = 0.1
        scan.range_min       = r_min
        scan.range_max       = r_max
        scan.ranges          = ranges.tolist()
        scan.intensities     = []
        return scan

    # ── Debug image ───────────────────────────────────────────────────────────

    def _publish_debug(self, bgr, floor_pts, overhead_pts, stamp,
                       proc_h, floor_frac, ovhd_top, ovhd_bot):
        vis = bgr.copy()
        h, w = vis.shape[:2]

        # Zone boundary lines
        floor_y    = int(h * (1.0 - floor_frac))
        ovhd_y_top = int(h * ovhd_top)
        ovhd_y_bot = int(h * ovhd_bot)
        cv2.line(vis, (0, floor_y),    (w, floor_y),    (0, 255, 0),  1)
        cv2.line(vis, (0, ovhd_y_top), (w, ovhd_y_top), (255, 128, 0), 1)
        cv2.line(vis, (0, ovhd_y_bot), (w, ovhd_y_bot), (255, 128, 0), 1)

        msg = self._bridge.cv2_to_imgmsg(vis, encoding='bgr8')
        msg.header.stamp = stamp
        self._debug_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
