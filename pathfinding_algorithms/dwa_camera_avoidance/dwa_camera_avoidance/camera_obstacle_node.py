#!/usr/bin/env python3
"""Monocular obstacle detector: BRIO image -> virtual LaserScan on /scan_camera.

Finds what the LiDAR plane (~0.17 m) misses: floor obstacles by back-projecting Canny
edges to the floor, overhead ones by intersecting pixel rays with an assumed height.
The scan uses frame base_scan, so Cartographer can read it as a second rangefinder.
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


# optical frame (z forward, x right, y down) -> robot base frame (REP 103), no tilt
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
    """Distance along d_robot to the plane z = z_target, or None if near-horizontal or behind."""
    dz = d_robot[2]
    if abs(dz) < 0.017:   # sin(1 deg)
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

        self.declare_parameter('camera_height',    0.28)   # m above floor
        self.declare_parameter('camera_tilt_deg',  5.0)    # positive = nose down
        self.declare_parameter('camera_x_offset',  0.08)   # m forward of robot centre
        self.declare_parameter('camera_y_offset',  0.0)    # m lateral offset

        self.declare_parameter('process_width',   640)
        self.declare_parameter('process_height',  480)

        self.declare_parameter('virtual_scan_fov_deg',               92.0)
        self.declare_parameter('virtual_scan_angle_increment_deg',    1.0)
        self.declare_parameter('virtual_scan_range_min',  0.05)
        self.declare_parameter('virtual_scan_range_max',  3.0)

        self.declare_parameter('floor_zone_fraction',    0.35)   # bottom N% of image
        self.declare_parameter('floor_canny_low',        40)
        self.declare_parameter('floor_canny_high',       120)
        self.declare_parameter('floor_min_height',       0.01)   # m above floor
        self.declare_parameter('floor_max_height',       0.14)   # m — below LiDAR ~0.17m

        self.declare_parameter('overhead_zone_top_fraction',    0.10)  # from image top
        self.declare_parameter('overhead_zone_bottom_fraction', 0.55)  # from image top
        self.declare_parameter('overhead_canny_low',   25)
        self.declare_parameter('overhead_canny_high',  80)
        self.declare_parameter('overhead_assume_height', 0.55)  # m — assumed obstacle height
        self.declare_parameter('overhead_min_height',    0.20)  # m above floor
        self.declare_parameter('overhead_max_height',    1.30)  # m above floor

        self.declare_parameter('min_edge_density',      8)    # min edge pixels per column
        self.declare_parameter('min_obstacle_columns',  4)    # adjacent columns an obstacle must span
        self.declare_parameter('floor_min_range',       0.30) # m, closer floor hits are texture
        self.declare_parameter('publish_debug',         False)
        self.declare_parameter('enable_overhead_detection', True)

        # A real obstacle's body continues above its base edge in the image while
        # floor texture is an isolated streak, so a candidate needs edge pixels
        # this high above its base. floor_support_min_edges: 0 disables the filter.
        self.declare_parameter('floor_support_min_height', 0.03)  # m
        self.declare_parameter('floor_support_min_edges',  2)     # edge px

        # Defaults match the real v4l2_camera; Gazebo publishes under /camera/...
        self.declare_parameter('image_topic',       '/image_raw/compressed')
        self.declare_parameter('camera_info_topic', '/camera_info')

        # scan_min_period keeps the sparse camera scan slower than the LiDAR in
        # Cartographer's matcher; stamp_lag: see _next_stamp.
        self.declare_parameter('scan_min_period', 0.20)   # s (~5 Hz)
        self.declare_parameter('stamp_lag',       0.45)   # s (wall-clock mode)

        self._camera_info: CameraInfo | None = None
        self._bridge = CvBridge()
        self._R_cam2robot: np.ndarray | None = None
        self._last_process_mono = 0.0
        self._stamp_lag_ns = int(
            self.get_parameter('stamp_lag').value * 1e9)

        img_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2,
        )
        # Compressed, as best-effort raw frames do not survive the robot's WiFi link
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

        # Empty scans until camera_info arrives, so Cartographer's queue doesn't
        # stall waiting for the second sensor
        self._warmup_timer = self.create_timer(1.0, self._warmup_scan_cb)

        self.get_logger().info('Camera obstacle node started.')

    def _next_stamp(self):
        """Wall-clock stamp immune to WSL2's backward steps (~1 s every ~30 s).

        Follows the wall clock while ahead, else advances at the monotonic rate,
        since Cartographer needs strictly increasing stamps. Sim time never steps
        back and diverges from monotonic time, so it only gets an increasing guard.
        """
        # Backdate by stamp_lag: image transport and detection take ~0.3-0.5 s, so
        # an un-lagged stamp sits ahead of the odom TF stream and Cartographer
        # matches the scan against extrapolated, wrong poses.
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

    def _info_cb(self, msg: CameraInfo):
        if self._camera_info is not None:
            return
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

        obstacle_points: list[tuple[float, float]] = []

        floor_pts = self._detect_floor_obstacles(
            gray, fx, fy, cx, cy, cam_origin, proc_h)
        obstacle_points.extend(floor_pts)

        # The fixed-height back-projection yields mostly phantoms where nothing
        # is truly overhead (e.g. Gazebo worlds), so it can be disabled.
        overhead_pts: list[tuple[float, float]] = []
        if self.get_parameter('enable_overhead_detection').value:
            overhead_pts = self._detect_overhead_obstacles(
                gray, fx, fy, cx, cy, cam_origin, proc_h)
            obstacle_points.extend(overhead_pts)

        # PC clock, matching scan_relay_node: the Pi's image stamps are on a
        # different clock and break Cartographer's queue ordering.
        wall_stamp = self._next_stamp()
        scan = self._build_virtual_scan(obstacle_points, wall_stamp)
        self._scan_pub.publish(scan)

        if self._debug_pub is not None:
            self._publish_debug(bgr, floor_pts, overhead_pts, wall_stamp,
                                proc_h, p('floor_zone_fraction').value,
                                p('overhead_zone_top_fraction').value,
                                p('overhead_zone_bottom_fraction').value)

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
            return []   # degenerate zone (params are dynamic)

        # blur suppresses floor texture
        blurred = cv2.GaussianBlur(zone_img, (7, 7), 0)
        edges   = cv2.Canny(blurred, canny_low, canny_high)

        candidates: list[tuple[int, float, float]] = []  # (col, robot_x, robot_y)
        w = edges.shape[1]

        for col in range(w):
            col_edges = np.nonzero(edges[:, col])[0]
            if len(col_edges) < min_dens:
                continue
            u = col

            # Nearest valid floor point: walk edges bottom-up, past the chassis and
            # floor texture. The topmost edge would back-project tall silhouettes
            # to phantom mid-range obstacles.
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
                    continue
                if dist >= r_max:
                    break             # higher rows only project farther
                if sup_min > 0:
                    # Only the upper half of the window counts, so a thick
                    # texture streak can't pass on its own pixels.
                    sup_window = max(4, min(60, int(fy * sup_height / dist)))
                    support = sum(1 for dr in range(sup_window // 2,
                                                    sup_window + 1)
                                  if (row_in_zone - dr) in col_edge_set)
                    if support < sup_min:
                        continue
                best = (col, float(point[0]), float(point[1]))
                break

            if best is not None:
                candidates.append(best)

        # isolated column hits are almost always texture
        if not candidates or min_cols <= 1:
            return [(x, y) for (_, x, y) in candidates]

        col_set = {c for (c, _, _) in candidates}
        points: list[tuple[float, float]] = []
        for (col, x, y) in candidates:
            span = sum(1 for dc in range(min_cols) if (col + dc) in col_set)
            if span >= min_cols:
                points.append((x, y))

        return points

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
            return []   # degenerate zone (params are dynamic)

        zone_img = gray[row_start:row_end, :]

        # horizontal-emphasis kernel for table-bottom edges
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
            # the lowest edge of an overhang is the nearest
            row_in_zone = int(col_edges[-1])
            v = row_start + row_in_zone
            u = col

            d_robot = _pixel_to_robot_dir(u, v, fx, fy, cx, cy, self._R_cam2robot)

            t = _ray_z_intersect(d_robot, cam_origin, z_target=assume_h)
            if t is None:
                continue

            point = cam_origin + t * d_robot
            if point[0] < 0.01:
                continue

            if not (min_height <= assume_h <= max_height):
                continue

            dist = math.hypot(point[0], point[1])
            if r_min < dist < r_max:
                points.append((float(point[0]), float(point[1])))

        return points

    def _empty_scan(self) -> LaserScan:
        """Scan with no returns, for Cartographer warmup."""
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
        # inf is "no return"; r_max would read as an arc of obstacles
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

        ranges = np.full(n_rays, np.inf, dtype=float)

        for (ox, oy) in points:
            bearing = math.atan2(oy, ox)
            if not (angle_min <= bearing <= angle_max):
                continue
            idx = int((bearing - angle_min) / inc_rad)
            if 0 <= idx < n_rays:
                dist = math.hypot(ox, oy)
                if dist < ranges[idx]:
                    ranges[idx] = dist

        scan = LaserScan()
        scan.header.stamp    = stamp
        scan.header.frame_id = 'base_scan'
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

    def _publish_debug(self, bgr, floor_pts, overhead_pts, stamp,
                       proc_h, floor_frac, ovhd_top, ovhd_bot):
        vis = bgr.copy()
        h, w = vis.shape[:2]

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
