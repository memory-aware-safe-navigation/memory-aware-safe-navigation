#!/usr/bin/env python3


import os
import time
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose
from gazebo_msgs.msg import ModelStates
from obstacle_detector.msg import Obstacles
from lidar_based_mcbf.basic_boundary_function.configs import CircularRobotSpecification, load_map_config

L_OCC      =  1.0
L_FREE     = -0.10
L_DYN_CLEAR= -0.5
L_MIN      = -3.0
L_MAX      =  9.0
OCC_THR    =  0.5

N_BEAMS   = 480
MAX_RANGE = 12.0
BIN_WIDTH = 2.0 * np.pi / N_BEAMS

ROS_TO_NP = {
    PointField.INT8:    np.int8,
    PointField.UINT8:   np.uint8,
    PointField.INT16:   np.int16,
    PointField.UINT16:  np.uint16,
    PointField.INT32:   np.int32,
    PointField.UINT32:  np.uint32,
    PointField.FLOAT32: np.float32,
    PointField.FLOAT64: np.float64,
}


def extract_xyz(msg: PointCloud2):
    field_map = {f.name: f for f in msg.fields}
    if not all(k in field_map for k in ('x', 'y', 'z')):
        return None
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)

    def read_field(name):
        f = field_map[name]
        dt = ROS_TO_NP[f.datatype]
        size = np.dtype(dt).itemsize
        return raw[:, f.offset:f.offset + size].copy().view(dt).reshape(-1).astype(np.float32)

    return read_field('x'), read_field('y'), read_field('z')


class OccupancyMapNode(Node):
    def __init__(self):
        super().__init__('occupancy_map_3d')

        pkg_share = get_package_share_directory('lidar_based_mcbf')
        econf_path = os.path.join(pkg_share, 'config', 'env_controller.yaml')
        with open(econf_path, 'r') as f:
            econf = yaml.safe_load(f)
        env_name = econf.get('env_name', 'large_hospital_1')
        map_cfg = load_map_config(env_name)
        x_min_def, x_max_def = map_cfg['x_min'], map_cfg['x_max']
        y_min_def, y_max_def = map_cfg['y_min'], map_cfg['y_max']

        self.declare_parameter('pointcloud_topic', '/cloud_all_fields_fullframe_global')
        self.declare_parameter('z_min',  0.0)
        self.declare_parameter('z_max',  1.2)
        self.declare_parameter('resolution', 0.1)
        self.declare_parameter('x_min', x_min_def)
        self.declare_parameter('x_max', x_max_def)
        self.declare_parameter('y_min', y_min_def)
        self.declare_parameter('y_max', y_max_def)
        self.declare_parameter('hit_thickness', 1)
        self.declare_parameter('visualize_rays', False)
        self.declare_parameter('dynamic_obs_buffer', 0.0)
        self.declare_parameter('obstacles_topic', '/obstacles')
        self.declare_parameter('dyn_clear_hz', 12.0)

        topic = self.get_parameter('pointcloud_topic').value
        self.rob = CircularRobotSpecification.from_yaml(
            os.path.join(pkg_share, 'config', 'spec_robot.yaml')).rob_name
        self.z_min = self.get_parameter('z_min').value
        self.z_max = self.get_parameter('z_max').value
        self.res   = self.get_parameter('resolution').value
        x_min = self.get_parameter('x_min').value
        x_max = self.get_parameter('x_max').value
        y_min = self.get_parameter('y_min').value
        y_max = self.get_parameter('y_max').value
        self.hit_thickness = max(1, int(self.get_parameter('hit_thickness').value))
        self.visualize_rays = bool(self.get_parameter('visualize_rays').value)
        self.dyn_buffer = float(self.get_parameter('dynamic_obs_buffer').value)
        self.human_radius = 0.5
        obstacles_topic = self.get_parameter('obstacles_topic').value
        dyn_clear_hz = float(self.get_parameter('dyn_clear_hz').value)
        self._obs_min_dt = 1.0 / dyn_clear_hz if dyn_clear_hz > 0.0 else 0.0
        self._last_obs_time = None

        self.origin = np.array([x_min, y_min], dtype=np.float32)
        self.nx = int(round((x_max - x_min) / self.res))
        self.ny = int(round((y_max - y_min) / self.res))
        self.log_odds = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.ever_updated = np.zeros((self.nx, self.ny), dtype=bool)
        self.prob_grid = np.full((self.ny, self.nx), -1, dtype=np.int8)
        self.iter_updated = np.zeros((self.nx, self.ny), dtype=bool)
        self._prev_dyn_mask = np.zeros((self.nx, self.ny), dtype=bool)

        beam_angles = -np.pi + (np.arange(N_BEAMS) + 0.5) * BIN_WIDTH
        self.beam_cos = np.cos(beam_angles).astype(np.float32)
        self.beam_sin = np.sin(beam_angles).astype(np.float32)

        self.robot_xy = None

        self.sub_cloud = self.create_subscription(PointCloud2, topic, self.cloud_callback, 1)
        self.sub_states = self.create_subscription(ModelStates, '/gazebo/model_states', self.states_callback, 1)
        self.sub_obstacles = self.create_subscription(Obstacles, obstacles_topic, self.obstacles_callback, 1)

        self.marker_pub = self.create_publisher(Marker, '/occu_marker', 1)
        self.occmap_pub = self.create_publisher(OccupancyGrid, '/occupancy_map', 1)
        self.ray_pub = self.create_publisher(Marker, '/ray_marker', 1)
        self.vacated_mask_pub = self.create_publisher(OccupancyGrid, '/dyn_vacated_mask', 1)

        self.get_logger().info(
            f"OccupancyMap3D: robot={self.rob}  z=[{self.z_min},{self.z_max}]  "
            f"grid={self.nx}x{self.ny}  res={self.res} m  beams={N_BEAMS}  "
            f"hit_thickness={self.hit_thickness}")

    def states_callback(self, msg: ModelStates):
        if self.rob not in msg.name:
            return
        idx = msg.name.index(self.rob)
        p = msg.pose[idx].position
        self.robot_xy = np.array([p.x, p.y], dtype=np.float32)

    def obstacles_callback(self, msg: Obstacles):
        now = time.monotonic()
        if self._last_obs_time is not None and now - self._last_obs_time < self._obs_min_dt:
            return
        self._last_obs_time = now

        r_cells = int(np.ceil((self.human_radius + self.dyn_buffer) / self.res))
        offsets = np.arange(-r_cells, r_cells + 1)
        ddx, ddy = np.meshgrid(offsets, offsets, indexing='ij')
        circle_shape = ddx ** 2 + ddy ** 2 <= r_cells ** 2

        curr_mask = np.zeros((self.nx, self.ny), dtype=bool)
        for circle in msg.circles:
            gx = int((circle.center.x - self.origin[0]) / self.res)
            gy = int((circle.center.y - self.origin[1]) / self.res)
            ix = np.clip(gx + ddx[circle_shape], 0, self.nx - 1)
            iy = np.clip(gy + ddy[circle_shape], 0, self.ny - 1)
            curr_mask[ix, iy] = True

        vacated = self._prev_dyn_mask & ~curr_mask
        if vacated.any():
            self.log_odds[vacated] = L_MIN
            self.ever_updated[vacated] = True
            self.iter_updated[vacated] = True

        grid_msg = OccupancyGrid()
        grid_msg.header.frame_id = 'mocap'
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.info.resolution = self.res
        grid_msg.info.width = self.nx
        grid_msg.info.height = self.ny
        grid_msg.info.origin = Pose()
        grid_msg.info.origin.position.x = float(self.origin[0])
        grid_msg.info.origin.position.y = float(self.origin[1])
        grid_msg.info.origin.orientation.w = 1.0
        grid_msg.data = np.where(vacated.T, np.int8(100), np.int8(0)).flatten().tolist()
        self.vacated_mask_pub.publish(grid_msg)

        self._prev_dyn_mask = curr_mask

    def _to_grid(self, xy: np.ndarray) -> np.ndarray:
        g = np.floor((xy - self.origin) / self.res).astype(np.int32)
        g[:, 0] = np.clip(g[:, 0], 0, self.nx - 1)
        g[:, 1] = np.clip(g[:, 1], 0, self.ny - 1)
        return g

    def _raytrace(self, gx0: int, gy0: int, gx1: int, gy1: int):
        cells = []
        dx =  abs(gx1 - gx0)
        dy = -abs(gy1 - gy0)
        sx = 1 if gx1 > gx0 else -1
        sy = 1 if gy1 > gy0 else -1
        err = dx + dy
        x, y = gx0, gy0
        while True:
            cells.append((x, y))
            if x == gx1 and y == gy1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x   += sx
            if e2 <= dx:
                err += dx
                y   += sy
        arr = np.array(cells, dtype=np.int32)
        k = min(self.hit_thickness, len(arr))
        return arr[:-k], arr[-k:]

    def _apply_free(self, cells: np.ndarray):
        if cells.shape[0] == 0:
            return
        cx = np.clip(cells[:, 0], 0, self.nx - 1)
        cy = np.clip(cells[:, 1], 0, self.ny - 1)
        self.log_odds[cx, cy] = np.clip(self.log_odds[cx, cy] + L_FREE, L_MIN, L_MAX)
        self.ever_updated[cx, cy] = True
        self.iter_updated[cx, cy] = True

    def _apply_hit(self, cells: np.ndarray):
        if cells.shape[0] == 0:
            return
        cx = np.clip(cells[:, 0], 0, self.nx - 1)
        cy = np.clip(cells[:, 1], 0, self.ny - 1)
        self.log_odds[cx, cy] = np.clip(self.log_odds[cx, cy] + L_OCC, L_MIN, L_MAX)
        self.ever_updated[cx, cy] = True
        self.iter_updated[cx, cy] = True

    def cloud_callback(self, msg: PointCloud2):
        if self.robot_xy is None:
            self.get_logger().warn(
                "Waiting for robot pose from /gazebo/model_states",
                throttle_duration_sec=2.0)
            return

        result = extract_xyz(msg)
        if result is None:
            return
        x, y, z = result

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        valid &= (z >= self.z_min) & (z <= self.z_max)
        x, y = x[valid], y[valid]

        rx, ry = float(self.robot_xy[0]), float(self.robot_xy[1])
        rob_g  = self._to_grid(self.robot_xy.reshape(1, 2))[0]
        gx0, gy0 = int(rob_g[0]), int(rob_g[1])

        hit_dx = np.full(N_BEAMS, np.nan, dtype=np.float32)
        hit_dy = np.full(N_BEAMS, np.nan, dtype=np.float32)

        if x.size > 0:
            dx = x - rx
            dy = y - ry
            dist = np.hypot(dx, dy)

            ok = dist > 0.05
            dx, dy, dist = dx[ok], dy[ok], dist[ok]

            if dist.size > 0:
                angles = np.arctan2(dy, dx)
                bins = np.floor((angles + np.pi) / BIN_WIDTH).astype(np.int32)
                bins = np.clip(bins, 0, N_BEAMS - 1)

                order = np.argsort(dist)
                bins_s = bins[order]
                dx_s = dx[order]
                dy_s = dy[order]
                _, first = np.unique(bins_s, return_index=True)
                hit_dx[bins_s[first]] = dx_s[first]
                hit_dy[bins_s[first]] = dy_s[first]

        hit_end_pts  = []
        miss_end_pts = []

        for b in range(N_BEAMS):
            if not np.isnan(hit_dx[b]):
                end_x = rx + float(hit_dx[b])
                end_y = ry + float(hit_dy[b])
                g_end = self._to_grid(np.array([[end_x, end_y]]))[0]
                free, hit = self._raytrace(gx0, gy0, int(g_end[0]), int(g_end[1]))
                self._apply_free(free)
                self._apply_hit(hit)
                if self.visualize_rays:
                    hit_end_pts.append((end_x, end_y))
            else:
                end_x = rx + MAX_RANGE * self.beam_cos[b]
                end_y = ry + MAX_RANGE * self.beam_sin[b]
                g_end = self._to_grid(np.array([[end_x, end_y]]))[0]
                free, _ = self._raytrace(gx0, gy0, int(g_end[0]), int(g_end[1]))
                self._apply_free(free)
                if self.visualize_rays:
                    miss_end_pts.append((end_x, end_y))

        if self.visualize_rays:
            stamp = self.get_clock().now().to_msg()

            def _ray_marker(mid, ends, r, g, b, a):
                m = Marker()
                m.header.frame_id = 'mocap'
                m.header.stamp = stamp
                m.ns = 'rays'
                m.id = mid
                m.type = Marker.LINE_LIST
                m.action = Marker.ADD
                m.pose.orientation.w = 1.0
                m.scale.x = 0.01
                m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a
                for ex, ey in ends:
                    p0 = Point(); p0.x = rx;  p0.y = ry;  p0.z = 0.1
                    p1 = Point(); p1.x = ex;  p1.y = ey;  p1.z = 0.1
                    m.points.append(p0)
                    m.points.append(p1)
                return m

            self.ray_pub.publish(_ray_marker(0, hit_end_pts,  0.0, 1.0, 0.0, 1.0))
            self.ray_pub.publish(_ray_marker(1, miss_end_pts, 0.5, 0.5, 0.5, 0.3))

        occ = np.argwhere(self.log_odds > OCC_THR)
        if occ.shape[0] > 0:
            marker = Marker()
            marker.header.frame_id = 'mocap'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'occupied_cells'
            marker.id = 0
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.res
            marker.scale.y = self.res
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            for ix, iy in occ:
                p = Point()
                p.x = float(self.origin[0] + (ix + 0.5) * self.res)
                p.y = float(self.origin[1] + (iy + 0.5) * self.res)
                p.z = 0.05
                marker.points.append(p)
            self.marker_pub.publish(marker)

        upd_i, upd_j = np.where(self.iter_updated)
        self.iter_updated[:] = False
        if upd_i.size > 0:
            lo_vals = self.log_odds[upd_i, upd_j].astype(np.float64)
            self.prob_grid[upd_j, upd_i] = (1.0 / (1.0 + np.exp(-lo_vals)) * 100.0).astype(np.int8)

        occ_msg = OccupancyGrid()
        occ_msg.header.frame_id = 'mocap'
        occ_msg.header.stamp = self.get_clock().now().to_msg()
        occ_msg.info.resolution = self.res
        occ_msg.info.width = self.nx
        occ_msg.info.height = self.ny
        occ_msg.info.origin = Pose()
        occ_msg.info.origin.position.x = float(self.origin[0])
        occ_msg.info.origin.position.y = float(self.origin[1])
        occ_msg.info.origin.position.z = 0.0
        occ_msg.info.origin.orientation.w = 1.0
        occ_msg.data = self.prob_grid.flatten().tolist()
        self.occmap_pub.publish(occ_msg)

def main(args=None):
    rclpy.init(args=args)
    node = OccupancyMapNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
