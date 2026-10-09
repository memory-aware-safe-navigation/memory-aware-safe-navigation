#!/usr/bin/env python3
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from obstacle_detector.msg import Obstacles
from std_msgs.msg import Float32, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from scipy.spatial.transform import Rotation as R

from ament_index_python.packages import get_package_share_directory

from lidar_based_mcbf.basic_boundary_function.configs import CircularRobotSpecification, EnvContrConfiguration
from lidar_based_mcbf.basic_boundary_function.env_approx import Env_Approx
from lidar_based_mcbf_msgs.msg import GPDF, GPDFSet, StageConstraint

class MapperNode(Node):
    def __init__(self):
        super().__init__('mapper_node')

        pkg_share = get_package_share_directory('lidar_based_mcbf')
        econf_path = os.path.join(pkg_share, 'config', 'env_controller.yaml')

        self.declare_parameter("econf_path", econf_path)
        self.declare_parameter("map_hz", 5.0)
        self.econf_path = self.get_parameter("econf_path").value
        self._map_period = 1.0 / float(self.get_parameter("map_hz").value)

        self.econf = EnvContrConfiguration.from_yaml(self.econf_path)

        self.SCENARIO = self.econf.env_name
        self.global_map_frame = "mocap"

        self.map_sub = self.create_subscription(OccupancyGrid, "/occupancy_map", self.callback_map, 1)
        self.obs_sub = self.create_subscription(Obstacles, "/obstacles_tracked", self.callback_obs_detector, 1)
        self.odom_sub = self.create_subscription(ModelStates, "/gazebo/model_states", self.callback_rob, 1)
        self.stage_sub = self.create_subscription(StageConstraint, "/stage_constraint", self.callback_stage, 1)

        self.gpdf_set_pub = self.create_publisher(GPDFSet, "/gpdf_sets", 10)
        self.xc_dxc_pub = self.create_publisher(MarkerArray, '/dyn_obs_xc', 10)
        self.xc_pub = self.create_publisher(Float32MultiArray, '/mapper/xc', 10)
        self.dxc_pub = self.create_publisher(Float32MultiArray, '/mapper/dxc', 10)
        self.loop_dt_pub = self.create_publisher(Float32, '/mapper_loop_dt', 10)
        self.cluster_time_pub = self.create_publisher(Float32, '/mapper_cluster_dt', 10)
        self.gpdf_time_pub = self.create_publisher(Float32, '/mapper_gpdf_dt', 10)
        self.update_time_pub = self.create_publisher(Float32, '/mapper_update_dt', 10)
        self.publish_time_pub = self.create_publisher(Float32, '/mapper_publish_dt', 10)
        self.points_pack_dt_pub = self.create_publisher(Float32, '/mapper_points_pack_dt', 10)
        self.disfield_dt_pub = self.create_publisher(Float32, '/mapper_disfield_dt', 10)

        spec_robot_fpath = os.path.join(pkg_share, 'config', 'spec_robot.yaml')
        spec_robot = CircularRobotSpecification.from_yaml(spec_robot_fpath)
        self.lidar_range = spec_robot.lidar_range
        self.rob_name = spec_robot.rob_name

        self.occupied_points: np.ndarray = None
        self.dynamic_obstacles = []
        self.rob_xy = np.zeros(2)
        self.rob_theta = 0.0
        self.is_confined = False
        self.running = True
        self._prev_xc_count = 0

        self.shared_lock = threading.Lock()
        self.env_approx = Env_Approx(self.shared_lock, original_env=self.SCENARIO)

        self.map_thread = threading.Thread(target=self._map_loop, daemon=True)
        self.map_thread.start()

    def _map_loop(self):
        self.get_logger().info("Map loop started")
        first = True
        while rclpy.ok() and self.running:
            t0 = time.monotonic()

            if self.occupied_points is None or len(self.occupied_points) == 0:
                time.sleep(self._map_period)
                continue

            with self.shared_lock:
                dyn_snap = list(self.dynamic_obstacles)

            self.env_approx.set_confined(self.is_confined)

            t_update = time.monotonic()
            log_t, gpdf_t = self.env_approx.update(
                self.occupied_points,
                dyn_snap,
                debug=False,
            )
            update_t = time.monotonic() - t_update

            if self.env_approx.monitor_time:
                ct = Float32(); ct.data = float(log_t); self.cluster_time_pub.publish(ct)
                gt = Float32(); gt.data = float(gpdf_t); self.gpdf_time_pub.publish(gt)
                ut = Float32(); ut.data = float(update_t); self.update_time_pub.publish(ut)

            t_publish = time.monotonic()
            self._publish_gpdf_set()
            self._publish_xc_dxc()
            pt = Float32(); pt.data = float(time.monotonic() - t_publish); self.publish_time_pub.publish(pt)

            if first:
                self.get_logger().info("First map built — controller may start")
                first = False

            elapsed = time.monotonic() - t0
            dt = Float32()
            dt.data = float(elapsed)
            self.loop_dt_pub.publish(dt)
            sleep_t = self._map_period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    _LOCAL_MAP_HALF_RANGE = 8.0

    def callback_map(self, msg):
        grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        occupied = np.argwhere(grid > 50)
        if occupied.size == 0:
            self.occupied_points = np.empty((0, 2))
            return
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        x = ox + (occupied[:, 1] + 0.5) * res
        y = oy + (occupied[:, 0] + 0.5) * res
        r = self._LOCAL_MAP_HALF_RANGE
        mask = (np.abs(x - self.rob_xy[0]) <= r) & (np.abs(y - self.rob_xy[1]) <= r)
        self.occupied_points = np.column_stack((x[mask], y[mask])) if mask.any() else np.empty((0, 2))

    def callback_obs_detector(self, data):
        obstacles = []
        for c in data.circles:
            vx, vy = c.velocity.x, c.velocity.y
            obstacles.append({"center": (c.center.x, c.center.y), "velocity": (vx, vy)})
        with self.shared_lock:
            self.dynamic_obstacles = obstacles

    def callback_rob(self, data):
        idx = data.name.index(self.rob_name)
        pose = data.pose[idx]
        self.rob_xy = np.array([pose.position.x, pose.position.y])
        q = pose.orientation
        self.rob_theta = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]

    def callback_stage(self, msg):
        self.is_confined = bool(msg.is_confined)

    _DIS_FIELD_TOP_K = 2
    _DIS_FIELD_GRID_COLS = 60
    _DIS_FIELD_GRID_ROWS = 60

    def _publish_gpdf_set(self):
        msg = GPDFSet()
        msg.header.frame_id = self.global_map_frame
        msg.header.stamp = self.get_clock().now().to_msg()

        if not self.env_approx.gpdf_env:
            self.gpdf_set_pub.publish(msg)
            return

        margin = self.econf.om_range + 0.5

        rob_query = self.rob_xy.reshape(1, 2)

        a = 0.2 if self.econf.dynamics == "differential" else 0.0
        shifted_query = np.array([[
            self.rob_xy[0] + a * np.cos(self.rob_theta),
            self.rob_xy[1] + a * np.sin(self.rob_theta),
        ]])

        h_robot_vals = []
        grad_shifted_vals = []
        for i, gpdf in enumerate(self.env_approx.gpdf_env):
            dis_r, _ = gpdf.dis_normal_func(rob_query)
            dis_r_shifted, _ = gpdf.dis_normal_func(shifted_query)
            h_r = float(min(dis_r[0],dis_r_shifted[0])) + self.env_approx.offset[i] - 1.0

            _, grad_s = gpdf.dis_normal_func(shifted_query)
            h_robot_vals.append(h_r)
            grad_shifted_vals.append(grad_s.flatten()[:2])

        filtered = [i for i, h in enumerate(h_robot_vals) if h <= self.lidar_range]

        if not filtered:
            self.gpdf_set_pub.publish(msg)
            return

        filtered_h = [h_robot_vals[i] for i in filtered]
        sorted_local = np.argsort(filtered_h)
        top_k_indices = set(filtered[j] for j in sorted_local[:self._DIS_FIELD_TOP_K])

        total_points_pack_t = 0.0
        total_disfield_t = 0.0

        for _, i in enumerate(filtered):
            gpdf = self.env_approx.gpdf_env[i]
            gp = GPDF()

            t0_pts = time.monotonic()
            for p in np.asarray(gpdf.pc_coords):
                pt = Point()
                pt.x, pt.y, pt.z = float(p[0]), float(p[1]), 0.0
                gp.points.append(pt)
            gp.weights = [float(w) for w in np.asarray(gpdf.gpdf_model).reshape(-1)]

            total_points_pack_t += time.monotonic() - t0_pts

            if i in top_k_indices:
                t0_dis = time.monotonic()
                coords = np.asarray(gpdf.pc_coords)
                xmin = coords[:, 0].min() - margin
                xmax = coords[:, 0].max() + margin
                ymin = coords[:, 1].min() - margin
                ymax = coords[:, 1].max() + margin
                _x = np.linspace(xmin, xmax, self._DIS_FIELD_GRID_COLS)
                _y = np.linspace(ymin, ymax, self._DIS_FIELD_GRID_ROWS)
                X, Y = np.meshgrid(_x, _y)
                all_xy = np.column_stack((X.ravel(), Y.ravel()))
                dis_flat, _ = self.env_approx.h_grad_vector(all_xy, obstacle_idx=i, dynamic_obstacle=False)
                gp.dis_field = dis_flat.astype(np.float32).tolist()
                gp.grid_xmin = float(xmin)
                gp.grid_ymin = float(ymin)
                gp.grid_dx = float((xmax - xmin) / (self._DIS_FIELD_GRID_COLS - 1))
                gp.grid_dy = float((ymax - ymin) / (self._DIS_FIELD_GRID_ROWS - 1))
                gp.grid_rows = int(X.shape[0])
                gp.grid_cols = int(X.shape[1])
                total_disfield_t += time.monotonic() - t0_dis

            gp.h_robot = h_robot_vals[i]
            gp.grad_h_shifted_x = float(grad_shifted_vals[i][0])
            gp.grad_h_shifted_y = float(grad_shifted_vals[i][1])

            if gpdf.centroid is not None:
                gp.has_centroid = True
                gp.centroid.x = float(gpdf.centroid[0])
                gp.centroid.y = float(gpdf.centroid[1])
                gp.centroid.z = 0.0
            else:
                gp.has_centroid = False

            msg.models.append(gp)

        ppt = Float32(); ppt.data = float(total_points_pack_t); self.points_pack_dt_pub.publish(ppt)
        dft = Float32(); dft.data = float(total_disfield_t); self.disfield_dt_pub.publish(dft)

        msg.num_dyn_circle = int(self.env_approx.num_dyn_circle)
        msg.offsets = [float(self.env_approx.offset[i]) for i in filtered]

        for x, y in self.env_approx.xc:
            pt = Point(); pt.x, pt.y, pt.z = float(x), float(y), 0.0; msg.xc.append(pt)
        for vx, vy in self.env_approx.dxc:
            pt = Point(); pt.x, pt.y, pt.z = float(vx), float(vy), 0.0; msg.dxc.append(pt)

        if self.env_approx.other_obs_pt_contour_single is not None:
            for x, y in self.env_approx.other_obs_pt_contour_single:
                pt = Point(); pt.x, pt.y, pt.z = float(x), float(y), 0.0; msg.obs_contour_pts.append(pt)

        self.gpdf_set_pub.publish(msg)

    def _publish_xc_dxc(self):
        xc = self.env_approx.xc
        dxc = self.env_approx.dxc
        n = len(xc)

        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        if n == 0:
            clear_all = Marker()
            clear_all.header.frame_id = self.global_map_frame
            clear_all.header.stamp = stamp
            clear_all.action = Marker.DELETEALL
            arr.markers.append(clear_all)
            self._prev_xc_count = 0
            self.xc_dxc_pub.publish(arr)
            self.xc_pub.publish(Float32MultiArray())
            self.dxc_pub.publish(Float32MultiArray())
            return

        for i in range(n, self._prev_xc_count):
            m = Marker()
            m.header.frame_id = self.global_map_frame
            m.header.stamp = stamp
            m.ns = "dxc"; m.id = i; m.action = Marker.DELETE
            arr.markers.append(m)
        self._prev_xc_count = n

        sphere = Marker()
        sphere.header.frame_id = self.global_map_frame
        sphere.header.stamp = stamp
        sphere.ns = "xc"; sphere.id = 0
        sphere.type = Marker.SPHERE_LIST; sphere.action = Marker.ADD
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.3
        sphere.color.a = 0.8; sphere.color.r = 1.0; sphere.color.g = 0.5
        for x, y in xc:
            p = Point(); p.x, p.y, p.z = float(x), float(y), 0.0; sphere.points.append(p)
        arr.markers.append(sphere)

        for i, ((x, y), (vx, vy)) in enumerate(zip(xc, dxc)):
            arrow = Marker()
            arrow.header.frame_id = self.global_map_frame
            arrow.header.stamp = stamp
            arrow.ns = "dxc"; arrow.id = i
            arrow.type = Marker.ARROW; arrow.action = Marker.ADD
            arrow.scale.x = 0.05; arrow.scale.y = 0.1; arrow.scale.z = 0.1
            arrow.color.a = 1.0; arrow.color.r = 1.0
            speed = np.hypot(vx, vy)
            dx, dy = (vx / speed, vy / speed) if speed > 1e-6 else (0.0, 0.0)
            p_tail = Point(); p_tail.x, p_tail.y, p_tail.z = float(x), float(y), 0.0
            p_tip = Point(); p_tip.x,  p_tip.y,  p_tip.z  = float(x + dx), float(y + dy), 0.0
            arrow.points = [p_tail, p_tip]
            arr.markers.append(arrow)

        self.xc_dxc_pub.publish(arr)

        xc_msg = Float32MultiArray()
        xc_msg.data = [v for x, y in xc for v in (float(x), float(y))]
        self.xc_pub.publish(xc_msg)

        dxc_msg = Float32MultiArray()
        dxc_msg.data = [v for vx, vy in dxc for v in (float(vx), float(vy))]
        self.dxc_pub.publish(dxc_msg)

    def destroy_node(self):
        self.running = False
        return super().destroy_node()


def main():
    rclpy.init()
    node = MapperNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
