#!/usr/bin/env python3


import os
import threading
import time
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import OccupancyGrid
from gazebo_msgs.msg import ModelStates
from visualization_msgs.msg import Marker, MarkerArray

from ament_index_python.packages import get_package_share_directory

from lidar_based_mcbf.basic_boundary_function.configs import (
    CircularRobotSpecification, EnvContrConfiguration,
)


class DWAControllerNode(Node):
    def __init__(self):
        super().__init__('dwa_controller_node')

        pkg_share = get_package_share_directory('lidar_based_mcbf')
        econf_path = os.path.join(pkg_share, 'config', 'env_controller.yaml')
        spec_robot_fpath = os.path.join(pkg_share, 'config', 'spec_robot.yaml')

        self.declare_parameter("econf_path", econf_path)
        self.declare_parameter("spec_robot_fpath", spec_robot_fpath)
        self.declare_parameter("occupancy_map_topic", "/occupancy_map")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("control_hz", 10.0)
        self.declare_parameter("predict_time", 1.5)
        self.declare_parameter("sim_dt", 0.1)
        self.declare_parameter("v_samples", 15)
        self.declare_parameter("w_samples", 21)
        self.declare_parameter("goal_tolerance", 0.4)
        self.declare_parameter("occ_prob_thresh", 65)
        self.declare_parameter("treat_unknown_as_occupied", False)
        self.declare_parameter("robot_radius", 0.4)
        self.declare_parameter("clearance_norm", 1.5)
        self.declare_parameter("weight_goal", 0.5)
        self.declare_parameter("weight_clearance", 0.2)
        self.declare_parameter("weight_velocity", 0.3)
        self.declare_parameter("weight_accel", 0.0)
        self.declare_parameter("publish_trajectories", True)

        self.econf_path = self.get_parameter("econf_path").value
        self.spec_robot_fpath = self.get_parameter("spec_robot_fpath").value
        self.econf = EnvContrConfiguration.from_yaml(self.econf_path)
        self.spec_robot = CircularRobotSpecification.from_yaml(self.spec_robot_fpath)
        self.rob_name = self.spec_robot.rob_name

        if self.econf.dynamics not in ("differential", "omni-directional"):
            raise RuntimeError('Undefined robot dynamics.')
        self.dynamics = self.econf.dynamics
        self.target = np.array(self.econf.target, dtype=float)

        self.robot_radius = float(self.get_parameter("robot_radius").value)

        self.control_hz = float(self.get_parameter("control_hz").value)
        self.predict_time = float(self.get_parameter("predict_time").value)
        self.sim_dt = float(self.get_parameter("sim_dt").value)
        self.n_v = max(1, int(self.get_parameter("v_samples").value))
        self.n_w = max(1, int(self.get_parameter("w_samples").value))
        self.goal_tol = float(self.get_parameter("goal_tolerance").value)
        self.occ_thresh = int(self.get_parameter("occ_prob_thresh").value)
        self.unknown_is_occupied = bool(self.get_parameter("treat_unknown_as_occupied").value)
        self.clearance_norm = float(self.get_parameter("clearance_norm").value)
        self.w_goal = float(self.get_parameter("weight_goal").value)
        self.w_clearance = float(self.get_parameter("weight_clearance").value)
        self.w_velocity = float(self.get_parameter("weight_velocity").value)
        self.w_accel = float(self.get_parameter("weight_accel").value)
        self.publish_trajectories = bool(self.get_parameter("publish_trajectories").value)

        occ_topic = self.get_parameter("occupancy_map_topic").value
        cmd_topic = self.get_parameter("cmd_vel_topic").value

        self.sub_map = self.create_subscription(OccupancyGrid, occ_topic, self.callback_map, 1)
        self.sub_odom = self.create_subscription(ModelStates, "/gazebo/model_states", self.callback_rob, 1)

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.traj_pub = self.create_publisher(MarkerArray, "/dwa_trajectories", 1)
        self.goal_pub = self.create_publisher(Marker, "/dwa_goal_marker", 1)
        self.feasibility_marker_pub = self.create_publisher(Marker, "/solver_status_marker", 10)

        self.rob_xy: Optional[np.ndarray] = None
        self.rob_theta = 0.0

        self.last_u1 = 0.0
        self.last_u2 = 0.0

        self.dist_grid: Optional[np.ndarray] = None
        self.grid_origin: Optional[np.ndarray] = None
        self.grid_res: Optional[float] = None
        self.grid_shape = None

        self._map_lock = threading.Lock()
        self._goal_reached = False
        self._running = True
        self.TIMEOUT = 4000
        self._kt = 0

        self._ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._ctrl_thread.start()

        self.get_logger().info(
            f"DWA baseline controller started. dynamics={self.dynamics} "
            f"target={self.target.tolist()} robot_radius={self.robot_radius:.2f} m "
            f"map_topic={occ_topic} cmd_topic={cmd_topic}")

    def callback_rob(self, msg: ModelStates):
        if self.rob_name not in msg.name:
            return
        idx = msg.name.index(self.rob_name)
        pose = msg.pose[idx]

        self.rob_xy = np.array([pose.position.x, pose.position.y])
        quat = [pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w]
        self.rob_theta = float(R.from_quat(quat).as_euler("xyz")[2])

    def callback_map(self, msg: OccupancyGrid):
        grid = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        if self.unknown_is_occupied:
            occupied = (grid < 0) | (grid >= self.occ_thresh)
        else:
            occupied = grid >= self.occ_thresh
        free = ~occupied

        dist_grid = distance_transform_edt(free, sampling=msg.info.resolution)

        with self._map_lock:
            self.dist_grid = dist_grid
            self.grid_origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
            self.grid_res = float(msg.info.resolution)
            self.grid_shape = dist_grid.shape

    def _clearance_at(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        with self._map_lock:
            dist_grid, origin, res, shape = self.dist_grid, self.grid_origin, self.grid_res, self.grid_shape
        if dist_grid is None:
            return np.zeros_like(xs)

        ny, nx = shape
        ix = np.floor((xs - origin[0]) / res).astype(np.int32)
        iy = np.floor((ys - origin[1]) / res).astype(np.int32)
        inbounds = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

        out = np.zeros_like(xs, dtype=np.float64)
        out[inbounds] = dist_grid[iy[inbounds], ix[inbounds]]
        return out

    def _dynamic_window(self):
        spec = self.spec_robot
        if self.dynamics == "differential":
            return (spec.lin_vel_min, spec.lin_vel_max), (-spec.ang_vel_max, spec.ang_vel_max)
        return (spec.lin_vel_min, spec.lin_vel_max), (spec.lin_vel_min, spec.lin_vel_max)

    def _simulate_and_score(self):
        window1, window2 = self._dynamic_window()
        u1_vals = np.linspace(window1[0], window1[1], self.n_v)
        u2_vals = np.linspace(window2[0], window2[1], self.n_w)
        U1, U2 = np.meshgrid(u1_vals, u2_vals, indexing='ij')
        u1 = U1.flatten()
        u2 = U2.flatten()
        n = u1.size

        x = np.full(n, self.rob_xy[0])
        y = np.full(n, self.rob_xy[1])
        th = np.full(n, self.rob_theta)

        n_steps = max(1, int(round(self.predict_time / self.sim_dt)))
        traj_x = np.empty((n_steps + 1, n))
        traj_y = np.empty((n_steps + 1, n))
        traj_x[0] = x
        traj_y[0] = y
        min_clear = np.full(n, np.inf)

        for k in range(n_steps):
            if self.dynamics == "differential":
                x = x + u1 * np.cos(th) * self.sim_dt
                y = y + u1 * np.sin(th) * self.sim_dt
                th = th + u2 * self.sim_dt
            else:
                x = x + u1 * self.sim_dt
                y = y + u2 * self.sim_dt
            traj_x[k + 1] = x
            traj_y[k + 1] = y
            min_clear = np.minimum(min_clear, self._clearance_at(x, y))

        collided = min_clear < self.robot_radius

        speed = u1 if self.dynamics == "differential" else np.hypot(u1, u2)

        dist_to_goal_start = float(np.linalg.norm(self.target - self.rob_xy))
        dist_to_goal_end = np.linalg.norm(
            self.target - np.stack([x, y], axis=1), axis=1)
        goal_cost = dist_to_goal_end / max(dist_to_goal_start, 1e-6)

        clearance_capped = np.minimum(min_clear, self.clearance_norm)
        clearance_cost = 1.0 - clearance_capped / self.clearance_norm

        v_span = max(self.spec_robot.lin_vel_max - self.spec_robot.lin_vel_min, 1e-6)
        velocity_cost = (self.spec_robot.lin_vel_max - speed) / v_span

        if self.dynamics == "differential":
            w2_span = max(2.0 * self.spec_robot.ang_vel_max, 1e-6)
        else:
            w2_span = v_span
        dv1 = np.abs(u1 - self.last_u1) / v_span
        dv2 = np.abs(u2 - self.last_u2) / w2_span
        accel_cost = 0.5 * (np.minimum(dv1, 1.0) + np.minimum(dv2, 1.0))

        cost = (self.w_goal * goal_cost
                + self.w_clearance * clearance_cost
                + self.w_velocity * velocity_cost
                + self.w_accel * accel_cost)

        return u1, u2, cost, collided, traj_x, traj_y

    def _control_loop(self):
        period = 1.0 / self.control_hz
        self.get_logger().info("DWA: waiting for robot pose and occupancy map...")
        while rclpy.ok() and self._running and (self.rob_xy is None or self.dist_grid is None):
            time.sleep(0.1)
        self.get_logger().info("DWA control loop starting")

        while rclpy.ok() and self._running:
            t0 = time.monotonic()

            if self.rob_xy is not None:
                if self._kt >= self.TIMEOUT:
                    self.get_logger().info("TIMEOUT — stopping DWA control loop")
                    self._publish_cmd(0.0, 0.0)
                    self._running = False
                    break

                dist_to_goal = float(np.linalg.norm(self.target - self.rob_xy))
                if dist_to_goal < self.goal_tol:
                    if not self._goal_reached:
                        self.get_logger().info("Target reached — stopping DWA control loop")
                        self._goal_reached = True
                    self._publish_cmd(0.0, 0.0)
                    self._publish_feasibility_marker(False)
                else:
                    self._goal_reached = False
                    u1, u2, cost, collided, traj_x, traj_y = self._simulate_and_score()
                    valid_cost = np.where(collided, np.inf, cost)

                    if np.all(collided):
                        self.get_logger().warn(
                            "DWA: no collision-free trajectory found — stopping",
                            throttle_duration_sec=1.0)
                        self._publish_cmd(0.0, 0.0)
                        best_idx = int(np.argmin(cost))
                        best_collided = True
                    else:
                        best_idx = int(np.argmin(valid_cost))
                        self._publish_cmd(float(u1[best_idx]), float(u2[best_idx]))
                        best_collided = False

                    if self.publish_trajectories:
                        self._publish_trajectory_markers(
                            traj_x, traj_y, cost, collided, best_idx, best_collided)

                    self._publish_feasibility_marker(bool(np.all(collided)))
                    self._kt += 1

                self._publish_goal_marker()

            elapsed = time.monotonic() - t0
            sleep_t = period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _publish_cmd(self, u1: float, u2: float):
        msg = Twist()
        if self.dynamics == "differential":
            msg.linear.x = u1
            msg.angular.z = u2
        else:
            msg.linear.x = u1
            msg.linear.y = u2
        self.cmd_pub.publish(msg)
        self.last_u1 = u1
        self.last_u2 = u2

    def _publish_trajectory_markers(self, traj_x, traj_y, cost, collided, best_idx, best_collided):
        stamp = self.get_clock().now().to_msg()
        marray = MarkerArray()

        valid_cost = cost[~collided]
        if valid_cost.size:
            c_min = float(valid_cost.min())
            c_max = float(valid_cost.max())
        else:
            c_min, c_max = float(cost.min()), float(cost.max())
        c_span = max(c_max - c_min, 1e-6)

        n = traj_x.shape[1]
        max_traj = 80
        idxs = np.arange(n)
        if n > max_traj:
            priority_cost = np.where(collided, np.inf, cost)
            order = np.argsort(priority_cost)
            keep = set(order[:max_traj // 2].tolist())
            rest = [int(i) for i in idxs if int(i) not in keep]
            if rest:
                extra = np.random.choice(rest, size=min(max_traj - len(keep), len(rest)), replace=False)
                keep.update(int(i) for i in extra)
            idxs = np.array(sorted(keep))

        cand = Marker()
        cand.header.frame_id = 'mocap'
        cand.header.stamp = stamp
        cand.ns = 'dwa_candidates'
        cand.id = 0
        cand.type = Marker.LINE_LIST
        cand.action = Marker.ADD
        cand.pose.orientation.w = 1.0
        cand.scale.x = 0.03
        for i in idxs:
            i = int(i)
            if collided[i]:
                color = ColorRGBA(r=1.0, g=0.3, b=0.0, a=0.45)
            else:
                frac = float(np.clip((cost[i] - c_min) / c_span, 0.0, 1.0))
                color = ColorRGBA(r=frac, g=1.0 - frac, b=0.0, a=0.9)
            xs = traj_x[:, i]
            ys = traj_y[:, i]
            for k in range(len(xs) - 1):
                cand.points.append(Point(x=float(xs[k]), y=float(ys[k]), z=0.06))
                cand.points.append(Point(x=float(xs[k + 1]), y=float(ys[k + 1]), z=0.06))
                cand.colors.append(color)
                cand.colors.append(color)
        marray.markers.append(cand)

        best = Marker()
        best.header.frame_id = 'mocap'
        best.header.stamp = stamp
        best.ns = 'dwa_best'
        best.id = 1
        best.type = Marker.LINE_STRIP
        best.action = Marker.ADD
        best.pose.orientation.w = 1.0
        best.scale.x = 0.07
        if best_collided:
            best.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        else:
            best.color = ColorRGBA(r=0.0, g=0.4, b=1.0, a=1.0)
        best.points = [Point(x=float(x), y=float(y), z=0.08)
                       for x, y in zip(traj_x[:, best_idx], traj_y[:, best_idx])]
        marray.markers.append(best)

        self.traj_pub.publish(marray)

    def _publish_feasibility_marker(self, is_infeasible: bool):
        marker = Marker()
        marker.header.frame_id = 'mocap'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "solver_status"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.rob_xy[0])
        marker.pose.position.y = float(self.rob_xy[1])
        marker.pose.position.z = 0.8
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.05
        marker.color.a = 1.0
        if is_infeasible:
            marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0
        else:
            marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0
        self.feasibility_marker_pub.publish(marker)

    def _publish_goal_marker(self):
        m = Marker()
        m.header.frame_id = 'mocap'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'target'
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = float(self.target[0])
        m.pose.position.y = float(self.target[1])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.3
        m.color = ColorRGBA(r=0.25, g=0.25, b=0.5, a=1.0)
        self.goal_pub.publish(m)

    def destroy_node(self):
        self._running = False
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DWAControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
