#!/usr/bin/env python3
import os
import threading
import time
from itertools import combinations
from types import SimpleNamespace
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node

from std_msgs.msg import Header, Bool, Float32
from geometry_msgs.msg import Point, Vector3Stamped
from gazebo_msgs.msg import ModelStates
from visualization_msgs.msg import Marker

from ament_index_python.packages import get_package_share_directory

from lidar_based_mcbf.basic_boundary_function.configs import CircularRobotSpecification, EnvContrConfiguration
from lidar_based_mcbf.basic_boundary_function.env_approx import Env_Approx
from lidar_based_mcbf_msgs.msg import GPDFSet, StageConstraint

class MapperConstraintNode(Node):
    def __init__(self):
        super().__init__('mapper_constraint_node')

        pkg_share = get_package_share_directory('lidar_based_mcbf')
        econf_path = os.path.join(pkg_share, 'config', 'env_controller.yaml')
        spec_robot_fpath = os.path.join(pkg_share, 'config', 'spec_robot.yaml')
        self.rob_name = CircularRobotSpecification.from_yaml(spec_robot_fpath).rob_name

        self.declare_parameter("econf_path", econf_path)
        self.declare_parameter("stage_hz", 10.0)
        self.declare_parameter("baseline_eval", False)
        self.econf_path = self.get_parameter("econf_path").value
        self._stage_period = 1.0 / float(self.get_parameter("stage_hz").value)
        self.baseline_eval = self.get_parameter("baseline_eval").get_parameter_value().bool_value

        self.econf = EnvContrConfiguration.from_yaml(self.econf_path)
        self.om_range = self.econf.om_range
        self.om_expand_threshold = self.econf.om_expand_threshold
        self.target = np.array(self.econf.target)
        self.global_map_frame = "mocap"

        self.stage_pub = self.create_publisher(StageConstraint, "/stage_constraint", 10)
        self.stage_dt_pub = self.create_publisher(Float32, "/controller_stage_dt", 10)
        self.target_marker_pub = self.create_publisher(Marker, "/target_marker", 10)
        self.temp_target_marker_pub = self.create_publisher(Marker, "/temp_target_marker", 10)

        self.odom_sub = self.create_subscription(
            ModelStates, "/gazebo/model_states", self.callback_rob, 1)
        self.gpdf_set_sub = self.create_subscription(
            GPDFSet, "/gpdf_sets", self.callback_gpdf_set, 1)
        self.e_prev_sub = self.create_subscription(
            Vector3Stamped, "/e_vec", self.callback_e_prev, 1)
        self.dynamic_obstacle_sub = self.create_subscription(
            Bool, "/dynamic_obstacle", self.callback_dynamic_obstacle, 1)

        self.shared_lock = threading.RLock()
        self.env_approx = Env_Approx(
            self.shared_lock, original_env=self.econf.env_name,
        )
        self.rob_xy: Optional[np.ndarray] = None
        self.rob_theta: float = 0.0
        self.e_prev: Optional[np.ndarray] = None
        self.dynamic_obstacle: bool = False
        self.no_map_yet = True

        self._stage3_latched = False
        self._is_confined_debounced = False
        self._confined_exit_pos: Optional[np.ndarray] = None
        self._confined_release_distance = 2.0
        self._target_confinement_range = 2.5

        self._temp_target: Optional[np.ndarray] = None
        self._temp_target_pullback = 0.5
        self._temp_target_reach_tol = 0.3
        self._disable_temp_target = False
        self._door_point_viz: Optional[np.ndarray] = None

        self.running = True
        self._stage_thread = threading.Thread(target=self._stage_loop, daemon=True)
        self._stage_thread.start()

    def callback_rob(self, data):
        idx = data.name.index(self.rob_name)
        pose = data.pose[idx]
        self.rob_xy = np.array([pose.position.x, pose.position.y])
        quat = [pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w]
        self.rob_theta = float(R.from_quat(quat).as_euler("xyz")[2])

    def callback_gpdf_set(self, msg):
        self.env_approx.update_from_msg(msg)
        self.no_map_yet = False

    def callback_e_prev(self, msg):
        if np.isnan(msg.vector.x) or np.isnan(msg.vector.y):
            self.e_prev = None
        else:
            self.e_prev = np.array([msg.vector.x, msg.vector.y])

    def callback_dynamic_obstacle(self, msg):
        self.dynamic_obstacle = bool(msg.data)

    def _stage_loop(self):
        self.get_logger().info("Stage loop started")
        while rclpy.ok() and self.running:
            t0 = time.monotonic()

            compute_start = time.time()
            result = self.compute_stage()
            stage_dt_msg = Float32()
            stage_dt_msg.data = float(time.time() - compute_start)
            self.stage_dt_pub.publish(stage_dt_msg)

            if result is not None:
                self._publish_stage(result)

            elapsed = time.monotonic() - t0
            sleep_t = self._stage_period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    @staticmethod
    def _compute_fixed_rect_mask_points(robot_pos: np.ndarray, heading: Optional[np.ndarray],
                                         target: Optional[np.ndarray], env: SimpleNamespace,
                                         stage3_latched: bool = False):
        if heading is None or target is None:
            return [], None, None, None

        robot_pos = np.asarray(robot_pos, dtype=float)[:2]
        heading = np.asarray(heading, dtype=float)[:2]

        heading_norm = float(np.linalg.norm(heading))
        if heading_norm < 1e-9:
            return [], None, None, None
        heading_unit = heading / heading_norm
        perp_unit = np.array([-heading_unit[1], heading_unit[0]])

        length, width = 3.5, 3.5
        fwd_min, fwd_max = -length / 2.0, length / 2.0
        lat_min, lat_max = -width / 2.0, width / 2.0

        yaw = float(np.arctan2(heading_unit[1], heading_unit[0]))
        center = robot_pos + heading_unit * ((fwd_min + fwd_max) / 2.0) + perp_unit * ((lat_min + lat_max) / 2.0)
        bounds = (float(center[0]), float(center[1]), yaw, length, width)

        if not env.gpdf_set:
            return [], bounds, None, None

        clusters_in_rect = []
        for idx, gpdf in enumerate(env.gpdf_set):
            pts = gpdf.pc_coords
            if pts is None or len(pts) == 0:
                continue
            pts = np.asarray(pts).reshape(-1, 2)
            rel = pts - robot_pos
            fwd = rel @ heading_unit
            lat = rel @ perp_unit
            mask = (fwd >= fwd_min) & (fwd <= fwd_max) & (lat >= lat_min) & (lat <= lat_max)
            if np.any(mask):
                clusters_in_rect.append({
                    'gpdf_idx': idx,
                    'pts_in_rect': pts[mask],
                    'pc_coords': pts,
                })

        min_gap = None
        closest_pair = None
        if len(clusters_in_rect) >= 2:
            for a, b in combinations(clusters_in_rect, 2):
                d = cdist(a['pts_in_rect'], b['pts_in_rect'])
                i, j = np.unravel_index(np.argmin(d), d.shape)
                dist = float(d[i, j])
                if dist <= 0.4:
                    continue
                if min_gap is None or dist < min_gap:
                    min_gap = dist
                    closest_pair = (a['pts_in_rect'][i].tolist(), b['pts_in_rect'][j].tolist())

        return clusters_in_rect, bounds, min_gap, closest_pair

    _CONFINE_NUM_BINS = 36
    _CONFINE_DIST_THRESHOLD = 3.0
    _CONFINE_BLOCKED_BINS = 34
    _CONFINE_MAX_RANGE = 8.0

    @staticmethod
    def _bin_confinement(center: np.ndarray, pts: Optional[np.ndarray],
                         num_bins: int = _CONFINE_NUM_BINS,
                         dist_threshold: float = _CONFINE_DIST_THRESHOLD,
                         blocked_bin_threshold: int = _CONFINE_BLOCKED_BINS,
                         max_range: float = _CONFINE_MAX_RANGE):
        center = np.asarray(center, dtype=float)[:2]
        bin_min_dist = np.full(num_bins, np.inf)

        if pts is None or len(pts) == 0:
            return False, 0, bin_min_dist, bin_min_dist < dist_threshold

        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        dx = pts[:, 0] - center[0]
        dy = pts[:, 1] - center[1]

        in_box = (np.abs(dx) <= max_range) & (np.abs(dy) <= max_range)
        dx = dx[in_box]
        dy = dy[in_box]
        d_sq = dx * dx + dy * dy

        in_range = d_sq <= max_range * max_range
        if not np.any(in_range):
            return False, 0, bin_min_dist, bin_min_dist < dist_threshold

        dx = dx[in_range]
        dy = dy[in_range]
        dists = np.sqrt(d_sq[in_range])

        bearings = np.arctan2(dy, dx)
        bin_width = 2.0 * np.pi / num_bins
        bin_idx = np.floor((bearings + np.pi) / bin_width).astype(int)
        np.clip(bin_idx, 0, num_bins - 1, out=bin_idx)

        np.minimum.at(bin_min_dist, bin_idx, dists)

        blocked_mask = bin_min_dist < dist_threshold
        blocked_count = int(np.sum(blocked_mask))
        return blocked_count >= blocked_bin_threshold, blocked_count, bin_min_dist, blocked_mask

    @staticmethod
    def _compute_room_confinement(robot_pos: np.ndarray, env: SimpleNamespace):
        obs_pts = getattr(env, 'other_obs_pt_contour_single', None)
        return MapperConstraintNode._bin_confinement(robot_pos, obs_pts)

    @staticmethod
    def _locate_door_point(robot_pos: np.ndarray, bin_min_dist: np.ndarray,
                            num_bins: int = 36, max_range: float = 8.0,
                            min_jump: float = 2.5):
        robot_pos = np.asarray(robot_pos, dtype=float)[:2]
        dists = np.where(np.isfinite(bin_min_dist), bin_min_dist, max_range)

        next_dists = np.roll(dists, -1)
        jumps = np.abs(next_dists - dists)

        i = int(np.argmax(jumps))
        jump = float(jumps[i])
        j = (i + 1) % num_bins
        far_idx = i if dists[i] > dists[j] else j

        if jump < min_jump:
            return None, jump, far_idx

        bin_width = 2.0 * np.pi / num_bins
        bearing = -np.pi + (far_idx + 0.5) * bin_width
        far_point = robot_pos + dists[far_idx] * np.array([np.cos(bearing), np.sin(bearing)])

        return far_point, jump, far_idx

    @staticmethod
    def _compute_target_confinement(target_pos: np.ndarray, env: SimpleNamespace):
        gpdf_set = getattr(env, 'gpdf_set', None)
        if not gpdf_set:
            return False, 0
        pt_list = [np.asarray(g.pc_coords, dtype=float).reshape(-1, 2)
                   for g in gpdf_set if g.pc_coords is not None and len(g.pc_coords) > 0]
        if not pt_list:
            return False, 0
        is_confined, count, _, _ = MapperConstraintNode._bin_confinement(
            target_pos, np.vstack(pt_list))
        return is_confined, count

    def compute_stage(self):
        if self.rob_xy is None or self.no_map_yet:
            return None

        if self.baseline_eval:
            self._stage3_latched = False
            self._is_confined_debounced = False
            self._confined_exit_pos = None
            self._temp_target = None
            self._door_point_viz = None

            robot_pos = np.asarray(self.rob_xy, dtype=float)[:2]
            if self.e_prev is not None:
                mask_dir = np.asarray(self.e_prev, dtype=float).flatten()[:2]
            else:
                mask_dir = np.array([np.cos(self.rob_theta), np.sin(self.rob_theta)])
            snap = self.env_approx.snapshot()
            _, rect_bounds, _, _ = self._compute_fixed_rect_mask_points(
                robot_pos, mask_dir, self.target[:2], snap)

            return {
                'stage': 1,
                'is_confined': False,
                'stage3_latched': False,
                'min_gap': None,
                'effective_om_expand_threshold': self.om_expand_threshold,
                'margin': 0.25,
                'rect_bounds': rect_bounds,
                'door_point': None,
                'door_point_viz': None,
                'temp_target': None,
            }

        robot_pos = np.asarray(self.rob_xy, dtype=float)[:2]

        if self.e_prev is not None:
            mask_dir = np.asarray(self.e_prev, dtype=float).flatten()[:2]
        else:
            mask_dir = np.array([np.cos(self.rob_theta), np.sin(self.rob_theta)])

        snap = self.env_approx.snapshot()

        clusters_in_rect, rect_bounds, min_gap, closest_cluster_pts = self._compute_fixed_rect_mask_points(
            robot_pos, mask_dir, self.target[:2], snap,
            stage3_latched=self._stage3_latched)

        door_point = None
        door_point_viz = None
        if len(clusters_in_rect) > 1:
            raw_is_confined = False
        elif not self.dynamic_obstacle:
            raw_is_confined, _, bin_min_dist, _ = self._compute_room_confinement(robot_pos, snap)
            if raw_is_confined:
                door_point, _, _ = self._locate_door_point(robot_pos, bin_min_dist)
            target_enclosed, _ = self._compute_target_confinement(self.target[:2], snap)
            dist_to_target = float(np.linalg.norm(robot_pos - self.target[:2]))
            if target_enclosed and dist_to_target <= self._target_confinement_range:
                raw_is_confined = True
                door_point = None
                self._temp_target = None
        else:
            raw_is_confined = False
        door_point_viz = door_point

        if raw_is_confined:
            self._confined_exit_pos = None
            is_confined = True
        elif self._is_confined_debounced:
            if self._confined_exit_pos is None:
                self._confined_exit_pos = robot_pos.copy()
            moved_dist = float(np.linalg.norm(robot_pos - self._confined_exit_pos))
            is_confined = moved_dist < self._confined_release_distance
            if not is_confined:
                self._confined_exit_pos = None
        else:
            is_confined = False
        self._is_confined_debounced = is_confined

        if not is_confined:
            self._door_point_viz = None
        elif door_point_viz is not None:
            self._door_point_viz = door_point_viz

        if not is_confined or self._disable_temp_target:
            self._temp_target = None
        elif self._temp_target is None and door_point is not None:
            to_door = door_point - robot_pos
            to_door_dist = float(np.linalg.norm(to_door))
            if to_door_dist > self._temp_target_pullback:
                self._temp_target = door_point - (to_door / to_door_dist) * self._temp_target_pullback

        if self._temp_target is not None:
            if float(np.linalg.norm(robot_pos - self._temp_target)) <= self._temp_target_reach_tol:
                self._temp_target = None

        if is_confined:
            self._stage3_latched = True
        else:
            if self._stage3_latched and (min_gap is None or min_gap > 2.0):
                self._stage3_latched = False

        if self._stage3_latched:
            stage = 2
        elif min_gap is not None and min_gap <= 1.5:
            stage = 2
            self._stage3_latched = True
        else:
            stage = 1

        if self.dynamic_obstacle:
            self._stage3_latched = False

        if stage == 2:
            effective_om_expand_threshold = self.om_range
            margin = 0.05
        else:
            effective_om_expand_threshold = self.om_expand_threshold
            margin = 0.25

        if self.dynamic_obstacle:
            effective_om_expand_threshold = 0.0

        return {
            'stage': stage,
            'is_confined': is_confined and not self.dynamic_obstacle,
            'stage3_latched': self._stage3_latched,
            'min_gap': min_gap,
            'effective_om_expand_threshold': effective_om_expand_threshold,
            'margin': margin,
            'rect_bounds': rect_bounds,
            'door_point': door_point,
            'door_point_viz': self._door_point_viz,
            'temp_target': self._temp_target,
        }

    def _publish_stage(self, result):
        msg = StageConstraint()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.global_map_frame

        msg.stage = int(result['stage'])
        msg.is_confined = bool(result['is_confined'])
        msg.stage3_latched = bool(result['stage3_latched'])
        msg.min_gap = float(result['min_gap']) if result['min_gap'] is not None else -1.0
        msg.effective_om_expand_threshold = float(result['effective_om_expand_threshold'])
        msg.margin = float(result['margin'])

        rect_bounds = result['rect_bounds']
        msg.has_rect_bounds = rect_bounds is not None
        if rect_bounds is not None:
            cx, cy, yaw, length, width = rect_bounds
            msg.rect_center = Point(x=float(cx), y=float(cy), z=0.0)
            msg.rect_yaw = float(yaw)
            msg.rect_length = float(length)
            msg.rect_width = float(width)

        temp_target = result['temp_target']
        msg.has_temp_target = temp_target is not None
        if temp_target is not None:
            msg.temp_target = Point(x=float(temp_target[0]), y=float(temp_target[1]), z=0.0)

        self.stage_pub.publish(msg)

        self._publish_target_marker(self.target[:2])
        self._publish_temp_target_marker(result['door_point_viz'])

    def _publish_target_marker(self, point):
        marker = Marker()
        marker.header.frame_id = self.global_map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "target"; marker.id = 0
        marker.type = Marker.CUBE; marker.action = Marker.ADD
        marker.pose.position.x = float(point[0]); marker.pose.position.y = float(point[1])
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.3
        marker.color.r = 0.25; marker.color.g = 0.25; marker.color.b = 0.5
        marker.color.a = 1.0
        self.target_marker_pub.publish(marker)

    def _publish_temp_target_marker(self, point: Optional[np.ndarray]):
        marker = Marker()
        marker.header.frame_id = self.global_map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "temp_target"; marker.id = 0
        marker.type = Marker.CUBE

        if point is None:
            marker.action = Marker.DELETE
            self.temp_target_marker_pub.publish(marker)
            return

        marker.action = Marker.ADD
        marker.pose.position.x = float(point[0]); marker.pose.position.y = float(point[1])
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.3
        marker.color.r = 0.6; marker.color.g = 0.6; marker.color.b = 0.8
        marker.color.a = 1.0
        self.temp_target_marker_pub.publish(marker)

    def destroy_node(self):
        self.running = False
        return super().destroy_node()


def main():
    rclpy.init()
    node = MapperConstraintNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
