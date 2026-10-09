#!/usr/bin/env python3
import os
import threading
import time
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
from skimage import measure

import rclpy
from rclpy.node import Node

from std_msgs.msg import Header, ColorRGBA, Float32
from geometry_msgs.msg import Point, Point32, PolygonStamped, Vector3, Vector3Stamped
from gazebo_msgs.msg import ModelStates
from visualization_msgs.msg import Marker, MarkerArray

from ament_index_python.packages import get_package_share_directory

from lidar_based_mcbf.basic_boundary_function.configs import CircularRobotSpecification, EnvContrConfiguration
from lidar_based_mcbf.basic_boundary_function.env_approx import Env_Approx
from lidar_based_mcbf.basic_boundary_function.onMan_approximation import OnMan_Approx
from lidar_based_mcbf.basic_boundary_function.gpdf_w_rh import ITER, MAX_OBS
from lidar_based_mcbf_msgs.msg import GPDFSet, StageConstraint, MCBFConstraint

class MCBFConstraintNode(Node):
    def __init__(self):
        super().__init__('mcbf_constraint_node')

        pkg_share = get_package_share_directory('lidar_based_mcbf')
        econf_path = os.path.join(pkg_share, 'config', 'env_controller.yaml')
        spec_robot_fpath = os.path.join(pkg_share, 'config', 'spec_robot.yaml')
        self.rob_name = CircularRobotSpecification.from_yaml(spec_robot_fpath).rob_name

        self.declare_parameter("econf_path", econf_path)
        self.declare_parameter("mcbf_hz", 10.0)
        self.declare_parameter("use_union_geodesic", False)
        self.declare_parameter("union_beta", 0.1)
        self.declare_parameter("rob_proj_tie_eps", 1.5)
        self.econf_path = self.get_parameter("econf_path").value
        self._period = 1.0 / float(self.get_parameter("mcbf_hz").value)
        self.use_union_geodesic = bool(self.get_parameter("use_union_geodesic").value)
        self.union_beta = float(self.get_parameter("union_beta").value)
        self.rob_proj_tie_eps = float(self.get_parameter("rob_proj_tie_eps").value)

        self.econf = EnvContrConfiguration.from_yaml(self.econf_path)
        self.target = np.array(self.econf.target, dtype=float)
        self.MCBF = self.econf.MCBF
        self.om_range = self.econf.om_range
        self.om_expand_threshold = self.econf.om_expand_threshold
        self.target_range = self.econf.target_range
        self.beta_coef = self.econf.beta_coef
        self.dynamics = self.econf.dynamics
        self.global_map_frame = "mocap"

        self.shift_a = 0.2 if self.dynamics == "differential" else 0.0

        self.mcbf_pub = self.create_publisher(MCBFConstraint, "/mcbf_constraint", 10)

        self.e_vec_pub = self.create_publisher(Vector3Stamped, "/e_vec", 10)
        self.e_vec_marker_pub = self.create_publisher(Marker, "/e_vec_marker", 10)
        self.boundary_points_pub = self.create_publisher(PolygonStamped, "/boundary_points", 10)
        self.proj_pub_rob = self.create_publisher(Marker, "/proj_points_rob", 10)
        self.proj_pub_rob_tie_candidates = self.create_publisher(Marker, "/proj_points_rob_tie_candidates", 10)
        self.proj_pub_target = self.create_publisher(Marker, "/proj_points_target", 10)
        self.geodesic_paths_pub = self.create_publisher(MarkerArray, "/geodesic_paths", 10)
        self.line_samples_pub = self.create_publisher(Marker, "/line_samples", 10)
        self.polygon_dt_publisher = self.create_publisher(Float32, "/controller_polygon_dt", 10)
        self.projection_dt_publisher = self.create_publisher(Float32, "/controller_projection_dt", 10)
        self.geodesic_dt_publisher = self.create_publisher(Float32, "/controller_geodesic_dt", 10)
        self.dist_nearest_proj_target_publisher = self.create_publisher(
            Float32, "/controller_dist_nearest_proj_target", 10)
        self.dist_perp_proj_target_publisher = self.create_publisher(
            Float32, "/controller_dist_perp_proj_target", 10)

        self.odom_sub = self.create_subscription(
            ModelStates, "/gazebo/model_states", self.callback_rob, 1)
        self.gpdf_set_sub = self.create_subscription(
            GPDFSet, "/gpdf_sets", self.callback_gpdf_set, 1)
        self.stage_sub = self.create_subscription(
            StageConstraint, "/stage_constraint", self.callback_stage, 1)

        self.shared_lock = threading.RLock()
        self.env_approx = Env_Approx(
            self.shared_lock, original_env=self.econf.env_name,
            radius=self.econf.radius)
        self.om = OnMan_Approx(self.env_approx, hold_time=self.econf.hold_time, w=self.econf.w)

        self.rob_xy: Optional[np.ndarray] = None
        self.rob_theta = 0.0
        self.no_map_yet = True
        self.e_prev: Optional[np.ndarray] = None
        self._latest_stage_msg: Optional[StageConstraint] = None

        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

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

    def callback_stage(self, msg):
        self._latest_stage_msg = msg

    def _current_stage_info(self) -> dict:
        msg = self._latest_stage_msg
        if msg is None:
            return {
                'is_confined': False,
                'effective_om_expand_threshold': self.om_expand_threshold,
                'active_target': self.target[:2],
            }
        active_target = (
            np.array([msg.temp_target.x, msg.temp_target.y])
            if msg.has_temp_target else self.target[:2]
        )
        return {
            'is_confined': bool(msg.is_confined),
            'effective_om_expand_threshold': float(msg.effective_om_expand_threshold),
            'active_target': active_target,
        }

    def _loop(self):
        self.get_logger().info("MCBF constraint loop started")
        while rclpy.ok() and self.running and (self.no_map_yet or self.rob_xy is None):
            time.sleep(self._period)

        while rclpy.ok() and self.running:
            t0 = time.monotonic()
            result = self.compute()
            self._publish(result)
            elapsed = time.monotonic() - t0
            sleep_t = self._period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    @staticmethod
    def _empty_result():
        return {
            'e_vec': None, 'beta': None, 'om_idx': -1,
            'boundary_points': None, 'rob_proj_points': None, 'rob_tie_points': None, 'target_proj_points': None,
            'dist_nearest_proj_target': None, 'dist_perp_proj_target': None,
            'geodesic_xi_list': None, 'geodesic_pi_list': None,
            'timing_polygon_dt': 0.0, 'timing_projection_dt': 0.0, 'timing_geodesic_dt': 0.0,
        }

    def compute(self) -> dict:
        env = self.env_approx
        om = self.om
        stage_info = self._current_stage_info()
        is_confined = stage_info['is_confined']
        effective_om_expand_threshold = stage_info['effective_om_expand_threshold']
        active_target = stage_info['active_target']

        result = self._empty_result()

        if not self.MCBF:
            return result

        with env.lock:
            snap = env.snapshot()

            state_xy = np.asarray(self.rob_xy, dtype=float)
            if self.dynamics == "differential":
                x = np.array([
                    state_xy[0] + self.shift_a * np.cos(self.rob_theta),
                    state_xy[1] + self.shift_a * np.sin(self.rob_theta),
                    self.rob_theta,
                ]).reshape(1, 3)
            else:
                x = np.array([state_xy[0], state_xy[1]]).reshape(1, -1)

            if self.use_union_geodesic:
                return self._compute_union(env, om, snap, state_xy, x, active_target)

            if snap.precomp_h_robot is not None and not env.mmp and snap.num_dyn_circle == 0:
                h_rc_set = snap.precomp_h_robot
                grad_set = snap.precomp_grad_shifted
            else:
                _, grad_set, _, _, _ = env.h_grad_set(x[:, :2], snap=snap)
                h_rc_set, *_ = env.h_grad_set(state_xy.reshape(1, 2), snap=snap)

            h_rank = h_rc_set
            _n_static = len(snap.gpdf_env)
            if (snap.precomp_h_robot is not None and _n_static > 0
                    and snap.precomp_h_robot.shape[0] == _n_static
                    and h_rc_set.shape[0] >= _n_static):
                h_rank = h_rc_set.copy()
                h_rank[-_n_static:] = snap.precomp_h_robot

            non_wall_count = len(snap.xc_ref)
            if non_wall_count <= 0:
                return result

            if non_wall_count > 2:
                _sub = h_rank[-non_wall_count:].flatten()
                _raw = np.argpartition(_sub, 2)[:2]
                min_h_idx_list = _raw[np.argsort(_sub[_raw])] + snap.num_dyn_circle
            elif non_wall_count == 2:
                _sub = h_rank[-non_wall_count:].flatten()
                min_h_idx_list = np.argsort(_sub) + snap.num_dyn_circle
            else:
                min_h_idx_list = np.array([snap.num_dyn_circle])

            selected_min_h_idx = None
            selected_on_b = False
            selected_h_om = None
            selected_gpdf_obj = None
            selected_boundary_points = None
            target_proj_points = None
            rob_proj_points = None
            rob_tie_points = None
            dist_nearest_proj_target = None
            dist_perp_proj_target = None
            beta = None
            timing_polygon_dt = 0.0
            timing_projection_dt = 0.0

            for slot_i, min_h_idx in enumerate(min_h_idx_list):
                if h_rc_set[min_h_idx] >= self.om_range:
                    continue

                gpdf_obj = snap.gpdf_set[min_h_idx - snap.num_dyn_circle]

                if len(gpdf_obj.pc_coords) == 0:
                    continue

                if (h_rc_set[min_h_idx] < effective_om_expand_threshold and np.linalg.norm(active_target - state_xy) >= self.target_range):
                    h_om = -snap.offset[min_h_idx - (env.num_dyn_mmp + snap.num_dyn_circle)]
                    on_b = True
                else:
                    h_om = h_rc_set[min_h_idx] - snap.offset[min_h_idx - (env.num_dyn_mmp + snap.num_dyn_circle)]
                    on_b = False

                if h_om < 0.0:
                    break

                if is_confined:
                    h_om = -snap.offset[min_h_idx - (env.num_dyn_mmp + snap.num_dyn_circle)]
                    on_b = True

                polygon_start = time.time()
                boundary_points = MCBFConstraintNode.get_polygon_from_gpdf(
                    env, gpdf_obj.pc_coords.reshape(-1, 2),
                    h_om=h_om,
                    extra_margin=float(abs(h_om)) + 0.2,
                    obstacle_idx=min_h_idx - snap.num_dyn_circle, dynamic_obstacle=False,
                    gpdf_obj=gpdf_obj)
                timing_polygon_dt += time.time() - polygon_start

                proj_start = time.time()
                _, proj_point, distances_from_start, total_length, dist_nearest, dist_perp, tie_points = (
                    MCBFConstraintNode.get_projection_info(
                        boundary_points, [state_xy.tolist(), active_target.tolist()],
                        tie_eps=self.rob_proj_tie_eps))
                timing_projection_dt += time.time() - proj_start

                minimal_distance = min(
                    abs(distances_from_start[0] - distances_from_start[1]),
                    total_length - abs(distances_from_start[0] - distances_from_start[1]))

                if target_proj_points is None or (
                        np.linalg.norm(active_target - target_proj_points)
                        > np.linalg.norm(active_target - proj_point[1])):
                    selected_min_h_idx = min_h_idx
                    selected_on_b = on_b
                    selected_h_om = h_om
                    selected_gpdf_obj = gpdf_obj
                    selected_boundary_points = boundary_points
                    target_proj_points = proj_point[1]
                    rob_proj_points = proj_point[0]
                    rob_tie_points = tie_points
                    dist_nearest_proj_target = dist_nearest
                    dist_perp_proj_target = dist_perp
                    beta = minimal_distance / (2 * ITER) * self.beta_coef

                if (np.linalg.norm(proj_point[-1] - state_xy) > total_length / 15
                        or np.linalg.norm(active_target - state_xy) < self.target_range):
                    break

            result.update({
                'boundary_points': selected_boundary_points,
                'rob_proj_points': rob_proj_points,
                'rob_tie_points': rob_tie_points,
                'target_proj_points': target_proj_points,
                'dist_nearest_proj_target': dist_nearest_proj_target,
                'dist_perp_proj_target': dist_perp_proj_target,
                'timing_polygon_dt': timing_polygon_dt,
                'timing_projection_dt': timing_projection_dt,
            })

            if selected_min_h_idx is None or h_rc_set[selected_min_h_idx] >= self.om_range:
                self.e_prev = None
                return result

            geo_start = time.time()
            e_vec_raw, pi_list, xi_list = om.geodesic_approx_phi_2D(
                x[:, :2], grad_set[selected_min_h_idx].reshape(1, 2), active_target.reshape(1, 2), beta,
                onM=selected_min_h_idx - snap.num_dyn_circle, checking_mode=True,
                on_boundary=selected_on_b,
                e_prev= self.e_prev,
                rob_proj_override=rob_proj_points,
                h_om=selected_h_om,
                gpdf_obj_override=selected_gpdf_obj)
            result['timing_geodesic_dt'] = time.time() - geo_start

            self.e_prev = e_vec_raw


            result.update({
                'e_vec': e_vec_raw.flatten(),
                'beta': beta,
                'om_idx': int(selected_min_h_idx),
                'geodesic_xi_list': xi_list,
                'geodesic_pi_list': pi_list,
            })
            return result

    def _compute_union(self, env, om, snap, state_xy, x, active_target) -> dict:
        result = self._empty_result()

        num_gpdf = len(snap.gpdf_set)
        num_non_wall = num_gpdf
        if num_non_wall <= 0:
            self.e_prev = None
            return result

        non_wall_idx = list(range(num_non_wall))
        if len(non_wall_idx) > MAX_OBS:
            dists = np.empty(len(non_wall_idx))
            for k, idx in enumerate(non_wall_idx):
                d, _ = snap.gpdf_set[idx].dis_normal_func(state_xy.reshape(1, 2))
                dists[k] = float(np.asarray(d).flatten()[0]) + snap.offset[idx]
            nearest = np.argsort(dists)[:MAX_OBS]
            non_wall_idx = [non_wall_idx[i] for i in sorted(nearest)]

        geo_start = time.time()
        h_om, grad_om = env.h_grad_uni(state_xy.reshape(1, 2), idx=non_wall_idx)
        h_om = float(np.asarray(h_om).flatten()[0])

        if h_om >= self.om_range:
            self.e_prev = None
            return result

        e_vec_output = om.geodesic_approx_phi_2D_uni(
            x[:, :2], grad_om.reshape(1, 2), active_target.reshape(1, 2),
            self.union_beta, onM=non_wall_idx, checking_mode=True,
            on_boundary=True, e_prev=self.e_prev, target_range=self.target_range)
        result['timing_geodesic_dt'] = time.time() - geo_start

        if e_vec_output is None:
            self.e_prev = None
            return result

        e_vec_raw, pi_list, xi_list = e_vec_output
        pi_list = np.asarray(pi_list).reshape(2, 1)
        self.e_prev = e_vec_raw

        if snap.precomp_h_robot is not None and not env.mmp and snap.num_dyn_circle == 0:
            h_rc_set = snap.precomp_h_robot
        else:
            h_rc_set, *_ = env.h_grad_set(state_xy.reshape(1, 2), snap=snap)
        gpdf_h = h_rc_set[snap.num_dyn_circle:snap.num_dyn_circle + num_non_wall]
        om_idx = int(np.argmin(gpdf_h)) + snap.num_dyn_circle if gpdf_h.size > 0 else -1

        result.update({
            'e_vec': e_vec_raw.flatten(),
            'beta': self.union_beta,
            'om_idx': om_idx,
            'geodesic_xi_list': xi_list,
            'geodesic_pi_list': pi_list,
        })
        return result

    @staticmethod
    def get_polygon_from_gpdf(env: Env_Approx, points_for_gpdf: np.ndarray,
                              h_om: float = 0.0, extra_margin: float = 0.0, resolution: float = 0.1,
                              obstacle_idx: int = -1, dynamic_obstacle: bool = False, gpdf_obj=None):
        if gpdf_obj is not None and gpdf_obj.dis_field is not None and gpdf_obj.grid_meta is not None:
            meta = gpdf_obj.grid_meta
            xmin = meta['xmin']
            ymin = meta['ymin']
            dx   = meta['dx']
            dy   = meta['dy']
            dis_mat = gpdf_obj.dis_field.astype(float)

            binary = (dis_mat >= h_om).astype(float)
            binary[[0, -1], :] = 1.0
            binary[:, [0, -1]] = 1.0
            contours = measure.find_contours(binary, level=0.5)
            if contours:
                contour = max(contours, key=len)
                contour_x = xmin + contour[:, 1] * dx
                contour_y = ymin + contour[:, 0] * dy
                return np.column_stack((contour_x, contour_y))
            else:
                return np.zeros((0, 2))


        xmin = points_for_gpdf[:, 0].min() - extra_margin
        xmax = points_for_gpdf[:, 0].max() + extra_margin
        ymin = points_for_gpdf[:, 1].min() - extra_margin
        ymax = points_for_gpdf[:, 1].max() + extra_margin

        _DIS_FIELD_GRID_COLS = 50
        _DIS_FIELD_GRID_ROWS = 50
        _x = np.linspace(xmin, xmax, _DIS_FIELD_GRID_COLS)
        _y = np.linspace(ymin, ymax, _DIS_FIELD_GRID_ROWS)
        X, Y = np.meshgrid(_x, _y)
        all_xy_coords = np.column_stack((X.ravel(), Y.ravel()))
        dis_mat, _ = env.h_grad_vector(all_xy_coords, obstacle_idx=obstacle_idx, dynamic_obstacle=dynamic_obstacle)
        dis_mat = dis_mat.reshape(X.shape)
        binary = (dis_mat >= h_om).astype(float)
        binary[[0, -1], :] = 1.0
        binary[:, [0, -1]] = 1.0
        contours = measure.find_contours(binary, level=0.5)
        contour = max(contours, key=len)
        contour_x = xmin + contour[:, 1] * resolution
        contour_y = ymin + contour[:, 0] * resolution
        boundary_points = np.column_stack((contour_x, contour_y))
        return boundary_points


    @staticmethod
    def get_projection_info(coords: List[tuple], points: List[tuple], tie_eps: float = 0.0):
        coords_np = np.asarray(coords, dtype=float)
        points_np = np.asarray(points, dtype=float)

        A = coords_np[:-1]
        B = coords_np[1:]
        AB = B - A

        segment_lengths = np.linalg.norm(AB, axis=1)
        cum_segment_lengths_with_zero = np.concatenate(([0], np.cumsum(segment_lengths)))
        total_length = float(np.sum(segment_lengths))

        AB_dot_AB = np.einsum('ij,ij->i', AB, AB)

        def nearest_point_proj(P: np.ndarray, bias_dfs: Optional[float] = None):
            AP = P[np.newaxis, :] - A
            AB_dot_AP = np.einsum('ij,ij->i', AB, AP)
            t = np.clip(AB_dot_AP / np.where(AB_dot_AB == 0.0, 1.0, AB_dot_AB), 0.0, 1.0)
            proj = A + t[:, np.newaxis] * AB
            dist = np.linalg.norm(proj - P[np.newaxis, :], axis=1)
            dfs_all = cum_segment_lengths_with_zero[:-1] + t * segment_lengths

            tie_points = None
            if bias_dfs is not None and tie_eps > 0.0:
                arc_diff = np.abs(dfs_all - bias_dfs)
                arc_diff = np.minimum(arc_diff, total_length - arc_diff)
                half_length = total_length / 2.0 if total_length > 0.0 else 1.0
                arc_frac = arc_diff / half_length
                cost = dist + tie_eps * arc_frac
                idx = int(np.argmin(cost))
                near_ties = np.nonzero(dist <= dist.min() + tie_eps)[0]
                tie_points = proj[near_ties] if near_ties.size > 0 else None
            else:
                idx = int(np.argmin(dist))

            return idx, proj[idx], dfs_all[idx], tie_points

        target_pt = points_np[-1]
        target_idx, target_proj, target_dfs, _ = nearest_point_proj(target_pt)

        min_indices = []
        projection_points = []
        distances_from_start = []
        rob_tie_points = None
        for i, P in enumerate(points_np):
            if i == 0 and len(points_np) > 1:
                idx, proj, dfs, rob_tie_points = nearest_point_proj(P, bias_dfs=target_dfs)
            elif i == len(points_np) - 1:
                idx, proj, dfs = target_idx, target_proj, target_dfs
            else:
                idx, proj, dfs, _ = nearest_point_proj(P)
            min_indices.append(idx)
            projection_points.append(proj)
            distances_from_start.append(dfs)

        dist_nearest_to_target = float(np.linalg.norm(target_proj - target_pt))

        return (np.asarray(min_indices), np.asarray(projection_points),
                np.asarray(distances_from_start), total_length,
                dist_nearest_to_target, None, rob_tie_points)

    def _publish(self, result: dict):
        stamp = self.get_clock().now().to_msg()

        msg = MCBFConstraint()
        msg.header = Header()
        msg.header.stamp = stamp
        msg.header.frame_id = self.global_map_frame
        e_vec = result['e_vec']
        msg.valid = e_vec is not None
        if e_vec is not None:
            msg.e_vec = Vector3(x=float(e_vec[0]), y=float(e_vec[1]), z=0.0)
            msg.beta = float(result['beta'])
            msg.om_idx = int(result['om_idx'])
        else:
            msg.e_vec = Vector3(x=float('nan'), y=float('nan'), z=0.0)
            msg.beta = float('nan')
            msg.om_idx = -1
        msg.dist_nearest_proj_target = (
            float(result['dist_nearest_proj_target'])
            if result['dist_nearest_proj_target'] is not None else -1.0)
        msg.dist_perp_proj_target = (
            float(result['dist_perp_proj_target'])
            if result['dist_perp_proj_target'] is not None else -1.0)
        self.mcbf_pub.publish(msg)

        for val, pub in [
            (result['timing_polygon_dt'], self.polygon_dt_publisher),
            (result['timing_projection_dt'], self.projection_dt_publisher),
            (result['timing_geodesic_dt'], self.geodesic_dt_publisher),
        ]:
            dt_msg = Float32()
            dt_msg.data = float(val)
            pub.publish(dt_msg)

        dist_nearest_msg = Float32()
        dist_nearest_msg.data = (
            float(result['dist_nearest_proj_target'])
            if result['dist_nearest_proj_target'] is not None else float('nan'))
        self.dist_nearest_proj_target_publisher.publish(dist_nearest_msg)

        dist_perp_msg = Float32()
        dist_perp_msg.data = (
            float(result['dist_perp_proj_target'])
            if result['dist_perp_proj_target'] is not None else float('nan'))
        self.dist_perp_proj_target_publisher.publish(dist_perp_msg)

        e_vec_msg = Vector3Stamped()
        e_vec_msg.header.frame_id = self.global_map_frame
        e_vec_msg.header.stamp = stamp
        if e_vec is not None:
            e_vec_msg.vector.x = float(e_vec[0])
            e_vec_msg.vector.y = float(e_vec[1])
        else:
            e_vec_msg.vector.x = float('nan')
            e_vec_msg.vector.y = float('nan')
        e_vec_msg.vector.z = 0.0
        self.e_vec_pub.publish(e_vec_msg)

        self._publish_evec_marker(e_vec if e_vec is not None else [0.0, 0.0], stamp)

        poly_msg = PolygonStamped()
        poly_msg.header.frame_id = self.global_map_frame
        poly_msg.header.stamp = stamp
        if result['boundary_points'] is not None:
            for px, py in result['boundary_points']:
                p = Point32(); p.x = float(px); p.y = float(py); p.z = 0.0
                poly_msg.polygon.points.append(p)
        self.boundary_points_pub.publish(poly_msg)

        self._publish_proj_marker(self.proj_pub_rob, result['rob_proj_points'], 0, stamp)
        self._publish_proj_tie_marker(result['rob_tie_points'], stamp)
        self._publish_proj_marker(self.proj_pub_target, result['target_proj_points'], 1, stamp)
        self._publish_geodesic_paths(result['geodesic_xi_list'], result['geodesic_pi_list'], stamp)
        self._publish_line_samples(stamp)

    def _publish_evec_marker(self, e_vec, stamp):
        marker = Marker()
        marker.header.frame_id = self.global_map_frame
        marker.header.stamp = stamp
        marker.ns = "e_vec"; marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.rob_xy[0])
        marker.pose.position.y = float(self.rob_xy[1])
        p_start = Point(); p_start.x = p_start.y = p_start.z = 0.0
        p_end = Point(); p_end.x = float(e_vec[0]); p_end.y = float(e_vec[1]); p_end.z = 0.0
        marker.points = [p_start, p_end]
        marker.scale.x = 0.05; marker.scale.y = 0.1; marker.scale.z = 0.1
        marker.color.a = 1.0; marker.color.b = 1.0
        self.e_vec_marker_pub.publish(marker)

    def _publish_proj_marker(self, pub, points, marker_id, stamp):
        marker = Marker()
        marker.header.frame_id = self.global_map_frame
        marker.header.stamp = stamp
        marker.ns = "proj_points"; marker.id = marker_id
        if points is None:
            marker.action = Marker.DELETE
        else:
            marker.type = Marker.SPHERE; marker.action = Marker.ADD
            marker.pose.position.x = float(points[0]); marker.pose.position.y = float(points[1])
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.3
            if marker_id == 0:
                marker.color.r = 0.5; marker.color.a = 1.0
            else:
                marker.color.g = 0.5; marker.color.b = 0.5; marker.color.a = 1.0
        pub.publish(marker)

    def _publish_proj_tie_marker(self, points, stamp):
        marker = Marker()
        marker.header.frame_id = self.global_map_frame
        marker.header.stamp = stamp
        marker.ns = "proj_points_tie_candidates"; marker.id = 0
        if points is None or len(points) == 0:
            marker.action = Marker.DELETE
        else:
            marker.type = Marker.POINTS; marker.action = Marker.ADD
            marker.scale.x = marker.scale.y = 0.15
            marker.color.r = 1.0; marker.color.g = 0.65; marker.color.a = 1.0
            for px, py in points:
                pt = Point(); pt.x = float(px); pt.y = float(py); pt.z = 0.0
                marker.points.append(pt)
        self.proj_pub_rob_tie_candidates.publish(marker)

    def _publish_geodesic_paths(self, xi_list, pi_list, stamp):
        arr = MarkerArray()
        if xi_list is None:
            for dir_idx in range(2):
                marker = Marker()
                marker.header.frame_id = self.global_map_frame
                marker.header.stamp = stamp
                marker.ns = "geodesic_paths"; marker.id = dir_idx
                marker.action = Marker.DELETE
                arr.markers.append(marker)
            self.geodesic_paths_pub.publish(arr)
            return
        chosen = int(np.argmin(pi_list[:, 0])) if pi_list is not None else -1
        for dir_idx in range(2):
            marker = Marker()
            marker.header.frame_id = self.global_map_frame
            marker.header.stamp = stamp
            marker.ns = "geodesic_paths"; marker.id = dir_idx
            marker.type = Marker.LINE_STRIP; marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.color.a = 1.0
            marker.color.b = 1.0 if dir_idx == chosen else 0.0
            for step in range(xi_list.shape[1]):
                p = Point()
                p.x = float(xi_list[dir_idx, step, 0, 0])
                p.y = float(xi_list[dir_idx, step, 0, 1])
                marker.points.append(p)
            arr.markers.append(marker)
        self.geodesic_paths_pub.publish(arr)

    def _publish_line_samples(self, stamp):
        pts = self.om.line_sample_pts
        h_vals = self.om.line_sample_h
        marker = Marker()
        marker.header.frame_id = self.global_map_frame
        marker.header.stamp = stamp
        marker.ns = "line_samples"; marker.id = 0
        marker.type = Marker.SPHERE_LIST; marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 0.12
        if pts is None or h_vals is None:
            self.line_samples_pub.publish(marker)
            return
        for (px, py), h in zip(pts, h_vals):
            p = Point(); p.x = float(px); p.y = float(py); p.z = 0.0
            marker.points.append(p)
            c = ColorRGBA()
            c.a = 1.0
            if float(h) < -0.01:
                c.r = 1.0; c.g = 0.0; c.b = 0.0
            else:
                c.r = 0.0; c.g = 0.0; c.b = 1.0
            marker.colors.append(c)
        self.line_samples_pub.publish(marker)

    def destroy_node(self):
        self.running = False
        return super().destroy_node()


def main():
    rclpy.init()
    node = MCBFConstraintNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
