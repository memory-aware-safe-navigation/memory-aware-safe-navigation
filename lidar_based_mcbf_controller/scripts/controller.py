#!/usr/bin/env python3
import os
import threading
import time
from typing import Optional
import numpy as np
import cvxpy as cp
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32, Bool, String
from geometry_msgs.msg import Twist, PoseStamped
from gazebo_msgs.msg import ModelStates
from visualization_msgs.msg import Marker, MarkerArray

from ament_index_python.packages import get_package_share_directory

from lidar_based_mcbf.basic_boundary_function.configs import (
    CircularRobotSpecification, EnvContrConfiguration,
)
from lidar_based_mcbf.basic_boundary_function.env_approx import Env_Approx
from lidar_based_mcbf_msgs.msg import GPDFSet, StageConstraint, MCBFConstraint

class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_medium')

        pkg_share = get_package_share_directory('lidar_based_mcbf')
        econf_path = os.path.join(pkg_share, 'config', 'env_controller.yaml')
        spec_robot_fpath = os.path.join(pkg_share, 'config', 'spec_robot.yaml')

        self.declare_parameter("econf_path", econf_path)
        self.declare_parameter("spec_robot_fpath", spec_robot_fpath)
        self.declare_parameter("n", 3)
        self.declare_parameter('disable_cmd', False)
        self.declare_parameter('control_hz', 10)
        self.declare_parameter('baseline_eval', False)

        self.econf_path = self.get_parameter("econf_path").value
        self.spec_robot_fpath = self.get_parameter("spec_robot_fpath").value
        self.n = self.get_parameter("n").value
        self.mapping_only = self.get_parameter('disable_cmd').get_parameter_value().bool_value
        control_hz = float(self.get_parameter('control_hz').value)
        self.baseline_eval = self.get_parameter('baseline_eval').get_parameter_value().bool_value

        self._odom_cb_group = MutuallyExclusiveCallbackGroup()
        self._gpdf_cb_group = MutuallyExclusiveCallbackGroup()
        self._ctrl_cb_group = MutuallyExclusiveCallbackGroup()
        self._stage_cb_group = MutuallyExclusiveCallbackGroup()
        self._u_nom_cb_group = MutuallyExclusiveCallbackGroup()
        self._mcbf_cb_group = MutuallyExclusiveCallbackGroup()

        self.sub_rob_odom = self.create_subscription(
            ModelStates, "/gazebo/model_states", self.callback_rob, 1,
            callback_group=self._odom_cb_group)
        self.gpdf_set_sub = self.create_subscription(
            GPDFSet, "/gpdf_sets", self.callback_gpdf_set, 1,
            callback_group=self._gpdf_cb_group)
        self.stage_sub = self.create_subscription(
            StageConstraint, "/stage_constraint", self.callback_stage, 1,
            callback_group=self._stage_cb_group)
        self.u_nom_sub = self.create_subscription(
            Twist, "/u_nom", self.callback_u_nom, 1,
            callback_group=self._u_nom_cb_group)
        self.mcbf_sub = self.create_subscription(
            MCBFConstraint, "/mcbf_constraint", self.callback_mcbf, 1,
            callback_group=self._mcbf_cb_group)

        self.dynamic_obstacle_publisher = self.create_publisher(Bool, "/dynamic_obstacle", 10)
        self.rob_pose_publisher = self.create_publisher(PoseStamped, "/rob_pose", 10)
        self.vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.state_publisher = self.create_publisher(Twist, "/cmd_states", 10)
        self.dt_publisher = self.create_publisher(Float32,  "/cmd_dt",  10)
        self.solve_dt_publisher = self.create_publisher(Float32, "/controller_solve_dt", 10)
        self.loop_gap_dt_publisher = self.create_publisher(Float32, "/controller_loop_gap_dt", 10)
        self.debug_publish_dt_publisher = self.create_publisher(Float32, "/controller_debug_publish_dt", 10)
        self.publish_dt_publisher = self.create_publisher(Float32, "/controller_publish_dt", 10)
        self.qp_dt_publisher = self.create_publisher(Float32, "/controller_qp_dt", 10)
        self.sleep_target_dt_publisher = self.create_publisher(Float32, "/controller_sleep_target_dt", 10)
        self.sleep_actual_dt_publisher = self.create_publisher(Float32, "/controller_sleep_actual_dt", 10)
        self.rect_clusters_pub = self.create_publisher(MarkerArray, '/rect_clusters', 10)
        self.feasibility_marker_pub = self.create_publisher(Marker, '/solver_status_marker', 10)
        self.infeasibility_debug_pub = self.create_publisher(String, '/controller_infeasibility_debug', 10)
        self.collision_marker_pub = self.create_publisher(Marker, '/collision_status_marker', 10)
        self.robot_h_pub = self.create_publisher(Float32, '/robot_h', 10)
        self.rect_mask_pub = self.create_publisher(Marker, '/rect_mask', 10)

        self.rob_xy = np.zeros(2)
        self.rob_quat = [0, 0, 0, 1]
        self.rob_x = np.zeros(self.n)
        self.rob_theta = 0.0
        self.rob_states = np.zeros(2)
        self.time = time.monotonic()
        self._last_loop_end_t = None
        self._last_solve_dt = 0.0
        self._spin_margin = 0.01

        self.no_map_yet = True
        self.global_map_frame = "mocap"
        self._latest_stage_msg: Optional[StageConstraint] = None
        self._latest_u_nom_msg: Optional[Twist] = None
        self._latest_mcbf_msg: Optional[MCBFConstraint] = None

        self.TIMEOUT = 4000

        self.econf = EnvContrConfiguration.from_yaml(self.econf_path)
        self.spec_robot = CircularRobotSpecification.from_yaml(self.spec_robot_fpath)
        self.rob_name = self.spec_robot.rob_name

        self.ROBOT_TARGET = np.array(self.econf.target)
        self.SCENARIO = self.econf.env_name

        self.shared_lock = threading.RLock()

        self.env_approx = Env_Approx(self.shared_lock, original_env=self.SCENARIO, radius=self.econf.radius)

        self.debug_info = {}
        self.timing_qp_dt = 0.0
        self._last_infeasibility_diagnostics = None

        self._kt = 0
        self._ctrl_running = True
        self._ctrl_period = 1.0 / control_hz
        self._ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._ctrl_thread.start()

        self.prev_u_mod = None

        self.get_logger().info(f"Control loop thread started at {control_hz:.1f} Hz target")

    def callback_rob(self, data):
        idx  = data.name.index(self.rob_name)
        pose = data.pose[idx]
        twist = data.twist[idx]

        speed = np.linalg.norm([twist.linear.x, twist.linear.y]) or 0.01
        self.rob_xy    = [pose.position.x, pose.position.y]
        self.rob_quat  = [pose.orientation.x, pose.orientation.y,
                          pose.orientation.z, pose.orientation.w]
        e = R.from_quat(self.rob_quat).as_euler("xyz")
        self.rob_theta = e[2]
        if self.n == 5:
            self.rob_x = [pose.position.x, pose.position.y, speed, e[2], twist.angular.z]
        else:
            self.rob_x = [pose.position.x, pose.position.y, e[2]]
        self.rob_states = [speed, twist.angular.z]

    def callback_gpdf_set(self, msg):
        self.env_approx.update_from_msg(msg)
        if self.no_map_yet:
            self.get_logger().info("First GPDFSet received — control loop starting")
            self.no_map_yet = False

    def callback_stage(self, msg):
        self._latest_stage_msg = msg
        if msg.has_rect_bounds:
            self.debug_info['rect_bounds'] = (
                msg.rect_center.x, msg.rect_center.y, msg.rect_yaw, msg.rect_length, msg.rect_width,
            )
        else:
            self.debug_info['rect_bounds'] = None

    def callback_u_nom(self, msg):
        self._latest_u_nom_msg = msg

    def callback_mcbf(self, msg):
        self._latest_mcbf_msg = msg

    def _get_u_nom(self) -> Optional[np.ndarray]:
        if self._latest_u_nom_msg is None:
            return None
        return np.array([self._latest_u_nom_msg.linear.x, self._latest_u_nom_msg.angular.z])

    def _current_stage_info(self) -> dict:
        msg = self._latest_stage_msg
        if msg is None:
            return {
                'stage': 1,
                'is_confined': False,
                'stage3_latched': False,
                'margin': 0.25,
                'effective_om_expand_threshold': self.econf.om_expand_threshold,
            }
        return {
            'stage': int(msg.stage),
            'is_confined': bool(msg.is_confined),
            'stage3_latched': bool(msg.stage3_latched),
            'margin': float(msg.margin),
            'effective_om_expand_threshold': float(msg.effective_om_expand_threshold),
        }

    def func_f(self, x):
        if self.econf.dynamics == "differential":
            return np.zeros((3, 1))
        elif self.econf.dynamics == "omni-directional":
            return np.zeros((2, 1))
        else:
            raise RuntimeError('Undefined robot dynamics.')

    def func_g(self, x, a=0):
        if self.econf.dynamics == "differential":
            g = np.zeros((3, 2))
            g[0][0] = np.cos(x[2])
            g[0][1] = -a*np.sin(x[2])
            g[1][0] = np.sin(x[2])
            g[1][1] = a*np.cos(x[2])
            g[2][1] = 1
            return g
        elif self.econf.dynamics == "omni-directional":
            return np.eye(2)
        else:
            raise RuntimeError('Undefined robot dynamics.')

    @staticmethod
    def process_shifted(x, a = 0):
        x_shited = np.zeros(3,)
        x_shited[0] = x[0] + a*np.cos(x[2])
        x_shited[1] = x[1] + a*np.sin(x[2])
        x_shited[2] = x[2]
        return x_shited

    def _solve_qp(self, u_nom: np.ndarray, stage_info: dict):
        env = self.env_approx
        dynamics = self.econf.dynamics
        MCBF = self.econf.MCBF

        stage = stage_info['stage']
        is_confined = stage_info['is_confined']
        margin = stage_info['margin']
        effective_om_expand_threshold = stage_info['effective_om_expand_threshold']
        self.debug_info['om_stage'] = stage
        self.debug_info['effective_om_range'] = effective_om_expand_threshold
        self.debug_info['is_confined'] = is_confined

        if stage == 2 or is_confined:
            alpha_coef = 0.7
            margin = 0.05
        else:
            alpha_coef = 0.5

        effective_ve = self.econf.ve

        with self.shared_lock:
            state = np.asarray(self.rob_x, dtype=float)
            state_xy = state[:2]
            if dynamics == "differential":
                a = 0.2
                x = self.process_shifted(state, a).reshape(1, 3)
            else:
                a = 0.0
                x = state.reshape(1, -1)

            if env.precomp_h_robot is not None and not env.mmp and env.num_dyn_circle == 0:
                h_rc_set = env.precomp_h_robot
                grad_set = env.precomp_grad_shifted
                gradt_set = np.zeros_like(h_rc_set)
            else:
                _, grad_set, gradt_set, _, _ = env.h_grad_set(x[:, :2])
                h_rc_set, *_ = env.h_grad_set(state_xy.reshape(1, 2))

            collision_h_threshold = -0.05
            num_dyn_total = env.num_dyn_mmp + env.num_dyn_circle
            dynamic_h_vals = h_rc_set[:num_dyn_total] if num_dyn_total > 0 else h_rc_set[:0]
            static_h_vals = h_rc_set[num_dyn_total:]
            collision_dynamic = bool(dynamic_h_vals.size and np.min(dynamic_h_vals) <= collision_h_threshold)
            collision_static = bool(static_h_vals.size and np.min(static_h_vals) <= collision_h_threshold)
            self.debug_info['collision_dynamic'] = collision_dynamic
            self.debug_info['collision_static'] = collision_static
            self.debug_info['collision'] = collision_dynamic or collision_static
            self.debug_info['h_robot'] = float(np.min(h_rc_set)) if h_rc_set.size else float('nan')

            dynamic_obstacle_fixed_margin = 0.2
            approach_dyn_vel_scaling = 1.5
            dynamic_obstacle = False
            if num_dyn_total > 0:
                h_c = h_rc_set[:len(h_rc_set) - len(env.xc_ref)]
                approaching_mask = gradt_set[:num_dyn_total] < -0.01
                gradt_set[:num_dyn_total][approaching_mask] = (
                    gradt_set[:num_dyn_total][approaching_mask] * approach_dyn_vel_scaling)
                nearby_idx = np.where(h_c.flatten() <= 4.0)[0]
                for idx in nearby_idx:
                    if (h_c[idx] - dynamic_obstacle_fixed_margin) + gradt_set[idx] < 2.5 and gradt_set[idx] < -0.1:
                        dynamic_obstacle = True
                        break
            self.debug_info['dynamic_obstacle'] = dynamic_obstacle

            if dynamic_obstacle:
                margin = 0.25
                alpha_coef = 0.5

            u_mod = cp.Variable(len(u_nom))
            dx = self.func_f(x.flatten()) + self.func_g(x.flatten(), a) @ u_mod

            active_obs_idx = np.where(h_rc_set.flatten() <= self.spec_robot.lidar_range)[0]
            obstacle_exprs = [
                (int(idx), (
                    grad_set[idx] @ dx[:2]
                    + alpha_coef * (h_rc_set.flatten()[idx] - margin)
                    + gradt_set.flatten()[idx]
                ))
                for idx in active_obs_idx
            ]
            obstacle_constraints = [(idx, expr >= 0) for idx, expr in obstacle_exprs]
            constraints = [c for _, c in obstacle_constraints]

            if dynamics == "differential":
                obj = cp.Minimize((u_mod[0] - u_nom[0]) ** 2 + 1E-4 * (u_mod[1] - u_nom[1]) ** 2)
                actuator_constraints = (
                                    [u_mod[1] <= self.spec_robot.ang_vel_max] + [u_mod[1] >= -self.spec_robot.ang_vel_max]
                                    + [u_mod[0] >= self.spec_robot.lin_vel_min] + [u_mod[0] <= self.spec_robot.lin_vel_max]
                                )
            else:
                obj = cp.Minimize((u_mod[0] - u_nom[0]) ** 2 + (u_mod[1] - u_nom[1]) ** 2)
                actuator_constraints = (
                    [u_mod[1] <= self.spec_robot.lin_vel_max] + [u_mod[1] >= self.spec_robot.lin_vel_min]
                    + [u_mod[0] >= self.spec_robot.lin_vel_min] + [u_mod[0] <= self.spec_robot.lin_vel_max]
                )
            constraints += actuator_constraints

            mcbf_msg = self._latest_mcbf_msg
            e_vec = None
            v_b_e = None
            if MCBF and mcbf_msg is not None and mcbf_msg.valid:
                e_vec = np.array([mcbf_msg.e_vec.x, mcbf_msg.e_vec.y])
                v_b_e = e_vec @ dx[:2]
                self.debug_info['om_idx'] = int(mcbf_msg.om_idx)
                self.debug_info['beta'] = float(mcbf_msg.beta)
                self.debug_info['e_vec'] = e_vec.tolist()
                self.debug_info['grad_value'] = (
                    grad_set[mcbf_msg.om_idx].tolist() if 0 <= mcbf_msg.om_idx < len(grad_set) else [])
            else:
                self.debug_info['om_idx'] = -1
                self.debug_info['beta'] = None
                self.debug_info['e_vec'] = None

            u_mod_value = np.zeros(u_nom.shape)
            prob = None
            mcbf_constraint = None
            qp_start = time.time()
            for i in range(2):
                if MCBF and e_vec is not None:
                    if i == 1:
                        effective_ve = effective_ve / 2
                    if dynamics == "differential":
                        mcbf_constraint = v_b_e >= effective_ve * self.spec_robot.lin_vel_max
                    else:
                        mcbf_constraint = v_b_e >= effective_ve * np.linalg.norm(
                            [self.spec_robot.lin_vel_max, self.spec_robot.lin_vel_min])

                    if dynamic_obstacle and i == 1:
                        prob = cp.Problem(obj, constraints)
                    else:
                        new_constraints = constraints + [mcbf_constraint]
                        prob = cp.Problem(obj, new_constraints)
                else:
                    prob = cp.Problem(obj, constraints)

                try:
                    prob.solve()
                except cp.SolverError:
                    prob.solve(solver=cp.SCS)

                if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    u_mod_value = u_mod.value
                    if not MCBF:
                        self.prev_u_mod = u_mod_value
                    else:
                        self.prev_u_mod = None
                    break
                else:
                    u_mod_value = np.zeros(u_nom.shape)
                    if not MCBF:
                        if self.prev_u_mod is not None:
                            u_mod_value = self.prev_u_mod
                        break
                    else:
                        self.prev_u_mod = None
            self.timing_qp_dt = time.time() - qp_start


            is_infeasible = prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            self._last_infeasibility_diagnostics = None

        status = {
            'isSuccess': False,
            'isSafe': True,
            'isInfeasible': is_infeasible,
        }
        return u_mod_value, status

    def _solve_slack_qp(self, u_mod, u_nom, obstacle_exprs, actuator_constraints, num_dyn_total,
                         dynamic_weight=50.0, static_weight=1.0, tracking_weight=1e-3):
        n = len(obstacle_exprs)
        if n == 0:
            return None, None

        slacks = cp.Variable(n, nonneg=True)
        weights = np.array([
            dynamic_weight if idx < num_dyn_total else static_weight
            for idx, _ in obstacle_exprs
        ])
        soft_constraints = list(actuator_constraints) + [
            expr + slacks[k] >= 0 for k, (_, expr) in enumerate(obstacle_exprs)
        ]
        obj = cp.Minimize(weights @ slacks + tracking_weight * cp.sum_squares(u_mod - u_nom))
        prob = cp.Problem(obj, soft_constraints)
        try:
            prob.solve()
        except cp.SolverError:
            prob.solve(solver=cp.SCS)

        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return u_mod.value, slacks.value
        return None, None


    def _control_loop(self):
        self.get_logger().info("Control loop thread started")

        while rclpy.ok() and self._ctrl_running and self.no_map_yet:
            time.sleep(self._ctrl_period)

        while rclpy.ok() and self._ctrl_running:

            loop_start_t = time.time()
            if self._last_loop_end_t is not None:
                loop_gap_msg = Float32()
                loop_gap_msg.data = float(loop_start_t - self._last_loop_end_t)
                self.loop_gap_dt_publisher.publish(loop_gap_msg)

            if len(self.env_approx.gpdf_set) == 0 and self.env_approx.num_dyn_circle == 0:
                u_nom = self._get_u_nom()
                if u_nom is None:
                    if not self.mapping_only:
                        self.publish_rob_speed(0.0, 0.0)
                    self._last_loop_end_t = time.time()
                    continue
                if not self.mapping_only:
                    self.publish_rob_speed(u_nom[0], u_nom[1])
                self._last_loop_end_t = time.time()
                continue

            if self._kt >= self.TIMEOUT:
                self.get_logger().info("TIMEOUT — stopping control loop")
                self._ctrl_running = False
                self.publish_rob_speed(0.0, 0.0)
                return

            if np.linalg.norm(np.asarray(self.rob_xy, dtype=float) - self.ROBOT_TARGET[:2]) < 0.4:
                self.get_logger().info("Target reached — stopping control loop")
                self._ctrl_running = False
                self.publish_rob_speed(0.0, 0.0)
                return

            u_nom = self._get_u_nom()
            if u_nom is None:
                self.publish_rob_speed(0.0, 0.0)
                self._last_loop_end_t = time.time()
                continue

            t_start = time.time()
            u_mod, controller_status = self._solve_qp(u_nom, self._current_stage_info())
            solve_ms = (time.time() - t_start) * 1e3
            solve_dt_msg = Float32()
            solve_dt_msg.data = float(solve_ms / 1e3)
            self.solve_dt_publisher.publish(solve_dt_msg)
            self._last_solve_dt = solve_dt_msg.data

            qp_dt_msg = Float32()
            qp_dt_msg.data = float(self.timing_qp_dt)
            self.qp_dt_publisher.publish(qp_dt_msg)

            if controller_status['isInfeasible']:
                if self.mapping_only:
                    self.publish_rob_speed(0.0, 0.0)
                else:
                    self.publish_rob_speed(u_mod[0], u_mod[1])
                self.publish_rob_states()
            else:
                if self.mapping_only:
                    self.publish_rob_speed(0.0, 0.0)
                else:
                    self.publish_rob_speed(u_mod[0], u_mod[1])
                self.publish_rob_states()

            t_pub_start = time.time()
            self.publish_rob_pose()
            self.publish_dynamic_obstacle()
            self.publish_rect_mask()
            self.publish_feasibility_marker(controller_status['isInfeasible'])
            self.publish_collision_marker()
            self.publish_robot_h()
            pub_dt_msg = Float32()
            pub_dt_msg.data = float(time.time() - t_pub_start)
            self.publish_dt_publisher.publish(pub_dt_msg)
            self._kt += 1
            self._last_loop_end_t = time.time()

    def publish_rob_speed(self, speed, angular_velocity):
        elapsed = time.monotonic() - self.time
        sleep_t = self._ctrl_period - elapsed
        if sleep_t > 0:
            t_sleep_start = time.perf_counter()
            deadline = t_sleep_start + sleep_t
            coarse_sleep = sleep_t - self._spin_margin
            if coarse_sleep > 0:
                time.sleep(coarse_sleep)
            while time.perf_counter() < deadline:
                pass
            actual_sleep = time.perf_counter() - t_sleep_start
        else:
            sleep_t = 0.0
            actual_sleep = 0.0

        sleep_target_msg = Float32()
        sleep_target_msg.data = float(sleep_t)
        self.sleep_target_dt_publisher.publish(sleep_target_msg)

        sleep_actual_msg = Float32()
        sleep_actual_msg.data = float(actual_sleep)
        self.sleep_actual_dt_publisher.publish(sleep_actual_msg)

        debug_dt_msg = Float32()
        debug_dt_msg.data = float(self._last_solve_dt + actual_sleep)
        self.debug_publish_dt_publisher.publish(debug_dt_msg)

        msg = Twist()
        msg.linear.x  = speed
        msg.angular.z = angular_velocity
        self.vel_publisher.publish(msg)
        t = Float32()
        now = time.monotonic()
        t.data = now - self.time
        self.dt_publisher.publish(t)
        self.time = now

    def publish_rob_states(self):
        msg = Twist()
        msg.linear.x  = self.rob_states[0]
        msg.angular.z = self.rob_states[1]
        msg.angular.x = self.rob_x[0]
        msg.angular.y = self.rob_x[1]
        self.state_publisher.publish(msg)

    def publish_dynamic_obstacle(self):
        msg = Bool()
        msg.data = bool(self.debug_info.get('dynamic_obstacle', False))
        self.dynamic_obstacle_publisher.publish(msg)

    def publish_rob_pose(self):
        msg = PoseStamped()
        msg.header.frame_id = self.global_map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(self.rob_xy[0])
        msg.pose.position.y = float(self.rob_xy[1])
        msg.pose.orientation.x = float(self.rob_quat[0])
        msg.pose.orientation.y = float(self.rob_quat[1])
        msg.pose.orientation.z = float(self.rob_quat[2])
        msg.pose.orientation.w = float(self.rob_quat[3])
        self.rob_pose_publisher.publish(msg)

    def publish_rect_mask(self):
        bounds = self.debug_info.get('rect_bounds')

        marker = Marker()
        marker.header.frame_id = self.global_map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rect_mask"; marker.id = 0

        if bounds is None:
            marker.action = Marker.DELETE
            self.rect_mask_pub.publish(marker)
            return

        if self.baseline_eval:
            stage_color = (1.0, 0.5, 0.0)
        else:
            stage = self.debug_info.get('om_stage', 1)
            if stage == 2 and self.debug_info.get('is_confined', False):
                stage_color = (0.6, 0.0, 0.8)
            else:
                stage_color = {
                    1: (0.0, 1.0, 0.0),
                    2: (1.0, 0.0, 0.0),
                }.get(stage, (0.2, 0.6, 1.0))

            if self.debug_info.get('dynamic_obstacle', False):
                stage_color = (0.0, 0.4, 0.9)

        center_x, center_y, yaw, length, width = bounds
        quat = R.from_euler('z', yaw).as_quat()

        marker.type = Marker.CUBE; marker.action = Marker.ADD
        marker.pose.position.x = center_x
        marker.pose.position.y = center_y
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        marker.scale.x = length
        marker.scale.y = width
        marker.scale.z = 0.02
        marker.color.r, marker.color.g, marker.color.b = stage_color
        marker.color.a = 0.15

        self.rect_mask_pub.publish(marker)

    def publish_feasibility_marker(self, is_infeasible: bool):
        marker = Marker()
        marker.header.frame_id = self.global_map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "solver_status"; marker.id = 0
        marker.type = Marker.CYLINDER; marker.action = Marker.ADD
        marker.pose.position.x = float(self.rob_xy[0])
        marker.pose.position.y = float(self.rob_xy[1])
        marker.pose.position.z = 0.8
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.4; marker.scale.y = 0.4; marker.scale.z = 0.05
        marker.color.a = 1.0
        if is_infeasible:
            marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0
        else:
            marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0
        self.feasibility_marker_pub.publish(marker)

    def publish_collision_marker(self):
        collision_static = self.debug_info.get('collision_static', False)
        collision_dynamic = self.debug_info.get('collision_dynamic', False)
        marker = Marker()
        marker.header.frame_id = self.global_map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "collision_status"; marker.id = 0
        marker.type = Marker.CUBE; marker.action = Marker.ADD
        marker.pose.position.x = float(self.rob_xy[0]-0.4)
        marker.pose.position.y = float(self.rob_xy[1])
        marker.pose.position.z = 1.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3; marker.scale.y = 0.3; marker.scale.z = 0.3
        marker.color.a = 1.0
        if collision_dynamic:
            marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 1.0
        elif collision_static:
            marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0
        else:
            marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0
        self.collision_marker_pub.publish(marker)

    def publish_robot_h(self):
        h_robot = self.debug_info.get('h_robot')
        if h_robot is None:
            return
        msg = Float32()
        msg.data = float(h_robot)
        self.robot_h_pub.publish(msg)

    def destroy_node(self):
        self._ctrl_running = False
        return super().destroy_node()

def main():
    rclpy.init()
    node = ControllerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
