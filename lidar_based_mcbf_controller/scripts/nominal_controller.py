#!/usr/bin/env python3
import os
import threading
import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates

from ament_index_python.packages import get_package_share_directory

from lidar_based_mcbf.basic_boundary_function.configs import CircularRobotSpecification, EnvContrConfiguration
from lidar_based_mcbf_msgs.msg import StageConstraint


class NominalControllerNode(Node):
    def __init__(self):
        super().__init__('nominal_controller_node')

        pkg_share = get_package_share_directory('lidar_based_mcbf')
        econf_path = os.path.join(pkg_share, 'config', 'env_controller.yaml')
        spec_robot_fpath = os.path.join(pkg_share, 'config', 'spec_robot.yaml')

        self.declare_parameter("econf_path", econf_path)
        self.declare_parameter("spec_robot_fpath", spec_robot_fpath)
        self.declare_parameter("nominal_hz", 10.0)
        self.econf_path = self.get_parameter("econf_path").value
        self.spec_robot_fpath = self.get_parameter("spec_robot_fpath").value
        self._nominal_period = 1.0 / float(self.get_parameter("nominal_hz").value)

        self.econf = EnvContrConfiguration.from_yaml(self.econf_path)
        self.spec_robot = CircularRobotSpecification.from_yaml(self.spec_robot_fpath)
        self.rob_name = self.spec_robot.rob_name

        self.target = np.array(self.econf.target)
        self.nominal_speed = self.econf.nominal_speed
        self.dt = self.spec_robot.ts
        self.dynamics = self.econf.dynamics

        self.u_nom_pub = self.create_publisher(Twist, "/u_nom", 10)

        self.stage_sub = self.create_subscription(
            StageConstraint, "/stage_constraint", self.callback_stage, 1)
        self.odom_sub = self.create_subscription(
            ModelStates, "/gazebo/model_states", self.callback_rob, 1)

        self.rob_xy: Optional[np.ndarray] = None
        self.rob_theta: float = 0.0
        self._temp_target: Optional[np.ndarray] = None

        self.running = True
        self._nominal_thread = threading.Thread(target=self._nominal_loop, daemon=True)
        self._nominal_thread.start()

    def callback_rob(self, data):
        idx = data.name.index(self.rob_name)
        pose = data.pose[idx]
        self.rob_xy = np.array([pose.position.x, pose.position.y])
        quat = [pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w]
        self.rob_theta = float(R.from_quat(quat).as_euler("xyz")[2])

    def callback_stage(self, msg):
        if msg.has_temp_target:
            self._temp_target = np.array([msg.temp_target.x, msg.temp_target.y])
        else:
            self._temp_target = None

    def compute_u_nom(self):
        if self.rob_xy is None:
            return None

        active_target = self._temp_target if self._temp_target is not None else self.target
        v_xy = active_target[:2] - self.rob_xy
        dist = float(np.linalg.norm(v_xy))

        if self.dynamics == "differential":
            theta_current = (self.rob_theta + np.pi) % (2 * np.pi) - np.pi
            theta_goal = float(np.arctan2(v_xy[1], v_xy[0]))
            delta_theta = theta_goal - theta_current
            if abs(delta_theta) > np.pi:
                delta_theta = -np.sign(delta_theta) * (2 * np.pi - abs(delta_theta))

            speed = min(dist, self.nominal_speed)
            omega = float(np.clip(delta_theta / self.dt,
                                -self.spec_robot.ang_vel_max,
                                self.spec_robot.ang_vel_max))
            return speed, omega

        elif self.dynamics == "omni-directional":
            return float(v_xy[0]), float(v_xy[1])

        else:
            raise RuntimeError('Undefined robot dynamics.')

    def _nominal_loop(self):
        self.get_logger().info("Nominal control loop started")
        while rclpy.ok() and self.running:
            t0 = time.monotonic()

            u_nom = self.compute_u_nom()
            if u_nom is not None:
                msg = Twist()
                msg.linear.x = float(u_nom[0])
                msg.angular.z = float(u_nom[1])
                self.u_nom_pub.publish(msg)

            elapsed = time.monotonic() - t0
            sleep_t = self._nominal_period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def destroy_node(self):
        self.running = False
        return super().destroy_node()


def main():
    rclpy.init()
    node = NominalControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
