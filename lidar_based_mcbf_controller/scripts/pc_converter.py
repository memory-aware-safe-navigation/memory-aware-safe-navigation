#!/usr/bin/env python3


import os
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from collections import deque

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
import tf2_ros
from tf2_ros import TransformBroadcaster
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import TransformStamped
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from tf2_ros import TransformException

from scipy.spatial.transform import Rotation as R
from ament_index_python.packages import get_package_share_directory
from lidar_based_mcbf.basic_boundary_function.configs import CircularRobotSpecification
import numpy as np

class Cloud2ToCloud(Node):
    def __init__(self):
        super().__init__('pointcloud_convert_node')

        self.set_parameters([
            rclpy.parameter.Parameter(
                'use_sim_time',
                rclpy.Parameter.Type.BOOL,
                True
            )
        ])

        self.tf_broadcaster = TransformBroadcaster(self)

        self.rob_xy = np.zeros(2,)
        self.rob_theta = 0.0

        self.pub = self.create_publisher(PointCloud2, "/cloud_all_fields_fullframe_global", 10)
        self.sub = self.create_subscription(PointCloud2, "/husky/points", self.callback_lidar, 1)

        self.annotated_pub = self.create_publisher(PointCloud2, "/annotated_pointcloud_global", 10)
        self.annotated_sub = self.create_subscription(PointCloud2, "/annotated_pointcloud", self.callback_annotated, 1)

        self.gazebo_sub = self.create_subscription(ModelStates, "/gazebo/model_states", self.callback_rob, 1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        spec_robot_fpath = os.path.join(get_package_share_directory('lidar_based_mcbf'), 'config', 'spec_robot.yaml')
        self.rob_name = CircularRobotSpecification.from_yaml(spec_robot_fpath).rob_name

        self.pending_msgs = deque()
        self.pending_annotated_msgs = deque()
        self.stale_msg_drop_time = 0.5
        self.tf_check_timer = self.create_timer(0.02, self.process_pending)

        self.get_logger().info("Converting local pointcloud to global pointcloud")

    def callback_rob(self, data):

        if self.rob_name not in data.name:
            self.get_logger().warn(f"{self.rob_name} not found in /gazebo/model_states")
            return

        idx = data.name.index(self.rob_name)
        pose = data.pose[idx]

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "mocap"
        t.child_frame_id = "base_link"

        t.transform.translation.x = float(pose.position.x)
        t.transform.translation.y = float(pose.position.y)
        t.transform.translation.z = float(pose.position.z)

        t.transform.rotation.x = float(pose.orientation.x)
        t.transform.rotation.y = float(pose.orientation.y)
        t.transform.rotation.z = float(pose.orientation.z)
        t.transform.rotation.w = float(pose.orientation.w)

        self.tf_broadcaster.sendTransform(t)


    def callback_lidar(self, msg: PointCloud2):
        self.pending_msgs.append((msg, self.get_clock().now()))

    def callback_annotated(self, msg: PointCloud2):
        self.pending_annotated_msgs.append((msg, self.get_clock().now()))

    def process_pending(self):
        self._process_pending_queue(self.pending_msgs, self.pub)
        self._process_pending_queue(self.pending_annotated_msgs, self.annotated_pub)

    def _process_pending_queue(self, pending_msgs, publisher):
        while pending_msgs:
            msg, enqueue_time = pending_msgs[0]
            stamp = rclpy.time.Time.from_msg(msg.header.stamp)

            age = (self.get_clock().now() - enqueue_time).nanoseconds * 1e-9
            if age > self.stale_msg_drop_time:
                pending_msgs.popleft()
                continue

            if not self.tf_buffer.can_transform(
                "mocap",
                msg.header.frame_id,
                stamp,
                timeout=rclpy.duration.Duration(seconds=0.0)
            ):
                return

            try:
                transform = self.tf_buffer.lookup_transform(
                    "mocap",
                    msg.header.frame_id,
                    stamp,
                    timeout=rclpy.duration.Duration(seconds=0.0)
                )
            except TransformException as ex:
                self.get_logger().warn(f"Could not transform cloud: {ex}")
                pending_msgs.popleft()
                return

            pending_msgs.popleft()
            cloud_global = do_transform_cloud(msg, transform)
            cloud_global.header.frame_id = "mocap"
            cloud_global.header.stamp = msg.header.stamp
            publisher.publish(cloud_global)

def main(args=None):
    rclpy.init(args=args)
    node = Cloud2ToCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass

if __name__ == '__main__':
    main()
