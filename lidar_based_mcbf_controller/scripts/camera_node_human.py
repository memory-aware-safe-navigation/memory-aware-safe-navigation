#!/usr/bin/env python3
import os

import rclpy
from rclpy.node import Node

import numpy as np
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge

from sensor_msgs.msg import PointCloud2, Image, CameraInfo, PointField
from sensor_msgs_py import point_cloud2
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Header

from tf2_ros import Buffer, TransformListener
from message_filters import Subscriber, ApproximateTimeSynchronizer
import time

from ament_index_python.packages import get_package_share_directory

from ultralytics import YOLO
from lidar_based_mcbf.basic_boundary_function.configs import CircularRobotSpecification
import cv2

class CameraSubscriber_human(Node):
    def __init__(self):
        super().__init__('camera_rgbd_human')

        self.bridge = CvBridge()

        pkg_share = get_package_share_directory('lidar_based_mcbf')
        spec_robot_fpath = os.path.join(pkg_share, 'config', 'spec_robot.yaml')
        self.rob_name = CircularRobotSpecification.from_yaml(spec_robot_fpath).rob_name
        default_model_path = os.path.join(pkg_share, 'models', 'yolo26n-seg.engine')
        self.declare_parameter('model_path', default_model_path)
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.model = YOLO(model_path)

        self.declare_parameter('semantic_max_range', 5.0)
        self.semantic_max_range = self.get_parameter('semantic_max_range').get_parameter_value().double_value

        self.human_detected = False

        self.image_hum = None

        self.publish_annotated_imgs = True
        self.annotated_img = None
        self.publish_overlay_msg = False

        self.create_subscription(CameraInfo, "/camera/camera_info", self.callback_caminfo, 1)
        self.imagesub_hum = Subscriber(self, Image, '/camera/image_raw')
        self.lidar_sub = Subscriber(self, PointCloud2, '/husky/points')
        self.sync2 = ApproximateTimeSynchronizer(
            [self.imagesub_hum, self.lidar_sub],
            queue_size=1,
            slop=0.5,
            allow_headerless=False
        )
        self.sync2.registerCallback(self.callback_human)

        self.annotated_pc_publisher = self.create_publisher(PointCloud2, '/annotated_pointcloud', 1)

        self.sub_rob_odom = self.create_subscription(ModelStates, "/gazebo/model_states", self.callback_rob, 1)

        self.n = 3
        self.obs_num = 3
        self.rob_x = np.zeros(self.n,)

        self.cam_matrix = None

        self.mask_list = []
        self.mask_list_correspondingID = []

        self.annotated_image_publisher = self.create_publisher(Image, "/annotated_rbg", 10)
        self.lidar_overlay_image_publisher = self.create_publisher(Image, "/overlay_rbg", 1)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.lidar_to_cam = self._lookup_matrix("front_rgbd_optical_frame", "lidar_link")

        self.get_logger().info("YOLO human detection node running")

    def _lookup_matrix(self, target_frame, source_frame, timeout_sec=5.0):
        transform = self.tf_buffer.lookup_transform(
            target_frame, source_frame, rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=timeout_sec)
        )
        return self._transform_to_matrix(transform)

    @staticmethod
    def _transform_to_matrix(transform):
        t = transform.transform.translation
        q = transform.transform.rotation
        T = np.eye(4)
        T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T[:3, 3] = [t.x, t.y, t.z]
        return T

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

    def callback_caminfo(self,data):
        if self.cam_matrix is None:
            self.cam_matrix = np.array(data.k).reshape(3,3)
            self.image_shape = [data.width, data.height]

    def callback_human(self, msg1, msg2):
        self.get_logger().info("Human callback triggered")

        if self.cam_matrix is None:
            self.get_logger().warn("Camera info not received yet; skipping frame", throttle_duration_sec=2.0)
            return

        start_time = time.time()
        self.mask_list.clear()
        self.mask_list_correspondingID.clear()
        self.human_detected = False

        self.image_hum = self.bridge.imgmsg_to_cv2(msg1, desired_encoding='bgr8')

        self.results = self.model.track(self.image_hum, classes = [0], persist = True)

        combined_mask = np.zeros((self.image_shape[1], self.image_shape[0]), dtype = np.uint8)

        if self.results[0].masks is not None and self.results[0].boxes is not None:
            for box, mask in zip(self.results[0].boxes, self.results[0].masks.data):

                if box.id is None:
                    continue

                obj_id = int(box.id.item())
                mask_binary = mask.cpu().numpy()
                mask_binary = (mask_binary > 0.7).astype(np.uint8)

                mask_binary_resized = cv2.resize(mask_binary, (self.image_shape[0], self.image_shape[1]), interpolation=cv2.INTER_NEAREST)

                self.mask_list.append(mask_binary_resized)
                self.mask_list_correspondingID.append(obj_id)

                self.human_detected = True

                combined_mask = np.logical_or(combined_mask, mask_binary_resized).astype(np.uint8)

        self.segmentation = combined_mask

        if self.publish_annotated_imgs:
            self.annotated_img = self.results[0].plot()
            annotated_img_msg = self.bridge.cv2_to_imgmsg(self.annotated_img, encoding='bgr8')
            self.annotated_image_publisher.publish(annotated_img_msg)

        if self.image_hum is not None:
            points = point_cloud2.read_points_numpy(msg2, field_names=("x", "y", "z"))
            xyz_array = points.T

            xyz_array = xyz_array[:, ~np.isnan(xyz_array).any(axis=0)]
            n_points = xyz_array.shape[1]

            xyz_homo = np.vstack((xyz_array, np.ones(n_points)))
            xyz_cam_frame = self.lidar_to_cam @ xyz_homo

            front_mask = xyz_cam_frame[2, :] >= 0
            front_idx = np.where(front_mask)[0]

            lidar_as_pixels = self.cam_matrix @ xyz_cam_frame[:3, front_mask]
            lidar_as_pixels = (lidar_as_pixels / xyz_cam_frame[2, front_mask]).astype(int)

            u = lidar_as_pixels[1]
            v = lidar_as_pixels[0]

            bounds_mask = (u >= 0) & (u < self.image_shape[1]) & (v >= 0) & (v < self.image_shape[0])
            u_valid = u[bounds_mask]
            v_valid = v[bounds_mask]

            visible_idx = front_idx[bounds_mask]

            if self.publish_overlay_msg:
                for i in range(len(v_valid)):
                    x = int(v_valid[i])
                    y = int(u_valid[i])
                    cv2.circle(self.image_hum, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
                overlay_img_msg = self.bridge.cv2_to_imgmsg(self.image_hum, encoding='bgr8')
                self.lidar_overlay_image_publisher.publish(overlay_img_msg)

            semantic_valid = np.zeros(n_points, dtype=bool)
            semantic_valid[visible_idx] = True

            is_human = np.zeros(n_points, dtype=bool)
            is_human_id = np.zeros(n_points, dtype=int)

            if self.human_detected:
                is_human[visible_idx] = self.segmentation[u_valid, v_valid] > 0

                for id, mask in zip(self.mask_list_correspondingID, self.mask_list):
                    human_mask = mask[u_valid, v_valid] > 0
                    is_human_id[visible_idx[human_mask]] = id

            far_mask = np.hypot(xyz_array[0, :], xyz_array[1, :]) > self.semantic_max_range
            semantic_valid[far_mask] = False
            is_human[far_mask] = False
            is_human_id[far_mask] = 0

            self.publish_labeled_point_cloud(
                xyz_array,
                is_human,
                is_human_id,
                semantic_valid,
                frame=msg2.header.frame_id,
                stamp=msg2.header.stamp)

    def publish_labeled_point_cloud(
            self,
            reconstructed_lidar_pts,
            is_human,
            is_human_id,
            semantic_valid,
            frame="lidar_link",
            stamp=None):
        header = Header()
        header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        header.frame_id = frame

        points = np.vstack((
            reconstructed_lidar_pts[0, :],
            reconstructed_lidar_pts[1, :],
            reconstructed_lidar_pts[2, :],
            is_human.astype(np.float32).reshape(1, -1),
            is_human_id.astype(np.float32).reshape(1, -1),
            semantic_valid.astype(np.float32).reshape(1, -1)
        )).T

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='is_human', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='is_human_id', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='semantic_valid', offset=20, datatype=PointField.FLOAT32, count=1)
        ]

        pc2_msg = point_cloud2.create_cloud(header, fields, points)

        self.annotated_pc_publisher.publish(pc2_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriber_human()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
