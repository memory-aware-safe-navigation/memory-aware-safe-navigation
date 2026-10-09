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
from std_msgs.msg import Header, Float32

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

        self.declare_parameter('semantic_max_range', 5.0)
        self.semantic_max_range = self.get_parameter('semantic_max_range').get_parameter_value().double_value

        self.declare_parameter('range_cap', False)
        self.range_cap = self.get_parameter('range_cap').get_parameter_value().bool_value

        self.publish_annotated_imgs = False
        self.publish_overlay_msg = False

        self.camera_names = ["front_rgbd", "left_rgbd", "rear_rgbd", "right_rgbd"]
        self.models = {name: YOLO(model_path) for name in self.camera_names}

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.lidar_to_cam = {
            name: self._lookup_matrix(f"{name}_optical_frame", "lidar_link")
            for name in self.camera_names
        }
        self.cam_matrix = None

        self.create_subscription(CameraInfo, "/front_rgbd/camera_info", self.callback_caminfo, 1)
        self.image_subs = [
            Subscriber(self, Image, f'/{name}/image_raw') for name in self.camera_names
        ]
        self.lidar_sub = Subscriber(self, PointCloud2, '/husky/points')
        self.sync2 = ApproximateTimeSynchronizer(
            [*self.image_subs, self.lidar_sub],
            queue_size=10,
            slop=0.4,
            allow_headerless=False
        )
        self.sync2.registerCallback(self.callback_human)

        self.annotated_pc_publisher = self.create_publisher(PointCloud2, '/annotated_pointcloud', 1)
        self.annotated_pc_dt_publisher = self.create_publisher(Float32, '/annotated_pc_dt', 10)
        self._last_pc_publish_t = None
        self.annotated_pc_proc_dt_publisher = self.create_publisher(Float32, '/annotated_pc_proc_dt', 10)

        self.sub_rob_odom = self.create_subscription(ModelStates, "/gazebo/model_states", self.callback_rob, 1)

        self.n = 3
        self.obs_num = 3
        self.rob_x = np.zeros(self.n,)

        self.annotated_image_publishers = {
            name: self.create_publisher(Image, f"/{name}/annotated_rgb", 10)
            for name in self.camera_names
        }
        self.lidar_overlay_image_publishers = {
            name: self.create_publisher(Image, f"/{name}/overlay_rgb", 1)
            for name in self.camera_names
        }

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

    def callback_human(self, msg_front, msg_left, msg_rear, msg_right, msg_lidar):

        if self.cam_matrix is None:
            self.get_logger().warn("Camera info not received yet; skipping frame", throttle_duration_sec=2.0)
            return

        start_time = time.time()

        image_msgs = [msg_front, msg_left, msg_rear, msg_right]
        images = [self.bridge.imgmsg_to_cv2(m, desired_encoding='bgr8') for m in image_msgs]

        results_per_cam = [
            self.models[name].track(image, classes=[0], persist=True, verbose=False)[0]
            for name, image in zip(self.camera_names, images)
        ]

        id_offset_step = 1000
        cameras = []

        for cam_idx, (name, image, results) in enumerate(zip(self.camera_names, images, results_per_cam)):
            combined_mask = np.zeros((self.image_shape[1], self.image_shape[0]), dtype=np.uint8)
            mask_list = []
            mask_ids = []
            human_detected = False

            if results.masks is not None and results.boxes is not None:
                for box, mask in zip(results.boxes, results.masks.data):

                    if box.id is None:
                        continue

                    obj_id = int(box.id.item()) + cam_idx * id_offset_step
                    mask_binary = mask.cpu().numpy()
                    mask_binary = (mask_binary > 0.7).astype(np.uint8)

                    mask_binary_resized = cv2.resize(mask_binary, (self.image_shape[0], self.image_shape[1]), interpolation=cv2.INTER_NEAREST)

                    mask_list.append(mask_binary_resized)
                    mask_ids.append(obj_id)

                    human_detected = True

                    combined_mask = np.logical_or(combined_mask, mask_binary_resized).astype(np.uint8)

            cameras.append({
                'name': name,
                'segmentation': combined_mask,
                'mask_list': mask_list,
                'mask_ids': mask_ids,
                'human_detected': human_detected,
            })

            if self.publish_annotated_imgs:
                annotated_img = results.plot()
                annotated_img_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
                self.annotated_image_publishers[name].publish(annotated_img_msg)

        points = point_cloud2.read_points_numpy(msg_lidar, field_names=("x", "y", "z"))
        xyz_array = points.T

        xyz_array = xyz_array[:, ~np.isnan(xyz_array).any(axis=0)]
        n_points = xyz_array.shape[1]

        xyz_homo = np.vstack((xyz_array, np.ones(n_points)))

        semantic_valid = np.zeros(n_points, dtype=bool)
        is_human = np.zeros(n_points, dtype=bool)
        is_human_id = np.zeros(n_points, dtype=int)
        is_mask_center = np.zeros(n_points, dtype=bool)

        for cam_idx, cam in enumerate(cameras):
            lidar_to_cam = self.lidar_to_cam[cam['name']]
            xyz_cam_frame = lidar_to_cam @ xyz_homo

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

            unclaimed = ~semantic_valid[visible_idx]
            visible_idx = visible_idx[unclaimed]
            u_valid = u_valid[unclaimed]
            v_valid = v_valid[unclaimed]

            if self.publish_overlay_msg:
                overlay_img = images[cam_idx].copy()
                for i in range(len(v_valid)):
                    x = int(v_valid[i])
                    y = int(u_valid[i])
                    cv2.circle(overlay_img, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
                overlay_img_msg = self.bridge.cv2_to_imgmsg(overlay_img, encoding='bgr8')
                self.lidar_overlay_image_publishers[cam['name']].publish(overlay_img_msg)

            semantic_valid[visible_idx] = True

            if cam['human_detected']:
                is_human[visible_idx] = cam['segmentation'][u_valid, v_valid] > 0

                for obj_id, mask in zip(cam['mask_ids'], cam['mask_list']):
                    human_mask = mask[u_valid, v_valid] > 0
                    is_human_id[visible_idx[human_mask]] = obj_id

                    if np.any(human_mask):
                        mask_ys, mask_xs = np.where(mask > 0)
                        mask_centroid = np.array([mask_ys.mean(), mask_xs.mean()])

                        matched_u = u_valid[human_mask]
                        matched_v = v_valid[human_mask]
                        d2 = (matched_u - mask_centroid[0]) ** 2 + (matched_v - mask_centroid[1]) ** 2
                        nearest_local = np.argmin(d2)

                        matched_global_idx = visible_idx[human_mask]
                        is_mask_center[matched_global_idx[nearest_local]] = True

        if self.range_cap:
            far_mask = np.hypot(xyz_array[0, :], xyz_array[1, :]) > self.semantic_max_range
            semantic_valid[far_mask] = False
            is_human[far_mask] = False
            is_human_id[far_mask] = 0
            is_mask_center[far_mask] = False

        self.publish_labeled_point_cloud(
            start_time,
            xyz_array,
            is_human,
            is_human_id,
            semantic_valid,
            is_mask_center,
            frame=msg_lidar.header.frame_id,
            stamp=msg_lidar.header.stamp)

    def publish_labeled_point_cloud(
            self,
            start_time,
            reconstructed_lidar_pts,
            is_human,
            is_human_id,
            semantic_valid,
            is_mask_center,
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
            semantic_valid.astype(np.float32).reshape(1, -1),
            is_mask_center.astype(np.float32).reshape(1, -1)
        )).T

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='is_human', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='is_human_id', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='semantic_valid', offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name='is_mask_center', offset=24, datatype=PointField.FLOAT32, count=1)
        ]

        pc2_msg = point_cloud2.create_cloud(header, fields, points)

        self.annotated_pc_publisher.publish(pc2_msg)

        proc_msg = Float32()
        proc_msg.data = float(time.time() - start_time)
        self.annotated_pc_proc_dt_publisher.publish(proc_msg)

        now = time.monotonic()
        if self._last_pc_publish_t is not None:
            dt_msg = Float32()
            dt_msg.data = float(now - self._last_pc_publish_t)
            self.annotated_pc_dt_publisher.publish(dt_msg)
        self._last_pc_publish_t = now

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
