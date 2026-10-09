#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from lidar_based_mcbf_msgs.msg import GPDFSet
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class GPDFVisualizer(Node):
    def __init__(self):
        super().__init__('gpdf_visualizer')

        self.sub = self.create_subscription(
            GPDFSet,
            '/gpdf_sets',
            self.callback_gpdf,
            10
        )

        self.pub = self.create_publisher(
            MarkerArray,
            '/gpdf_markers',
            10
        )

        self.get_logger().info("Convert GPDF/Wall into marker visual")

    def callback_gpdf(self, msg):
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        tot_num_gpdfs = len(msg.models)

        for i, gp in enumerate(msg.models):
            marker = Marker()
            marker.header = msg.header
            marker.ns = "gpdf"
            marker.id = i
            marker.type = Marker.POINTS
            marker.action = Marker.ADD

            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.1
            marker.scale.y = 0.1

            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            marker.points = gp.points

            marker_array.markers.append(marker)

        self.pub.publish(marker_array)

def main():
    rclpy.init()
    node = GPDFVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()