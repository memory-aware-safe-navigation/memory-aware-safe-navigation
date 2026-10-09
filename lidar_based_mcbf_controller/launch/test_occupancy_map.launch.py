from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    pc_convert_node = Node(
        package='lidar_based_mcbf',
        executable='pc_converter.py',
        name='pointcloud_convert_node',
        output='screen',
    )

    occupancy_map_node = Node(
        package='lidar_based_mcbf',
        executable='occupancy_map_3d.py',
        name='occupancy_map_3d',
        output='screen',
    )

    return LaunchDescription([
        pc_convert_node,
        occupancy_map_node,
    ])
