from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.parameter_descriptions import ParameterValue

import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    rviz_show = LaunchConfiguration('rviz_show')
    declare_rviz_show = DeclareLaunchArgument(
        'rviz_show',
        default_value='false',
        description='Whether to launch rviz alongside the DWA controller'
    )

    record_bag = LaunchConfiguration('record_bag')
    bag_output = LaunchConfiguration('bag_output')
    declare_record_bag = DeclareLaunchArgument(
        'record_bag',
        default_value='false',
        description='Record all topics needed to replay dwa_visual.rviz later',
    )
    declare_bag_output = DeclareLaunchArgument(
        'bag_output',
        default_value='bags/dwa_control/run',
        description='Output bag path/prefix used when record_bag is true',
    )

    dwa_bag_record = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '-o', bag_output,
            '/tf',
            '/tf_static',
            '/clock',
            '/robot_description',
            '/occupancy_map',
            '/dwa_trajectories',
            '/dwa_goal_marker',
            '/solver_status_marker',
            '/gazebo/model_states',
        ],
        output='screen',
        condition=IfCondition(record_bag),
    )

    use_semantic_humans = LaunchConfiguration('use_semantic_humans')
    semantic_model_path = LaunchConfiguration('semantic_model_path')
    default_semantic_model_path = os.path.join(
        get_package_share_directory('lidar_based_mcbf'),
        'models',
        'yolo26n-seg.engine',
    )
    declare_use_semantic_humans = DeclareLaunchArgument(
        'use_semantic_humans',
        default_value='false',
        description='Use YOLO-labeled point clouds to correct human obstacle detections'
    )
    declare_semantic_model_path = DeclareLaunchArgument(
        'semantic_model_path',
        default_value=default_semantic_model_path,
        description='YOLO segmentation model path used when use_semantic_humans is true'
    )
    use_obstacle_detector = LaunchConfiguration('use_obstacle_detector')
    declare_use_obstacle_detector = DeclareLaunchArgument(
        'use_obstacle_detector',
        default_value='true',
        description='Whether to launch the obstacle_detector extractor/tracker nodes'
    )

    pc_convert_node = Node(
        package='lidar_based_mcbf',
        executable='pc_converter.py',
        name='pointcloud_convert_node',
        output='screen',
    )

    camera_human_node = Node(
        package='lidar_based_mcbf',
        executable='camera_node_human_multicam.py',
        name='camera_rgbd_human',
        parameters=[{
            'use_sim_time': True,
            'model_path': semantic_model_path,
        }],
        output='screen',
        prefix='nice -n 10',
        condition=IfCondition(use_semantic_humans),
    )

    occupancy_map_node = Node(
        package='lidar_based_mcbf',
        executable='occupancy_map_3d.py',
        name='occupancy_map_3d',
        output='screen',
        prefix='nice -n 10',
    )

    dwa_controller_node = Node(
        package='lidar_based_mcbf',
        executable='dwa_controller.py',
        name='dwa_controller_node',
        output='screen',
    )

    obs_detector_node = Node(
        package='obstacle_detector',
        executable='obstacle_extractor_node',
        name='obstacle_extractor',
        parameters=[{
            'active': True,
            'use_scan': False,
            'use_pcl': True,
            'pointcloud_topic': ParameterValue(PythonExpression([
                "'/annotated_pointcloud_global' if '",
                use_semantic_humans,
                "'.lower() == 'true' else '/cloud_all_fields_fullframe_global'"
            ]), value_type=str),
            'use_semantic_labels': ParameterValue(use_semantic_humans, value_type=bool),
            'use_split_and_merge': True,
            'circles_from_visibles': True,
            'discard_converted_segments': True,
            'transform_coordinates': True,
            'min_group_points': 3,
            'max_group_distance': 0.6,
            'distance_proportion': 0.0314,
            'max_split_distance': 0.5,
            'max_merge_separation': 0.6,
            'max_merge_spread': 0.35,
            'max_circle_radius': 0.75,
            'min_circle_radius': 0.12,
            'radius_enlargement': 0.3,
            'frame_id': 'mocap',
            'processing_frame_id': 'base_link',
            'sort_input_points': True,
            'height_lidar_to_ground': 0.666,
            'z_filter_half_range': 0.65,
            'min_x_limit': -20.0,
            'max_x_limit':  20.0,
            'min_y_limit': -40.0,
            'max_y_limit':  30.0,
        }],
        condition=IfCondition(use_obstacle_detector),
    )

    obs_tracker_node = Node(
        package='obstacle_detector',
        executable='obstacle_tracker_node',
        name='obstacle_tracker',
        parameters=[{
            'use_sim_time': True,
            'active': True,
            'loop_rate': 100.0,
            'tracking_duration': 1.5,
            'min_correspondence_cost': 1.2,
            'std_correspondence_dev': 0.35,
            'process_variance': 0.10,
            'process_rate_variance': 0.06,
            'measurement_variance': 0.08,
            'velocity_filter_threshold': ParameterValue(PythonExpression([
                "0.0 if '",
                use_semantic_humans,
                "'.lower() == 'true' else 0.12"
            ]), value_type=float),
            'velocity_filter_buffer_size': 80,
            'min_measurement_radius': 0.45,
            'max_measurement_radius': 1.00,
            'semantic_clamp_measurement_radius': ParameterValue(use_semantic_humans, value_type=bool),
            'min_publish_radius': 0.45,
            'max_publish_radius': 0.75,
            'publish_after_miss_duration': 5.5,
            'dynamic_publish_grace_duration': 5.5,
            'dynamic_publish_min_corrections': 1,
            'sensor_rate': 10.0,
            'frame_id': 'mocap',
        }],
        remappings=[
            ('tracked_obstacles', 'obstacles')
        ],
        condition=IfCondition(use_obstacle_detector),
    )

    rviz_config = os.path.join(
        get_package_share_directory('lidar_based_mcbf'),
        'resources',
        'dwa_visual.rviz'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='dwa_rviz',
        output='screen',
        arguments=['-d', rviz_config],
        condition=IfCondition(rviz_show)
    )

    return LaunchDescription([
        declare_rviz_show,
        declare_record_bag,
        declare_bag_output,
        declare_use_semantic_humans,
        declare_semantic_model_path,
        declare_use_obstacle_detector,
        dwa_bag_record,
        camera_human_node,
        pc_convert_node,
        occupancy_map_node,
        dwa_controller_node,
        rviz_node,
        obs_detector_node,
        obs_tracker_node,
    ])
