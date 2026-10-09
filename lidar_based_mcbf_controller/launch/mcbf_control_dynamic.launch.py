from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.parameter_descriptions import ParameterValue

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    record_bag = LaunchConfiguration('record_bag')
    bag_output = LaunchConfiguration('bag_output')
    use_semantic_humans = LaunchConfiguration('use_semantic_humans')
    semantic_model_path = LaunchConfiguration('semantic_model_path')
    default_semantic_model_path = os.path.join(
        get_package_share_directory('lidar_based_mcbf'),
        'models',
        'yolo26n-seg.engine',
    )
    declare_record_bag = DeclareLaunchArgument(
        'record_bag',
        default_value='false',
        description='Whether to record rosbag'
    )
    declare_bag_output = DeclareLaunchArgument(
        'bag_output',
        default_value='bags/mcbf_control/run',
        description='Output bag path/prefix used when record_bag is true',
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

    rviz_show = LaunchConfiguration('rviz_show')
    declare_rviz_show = DeclareLaunchArgument(
        'rviz_show',
        default_value='false',
        description='Whether to launch rviz alongside the controller'
    )

    disable_cmd_vel = LaunchConfiguration('disable_cmd')
    declare_disable_cmd = DeclareLaunchArgument(
        'disable_cmd',
        default_value='false',
        description='Whether to disable command sending'
    )

    obs_detector_bag = ExecuteProcess(
        cmd = [
            'ros2', 'bag', 'record',
            '-o', bag_output,
            '/gazebo/model_states',
            '/raw_obstacles',
            '/filtered_pc',
            '/cloud_all_fields_fullframe_global',
            '/clock',
            '/tf',
            '/tf_static',
            '/robot_description',
            '/obstacles',
            '/cmd_vel',
            '/cmd_dt',
            '/cmd_states',
            '/boundary_points',
            '/e_vec_marker',
            '/rob_pose',
            '/gpdf_sets',
            '/gpdf_markers',
            '/mapper_loop_dt',
            '/mapper_cluster_dt',
            '/mapper_gpdf_dt',
            '/mapper_update_dt',
            '/mapper_publish_dt',
            '/mapper_points_pack_dt',
            '/mapper_disfield_dt',
            '/controller_solve_dt',
            '/controller_publish_dt',
            '/controller_stage_dt',
            '/controller_polygon_dt',
            '/controller_projection_dt',
            '/controller_geodesic_dt',
            '/controller_qp_dt',
            '/mapper/xc',
            '/mapper/dxc'
        ],
        output='screen',
        condition=IfCondition(record_bag)
    )

    pc_convert_node = Node(
        package='lidar_based_mcbf',
        executable='pc_converter.py',
        name='pointcloud_convert_node',
        output='screen',
    )

    camera_human_node = Node(
        package='lidar_based_mcbf',
        executable='camera_node_human.py',
        name='camera_rgbd_human',
        parameters=[{
            'use_sim_time': True,
            'model_path': semantic_model_path,
        }],
        output='screen',
        condition=IfCondition(use_semantic_humans),
    )

    occupancy_map_node = Node(
        package='lidar_based_mcbf',
        executable='occupancy_map_3d.py',
        name='occupancy_map_3d',
        output='screen',
    )

    mapper_node = Node(
        package='lidar_based_mcbf',
        executable='mapper.py',
        name='mapper_node',
        output='screen',
    )

    mapper_constraint_node = Node(
        package='lidar_based_mcbf',
        executable='mapper_constraint.py',
        name='mapper_constraint_node',
        output='screen',
    )

    mcbf_constraint_node = Node(
        package='lidar_based_mcbf',
        executable='mcbf_constraint_node.py',
        name='mcbf_constraint_node',
        output='screen',
    )

    nominal_controller_node = Node(
        package='lidar_based_mcbf',
        executable='nominal_controller.py',
        name='nominal_controller_node',
        output='screen',
    )

    controller_node = Node(
        package='lidar_based_mcbf',
        executable='controller.py',
        name='controller_medium',
        parameters=[{
            'disable_cmd': disable_cmd_vel,
        }],
        output='screen',
    )

    gpdf_visual_node = Node(
        package='lidar_based_mcbf',
        executable='gpdf_visual.py',
        name='gpdf_visualizer',
        output='screen',
    )

    rviz_config = os.path.join(
        get_package_share_directory('lidar_based_mcbf'),
        'resources',
        'mcbf_visual_human.rviz'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='mcbf_rviz',
        output='screen',
        arguments=['-d', rviz_config],
        condition=IfCondition(rviz_show)
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
        }]
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
            'velocity_filter_threshold': 0.12,
            'velocity_filter_buffer_size': 80,
            'min_measurement_radius': 0.45,
            'max_measurement_radius': 1.00,
            'semantic_clamp_measurement_radius': ParameterValue(use_semantic_humans, value_type=bool),
            'min_publish_radius': 0.45,
            'max_publish_radius': 0.65,
            'publish_after_miss_duration': 4.5,
            'dynamic_publish_grace_duration': 4.5,
            'dynamic_publish_min_corrections': 1,
            'sensor_rate': 10.0,
            'frame_id': 'mocap',
        }],
        remappings=[
            ('tracked_obstacles', 'obstacles')
        ]
    )

    return LaunchDescription([
        declare_record_bag,
        declare_bag_output,
        declare_rviz_show,
        declare_disable_cmd,
        declare_use_semantic_humans,
        declare_semantic_model_path,
        camera_human_node,
        pc_convert_node,
        occupancy_map_node,
        mapper_node,
        mapper_constraint_node,
        mcbf_constraint_node,
        nominal_controller_node,
        controller_node,
        gpdf_visual_node,
        rviz_node,
        obs_detector_node,
        obs_tracker_node,
        obs_detector_bag
    ])
