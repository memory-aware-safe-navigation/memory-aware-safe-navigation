from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.parameter_descriptions import ParameterValue

import os
from ament_index_python.packages import get_package_share_directory


def _validate_record_bag_args(context, *args, **kwargs):
    record_bag_val = LaunchConfiguration('record_bag').perform(context).lower() == 'true'
    record_bag_perf_val = LaunchConfiguration('record_bag_perf').perform(context).lower() == 'true'
    if record_bag_val and record_bag_perf_val:
        raise RuntimeError(
            "record_bag and record_bag_perf cannot both be set to true at the same time"
        )
    return []


def _validate_rviz_args(context, *args, **kwargs):
    rviz_show_val = LaunchConfiguration('rviz_show').perform(context).lower() == 'true'
    rviz_show_lidar_val = LaunchConfiguration('rviz_show_lidar').perform(context).lower() == 'true'
    if rviz_show_val and rviz_show_lidar_val:
        raise RuntimeError(
            "rviz_show and rviz_show_lidar cannot both be set to true at the same time"
        )
    return []


def generate_launch_description():

    record_bag = LaunchConfiguration('record_bag')
    bag_output = LaunchConfiguration('bag_output')
    record_bag_perf = LaunchConfiguration('record_bag_perf')
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
    declare_record_bag_perf = DeclareLaunchArgument(
        'record_bag_perf',
        default_value='false',
        description='Whether to record a perf-only rosbag (same as record_bag but excludes '
                     'rviz-only visualization topics). Cannot be set true together with record_bag.'
    )
    declare_bag_output = DeclareLaunchArgument(
        'bag_output',
        default_value='bags/mcbf_control_multicam/run',
        description='Output bag path/prefix used when record_bag or record_bag_perf is true',
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
        description='Whether to launch rviz using the base (no-lidar) config. '
                     'Cannot be set true together with rviz_show_lidar.'
    )

    rviz_show_lidar = LaunchConfiguration('rviz_show_lidar')
    declare_rviz_show_lidar = DeclareLaunchArgument(
        'rviz_show_lidar',
        default_value='false',
        description='Whether to launch rviz using the config that also displays the '
                     'annotated lidar pointcloud. Cannot be set true together with rviz_show.'
    )

    disable_cmd_vel = LaunchConfiguration('disable_cmd')
    declare_disable_cmd = DeclareLaunchArgument(
        'disable_cmd',
        default_value='false',
        description='Whether to disable command sending'
    )

    baseline_eval = LaunchConfiguration('baseline_eval')
    declare_baseline_eval = DeclareLaunchArgument(
        'baseline_eval',
        default_value='false',
        description='Whether to run the pure CBF baseline evaluation (fixed alpha=0.5, margin=0.25 regardless of stage)'
    )

    raycast_show = LaunchConfiguration('raycast_show')
    declare_raycast_show = DeclareLaunchArgument(
        'raycast_show',
        default_value='false',
        description='Whether occupancy_map_3d publishes /ray_marker showing the raycast hit/miss lines'
    )

    obs_detector_bag = ExecuteProcess(
        cmd = [
            'ros2', 'bag', 'record',
            '-o', bag_output,
            '/gazebo/model_states',
            '/raw_obstacles',
            '/clock',
            '/tf',
            '/tf_static',
            '/robot_description',
            '/obstacles',
            '/obstacles_tracked',
            '/obstacles_track_list',
            '/cmd_vel',
            '/cmd_dt',
            '/cmd_states',
            '/boundary_points',
            '/e_vec_marker',
            '/rob_pose',
            '/robot_h',
            '/gpdf_sets',
            '/gpdf_markers',
            '/dyn_obs_xc',
            '/occupancy_map',
            '/dyn_vacated_mask',
            '/target_marker',
            '/temp_target_marker',
            '/line_samples',
            '/geodesic_paths',
            '/proj_points_rob',
            '/proj_points_target',
            '/solver_status_marker',
            '/rect_mask',
            '/collision_status_marker',
            '/mapper_loop_dt',
            '/mapper_cluster_dt',
            '/mapper_gpdf_dt',
            '/mapper_update_dt',
            '/mapper_publish_dt',
            '/mapper_points_pack_dt',
            '/mapper_disfield_dt',
            '/controller_solve_dt',
            '/controller_loop_gap_dt',
            '/controller_debug_publish_dt',
            '/controller_publish_dt',
            '/controller_stage_dt',
            '/controller_polygon_dt',
            '/controller_projection_dt',
            '/controller_geodesic_dt',
            '/controller_qp_dt',
            '/controller_sleep_target_dt',
            '/controller_sleep_actual_dt',
            '/annotated_pc_dt',
            '/annotated_pc_proc_dt',
            '/mapper/xc',
            '/mapper/dxc'
        ],
        output='screen',
        prefix='nice -n 15',
        condition=IfCondition(record_bag)
    )

    obs_detector_bag_perf = ExecuteProcess(
        cmd = [
            'ros2', 'bag', 'record',
            '-o', bag_output,
            '/gazebo/model_states',
            '/clock',
            '/tf',
            '/tf_static',
            '/obstacles',
            '/obstacles_tracked',
            '/obstacles_track_list',
            '/cmd_vel',
            '/cmd_dt',
            '/cmd_states',
            '/robot_h',
            '/gpdf_sets',
            '/occupancy_map',
            '/mapper_loop_dt',
            '/mapper_cluster_dt',
            '/mapper_gpdf_dt',
            '/mapper_update_dt',
            '/mapper_publish_dt',
            '/mapper_points_pack_dt',
            '/mapper_disfield_dt',
            '/controller_solve_dt',
            '/controller_loop_gap_dt',
            '/controller_debug_publish_dt',
            '/controller_publish_dt',
            '/controller_stage_dt',
            '/controller_polygon_dt',
            '/controller_projection_dt',
            '/controller_geodesic_dt',
            '/controller_qp_dt',
            '/controller_sleep_target_dt',
            '/controller_sleep_actual_dt',
            '/annotated_pc_dt',
            '/annotated_pc_proc_dt',
            '/mapper/xc',
            '/mapper/dxc'
        ],
        output='screen',
        prefix='nice -n 15',
        condition=IfCondition(record_bag_perf)
    )

    pc_convert_node = Node(
        package='lidar_based_mcbf',
        executable='pc_converter.py',
        name='pointcloud_convert_node',
        output='screen',
    )

    base_footprint_visual_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_footprint_visual_tf',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'base_link', '--child-frame-id', 'base_footprint'],
        parameters=[{'use_sim_time': True}],
    )

    base_cover_link_visual_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_cover_link_visual_tf',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--roll', '0', '--pitch', '0', '--yaw', '0',
                   '--frame-id', 'base_link', '--child-frame-id', 'base_cover_link'],
        parameters=[{'use_sim_time': True}],
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
        parameters=[{
            'visualize_rays': ParameterValue(raycast_show, value_type=bool),
        }],
        output='screen',
        prefix='nice -n 10',
    )

    obstacle_track_list_node = Node(
        package='lidar_based_mcbf',
        executable='obstacle_track_list.py',
        name='obstacle_track_list_node',
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
        parameters=[{
            'baseline_eval': baseline_eval,
        }],
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
            'baseline_eval': baseline_eval,
        }],
        output='screen',
    )

    gpdf_visual_node = Node(
        package='lidar_based_mcbf',
        executable='gpdf_visual.py',
        name='gpdf_visualizer',
        output='screen',
        prefix='nice -n 15',
    )

    rviz_config_lidar = os.path.join(
        get_package_share_directory('lidar_based_mcbf'),
        'resources',
        'mcbf_visual_human.rviz'
    )
    rviz_config_no_lidar = os.path.join(
        get_package_share_directory('lidar_based_mcbf'),
        'resources',
        'mcbf_visual_human_no_lidar.rviz'
    )
    rviz_config = PythonExpression([
        "'", rviz_config_lidar, "' if '",
        rviz_show_lidar,
        "'.lower() == 'true' else '", rviz_config_no_lidar, "'"
    ])
    rviz_enabled = PythonExpression([
        "'true' if ('", rviz_show, "'.lower() == 'true' or '",
        rviz_show_lidar, "'.lower() == 'true') else 'false'"
    ])

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='mcbf_rviz',
        output='screen',
        arguments=['-d', rviz_config],
        prefix='nice -n 15',
        condition=IfCondition(rviz_enabled)
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
        ]
    )

    return LaunchDescription([
        declare_record_bag,
        declare_bag_output,
        declare_record_bag_perf,
        OpaqueFunction(function=_validate_record_bag_args),
        declare_rviz_show,
        declare_rviz_show_lidar,
        OpaqueFunction(function=_validate_rviz_args),
        declare_disable_cmd,
        declare_baseline_eval,
        declare_raycast_show,
        declare_use_semantic_humans,
        declare_semantic_model_path,
        camera_human_node,
        pc_convert_node,
        base_footprint_visual_tf,
        base_cover_link_visual_tf,
        occupancy_map_node,
        obstacle_track_list_node,
        mapper_node,
        mapper_constraint_node,
        mcbf_constraint_node,
        nominal_controller_node,
        controller_node,
        gpdf_visual_node,
        rviz_node,
        obs_detector_node,
        obs_tracker_node,
        obs_detector_bag,
        obs_detector_bag_perf
    ])
