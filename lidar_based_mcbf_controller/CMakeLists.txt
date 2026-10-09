cmake_minimum_required(VERSION 3.8)
project(lidar_based_mcbf)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/pc_converter.py
  scripts/mapper.py
  scripts/mapper_constraint.py
  scripts/obstacle_track_list.py
  scripts/mcbf_constraint_node.py
  scripts/nominal_controller.py
  scripts/controller.py
  scripts/gpdf_visual.py
  scripts/occupancy_map_3d.py
  scripts/dwa_controller.py
  scripts/camera_node_human.py
  scripts/camera_node_human_multicam.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  resources
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  models
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
