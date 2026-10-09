# Memory-Aware Multi-Sensor Perception for Efficient and Safe Navigation in Dynamic Environments

ROS 2 (Humble) implementation of a memory-aware, multi-sensor navigation framework that combines LiDAR and RGB
perception, an online-learned distance-field map of the environment, and a stage-adaptive
Modulated Control Barrier Function Quadratic Program (MCBF-QP) controller.

## What this package does

The robot navigates to a goal in a **previously unmapped, dynamic indoor or outdoor environment** using only onboard
sensing. No prior map is needed. The package implements the pipeline from the paper, except onboard state estimation (see step 1):

1. **Robot state (state estimation is not included).** Observations and the robot state are expressed in a persistent
   reference frame that starts at the robot's initial pose. This package does **not** contain a state estimator: the
   nodes read the robot pose from `/gazebo/model_states`, which is the simulator's **ground-truth state** in our
   simulation experiments. `pc_converter.py` uses that pose to transform the raw pointcloud into the reference frame. For
   hardware, provide the pose from a localization package of your choice (see
   [Implementation](#simulation-or-hardware)). In our hardware experiments the pose came from
   [FAST-LIO2](https://github.com/hku-mars/FAST_LIO) ([ROS 2 branch](https://github.com/hku-mars/FAST_LIO/tree/ROS2))
   fused with wheel velocity in an EKF. Neither FAST-LIO2 nor the EKF is part of this repository.
2. **Dynamic obstacle detection.** Two branches run in parallel and are cross-validated.
   - *LiDAR branch:* z-filter the pointcloud, cluster it, and fit lines (likely static infrastructure) and circles
     (candidate dynamic obstacles) with `obstacle_detector`, then track them and estimate their velocities.
   - *Camera branch:* YOLO instance segmentation (`camera_node_human*.py`, enabled with `use_semantic_humans:=true`)
     labels human points in the pointcloud, suppressing false positives and recovering missed detections.
3. **Occupancy grid mapping** (`occupancy_map_3d.py`). Ray-traced 2D log-odds grid. Cells vacated by tracked dynamic
   obstacles are reset explicitly, so no stale occupancy is left behind.
4. **Online environment representation learning** (`mapper.py`, `mapper_constraint.py`). Occupied cells near the
   robot are clustered (DBSCAN), boundaries are extracted (marching squares), dynamic and static points are
   separated, and a Gaussian Process Distance Field (GPDF, JAX/JIT) is learned for each obstacle. This gives a
   persistent, differentiable distance field for static infrastructure and separate ones for dynamic obstacles.
5. **Stage-adaptive MCBF-QP control** (`mcbf_constraint_node.py`, `nominal_controller.py`, `controller.py`). The perception-based controller is adapted from prior MCBF-QP work. The proposed controller selects one of three stages, *dynamic-obstacle*, *narrow-space* or *normal*, and adapts its CBF parameters and guidance accordingly.

The package also contains a DWA baseline (`dwa_control.launch.py`) that runs on the same perception stack.

## Repository layout

This repository contains two ROS 2 packages, `lidar_based_mcbf_controller/` (the controller and perception nodes) and
`lidar_based_mcbf_msgs/` (the custom messages). Paths below are relative to `lidar_based_mcbf_controller/`.

| Path | Contents |
|---|---|
| `launch/` | Launch files (see [Launch files](#launch-files)) |
| `scripts/` | ROS 2 nodes |
| `lidar_based_mcbf/` | Python library (`basic_boundary_function/`): GPDF learning, environment approximation (`env_approx.py`), on-manifold geodesic approximation (`onMan_approximation.py`) and config loaders (`configs.py`) |
| `config/` | `env_controller.yaml` (environment, goal and controller settings), `env_maps.yaml` (per-environment map bounds and sampling), `gpdf.yaml`, `spec_robot.yaml` (robot name and geometry / limits), `autotune_2D.yaml` (2D controller parameters) |
| `models/` | Optional YOLO segmentation engine (`yolo26n-seg.engine` by default). Git-ignored, so place your own here (see [YOLO model](#yolo-model-if-only-using-lidar-for-geometric-based-dynamic-obstacle-tracking-this-is-not-needed-required-if-using-camera-for-image-based-detection)) |
| `resources/` | RViz configs |

### Nodes (`scripts/`)

| Node | Role |
|---|---|
| `pc_converter.py` | Converts the LiDAR pointcloud into the reference frame and publishes the reference-frame TF from the robot pose |
| `camera_node_human.py`, `camera_node_human_multicam.py` | YOLO human segmentation; labels LiDAR points (single / multi camera) |
| `occupancy_map_3d.py` | Ray-traced occupancy grid with dynamic-obstacle clearing |
| `mapper.py`, `mapper_constraint.py` | Clustering and GPDF learning, and the resulting barrier constraints |
| `mcbf_constraint_node.py` | Builds the MCBF constraints and stage selection |
| `nominal_controller.py` | Nominal (goal-reaching) controller |
| `controller.py` | Solves the MCBF-QP and publishes `/cmd_vel` |
| `obstacle_track_list.py` | Matches tracker output (`/obstacles`) with raw detections (`/raw_obstacles`), keeps a track list, and publishes the active dynamic obstacles on `/obstacles_tracked`, which the mapper subscribes to (started by the multicam launch) |
| `gpdf_visual.py` | GPDF boundary-set visualization |
| `dwa_controller.py` | DWA baseline controller |

## Launch files

| Launch file | Purpose |
|---|---|
| `mcbf_control_dynamic.launch.py` | Full MCBF-QP pipeline for a single-camera setup (occupancy map, GPDF mapper, MCBF constraints, controller, obstacle detector/tracker, optional semantic humans, RViz, rosbag) |
| `mcbf_control_dynamic_multicam.launch.py` | Same as `mcbf_control_dynamic.launch.py` but with multiple cameras |
| `dwa_control.launch.py` | DWA baseline on the same occupancy map and obstacle tracker |
| `test_occupancy_map.launch.py` | Runs only `pc_converter` and `occupancy_map_3d`, for testing the occupancy map |

These launch files **do not start the simulator or the robot**. Start those first (see below), or run on hardware.

## Setup

### Requirements
- Ubuntu 22.04, ROS 2 Humble, Gazebo (Classic) for simulation
- Python 3 packages: `numpy==1.26.4`, `scipy==1.15.3`, `scikit-learn==1.7.2`, `scikit-image==0.25.2`,
  `cvxpy==1.7.5`, `jax==0.6.2`, `opencv-python==4.9.0.80`, `pyyaml==5.4.1`, `matplotlib==3.5.1`, `pillow==9.0.1`,
  `ultralytics==8.4.105` (YOLO; running a TensorRT `.engine` model needs an NVIDIA GPU)
- ROS packages: `rclpy`, `tf2_ros`, `tf2_geometry_msgs`, `cv_bridge`, `message_filters`, `gazebo_msgs`, and the
  workspace packages below

### Workspace packages
Clone this repository into the workspace `src/` folder (it provides the first two packages below), along with the other required packages.

**Required**

| Package | Why it is needed |
|---|---|
| `lidar_based_mcbf` (`lidar_based_mcbf_controller/` in this repo) | The controller and perception nodes |
| `lidar_based_mcbf_msgs` (`lidar_based_mcbf_msgs/` in this repo) | Custom messages |
| `obstacle_detector_lidar` | Provides `obstacle_detector` (`obstacle_extractor_node`, `obstacle_tracker_node`). It is based on the original [`obstacle_detector`](https://github.com/tysik/obstacle_detector) by its original author, Mateusz Przybyla (tysik), and we only modified it to accept pointclouds and to incorporate camera (semantic) detection information |

**Optional (simulation only).** These are what we used for the simulation experiments. You do not have to use them, and
any other simulator, robot and world that provides the topics the nodes expect will work. If you want to reproduce our
setup, they can be found here:

| Package | Purpose | Source |
|---|---|---|
| `pmb2_robot` (`pmb2_bringup`, ...) | Simulated PMB2 differential-drive robot | <https://github.com/pal-robotics/pmb2_robot> |
| `aws_sim_env` | Hospital and small-warehouse Gazebo worlds adapted from AWS RoboMaker | <https://github.com/aws-robotics/aws-robomaker-hospital-world>, <https://github.com/aws-robotics/aws-robomaker-small-warehouse-world> |

### YOLO model (If only using LiDAR for geometric-based dynamic obstacle tracking, this is not needed; Required if using camera for image-based detection)

The YOLO model is only needed with `use_semantic_humans:=true`. It is used to detect and track **people only**: the
nodes run Ultralytics `model.track(..., classes=[0])`, so any YOLO model you prefer works as long as it is an
**instance-segmentation** model (it must output masks) and its class 0 is `person` (as in COCO-trained models). Other
classes are ignored. We used YOLO26n-seg, converted to a TensorRT engine file (`yolo26n-seg.engine`). That file is not
tracked in git, so place your own engine in `models/` or pass its path with `semantic_model_path:=<path>`. Visit [Ultralytics](https://www.ultralytics.com/) for instruction on preparing engine files based on existing models and fine-tune models for specific class.

### Build

Install the Python packages listed under [Requirements](#requirements). `rosdep` only installs ROS and system
dependencies, so these need `pip`:
```bash
pip install numpy==1.26.4 scipy==1.15.3 scikit-learn==1.7.2 scikit-image==0.25.2 cvxpy==1.7.5 jax==0.6.2 opencv-python==4.9.0.80 pyyaml==5.4.1 matplotlib==3.5.1 pillow==9.0.1 ultralytics==8.4.105
```

Then build from the root of your workspace (`<your_ws>` is the folder whose `src/` contains the packages above, for
example `~/lidar_ws`). If `rosdep` has never been set up on this machine, run `sudo rosdep init` and `rosdep update`
first.
```bash
source /opt/ros/humble/setup.bash
cd <your_ws>
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

If you add a YOLO engine to `models/` after building, re-run `colcon build` so it is installed, or pass its path with
`semantic_model_path:=<path>`.

## Implementation

### Simulation (or hardware)

Terminal 1: launch your own simulation environment (or your robot's hardware drivers). The nodes expect:

- **LiDAR pointcloud:** a `sensor_msgs/PointCloud2` on `/husky/points`. Remap it, or edit the topic in
  `pc_converter.py`, to match your LiDAR driver.
- **Robot state:** a `gazebo_msgs/ModelStates` message on `/gazebo/model_states` whose `name` list contains your robot.
  - With ground-truth state from the simulator, set `rob_name` in `config/spec_robot.yaml` to your robot's model name.
  - Otherwise use any localization package you like, and publish its pose estimate as a `ModelStates` message on the
    same topic, using the same `rob_name`.
- **Velocity commands:** the controller publishes `geometry_msgs/Twist` on `/cmd_vel`.
- **Cameras (only with `use_semantic_humans:=true`):** the RGB image and camera-info topics that
  `camera_node_human*.py` subscribe to.

Also set `env_name` and `target` in `config/env_controller.yaml` (see [Settings](#settings)), and keep the robot's
start and the goal inside the mapped area.

#### Proposed Method

Terminal 2: the MCBF-QP navigation stack (perception, mapping and controller). Wait until the simulation (or drivers) is up,
then launch it:
```bash
ros2 launch lidar_based_mcbf mcbf_control_dynamic_multicam.launch.py rviz_show:=true
```
For a single-camera setup, launch `mcbf_control_dynamic.launch.py` instead.

#### DWA baseline

With your simulation running in Terminal 1, launch the DWA baseline in Terminal 2 instead of the MCBF-QP stack:
```bash
ros2 launch lidar_based_mcbf dwa_control.launch.py rviz_show:=true
```

#### Testing the occupancy map alone

To check only the occupancy map, launch just the pointcloud converter and the occupancy grid (with the simulation or a
bag playing):
```bash
ros2 launch lidar_based_mcbf test_occupancy_map.launch.py
```

### Launch arguments

| Argument | Default | Launch files | Description |
|---|---|---|---|
| `use_semantic_humans` | `false` | `dwa_control.launch.py`<br>`mcbf_control_dynamic.launch.py`<br>`mcbf_control_dynamic_multicam.launch.py` | Use YOLO-labeled pointclouds to correct human detections |
| `semantic_model_path` | `models/yolo26n-seg.engine` | `dwa_control.launch.py`<br>`mcbf_control_dynamic.launch.py`<br>`mcbf_control_dynamic_multicam.launch.py` | YOLO model used when semantic humans are on |
| `rviz_show` | `false` | `dwa_control.launch.py`<br>`mcbf_control_dynamic.launch.py`<br>`mcbf_control_dynamic_multicam.launch.py` | Launch RViz |
| `disable_cmd` | `false` | `mcbf_control_dynamic.launch.py`<br>`mcbf_control_dynamic_multicam.launch.py` | Run everything but do not send velocity commands |
| `record_bag` | `false` | `dwa_control.launch.py`<br>`mcbf_control_dynamic.launch.py`<br>`mcbf_control_dynamic_multicam.launch.py` | Record controller / mapper / timing topics (DWA: replay topics) to a rosbag |
| `record_bag_perf` | `false` | `mcbf_control_dynamic_multicam.launch.py` | Record a performance-only bag (cannot be combined with `record_bag`) |
| `bag_output` | `bags/<name>/run` | `dwa_control.launch.py`<br>`mcbf_control_dynamic.launch.py`<br>`mcbf_control_dynamic_multicam.launch.py` | Output path of the rosbag, used with `record_bag` (and `record_bag_perf`). `<name>` is `dwa_control`, `mcbf_control` or `mcbf_control_multicam` respectively. `ros2 bag` fails if the folder already exists, so use a new path for each run |
| `use_obstacle_detector` | `true` | `dwa_control.launch.py` | Launch the obstacle extractor / tracker |
| `rviz_show_lidar` | `false` | `mcbf_control_dynamic_multicam.launch.py` | RViz config that also shows LiDAR data |
| `baseline_eval` | `false` | `mcbf_control_dynamic_multicam.launch.py` | Pure CBF baseline (fixed `alpha=0.5`, `margin=0.25`, regardless of stage) |
| `raycast_show` | `false` | `mcbf_control_dynamic_multicam.launch.py` | Publish the raycast markers |

Run `ros2 launch lidar_based_mcbf <file> --show-args` for the full list.

### Settings
- `config/env_controller.yaml`: environment name (`env_name`), goal (`target`), `dynamics`, `nominal_speed`, and
  `MCBF` (`True` for the MCBF-QP, `False` for the regular CBF-QP). `env_name` selects the mapping area and clustering
  settings, and must be one of the supported environments below. Any other name raises an
  `Unknown environment name` error at startup.

  | `env_name` | Environment | Mapped area (x range, y range, in the reference frame) | Size |
  |---|---|---|---|
  | `aws_hospital` | AWS RoboMaker hospital world (simulation, default) | x in [-15, 15] m, y in [-35, 22] m | 30 m x 57 m |
  | `small_warehouse` | AWS RoboMaker small warehouse world (simulation) | x in [-10, 10] m, y in [-10, 10] m | 20 m x 20 m |
  | `large_hospital_1` | Large hospital map | x in [-21, 11] m, y in [-13, 13] m | 32 m x 26 m |
  | `vicon` | Vicon motion-capture arena | x in [-3, 4] m, y in [-3, 4] m | 7 m x 7 m |

  The occupancy map only covers this area, given in the global `mocap` frame (Gazebo world coordinates in
  simulation). Keep the robot, its start and the goal `target` inside it. To use a new environment, add its bounds and
  clustering settings (`x_min`, `x_max`, `y_min`, `y_max`, `other_num_samples`) to `config/env_maps.yaml`; no code
  changes are needed.
- `config/spec_robot.yaml`: `rob_name` (the robot's model name in `/gazebo/model_states`, `pmb2` by default, shared by
  all nodes), physical size, margins and velocity / acceleration limits.
- `config/gpdf.yaml`: GPDF settings. `config/autotune_2D.yaml`: 2D controller parameters.
