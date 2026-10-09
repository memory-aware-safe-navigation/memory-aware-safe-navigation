# lidar_based_mcbf_msgs

Custom ROS 2 messages for [`lidar_based_mcbf`](../lidar_based_mcbf_controller). Build it in the same workspace as the
controller package; it has no nodes of its own.

| Message | Purpose |
|---|---|
| `GPDF` | One learned obstacle distance field (points, weights, distance-field grid, robot barrier value and gradient) |
| `GPDFSet` | The set of `GPDF` models from the mapper, plus dynamic-obstacle centers and velocities |
| `StageConstraint` | Adaptive stage and constraint parameters used by the controller |
| `MCBFConstraint` | Geodesic direction (`e_vec`) and related terms for the MCBF-QP |

Dependencies: `std_msgs`, `geometry_msgs`.
