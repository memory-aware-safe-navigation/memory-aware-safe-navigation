std_msgs/Header header

# Adaptive "stage" (1=tight, 2=relaxed) computed from the fixed
# heading-aligned rectangle mask + room-confinement checks. Ported from
# OnManCBFController._compute_fixed_rect_mask_points /
# _compute_room_confinement in cbf_control.py — now owned by
# mapper_constraint.py so both the optimization node and the nominal
# controller node can consume it without recomputing it themselves.
int32 stage

bool is_confined
bool stage3_latched

# -1.0 when no gap was found between two clusters inside the rect mask
float64 min_gap

# Margin/threshold the optimizer should use as a baseline for this stage.
# The optimizer may still override these (e.g. for a dynamic-obstacle
# encounter) using state this node doesn't have visibility into.
float64 effective_om_expand_threshold
float64 margin

# Heading-aligned rectangle mask bounds, for downstream visualization
# (mirrors controller.py's publish_rect_mask) and reuse by subscribers.
bool has_rect_bounds
geometry_msgs/Point rect_center
float64 rect_yaw
float64 rect_length
float64 rect_width

# Temporary sub-goal for exiting a confined room: the detected doorway
# point pulled back toward the robot by a fixed margin. Latched once (set
# the first time a valid doorway is seen while confined) and held fixed
# until reached or until is_confined goes false. Consumed by
# nominal_controller.py and mcbf_constraint_node.py in place of the real
# target while active.
bool has_temp_target
geometry_msgs/Point temp_target
