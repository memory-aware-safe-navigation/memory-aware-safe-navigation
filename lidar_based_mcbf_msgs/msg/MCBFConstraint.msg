std_msgs/Header header

# Output of the geodesic/MCBF e_vec computation formerly inlined in
# OnManCBFController.get_opt_problem_2D (cbf_control.py, the "if self.MCBF and
# not dynamic_obstacle and not disable_mcbf_forwall" block). Now owned by
# mcbf_constraint_node.py so the QP-solving node only has to combine e_vec
# with its own decision-variable expression (v_b_e = e_vec . dx[:2]) instead
# of recomputing the polygon/projection/geodesic pipeline itself.

# False when the block did not run this cycle (dynamic obstacle in play,
# wall-only scene with <1 non-wall cluster, or no obstacle within om_range) —
# e_vec/beta/om_idx are all zero-valued and must not be used to build a
# constraint when this is false.
bool valid

# Geodesic direction vector (z unused, 2D only). NaN-filled when not valid,
# matching the existing /e_vec NaN convention consumed by mapper_constraint.py.
geometry_msgs/Vector3 e_vec

float64 beta

# Index into om.env.gpdf_set of the obstacle cluster the geodesic was built
# against (selected_min_h_idx in the original code). -1 when not valid.
int32 om_idx

# -1.0 when not valid / not applicable (mirrors StageConstraint's min_gap convention)
float64 dist_nearest_proj_target
float64 dist_perp_proj_target

# Timing breakdown (polygon/projection/geodesic) is published directly by
# mcbf_constraint_node.py as Float32 on /controller_polygon_dt,
# /controller_projection_dt, /controller_geodesic_dt — the same topics
# controller.py used to publish under — rather than bundled here.
