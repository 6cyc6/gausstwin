import warp as wp

Vec2 = list[float] | tuple[float, float] | wp.vec2
"""A 2D vector represented as a list or tuple of 2 floats."""
Vec3 = list[float] | tuple[float, float, float] | wp.vec3
"""A 3D vector represented as a list or tuple of 3 floats."""
Vec4 = list[float] | tuple[float, float, float, float] | wp.vec4
"""A 4D vector represented as a list or tuple of 4 floats."""
Quat = list[float] | tuple[float, float, float, float] | wp.quat
"""A quaternion represented as a list or tuple of 4 floats (in XYZW order)."""
Mat33 = list[float] | wp.mat33
"""A 3x3 matrix represented as a list of 9 floats or a ``warp.mat33``."""
Transform = tuple[Vec3, Quat] | wp.transform
"""A 3D transformation represented as a tuple of 3D translation and rotation quaternion (in XYZW order)."""


# ======================================== Transform 3DGS =============================================== #
@wp.kernel
def transform_gaussian_means_kernel(
    means: wp.array(dtype=wp.vec3f), # type: ignore
    # gaussian_indices: wp.array(dtype=wp.int32), # type: ignore
    disp_indices: wp.array(dtype=wp.int32), # type: ignore
    rotations: wp.array(dtype=wp.quatf), # type: ignore
    displacements: wp.array(dtype=wp.vec3f), # type: ignore
    centers: wp.array(dtype=wp.vec3f), # type: ignore
    means_out: wp.array(dtype=wp.vec3f), # type: ignore
):
    tid = wp.tid()
    # mean_idx = gaussian_indices[tid]
    obj_idx = disp_indices[tid]

    means_centered = means[tid] - centers[obj_idx]
    # apply transformation
    means_rotated = wp.quat_rotate(rotations[obj_idx], means_centered)

    means_out[tid] = means_rotated + displacements[obj_idx] + centers[obj_idx]


@wp.kernel
def transform_gaussians_kernel(
    gaussian_pos: wp.array(dtype=wp.vec3),  # type: ignore
    gaussian_quats: wp.array(dtype=wp.quatf), # type: ignore wxyz
    indices: wp.array(dtype=wp.int32), # type: ignore
    rb_pos: wp.array(dtype=wp.vec3), # type: ignore
    rb_quats: wp.array(dtype=wp.quatf), # type: ignore xyzw
    transformed_pos: wp.array(dtype=wp.vec3), # type: ignore
    transformed_quats: wp.array(dtype=wp.vec4f) # type: ignore
): 
    tid = wp.tid()  
    idx = indices[tid]
    if idx == -1:
        return
    
    # Fetch body poses
    q = rb_quats[idx] # xyzw
    t = rb_pos[idx]

    # transform means
    means = gaussian_pos[tid]
    mean_rot = wp.quat_rotate(q, means)
    transformed_pos[tid] = mean_rot + t

    # transform quaternions
    quats_wxyz = gaussian_quats[tid]
    quats = wp.quat(quats_wxyz[1], quats_wxyz[2], quats_wxyz[3], quats_wxyz[0]) # wxyz -> xyzw
    # quat_trans = quat_mul(q, quats) # xyzw
    quat_trans = wp.normalize(q * quats)
    transformed_quats[tid] = wp.vec4(quat_trans[3], quat_trans[0], quat_trans[1], quat_trans[2])  # type: ignore

# ------------------- Rigid Body ------------------- #
# transform rigid body 3DGS
@wp.kernel
def transform_rigid_body_gaussians_kernel(
    gaussian_pos: wp.array(dtype=wp.vec3),  # type: ignore
    gaussian_quats: wp.array(dtype=wp.quatf), # type: ignore wxyz
    indices: wp.array(dtype=wp.int32), # type: ignore
    rb_state:  wp.array(dtype=wp.transform), # type: ignore
    disp_ground: float,
    transformed_pos: wp.array(dtype=wp.vec3), # type: ignore
    transformed_quats: wp.array(dtype=wp.vec4f) # type: ignore
): 
    tid = wp.tid()  
    bid = indices[tid]
    if bid == -1:
        return
    
    # fetch body poses
    tf = rb_state[bid]
    rb_t = wp.transform_get_translation(tf)
    rb_q = wp.transform_get_rotation(tf)

    # transform means
    disp_ground_h = wp.vec3(0.0, 0.0, disp_ground)
    means = gaussian_pos[tid]
    mean_rot = wp.quat_rotate(rb_q, means)
    transformed_pos[tid] = mean_rot + rb_t + disp_ground_h

    # transform quaternions
    q = gaussian_quats[tid]
    quats = wp.quatf(q[1], q[2], q[3], q[0]) # wxyz -> xyzw
    quat_trans = rb_q * quats
    quat_trans = wp.normalize(quat_trans)
    
    transformed_quats[tid] = wp.vec4(quat_trans[3], quat_trans[0], quat_trans[1], quat_trans[2]) # xyzw -> wxyz


# ------------------- Rope ------------------- #
@wp.kernel
def transform_rigid_body_gaussians_part_kernel(
    gaussian_pos: wp.array(dtype=wp.vec3),  # type: ignore
    gaussian_quats: wp.array(dtype=wp.quatf), # type: ignore wxyz
    gs_indices: wp.array(dtype=wp.int32), # type: ignore
    body_indices: wp.array(dtype=wp.int32), # type: ignore
    rb_state:  wp.array(dtype=wp.transform), # type: ignore
    disp_ground: float,
    transformed_pos: wp.array(dtype=wp.vec3), # type: ignore
    transformed_quats: wp.array(dtype=wp.vec4f) # type: ignore
): 
    tid = wp.tid()  
    gsid = gs_indices[tid]
    bid = body_indices[tid]

    if bid == -1:
        return
    
    # Fetch rigid body gaussians
    gs_means = gaussian_pos[gsid]
    gs_q = gaussian_quats[gsid]

    # Fetch body poses
    tf = rb_state[bid]
    rb_t = wp.transform_get_translation(tf)
    rb_q = wp.transform_get_rotation(tf)

    # transform means
    disp_ground_h = wp.vec3(0.0, 0.0, disp_ground)
    mean_rot = wp.quat_rotate(rb_q, gs_means)
    transformed_pos[gsid] = mean_rot + rb_t + disp_ground_h

    # transform quaternions
    quats = wp.quatf(gs_q[1], gs_q[2], gs_q[3], gs_q[0])
    quat_trans = rb_q * quats
    quat_trans = wp.normalize(quat_trans)
    # transformed_quats[tid] = quat_trans 
    transformed_quats[gsid] = wp.vec4(quat_trans[3], quat_trans[0], quat_trans[1], quat_trans[2])  # type: ignore


@wp.kernel
def transform_rope_gaussians_part_kernel(
    gaussian_pos: wp.array(dtype=wp.vec3),  # type: ignore
    gaussian_quats: wp.array(dtype=wp.quatf), # type: ignore wxyz
    gs_indices: wp.array(dtype=wp.int32), # type: ignore
    rope_indices: wp.array(dtype=wp.int32), # type: ignore
    rope_quat_indices: wp.array(dtype=wp.int32), # type: ignore
    rope_pos:  wp.array(dtype=wp.vec3), # type: ignore
    rope_quats:  wp.array(dtype=wp.quatf), # type: ignore
    ground_h: float,
    transformed_pos: wp.array(dtype=wp.vec3), # type: ignore
    transformed_quats: wp.array(dtype=wp.vec4f), # type: ignore
): 
    tid = wp.tid()  
    gsid = gs_indices[tid]
    pid = rope_indices[tid]
    pqid = rope_quat_indices[tid]

    if pid == -1:
        return
    
    # Fetch rope gaussians
    gs_means = gaussian_pos[gsid]
    gs_q = gaussian_quats[gsid]

    # Fetch rope poses
    rb_t = rope_pos[pid]
    rb_q = rope_quats[pqid]

    # transform means
    ground_h_disp = wp.vec3(0.0, 0.0, ground_h)
    mean_rot = wp.quat_rotate(rb_q, gs_means)
    transformed_pos[gsid] = mean_rot + rb_t + ground_h_disp

    # transform quaternions
    quats = wp.quatf(gs_q[1], gs_q[2], gs_q[3], gs_q[0])
    quat_trans = rb_q * quats
    quat_trans = wp.normalize(quat_trans)
    # transformed_quats[tid] = quat_trans 
    transformed_quats[gsid] = wp.vec4(quat_trans[3], quat_trans[0], quat_trans[1], quat_trans[2])  # type: ignore


@wp.kernel
def transform_tracked_particles_kernel(
    pts_body: wp.array(dtype=wp.vec3), # type: ignore
    positions: wp.array(dtype=wp.vec3), # type: ignore
    quats: wp.array(dtype=wp.quatf), # type: ignore xyzw
    obj_centers: wp.array(dtype=wp.vec3), # type: ignore
    indices: wp.array(dtype=wp.int32), # type: ignore
    out_pts: wp.array(dtype=wp.vec3), # type: ignore
):
    """Transform tracked particles based on robot link poses."""
    tid = wp.tid()  
    idx = indices[tid]

    # Fetch link transform
    q = wp.normalize(quats[idx])
    t = positions[idx]
    t_centers = obj_centers[idx]

    # Rotate + translate mean
    pts = pts_body[tid]
    pts_rotated = wp.quat_rotate(q, pts)
    
    out_pts[tid] = pts_rotated + t + t_centers
    
    # delta = pts_rotated + t - t_centers
    # out_pts[tid] = delta * 5.0 + t_centers


# for evaluation
@wp.kernel
def transform_gaussians_kernel_eval(
    gaussian_pos: wp.array(dtype=wp.vec3),  # type: ignore
    gaussian_quats: wp.array(dtype=wp.quatf), # type: ignore wxyz
    indices: wp.array(dtype=wp.int32), # type: ignore
    rb_pos: wp.array(dtype=wp.vec3), # type: ignore
    rb_quats: wp.array(dtype=wp.quatf), # type: ignore xyzw
    transformed_pos: wp.array(dtype=wp.vec3), # type: ignore
    transformed_quats: wp.array(dtype=wp.vec4f) # type: ignore
): 
    tid = wp.tid()  
    idx = indices[tid]
    if idx == -1:
        return
    
    # Fetch body poses
    q = rb_quats[idx] # xyzw
    t = rb_pos[idx]

    # transform means
    means = gaussian_pos[tid]
    mean_rot = wp.quat_rotate(q, means)
    transformed_pos[tid] = mean_rot + t

    # transform quaternions
    quats_wxyz = gaussian_quats[tid]
    quats = wp.quat(quats_wxyz[1], quats_wxyz[2], quats_wxyz[3], quats_wxyz[0]) # wxyz -> xyzw
    # quat_trans = quat_mul(q, quats) # xyzw
    quat_trans = wp.normalize(q * quats)
    transformed_quats[tid] = wp.vec4(quat_trans[3], quat_trans[0], quat_trans[1], quat_trans[2])  # type: ignore
    
# ============================================= Velocity Damping ==================================================== #
@wp.kernel
def velocity_damping_particle_kernel(
    factor: float,
    qd_0: wp.array(dtype=wp.vec3), # type: ignore
    qd_1: wp.array(dtype=wp.vec3), # type: ignore
):
    tid = wp.tid()
    qd_0[tid] = qd_0[tid] * factor
    qd_1[tid] = qd_1[tid] * factor
    

@wp.kernel
def velocity_damping_rigid_body_kernel(
    factor: float,
    body_qd_0: wp.array(dtype=wp.spatial_vector), # type: ignore
    body_qd_1: wp.array(dtype=wp.spatial_vector), # type: ignore
):
    tid = wp.tid()
    for j in range(6):
        body_qd_0[tid][j] = body_qd_0[tid][j] * factor
        body_qd_1[tid][j] = body_qd_1[tid][j] * factor


@wp.kernel
def velocity_damping_rod_kernel(
    factor: float,
    rod_qd_0: wp.array(dtype=wp.vec3), # type: ignore
    rod_qd_1: wp.array(dtype=wp.vec3), # type: ignore
):
    tid = wp.tid()
    rod_qd_0[tid] = rod_qd_0[tid] * factor
    rod_qd_1[tid] = rod_qd_1[tid] * factor
    
    # tid = wp.tid()
    # for j in range(3):
    #     rod_qd_0[tid][j] = rod_qd_0[tid][j] * factor
    #     rod_qd_1[tid][j] = rod_qd_1[tid][j] * factor

# ============================================= Visual Forces ==================================================== #
# ------------------------------------------- for rope ------------------------------------------ # 
@wp.kernel
def set_visual_rope_force_kernel(
    kp: float,
    prev_mean: wp.array(dtype=wp.vec3f), # type: ignore
    target_mean: wp.array(dtype=wp.vec3f), # type: ignore
    part_indices: wp.array(dtype=wp.int32), # type: ignore
    weights: wp.array(dtype=wp.float32), # type: ignore
    particle_f: wp.array(dtype=wp.vec3f),  # type: ignore
):
    tid = wp.tid()
    pid = part_indices[tid]
    
    ws = weights[tid]
    delta = target_mean[tid] - prev_mean[tid]
    dist = wp.length(delta)
    
    if dist >= 0.002:
        f = delta * kp * ws
    else:
        f = delta * 0.0

    wp.atomic_add(particle_f, pid, f)


# ----------------------------------- for rigid body dynamics ----------------------------------- # 
@wp.kernel
def set_visual_body_force_kernel(
    kp_f: float,
    kp_m: float,
    prev_means: wp.array(dtype=wp.vec3f),  # type: ignore
    target_means: wp.array(dtype=wp.vec3f),  # type: ignore
    weights: wp.array(dtype=wp.float32), # type: ignore
    body_ids_wp: wp.array(dtype=wp.int32),  # type: ignore
    body_ids_opt: wp.array(dtype=wp.int32),  # type: ignore
    body_q: wp.array(dtype=wp.transform),  # type: ignore
    forces: wp.array(dtype=wp.vec3), # type: ignore
    moments: wp.array(dtype=wp.vec3), # type: ignore
):
    tid = wp.tid()
    bid_wp = body_ids_wp[tid]
    bid_opt = body_ids_opt[tid]
    
    # get visual force
    delta = target_means[tid] - prev_means[tid]
    f = delta * weights[tid]
    # get moment 
    body_trans = body_q[bid_wp]
    com = wp.transform_get_translation(body_trans)
    r = prev_means[tid] - com
    moment_f = wp.cross(r, f)
    
    # accumulate body force and moment
    wp.atomic_add(forces, bid_opt, f * kp_f)
    wp.atomic_add(moments, bid_opt, moment_f * kp_m) # type: ignore


@wp.kernel
def set_visual_body_force_new_kernel(
    kp_f: float,
    kp_m: float,
    prev_means: wp.array(dtype=wp.vec3f),  # type: ignore
    target_means: wp.array(dtype=wp.vec3f),  # type: ignore
    opacities: wp.array(dtype=wp.float32), # type: ignore
    weights: wp.array(dtype=wp.float32), # type: ignore
    body_ids_wp: wp.array(dtype=wp.int32),  # type: ignore
    body_ids_opt: wp.array(dtype=wp.int32),  # type: ignore
    body_q: wp.array(dtype=wp.transform),  # type: ignore
    forces: wp.array(dtype=wp.vec3), # type: ignore
    moments: wp.array(dtype=wp.vec3), # type: ignore
):
    tid = wp.tid()
    bid_wp = body_ids_wp[tid]
    bid_opt = body_ids_opt[tid]
    opcity = opacities[tid]
    weight = weights[tid]
    
    # get visual force
    delta = target_means[tid] - prev_means[tid]
    f = delta * weight * opcity
    # f = delta * opcity
    # get moment 
    body_trans = body_q[bid_wp]
    com = wp.transform_get_translation(body_trans)
    r = prev_means[tid] - com
    moment_f = wp.cross(r, f)
    
    # accumulate body force and moment
    wp.atomic_add(forces, bid_opt, f * kp_f)
    wp.atomic_add(moments, bid_opt, moment_f * kp_m) # type: ignore


@wp.kernel
def apply_body_force_kernel(
    total_force: wp.array(dtype=wp.vec3f),  # type: ignore
    total_moment: wp.array(dtype=wp.vec3f),  # type: ignore
    body_ids: wp.array(dtype=wp.int32),  # type: ignore
    body_f: wp.array(dtype=wp.spatial_vectorf),  # type: ignore
):
    tid = wp.tid()
    bid = body_ids[tid]
    body_f[bid] = wp.spatial_vector(total_moment[tid], total_force[tid])  # type: ignore