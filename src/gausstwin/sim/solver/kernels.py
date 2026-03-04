import warp as wp

from warp.sim.integrator_xpbd import PARTICLE_FLAG_ACTIVE


# ==================================== Solve Constraints ==================================== #
@wp.kernel
def solve_particle_particle_contacts(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3), # type: ignore
    particle_v: wp.array(dtype=wp.vec3), # type: ignore
    particle_invmass: wp.array(dtype=float), # type: ignore
    particle_radius: wp.array(dtype=float), # type: ignore
    particle_flags: wp.array(dtype=wp.uint32), # type: ignore
    obj_indices: wp.array(dtype=int), # type: ignore
    k_mu: float,
    k_cohesion: float,
    max_radius: float,
    dt: float,
    relaxation: float,
    deltas: wp.array(dtype=wp.vec3), # type: ignore
):
    """Solve particle-particle constraint."""
    tid = wp.tid()
    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        # hash grid has not been built yet
        return
    if (particle_flags[i] & PARTICLE_FLAG_ACTIVE) == 0:
        return

    x = particle_x[i]
    v = particle_v[i]
    radius = particle_radius[i]
    w1 = particle_invmass[i]
    obj_idx = obj_indices[i]

    # particle contact
    query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion)
    index = int(0)

    delta = wp.vec3(0.0)

    while wp.hash_grid_query_next(query, index):
        if (particle_flags[index] & PARTICLE_FLAG_ACTIVE) != 0 and index != i:
            # check if two points are from the same shape   
            if obj_indices[index] == obj_idx and obj_idx > -0.5 and (index - i <= 2 or i - index <= 2):
                continue  # Skip if they belong to the same rope and are adjacent

            # compute distance to point
            n = x - particle_x[index]
            d = wp.length(n)
            err = d - radius - particle_radius[index]

            # compute inverse masses
            w2 = particle_invmass[index]
            denom = w1 + w2

            if err <= k_cohesion and denom > 0.0:
                n = n / d
                vrel = v - particle_v[index]

                # normal
                lambda_n = err
                delta_n = n * lambda_n

                # friction
                vn = wp.dot(n, vrel)
                vt = v - n * vn

                lambda_f = wp.max(k_mu * lambda_n, -wp.length(vt) * dt)
                lambda_f = wp.max(k_mu * lambda_n, -wp.length(vt) * dt)
                delta_f = wp.normalize(vt) * lambda_f
                delta += (delta_f - delta_n) / denom 

    wp.atomic_add(deltas, i, delta * w1 * relaxation)


# ===================================== ELASTIC ROD ==================================== #
# solve rod length constraints
@wp.kernel
def solve_rod_length_constraints(
    particle_x: wp.array(dtype=wp.vec3), # type: ignore
    particle_inv_mass: wp.array(dtype=float), # type: ignore
    rod_edge_particle_indices: wp.array(dtype=int), # type: ignore
    rod_rest_lengths: wp.array(dtype=float), # type: ignore
    stiffness: float,
    delta_x: wp.array(dtype=wp.vec3), # type: ignore
):
    """
    Solve distance constraints for rod segments to maintain consistent length.
    This ensures rod segments don't stretch or compress beyond their rest length.
    """
    tid = wp.tid()

    # get particle indices of each segment
    i = rod_edge_particle_indices[tid * 2 + 0]
    j = rod_edge_particle_indices[tid * 2 + 1]

    xi = particle_x[i]
    xj = particle_x[j]
    
    wi = particle_inv_mass[i]
    wj = particle_inv_mass[j]
    
    # Skip if both particles have infinite mass
    if wi == 0.0 and wj == 0.0:
        return
    
    rest_length = rod_rest_lengths[tid]
    
    eps = 1e-9
    
    # Compute current distance and direction
    delta = xi - xj
    current_length = wp.length(delta)
    
    # Avoid division by zero
    if current_length < eps:
        return
    
    # Constraint: C = current_length - rest_length
    constraint_value = current_length - rest_length
    
    # Skip if constraint is already satisfied (within tolerance)
    if wp.abs(constraint_value) < eps:
        return
    
    # Normalized direction vector
    n = delta / current_length
    
    # PBD constraint correction
    # λ = -C / (∇C^T M^-1 ∇C)
    # For distance constraints: ∇C^T M^-1 ∇C = wi + wj
    w_total = wi + wj
    lambda_correction = constraint_value / w_total
    
    if w_total < eps:
        return
    
    correction_i = -lambda_correction * wi * n * stiffness
    correction_j = lambda_correction * wj * n * stiffness
    
    # Atomically add corrections to avoid race conditions
    wp.atomic_add(delta_x, i, correction_i)
    wp.atomic_add(delta_x, j, correction_j)


# solve rod stretch/shear constraints
@wp.kernel
def solve_rod_stretch_shear_constraints(
    particle_x: wp.array(dtype=wp.vec3), # type: ignore
    rod_quaternions: wp.array(dtype=wp.quat), # type: ignore
    rod_indicies: wp.array(dtype=int), # type: ignore
    inv_mass: wp.array(dtype=float), # type: ignore
    rod_edge_particle_indices: wp.array(dtype=int), # type: ignore
    inv_mass_q: wp.array(dtype=float), # type: ignore
    rest_lengths: wp.array(dtype=float), # type: ignore
    stiffness: wp.array(dtype=wp.vec3), # type: ignore                 
    delta_x: wp.array(dtype=wp.vec3), # type: ignore
    delta_q: wp.array(dtype=wp.quat), # type: ignore

):
    """
    Reference Paper:
    Jan Bender, Matthias Müller, and Miles Macklin. "A survey on position based dynamics, 2017." 
    Proceedings of the European Association for Computer Graphics: Tutorials (2017): 1-31.
    section 5.8.3
    Reference C++ code: 
    https://github.com/InteractiveComputerGraphics/PositionBasedDynamics/blob/master/PositionBasedDynamics/PositionBasedElasticRods.cpp
    """
    tid = wp.tid()
    rod_idx = rod_indicies[tid]

    # get particle indices of each segment
    i = rod_edge_particle_indices[tid * 2 + 0]
    j = rod_edge_particle_indices[tid * 2 + 1]

    xi = particle_x[i]
    xj = particle_x[j]

    wi = inv_mass[i]
    wj = inv_mass[j]

    q0 = rod_quaternions[tid]
    wq = inv_mass_q[tid]

    rest_len = rest_lengths[tid]

    ks = stiffness[rod_idx]  

    eps = 1e-6

    # Compute d3(q) = R(q) * e3, e3 = (0,0,1)
    # d3 = q0 * e_3 * q0_conjugate
    # here, we use wp.quat_rotate() 
    e3 = wp.vec3(0.0, 0.0, 1.0)
    d3 = wp.quat_rotate(q0, e3)

    # Formula: C_s = (p2 - p1) / l - d3(q)
    gamma = (xj - xi) / (rest_len + eps) - d3

    # get weight factors for each delta
    # In Paper formula (37)
    # w1 * l / (w1 + w2 + 4wq * l * l)
    # w2 * l / (w1 + w2 + 4wq * l * l)
    # wq * l * l / (w1 + w2 + 4wq * l * l)
    # Here
    # denom = 1 / [l / (w1 + w2 + 4wq * l * l)]
    denom = (wi + wj) / (rest_len + eps) + wq * 4.0 * rest_len + eps
    gamma /= denom

    # apply stiffness
    if wp.abs(ks[0] - ks[1]) < eps and wp.abs(ks[0] - ks[2]) < eps: # if all ks are approx. equal
        gamma_scaled = wp.cw_mul(gamma, ks)
    else:
        # transform to material frame
        gamma_local = wp.quat_rotate_inv(q0, gamma)
        # scale each axis
        gamma_local_scaled = wp.cw_mul(gamma_local, ks)
        # transform back
        gamma_scaled = wp.quat_rotate(q0, gamma_local_scaled)

    # Position corrections
    corr0 = wi * gamma_scaled 
    corr1 = -wj * gamma_scaled 

    wp.atomic_add(delta_x, i, corr0)
    wp.atomic_add(delta_x, j, corr1)

    # Quaternion correction
    # Compute q*e3_conj efficiently
    # qx, qy, qz, qw = q0[0], q0[1], q0[2], q0[3]
    # q_e3_bar = wp.quat(-qy, qx, -qw, qz)
    q_e3_bar = wp.quat(-q0[1], q0[0], -q0[3], q0[2])

    gamma_quat = wp.quat(gamma_scaled[0], gamma_scaled[1], gamma_scaled[2], 0.0) # type: ignore
    corrq = gamma_quat * q_e3_bar

    corrq = corrq * 2.0 * wq * rest_len 

    wp.atomic_add(delta_q, tid, corrq)



@wp.func
def compute_diff_darboux_vec(
    omega: wp.quat,
    rest_darboux: wp.quat
) -> wp.vec3:
    # displacement towards the nearest rest pose
    # C_b(q1, q2) = darboux_vec - alpha * rest_darboux_vec
    # alpha = 1  if length(omega_minus) <= length(omega_plus)
    # alpha = -1 if length(omega_minus) >  length(omega_plus)
    plus = omega + rest_darboux
    minus = omega - rest_darboux

    omega_plus = wp.vec3(plus[0], plus[1], plus[2])
    omega_minus = wp.vec3(minus[0], minus[1], minus[2])

    if wp.length(omega_minus) > wp.length(omega_plus):
        return omega_plus  # choose the closer vector
    else:
        return omega_minus
    

# solve rod bend/twist constraints
@wp.kernel
def solve_rod_bend_twist_constraints(
    rod_quaternions: wp.array(dtype=wp.quat), # type:ignore
    inv_mass_q: wp.array(dtype=float), # type:ignore
    rest_darboux: wp.array(dtype=wp.quat), # type:ignore
    bending_twisting_ks: wp.array(dtype=wp.vec3), # type:ignore
    rod_edge_seg_indices: wp.array(dtype=int), # type: ignore
    rod_indicies: wp.array(dtype=int), # type: ignore
    delta_q: wp.array(dtype=wp.quat), # type:ignore
):
    """
    Reference Paper:
    Bender, Jan, Matthias Müller, and Miles Macklin. "A survey on position based dynamics, 2017." 
    Proceedings of the European Association for Computer Graphics: Tutorials (2017): 1-31.
    section 5.8.3
    Reference C++ code: 
    https://github.com/InteractiveComputerGraphics/PositionBasedDynamics/blob/master/PositionBasedDynamics/PositionBasedElasticRods.cpp
    """
    tid = wp.tid()
    rod_idx = rod_indicies[tid]

    # get segment indices for each darboux vector
    i = rod_edge_seg_indices[tid * 2 + 0]
    j = rod_edge_seg_indices[tid * 2 + 1]

    q0 = rod_quaternions[i]
    q1 = rod_quaternions[j]

    invMassq0 = inv_mass_q[i]
    invMassq1 = inv_mass_q[j]

    ks = bending_twisting_ks[rod_idx]
    rest_darboux_vec = rest_darboux[tid]

    eps = 1e-6

    # compute discrete Darboux vector
    omega = wp.quat_inverse(q0) * q1

    # compute C_b(q1, q2) = darboux_vec - rest_darboux_vec
    diff_darboux_vec = compute_diff_darboux_vec(omega, rest_darboux_vec)

    # apply bending and twisting stiffness per axis
    denom = 1.0 / (invMassq0 + invMassq1 + eps)
    omega_quat = wp.quat(
        diff_darboux_vec[0] * ks[0] * denom, 
        diff_darboux_vec[1] * ks[1] * denom, 
        diff_darboux_vec[2] * ks[2] * denom, 
        0.0
    )

    # Compute quaternion corrections
    corrq0 = q1 * omega_quat * invMassq0
    corrq1 = q0 * omega_quat * (-invMassq1)

    # Atomically accumulate the quaternion corrections
    wp.atomic_add(delta_q, i, corrq0)
    wp.atomic_add(delta_q, j, corrq1)
    
# ==================================== State Update ==================================== #
# ---------------- Rods ---------------- #
@wp.kernel
def apply_rod_delta_kernel(
    rod_q: wp.array(dtype=wp.quat), # type: ignore
    delta_q: wp.array(dtype=wp.quat), # type: ignore
    rod_q_out: wp.array(dtype=wp.quat), # type: ignore
):
    tid = wp.tid()
    q = rod_q[tid]
    dq = delta_q[tid]
    
    q_corr = q + dq
    q_corr = wp.normalize(q_corr)
    
    rod_q_out[tid] = q_corr


@wp.kernel
def update_rod_qd_kernel(
    rod_q_orig: wp.array(dtype=wp.quat), # type: ignore
    rod_q_pred: wp.array(dtype=wp.quat), # type: ignore
    dt: float,
    rod_qd_out: wp.array(dtype=wp.vec3), # type: ignore
):
    tid = wp.tid()
    q0 = rod_q_orig[tid]
    q1 = rod_q_pred[tid]
    
    # relative rotation from q0 -> q1
    qr = q1 * wp.quat_inverse(q0)

    # shortest rotation
    if qr[3] < 0.0:
        qr = -qr

    axis, angle = wp.quat_to_axis_angle(qr)

    if wp.abs(angle) < 1.0e-6:
        omega = wp.vec3(0.0, 0.0, 0.0)
    else:
        omega = (angle / dt) * axis

    rod_qd_out[tid] = omega
    

@wp.kernel
def apply_rod_deltas_kernel(
    q_orig: wp.array(dtype=wp.quat), # type: ignore
    q_pred: wp.array(dtype=wp.quat), # type: ignore
    delta_q: wp.array(dtype=wp.quat), # type: ignore
    dt: float,
    # outputs
    q_out: wp.array(dtype=wp.quat), # type: ignore 
    qd_out: wp.array(dtype=wp.vec3), # type: ignore
):
    tid = wp.tid()
    q0 = q_orig[tid]
    u = q_pred[tid]
    
    dq = delta_q[tid]
    
    q_new = u + dq
    q_new = wp.normalize(q_new)
    q1 = 2.0 * wp.quat_inverse(q0) * q_new / dt
    qd_new = wp.vec3(q1[0], q1[1], q1[2])

    # enforce velocity limit to prevent instability
    v_new_mag = wp.length(qd_new)
    if v_new_mag > 1e3:
        qd_new *= 1e3 / v_new_mag
        
    q_out[tid] = q_new
    qd_out[tid] = qd_new
    

# ---------------- Rigid Bodies ---------------- #
@wp.kernel
def remove_robot_deltas_kernel(
    delta: wp.array(dtype=wp.vec3), # type: ignore
    particle_is_robot: wp.array(dtype=wp.bool), # type: ignore
):
    tid = wp.tid()
    if particle_is_robot[tid]:
        delta[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def zero_robot_deltas_kernel(
    delta: wp.array(dtype=wp.vec3), # type: ignore
    particle_is_robot_indices: wp.array(dtype=wp.int32), # type: ignore
):
    tid = wp.tid()
    rid = particle_is_robot_indices[tid]
    
    delta[rid] = wp.vec3(0.0, 0.0, 0.0)
        

@wp.kernel
def apply_body_deltas(
    q_in: wp.array(dtype=wp.transform), # type: ignore
    qd_in: wp.array(dtype=wp.spatial_vector), # type: ignore
    body_com: wp.array(dtype=wp.vec3), # type: ignore
    body_I: wp.array(dtype=wp.mat33), # type: ignore
    body_inv_m: wp.array(dtype=float), # type: ignore
    body_inv_I: wp.array(dtype=wp.mat33), # type: ignore
    deltas: wp.array(dtype=wp.spatial_vector), # type: ignore
    constraint_inv_weights: wp.array(dtype=float), # type: ignore
    dt: float,
    is_robot: wp.array(dtype=wp.bool), # type: ignore
    # outputs
    q_out: wp.array(dtype=wp.transform), # type: ignore
    qd_out: wp.array(dtype=wp.spatial_vector), # type: ignore
):
    tid = wp.tid()
    if is_robot[tid]:
        return  # Skip robot bodies
    
    inv_m = body_inv_m[tid]
    if inv_m == 0.0:
        q_out[tid] = q_in[tid]
        qd_out[tid] = qd_in[tid]
        return
    inv_I = body_inv_I[tid]

    tf = q_in[tid]
    delta = deltas[tid]

    p0 = wp.transform_get_translation(tf)
    q0 = wp.transform_get_rotation(tf)

    weight = 1.0
    if constraint_inv_weights:
        inv_weight = constraint_inv_weights[tid]
        if inv_weight > 0.0:
            weight = 1.0 / inv_weight

    dp = wp.spatial_bottom(delta) * (inv_m * weight)
    dq = wp.spatial_top(delta) * weight
    dq = wp.quat_rotate(q0, inv_I * wp.quat_rotate_inv(q0, dq))

    # update orientation
    q1 = q0 + 0.5 * wp.quat(dq * dt, 0.0) * q0
    q1 = wp.normalize(q1)

    # update position
    com = body_com[tid]
    x_com = p0 + wp.quat_rotate(q0, com)
    p1 = x_com + dp * dt
    p1 -= wp.quat_rotate(q1, com)

    q_out[tid] = wp.transform(p1, q1)

    v0 = wp.spatial_bottom(qd_in[tid])
    w0 = wp.spatial_top(qd_in[tid])

    # update linear and angular velocity
    v1 = v0 + dp
    # angular part (compute in body frame)
    wb = wp.quat_rotate_inv(q0, w0 + dq)
    tb = -wp.cross(wb, body_I[tid] * wb)  # coriolis forces
    w1 = wp.quat_rotate(q0, wb + inv_I * tb * dt)

    # XXX this improves gradient stability
    if wp.length(v1) < 1e-4:
        v1 = wp.vec3(0.0)
    if wp.length(w1) < 1e-4:
        w1 = wp.vec3(0.0)

    qd_out[tid] = wp.spatial_vector(w1, v1)
    
# ---------------- Particles ---------------- #
@wp.kernel
def apply_particle_deltas(
    x_orig: wp.array(dtype=wp.vec3), # type: ignore
    x_pred: wp.array(dtype=wp.vec3), # type: ignore
    particle_flags: wp.array(dtype=wp.uint32), # type: ignore
    obj_indices: wp.array(dtype=wp.int32), # type: ignore
    delta: wp.array(dtype=wp.vec3), # type: ignore
    dt: float,
    v_max: float,
    x_out: wp.array(dtype=wp.vec3), # type: ignore
    v_out: wp.array(dtype=wp.vec3), # type: ignore
):
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return
    obj_id = obj_indices[tid]
    if obj_id < 0:
        return  # skip particles that do not belong to an object

    x0 = x_orig[tid]
    xp = x_pred[tid]

    # constraint deltas
    d = delta[tid]

    x_new = xp + d
    v_new = (x_new - x0) / dt

    # enforce velocity limit to prevent instability
    v_new_mag = wp.length(v_new)
    if v_new_mag > v_max:
        v_new *= v_max / v_new_mag

    x_out[tid] = x_new
    v_out[tid] = v_new
    