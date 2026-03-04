import warp as wp
import warp.sim

from warp.sim import Control
from warp.sim.integrator_xpbd import solve_body_joints, update_body_velocities, bending_constraint, \
solve_particle_ground_contacts, apply_joint_actions, solve_body_contact_positions, \
solve_particle_shape_contacts, solve_springs, solve_tetrahedra, PARTICLE_FLAG_ACTIVE, \
apply_particle_shape_restitution, apply_particle_ground_restitution, apply_rigid_restitution, apply_body_delta_velocities

from .kernels import *
from ..model_builder.model import Model, State
from typing_extensions import override


# ==================================== INTEGRATOR ==================================== #
@wp.kernel
def integrate_particles(
    x: wp.array(dtype=wp.vec3), # type: ignore
    v: wp.array(dtype=wp.vec3), # type: ignore
    f: wp.array(dtype=wp.vec3), # type: ignore
    w: wp.array(dtype=float), # type: ignore
    particle_flags: wp.array(dtype=wp.uint32), # type: ignore
    obj_indices: wp.array(dtype=int), # type: ignore
    gravity: wp.vec3,
    dt: float,
    v_max: float,
    x_new: wp.array(dtype=wp.vec3), # type: ignore
    v_new: wp.array(dtype=wp.vec3), # type: ignore
): 
    tid = wp.tid()
    if (particle_flags[tid] & PARTICLE_FLAG_ACTIVE) == 0:
        return
    
    # do not integrate particles that are labeled as -1 (e.g., fixed particles)
    obj_idx = obj_indices[tid]
    if obj_idx < 0:
        return

    x0 = x[tid]
    v0 = v[tid]
    f0 = f[tid]

    inv_mass = w[tid]

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v1 = v0 + (f0 * inv_mass + gravity * wp.step(-inv_mass)) * dt
    # enforce velocity limit to prevent instability
    v1_mag = wp.length(v1)
    if v1_mag > v_max:
        v1 *= v_max / v1_mag
    x1 = x0 + v1 * dt

    x_new[tid] = x1
    v_new[tid] = v1


@wp.func
def integrate_rigid_body(
    q: wp.transform,
    qd: wp.spatial_vector,
    f: wp.spatial_vector,
    com: wp.vec3,
    inertia: wp.mat33,
    inv_mass: float,
    inv_inertia: wp.mat33,
    gravity: wp.vec3,
    angular_damping: float,
    dt: float,
):
    # unpack transform
    x0 = wp.transform_get_translation(q)
    r0 = wp.transform_get_rotation(q)

    # unpack spatial twist
    w0 = wp.spatial_top(qd)
    v0 = wp.spatial_bottom(qd)

    # unpack spatial wrench
    t0 = wp.spatial_top(f)
    f0 = wp.spatial_bottom(f)

    x_com = x0 + wp.quat_rotate(r0, com) # type: ignore

    # linear part
    v1 = v0 + (f0 * inv_mass + gravity * wp.nonzero(inv_mass)) * dt # type: ignore
    x1 = x_com + v1 * dt

    # angular part (compute in body frame)
    wb = wp.quat_rotate_inv(r0, w0)
    tb = wp.quat_rotate_inv(r0, t0) - wp.cross(wb, inertia * wb)  # coriolis forces  # type: ignore

    w1 = wp.quat_rotate(r0, wb + inv_inertia * tb * dt)
    r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)

    # angular damping
    w1 *= 1.0 - angular_damping * dt  # type: ignore

    q_new = wp.transform(x1 - wp.quat_rotate(r1, com), r1)
    qd_new = wp.spatial_vector(w1, v1)  # type: ignore

    return q_new, qd_new


# semi-implicit euler integration
@wp.kernel
def integrate_bodies(
    body_q: wp.array(dtype=wp.transform),  # type: ignore
    body_qd: wp.array(dtype=wp.spatial_vector),  # type: ignore
    body_f: wp.array(dtype=wp.spatial_vector),  # type: ignore
    body_com: wp.array(dtype=wp.vec3),  # type: ignore
    m: wp.array(dtype=float),  # type: ignore
    i: wp.array(dtype=wp.mat33),  # type: ignore
    inv_m: wp.array(dtype=float),  # type: ignore
    inv_i: wp.array(dtype=wp.mat33),  # type: ignore
    gravity_factor: wp.array(dtype=float),  # type: ignore
    gravity: wp.vec3,
    angular_damping: float,
    dt: float,
    is_robot: wp.array(dtype=wp.bool),  # type: ignore
    # outputs
    body_q_new: wp.array(dtype=wp.transform),  # type: ignore
    body_qd_new: wp.array(dtype=wp.spatial_vector),  # type: ignore
):
    tid = wp.tid()

    if is_robot[tid]:
        return  # Skip integration for robot bodies
    
    # positions
    q = body_q[tid]
    qd = body_qd[tid]
    f = body_f[tid]

    # masses
    inv_mass = inv_m[tid]  # 1 / mass

    inertia = i[tid]
    inv_inertia = inv_i[tid]  # inverse of 3x3 inertia matrix

    com = body_com[tid]

    q_new, qd_new = integrate_rigid_body(
        q,
        qd,
        f,
        com,
        inertia,
        inv_mass,
        inv_inertia,
        gravity * gravity_factor[tid],
        angular_damping,
        dt,
    )

    body_q_new[tid] = q_new
    body_qd_new[tid] = qd_new


@wp.kernel
def integrate_rods_kernel(
    rod_q: wp.array(dtype=wp.quat),  # type: ignore
    rod_qd: wp.array(dtype=wp.vec3),  # type: ignore
    dt: float,
    rod_q_new: wp.array(dtype=wp.quat),  # type: ignore
):
    tid = wp.tid()

    q = rod_q[tid]
    w = rod_qd[tid]
    
    if wp.length(w) < 1.0e-6:
        q_new = q
    else:
        q_new = q + 0.5 * dt * q * wp.quat(w[0], w[1], w[2], 0.0)
    q_new = wp.normalize(q_new)

    rod_q_new[tid] = q_new
    

class XPBDIntegrator(warp.sim.XPBDIntegrator):
    """
    Override integrate bodies to make robot not affected by gravity
    """
    # For bodies
    @override
    def integrate_bodies(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        dt: float,
        angular_damping: float = 0.0,
    ):
        """
        Integrate the rigid bodies of the model.

        Args:
            model (Model): The model to integrate.
            state_in (State): The input state.
            state_out (State): The output state.
            dt (float): The time step (typically in seconds).
            angular_damping (float, optional): The angular damping factor. Defaults to 0.0.
        """
        if model.body_count:
            wp.launch(
                kernel=integrate_bodies,
                dim=model.body_count,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    state_in.body_f,
                    model.body_com,
                    model.body_mass,
                    model.body_inertia,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    model.gravity_factor,  
                    model.gravity,
                    angular_damping,
                    dt,
                    model.body_is_robot,  
                ],
                outputs=[state_out.body_q, state_out.body_qd],
                device=model.device,
            )
    
    
    @override
    def apply_body_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        body_deltas: wp.array,
        dt: float,
        rigid_contact_inv_weight: wp.array = None,
    ):
        with wp.ScopedTimer("apply_body_deltas", False):
            if state_in.requires_grad:
                body_q = state_out.body_q
                body_qd = state_out.body_qd
                new_body_q = wp.clone(body_q)
                new_body_qd = wp.clone(body_qd)
                self._body_delta_counter += 1
            else:
                if self._body_delta_counter == 0:
                    body_q = state_out.body_q
                    body_qd = state_out.body_qd
                    new_body_q = state_in.body_q
                    new_body_qd = state_in.body_qd
                else:
                    body_q = state_in.body_q
                    body_qd = state_in.body_qd
                    new_body_q = state_out.body_q
                    new_body_qd = state_out.body_qd
                self._body_delta_counter = 1 - self._body_delta_counter

            wp.launch(
                kernel=apply_body_deltas,
                dim=model.body_count,
                inputs=[
                    body_q,
                    body_qd,
                    model.body_com,
                    model.body_inertia,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    body_deltas,
                    rigid_contact_inv_weight,
                    dt,
                    model.body_is_robot,
                ],
                outputs=[
                    new_body_q,
                    new_body_qd,
                ],
                device=model.device,
            )

            if state_in.requires_grad:
                state_out.body_q = new_body_q
                state_out.body_qd = new_body_qd

        return new_body_q, new_body_qd
    
    # for particles
    @override
    def integrate_particles(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        dt: float,
    ):
        """
        Integrate the particles of the model.

        Args:
            model (Model): The model to integrate.
            state_in (State): The input state.
            state_out (State): The output state.
            dt (float): The time step (typically in seconds).
        """
        if model.particle_count:
            wp.launch(
                kernel=integrate_particles,
                dim=model.particle_count,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    state_in.particle_f,
                    model.particle_inv_mass,
                    model.particle_flags,
                    model.particle_obj_ids,
                    model.gravity,
                    dt,
                    model.particle_max_velocity,
                ],
                outputs=[state_out.particle_q, state_out.particle_qd],
                device=model.device,
            )
    
    @override
    def apply_particle_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        particle_deltas: wp.array,
        dt: float,
    ):
        if state_in.requires_grad:
            particle_q = state_out.particle_q
            # allocate new particle arrays so gradients can be tracked correctly without overwriting
            new_particle_q = wp.empty_like(state_out.particle_q)
            new_particle_qd = wp.empty_like(state_out.particle_qd)
            self._particle_delta_counter += 1
        else:
            if self._particle_delta_counter == 0:
                particle_q = state_out.particle_q
                new_particle_q = state_in.particle_q
                new_particle_qd = state_in.particle_qd
            else:
                particle_q = state_in.particle_q
                new_particle_q = state_out.particle_q
                new_particle_qd = state_out.particle_qd
            self._particle_delta_counter = 1 - self._particle_delta_counter

        wp.launch(
            kernel=apply_particle_deltas,
            dim=model.particle_count,
            inputs=[
                self.particle_q_init,
                particle_q,
                model.particle_flags,
                model.particle_obj_ids,
                particle_deltas,
                dt,
                model.particle_max_velocity,
            ],
            outputs=[new_particle_q, new_particle_qd],
            device=model.device,
        )

        if state_in.requires_grad:
            state_out.particle_q = new_particle_q
            state_out.particle_qd = new_particle_qd

        return new_particle_q, new_particle_qd
    
    # for rod
    def integrate_rods(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        dt: float,
    ):
        """
        Integrate the elastic rods of the model.

        Args:
            model (Model): The model to integrate.
            state_in (State): The input state.
            state_out (State): The output state.
            dt (float): The time step (typically in seconds).
        """
        if model.rod_count:
            wp.launch(
                kernel=integrate_rods_kernel,
                dim=model.rod_seg_count,
                inputs=[
                    state_in.rod_q, 
                    state_in.rod_qd,
                    dt
                ],
                outputs=[state_out.rod_q],
                device=model.device,
            )
    
    
    def apply_rod_deltas(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        rod_deltas: wp.array,
        dt: float,
    ):
        with wp.ScopedTimer("apply_rod_deltas", False):
            if state_in.requires_grad:
                rod_q = state_out.rod_q
                # allocate new rod arrays so gradients can be tracked correctly without overwriting
                new_rod_q = wp.empty_like(state_out.rod_q)
                new_rod_qd = wp.empty_like(state_out.rod_qd)
                self._rod_delta_counter += 1
            else:
                if self._rod_delta_counter == 0:
                    rod_q = state_out.rod_q
                    new_rod_q = state_in.rod_q
                    new_rod_qd = state_in.rod_qd
                else:
                    rod_q = state_in.rod_q
                    new_rod_q = state_out.rod_q
                    new_rod_qd = state_out.rod_qd
                self._rod_delta_counter = 1 - self._rod_delta_counter
            
            wp.launch(
                kernel=apply_rod_deltas_kernel,
                dim=model.rod_seg_count,
                inputs=[
                    self.rod_q_init,
                    rod_q,
                    rod_deltas,
                    dt
                ],
                outputs=[new_rod_q, new_rod_qd],
                device=model.device,
            )

            if state_in.requires_grad:
                state_out.rod_q = new_rod_q
                state_out.rod_qd = new_rod_qd

        return new_rod_q, new_rod_qd
            
    
    @override
    def simulate(self, model: Model, state_in: State, state_out: State, dt: float, control: Control = None):
        requires_grad = state_in.requires_grad
        self._particle_delta_counter = 0
        self._body_delta_counter = 0
        self._rod_delta_counter = 0

        # for particles
        particle_q = None
        particle_qd = None
        particle_deltas = None
        
        # for elastic rods
        rod_q = None
        rod_qd = None
        rod_deltas = None
        
        # for rigid body
        body_q = None
        body_qd = None
        body_deltas = None

        rigid_contact_inv_weight = None

        if model.rigid_contact_max > 0:
            if self.rigid_contact_con_weighting:
                rigid_contact_inv_weight = wp.zeros_like(model.rigid_contact_thickness) 
            rigid_contact_inv_weight_init = None

        if control is None:
            control = model.control(clone_variables=False)
        
        with wp.ScopedTimer("simulate", False):
            # ------------ Integration Step ------------ #
            # particles
            if model.particle_count:
                particle_q = state_out.particle_q
                particle_qd = state_out.particle_qd
                
                self.particle_q_init = wp.clone(state_in.particle_q) 
                if self.enable_restitution:
                    self.particle_qd_init = wp.clone(state_in.particle_qd) 
                particle_deltas = wp.empty_like(state_out.particle_qd) 

                self.integrate_particles(model, state_in, state_out, dt)

            # rods
            if model.rod_count:
                rod_q = state_out.rod_q
                rod_qd = state_out.rod_qd
                
                self.rod_q_init = wp.clone(state_in.rod_q) 
                if self.enable_restitution:
                    self.rod_qd_init = wp.clone(state_in.rod_qd) 
                rod_deltas = wp.empty_like(state_out.rod_q)
                
                self.integrate_rods(model, state_in, state_out, dt)
            
            # rigid bodies
            if model.body_count:
                body_q = state_out.body_q
                body_qd = state_out.body_qd

                if self.compute_body_velocity_from_position_delta or self.enable_restitution:
                    body_q_init = wp.clone(state_in.body_q)
                    body_qd_init = wp.clone(state_in.body_qd)

                body_deltas = wp.empty_like(state_out.body_qd) 

                if model.joint_count:
                    wp.launch(
                        kernel=apply_joint_actions,
                        dim=model.joint_count,
                        inputs=[
                            state_in.body_q,
                            model.body_com,
                            model.joint_type,
                            model.joint_parent,
                            model.joint_child,
                            model.joint_X_p,
                            model.joint_X_c,
                            model.joint_axis_start,
                            model.joint_axis_dim,
                            model.joint_axis,
                            model.joint_axis_mode,
                            control.joint_act,
                        ],
                        outputs=[state_in.body_f],
                        device=model.device,
                    )

                self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)

            spring_constraint_lambdas = None
            if model.spring_count:
                spring_constraint_lambdas = wp.empty_like(model.spring_rest_length) 
            edge_constraint_lambdas = None
            if model.edge_count:
                edge_constraint_lambdas = wp.empty_like(model.edge_rest_angle) 
            
            # ------------ Solver Iterations ------------ #
            for i in range(self.iterations):
                with wp.ScopedTimer(f"iteration_{i}", False):
                    if model.body_count:
                        if requires_grad and i > 0:
                            body_deltas = wp.zeros_like(body_deltas) 
                        else:
                            body_deltas.zero_()
                    
                    if model.rod_count:
                        if requires_grad and i > 0:
                            rod_deltas = wp.zeros_like(rod_deltas) 
                        else:
                            rod_deltas.zero_()

                    if model.particle_count:
                        if requires_grad and i > 0:
                            particle_deltas = wp.zeros_like(particle_deltas) 
                        else:
                            particle_deltas.zero_()
                    
                        # particle ground contact
                        if model.ground:
                            wp.launch(
                                kernel=solve_particle_ground_contacts,
                                dim=model.particle_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    model.soft_contact_ke,
                                    model.soft_contact_kd,
                                    model.soft_contact_kf,
                                    model.soft_contact_mu,
                                    model.ground_plane,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # particle-rigid body contacts (besides ground plane)
                        if model.shape_count > 1:
                            wp.launch(
                                kernel=solve_particle_shape_contacts,
                                dim=model.soft_contact_max,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    body_q,
                                    body_qd,
                                    model.body_com,
                                    model.body_inv_mass,
                                    model.body_inv_inertia,
                                    model.shape_body,
                                    model.shape_materials,
                                    model.soft_contact_mu,
                                    model.particle_adhesion,
                                    model.soft_contact_count,
                                    model.soft_contact_particle,
                                    model.soft_contact_shape,
                                    model.soft_contact_body_pos,
                                    model.soft_contact_body_vel,
                                    model.soft_contact_normal,
                                    model.soft_contact_max,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                # outputs
                                outputs=[particle_deltas, body_deltas],
                                device=model.device,
                            )

                        if model.particle_max_radius > 0.0 and model.particle_count > 1:
                            # assert model.particle_grid.reserved, "model.particle_grid must be built, see HashGrid.build()"
                            wp.launch(
                                kernel=solve_particle_particle_contacts,
                                dim=model.particle_count,
                                inputs=[
                                    model.particle_grid.id, # type: ignore
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.particle_radius,
                                    model.particle_flags,
                                    model.particle_obj_ids, 
                                    model.particle_mu,
                                    model.particle_cohesion,
                                    model.particle_max_radius,
                                    dt,
                                    self.soft_contact_relaxation,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # distance constraints
                        if model.spring_count:
                            spring_constraint_lambdas.zero_()
                            wp.launch(
                                kernel=solve_springs,
                                dim=model.spring_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.spring_indices,
                                    model.spring_rest_length,
                                    model.spring_stiffness,
                                    model.spring_damping,
                                    dt,
                                    spring_constraint_lambdas,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # bending constraints
                        if model.edge_count:
                            edge_constraint_lambdas.zero_()
                            wp.launch(
                                kernel=bending_constraint,
                                dim=model.edge_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.edge_indices,
                                    model.edge_rest_angle,
                                    model.edge_bending_properties,
                                    dt,
                                    edge_constraint_lambdas,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                        # tetrahedral FEM
                        if model.tet_count:
                            wp.launch(
                                kernel=solve_tetrahedra,
                                dim=model.tet_count,
                                inputs=[
                                    particle_q,
                                    particle_qd,
                                    model.particle_inv_mass,
                                    model.tet_indices,
                                    model.tet_poses,
                                    model.tet_activations,
                                    model.tet_materials,
                                    dt,
                                    self.soft_body_relaxation,
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )
                        
                        # =========================== Elastic Rod Constraints =========================== #
                        if model.rod_count > 0:
                            wp.launch(
                                kernel=solve_rod_stretch_shear_constraints,
                                dim=model.rod_seg_count,
                                inputs=[
                                    particle_q, 
                                    rod_q, 
                                    model.rod_stiffness_stretch_indices,
                                    model.particle_inv_mass, 
                                    model.rod_seg_particle_indices, 
                                    model.rod_seg_inv_mass_q,
                                    model.rod_seg_rest_lengths,
                                    model.rod_stiffness_stretch, 
                                ],
                                outputs=[particle_deltas, rod_deltas]
                            )
                            
                            wp.launch(
                                kernel=solve_rod_bend_twist_constraints,
                                dim=model.rod_darboux_count,
                                inputs=[
                                    rod_q,
                                    model.rod_seg_inv_mass_q,
                                    model.rod_rest_darbouxs,
                                    model.rod_stiffness_bend,
                                    model.rod_darboux_seg_indices,
                                    model.rod_stiffness_bend_indices,
                                ],
                                outputs=[rod_deltas]
                            )

                            wp.launch(
                                kernel=solve_rod_length_constraints,
                                dim=model.rod_seg_count,
                                inputs=[
                                    particle_q,
                                    model.particle_inv_mass,
                                    model.rod_seg_particle_indices,
                                    model.rod_seg_rest_lengths,
                                    0.2 # 0.1
                                ],
                                outputs=[particle_deltas],
                                device=model.device,
                            )

                            # apply rod deltas
                            rod_q, rod_qd = self.apply_rod_deltas(
                                model, state_in, state_out, rod_deltas, dt
                            )

                        # apply particle deltas
                        particle_q, particle_qd = self.apply_particle_deltas(
                            model, state_in, state_out, particle_deltas, dt
                        )

                        
                    # handle rigid bodies
                    # ----------------------------
                    if model.joint_count:
                        wp.launch(
                            kernel=solve_body_joints,
                            dim=model.joint_count,
                            inputs=[
                                body_q,
                                body_qd,
                                model.body_com,
                                model.body_inv_mass,
                                model.body_inv_inertia,
                                model.joint_type,
                                model.joint_enabled,
                                model.joint_parent,
                                model.joint_child,
                                model.joint_X_p,
                                model.joint_X_c,
                                model.joint_limit_lower,
                                model.joint_limit_upper,
                                model.joint_axis_start,
                                model.joint_axis_dim,
                                model.joint_axis_mode,
                                model.joint_axis,
                                control.joint_act,
                                model.joint_target_ke,
                                model.joint_target_kd,
                                model.joint_linear_compliance,
                                model.joint_angular_compliance,
                                self.joint_angular_relaxation,
                                self.joint_linear_relaxation,
                                dt,
                            ],
                            outputs=[body_deltas],
                            device=model.device,
                        )

                        body_q, body_qd = self.apply_body_deltas(model, state_in, state_out, body_deltas, dt)
                    # Solve rigid contact constraints
                    if model.rigid_contact_max and (
                        (model.ground and model.shape_ground_contact_pair_count) or model.shape_contact_pair_count
                    ):
                        if self.rigid_contact_con_weighting:
                            rigid_contact_inv_weight.zero_()
                        body_deltas.zero_()

                        wp.launch(
                            kernel=solve_body_contact_positions,
                            dim=model.rigid_contact_max,
                            inputs=[
                                body_q,
                                body_qd,
                                model.body_com,
                                model.body_inv_mass,
                                model.body_inv_inertia,
                                model.shape_body,
                                model.rigid_contact_count,
                                model.rigid_contact_point0,
                                model.rigid_contact_point1,
                                model.rigid_contact_offset0,
                                model.rigid_contact_offset1,
                                model.rigid_contact_normal,
                                model.rigid_contact_thickness,
                                model.rigid_contact_shape0,
                                model.rigid_contact_shape1,
                                model.shape_materials,
                                self.rigid_contact_relaxation,
                                dt,
                                model.rigid_contact_torsional_friction,
                                model.rigid_contact_rolling_friction,
                            ],
                            outputs=[
                                body_deltas,
                                rigid_contact_inv_weight,
                            ],
                            device=model.device,
                        )

                        if self.enable_restitution and i == 0:
                            # remember contact constraint weighting from the first iteration
                            if self.rigid_contact_con_weighting:
                                rigid_contact_inv_weight_init = wp.clone(rigid_contact_inv_weight)
                            else:
                                rigid_contact_inv_weight_init = None

                        body_q, body_qd = self.apply_body_deltas(
                            model, state_in, state_out, body_deltas, dt, rigid_contact_inv_weight
                        )

            if model.particle_count:
                if particle_q.ptr != state_out.particle_q.ptr:
                    state_out.particle_q.assign(particle_q)
                    state_out.particle_qd.assign(particle_qd)

            if model.body_count:
                if body_q.ptr != state_out.body_q.ptr:
                    state_out.body_q.assign(body_q)
                    state_out.body_qd.assign(body_qd)
                    
            # update rod 
            if model.rod_count:
                if rod_q.ptr != state_out.rod_q.ptr:
                    state_out.rod_q.assign(rod_q)
                    state_out.rod_qd.assign(rod_qd)
                
            # update body velocities from position changes
            if self.compute_body_velocity_from_position_delta and model.body_count and not requires_grad:
                # causes gradient issues (probably due to numerical problems
                # when computing velocities from position changes)
                if requires_grad:
                    out_body_qd = wp.clone(state_out.body_qd)
                else:
                    out_body_qd = state_out.body_qd

                # update body velocities
                wp.launch(
                    kernel=update_body_velocities,
                    dim=model.body_count,
                    inputs=[state_out.body_q, body_q_init, model.body_com, dt],
                    outputs=[out_body_qd],
                    device=model.device,
                )

            if self.enable_restitution:
                if model.particle_count:
                    wp.launch(
                        kernel=apply_particle_shape_restitution,
                        dim=model.particle_count,
                        inputs=[
                            particle_q,
                            particle_qd,
                            self.particle_q_init,
                            self.particle_qd_init,
                            model.particle_inv_mass,
                            model.particle_radius,
                            model.particle_flags,
                            body_q,
                            body_qd,
                            model.body_com,
                            model.body_inv_mass,
                            model.body_inv_inertia,
                            model.shape_body,
                            model.shape_materials,
                            model.particle_adhesion,
                            model.soft_contact_restitution,
                            model.soft_contact_count,
                            model.soft_contact_particle,
                            model.soft_contact_shape,
                            model.soft_contact_body_pos,
                            model.soft_contact_body_vel,
                            model.soft_contact_normal,
                            model.soft_contact_max,
                            dt,
                            self.soft_contact_relaxation,
                        ],
                        outputs=[state_out.particle_qd],
                        device=model.device,
                    )
                    if model.ground:
                        wp.launch(
                            kernel=apply_particle_ground_restitution,
                            dim=model.particle_count,
                            inputs=[
                                particle_q,
                                particle_qd,
                                self.particle_q_init,
                                self.particle_qd_init,
                                model.particle_inv_mass,
                                model.particle_radius,
                                model.particle_flags,
                                model.particle_adhesion,
                                model.soft_contact_restitution,
                                model.ground_plane,
                                dt,
                                self.soft_contact_relaxation,
                            ],
                            outputs=[state_out.particle_qd],
                            device=model.device,
                        )

                if model.body_count:
                    body_deltas.zero_()
                    wp.launch(
                        kernel=apply_rigid_restitution,
                        dim=model.rigid_contact_max,
                        inputs=[
                            state_out.body_q,
                            state_out.body_qd,
                            body_q_init,
                            body_qd_init,
                            model.body_com,
                            model.body_inv_mass,
                            model.body_inv_inertia,
                            model.shape_body,
                            model.rigid_contact_count,
                            model.rigid_contact_normal,
                            model.rigid_contact_shape0,
                            model.rigid_contact_shape1,
                            model.shape_materials,
                            model.rigid_contact_point0,
                            model.rigid_contact_point1,
                            model.rigid_contact_offset0,
                            model.rigid_contact_offset1,
                            model.rigid_contact_thickness,
                            rigid_contact_inv_weight_init,
                            model.gravity,
                            dt,
                        ],
                        outputs=[
                            body_deltas,
                        ],
                        device=model.device,
                    )

                    wp.launch(
                        kernel=apply_body_delta_velocities,
                        dim=model.body_count,
                        inputs=[
                            body_deltas,
                        ],
                        outputs=[state_out.body_qd],
                        device=model.device,
                    )

            return state_out
        