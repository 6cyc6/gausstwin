import copy

import numpy as np
import warp as wp
import warp.sim

from .model import Model
from types import MethodType
from typing import List, Tuple
from scipy.spatial.transform import Rotation as R
from warp.sim.graph_coloring import combine_independent_particle_coloring

Vec3 = List[float]
Vec4 = List[float]
Quat = List[float]
Mat33 = List[float]
Transform = Tuple[Vec3, Quat]

# Particle flags
PARTICLE_FLAG_ACTIVE = wp.constant(wp.uint32(1 << 0))

def flag_to_int(flag):
    """Converts a flag to an integer."""
    if type(flag) in wp.types.int_types:
        return flag.value
    return int(flag)


class ModelBuilder(warp.sim.ModelBuilder):
    """Builder for :class:`Model` objects.

    The builder is used to create a model and add particles, rigid bodies, and other
    components to it. The builder is created using the :func:`Model.builder()` function.
    """
    def __init__(self, up_vector=(0.0, 0.0, 1.0), gravity=-9.80665):
        super().__init__(up_vector=up_vector, gravity=gravity)
        self.default_particle_radius = 0.005
        
        # for elastic rod
        self.rod_q = []                          # orientations per rod segment
        self.rod_qd = []
        self.rod_seg_inv_mass_q = []             # inverse masses for quaternions
        self.rod_seg_rest_lengths = []           # rest lengths per edge
        self.rod_seg_particle_indices = []       # which particles each segment links (particle_idx_0, particle_idx_1)
        self.rod_rest_darbouxs = []              # darboux vector between each segment
        self.rod_darboux_seg_indices = []        # which segments a darboux vector links (seg_idx_0, seg_idx_1)
        self.rod_stiffness_bend = []             # bending stiffness per rod
        self.rod_stiffness_stretch = []          # stretch stiffness per rod
        self.rod_stiffness_stretch_indices = []  # which rod each segment belongs to (rod_idx)
        self.rod_stiffness_bend_indices = []     # which rod each segment belongs to (rod_idx)
        
        # for particle
        self.particle_obj_ids = []               # object IDs for collision filtering
        # for rigid bodies
        self.body_is_robot = []                  # boolean list indicating if body is part of robot
        self.gravity_factor = []                 # per-body gravity factor


    @property
    def rod_count(self):
        """
        The number of rods in the model.
        """
        return len(self.rod_stiffness_bend)

    @property
    def rod_seg_count(self):
        """
        The number of rod segments in the model.
        """
        return len(self.rod_q)

    @property
    def rod_darboux_count(self):
        """
        The number of rod darbouxs in the model.
        """
        return len(self.rod_rest_darbouxs)

    @property
    def obj_count(self):
        """
        The number of objects in the model.
        """
        return len(self.particle_obj_ids)


    # particles
    def add_particle(
        self,
        pos: Vec3,
        vel: Vec3,
        mass: float,
        radius: float = None, # type: ignore
        obj_id: int = -1,
        flags: wp.uint32 = PARTICLE_FLAG_ACTIVE,
    ) -> int:
        """Adds a single particle to the model

        Args:
            pos: The initial position of the particle
            vel: The initial velocity of the particle
            mass: The mass of the particle
            radius: The radius of the particle used in collision handling. If None, the radius is set to the default value (:attr:`default_particle_radius`).
            flags: The flags that control the dynamical behavior of the particle, see PARTICLE_FLAG_* constants

        Note:
            Set the mass equal to zero to create a 'kinematic' particle that does is not subject to dynamics.

        Returns:
            The index of the particle in the system
        """
        self.particle_q.append(pos)
        self.particle_qd.append(vel)
        self.particle_mass.append(mass)
        if radius is None:
            radius = self.default_particle_radius
        self.particle_radius.append(radius)
        self.particle_flags.append(flags)
        self.particle_obj_ids.append(obj_id)
        
        particle_id = self.particle_count - 1

        return particle_id


    # register a rigid body and return its index.
    def add_body(
        self,
        origin: Transform | None = None,
        armature: float = 0.0,
        com: Vec3 | None = None,
        I_m: Mat33 | None = None,
        m: float = 0.0,
        is_robot: bool = False, # type: ignore
        name: str | None = None,
        enable_gravity: bool = True,
    ) -> int:
        """Adds a rigid body to the model.

        Args:
            origin: The location of the body in the world frame
            armature: Artificial inertia added to the body
            com: The center of mass of the body w.r.t its origin
            I_m: The 3x3 inertia tensor of the body (specified relative to the center of mass)
            m: Mass of the body
            name: Name of the body (optional)

        Returns:
            The index of the body in the model

        Note:
            If the mass (m) is zero then the body is treated as kinematic with no dynamics

        """

        if origin is None:
            origin = wp.transform()

        if com is None:
            com = wp.vec3()

        if I_m is None:
            I_m = wp.mat33()

        body_id = len(self.body_mass)

        # body data
        inertia = I_m + wp.mat33(np.eye(3)) * armature
        self.body_inertia.append(inertia)
        self.body_mass.append(m)
        self.body_com.append(com)

        if m > 0.0:
            self.body_inv_mass.append(1.0 / m)
        else:
            self.body_inv_mass.append(0.0)

        if any(x for x in inertia):
            self.body_inv_inertia.append(wp.inverse(inertia))
        else:
            self.body_inv_inertia.append(inertia)

        self.body_q.append(origin)
        self.body_qd.append(wp.spatial_vector())

        self.body_name.append(name or f"body {body_id}")
        self.body_shapes[body_id] = []
        
        # if the body belongs to a robot
        self.body_is_robot.append(is_robot)
        self.gravity_factor.append(1.0 if enable_gravity else 0.0)
        
        return body_id


    def add_rod(
        self, 
        pos, 
        mass, 
        mass_q_factor=1.0,
        poses=None,
        radius=0.005,
        stiffness_stretch=[0.01, 0.01, 1.0], 
        stiffness_bend=[0.01, 0.01, 0.02], 
        dh=0.001,
    ):
        """
        Adds an elastic rod made of particles with orientations/quaternions per segment.
        each Rod:
        N particles
        N - 1 segments
        N - 2 darboux vector

        Args:
            pos: list of positions (length N)
            mass: list of mass of each particle (length N)
            mass_q_factor: scaling factor for the inverse mass of the quaternion (relative to particle mass)
            radius: particle radius
            stiffness_stretch: stretch stiffness 
            stiffness_bend: bend stiffness 
            dh: offset in height to avoid initial penetration with ground

        Returns:
            rod_id (int): ID for the new rod
        """
        start_particle_idx = self.particle_count
        start_seg_idx = self.rod_seg_count
        start_rod_idx = self.rod_count
        n_particles = len(pos)
        n_segments = n_particles - 1

        # add particles 
        for i in range(n_particles):
            pos_i = pos[i]
            pos_i[-1] = pos_i[-1] + dh 
            # if i == 0 or i == n_particles - 1:
            #     mass_i = 0.5 * mass[i]  # make the end particles half mass
            # else:
            #     mass_i = mass[i]
            mass_i = mass[i]
            vel_i = (0.0, 0.0, 0.0)
            self.add_particle(pos_i, vel_i, mass_i, radius, obj_id=start_rod_idx)

        # add edges for rod segments
        for i in range(n_segments):
            # save indices of particles each segment connects
            self.rod_seg_particle_indices.append(start_particle_idx + i)
            self.rod_seg_particle_indices.append(start_particle_idx + i + 1)
                                                    
            # save rest lengths of each segment
            z_dir = np.array(pos[i+1]) - np.array(pos[i])
            rest_len = np.linalg.norm(z_dir)
            self.rod_seg_rest_lengths.append(rest_len)
            # save quaternions 
            if poses is not None:
                pose = np.array(poses[i])
                r = R.from_matrix(pose[:3, :3]) 
                q = r.as_quat()  # returns [x, y, z, w]
            else:
                # if no pose is given, compute from the particle positions. Assume the rod is on the table, so x axis is (0, 0, 1), z axis is along the rod
                rot_mat = np.eye(3)
                z_axis = z_dir / rest_len
                x_axis = np.array([0.0, 0.0, 1.0]) # assume x points up
                y_axis = np.cross(z_axis, x_axis)
                y_axis /= np.linalg.norm(y_axis)
                rot_mat[:, 0] = x_axis
                rot_mat[:, 1] = y_axis
                rot_mat[:, 2] = z_axis
                r = R.from_matrix(rot_mat)
                q = r.as_quat()  # returns [x, y, z, w]
            self.rod_q.append(q.tolist())  
            # inverse masses 
            # if i == 0 or i == n_segments - 1:
            #     self.rod_seg_inv_mass_q.append(1.0 / (mass[i] * 0.001 * mass_q_factor * 1.5))  # make the end segment stiffer
            # else:
            #     self.rod_seg_inv_mass_q.append(1.0 / (mass[i] * 0.001 * mass_q_factor))  
            self.rod_seg_inv_mass_q.append(1.0 / (mass[i] * 0.001 * mass_q_factor))  # scale by small number to make it stiffer. 0.001 is computed by comparing the physiccal properties of the rope
            
            # rod segment velocities (local frame)
            self.rod_qd.append([0.0, 0.0, 0.0])
            
            # add darboux constraint (bend twist)
            if i != 0:
                q1 = R.from_quat(self.rod_q[-2])
                q2 = R.from_quat(self.rod_q[-1])
                rest_darboux_vec = (q1.inv() * q2).as_quat()
                identity = np.array([0.0, 0.0, 0.0, 1.0])

                omega_plus = rest_darboux_vec + identity
                omega_minus = rest_darboux_vec - identity

                if np.dot(omega_minus, omega_minus) > np.dot(omega_plus, omega_plus):
                    rest_darboux_vec *= -1.0
                # save darbouxs vector
                self.rod_rest_darbouxs.append(rest_darboux_vec.tolist())
                # save indicies of segments each darboux vector links
                self.rod_darboux_seg_indices.append(start_seg_idx + i - 1)
                self.rod_darboux_seg_indices.append(start_seg_idx + i)

        # rod stiffness indices
        self.rod_stiffness_bend_indices.extend([start_rod_idx] * (n_segments - 1))
        self.rod_stiffness_stretch_indices.extend([start_rod_idx] * n_segments)

        # Store stiffness parameters
        self.rod_stiffness_stretch.append(stiffness_stretch)
        self.rod_stiffness_bend.append(stiffness_bend)
        
        rod_id = self.rod_count - 1
        
        return rod_id


    def _create_ground_plane(self) -> None:
        # self._ground_params["has_ground_collision"] = True
        ground_id = self.add_shape_plane(**self._ground_params)
        self._ground_created = True
        # disable ground collisions as they will be treated separately
        for i in range(self.shape_count - 1):
            self.shape_collision_filter_pairs.add((i, ground_id))


    def add_builder(
        self,
        builder,
        xform: Transform | None = None,
        update_num_env_count: bool = True,
        separate_collision_group: bool = True,
    ):
        """Copies the data from `builder`, another `ModelBuilder` to this `ModelBuilder`.

        Args:
            builder (ModelBuilder): a model builder to add model data from.
            xform (:ref:`transform <transform>`): offset transform applied to root bodies.
            update_num_env_count (bool): if True, the number of environments is incremented by 1.
            separate_collision_group (bool): if True, the shapes from the articulations in `builder` will all be put into a single new collision group, otherwise, only the shapes in collision group > -1 will be moved to a new group.
        """

        start_particle_idx = self.particle_count
        if builder.particle_count:
            self.particle_max_velocity = builder.particle_max_velocity
            if xform is not None:
                pos_offset = wp.transform_get_translation(xform)
            else:
                pos_offset = np.zeros(3)
            self.particle_q.extend((np.array(builder.particle_q) + pos_offset).tolist())
            # other particle attributes are added below

        if builder.obj_count:
            start_obj_idx = self.obj_count
            # Offset obj_ids by start_obj_idx, preserving negative values
            obj_ids = np.array(builder.particle_obj_ids, dtype=np.int32)
            mask = obj_ids >= 0
            obj_ids[mask] += start_obj_idx
            self.particle_obj_ids.extend(obj_ids.tolist())
        
        if builder.rod_count:
            start_rod_idx = self.rod_count
            start_seg_idx = self.rod_seg_count
            start_darboux_idx = self.rod_darboux_count
            
            self.rod_seg_particle_indices.extend((np.array(builder.rod_seg_particle_indices, dtype=np.int32) + start_particle_idx).tolist())
            self.rod_darboux_seg_indices.extend((np.array(builder.rod_darboux_seg_indices, dtype=np.int32) + start_seg_idx).tolist())
            self.rod_stiffness_bend_indices.extend((np.array(builder.rod_stiffness_bend_indices, dtype=np.int32) + start_darboux_idx).tolist())
            self.rod_stiffness_stretch_indices.extend((np.array(builder.rod_stiffness_stretch_indices, dtype=np.int32) + start_seg_idx).tolist())
            
        if builder.spring_count:
            self.spring_indices.extend((np.array(builder.spring_indices, dtype=np.int32) + start_particle_idx).tolist())
        if builder.edge_count:
            # Update edge indices by adding offset, preserving -1 values
            edge_indices = np.array(builder.edge_indices, dtype=np.int32)
            mask = edge_indices != -1
            edge_indices[mask] += start_particle_idx
            self.edge_indices.extend(edge_indices.tolist())
        if builder.tri_count:
            self.tri_indices.extend((np.array(builder.tri_indices, dtype=np.int32) + start_particle_idx).tolist())
        if builder.tet_count:
            self.tet_indices.extend((np.array(builder.tet_indices, dtype=np.int32) + start_particle_idx).tolist())

        builder_coloring_translated = [group + start_particle_idx for group in builder.particle_color_groups]
        self.particle_color_groups = combine_independent_particle_coloring(
            self.particle_color_groups, builder_coloring_translated
        )

        start_body_idx = self.body_count
        start_shape_idx = self.shape_count
        for s, b in enumerate(builder.shape_body):
            if b > -1:
                new_b = b + start_body_idx
                self.shape_body.append(new_b)
                self.shape_transform.append(builder.shape_transform[s])
            else:
                self.shape_body.append(-1)
                # apply offset transform to root bodies
                if xform is not None:
                    self.shape_transform.append(xform * wp.transform(*builder.shape_transform[s]))
                else:
                    self.shape_transform.append(builder.shape_transform[s])

        for b, shapes in builder.body_shapes.items():
            self.body_shapes[b + start_body_idx] = [s + start_shape_idx for s in shapes]

        if builder.joint_count:
            joint_X_p = copy.deepcopy(builder.joint_X_p)
            joint_q = copy.deepcopy(builder.joint_q)
            if xform is not None:
                for i in range(len(joint_X_p)):
                    if builder.joint_type[i] == wp.sim.JOINT_FREE:
                        qi = builder.joint_q_start[i]
                        xform_prev = wp.transform(joint_q[qi : qi + 3], joint_q[qi + 3 : qi + 7])
                        tf = xform * xform_prev
                        joint_q[qi : qi + 3] = tf.p
                        joint_q[qi + 3 : qi + 7] = tf.q
                    elif builder.joint_parent[i] == -1:
                        joint_X_p[i] = xform * wp.transform(*joint_X_p[i])
            self.joint_X_p.extend(joint_X_p)
            self.joint_q.extend(joint_q)

            # offset the indices
            self.articulation_start.extend([a + self.joint_count for a in builder.articulation_start])
            self.joint_parent.extend([p + self.body_count if p != -1 else -1 for p in builder.joint_parent])
            self.joint_child.extend([c + self.body_count for c in builder.joint_child])

            self.joint_q_start.extend([c + self.joint_coord_count for c in builder.joint_q_start])
            self.joint_qd_start.extend([c + self.joint_dof_count for c in builder.joint_qd_start])

            self.joint_axis_start.extend([a + self.joint_axis_total_count for a in builder.joint_axis_start])

        for i in range(builder.body_count):
            if xform is not None:
                self.body_q.append(xform * wp.transform(*builder.body_q[i]))
            else:
                self.body_q.append(builder.body_q[i])

        # apply collision group
        if separate_collision_group:
            self.shape_collision_group.extend([self.last_collision_group + 1 for _ in builder.shape_collision_group])
        else:
            self.shape_collision_group.extend(
                [(g + self.last_collision_group if g > -1 else -1) for g in builder.shape_collision_group]
            )
        shape_count = self.shape_count
        for i, j in builder.shape_collision_filter_pairs:
            self.shape_collision_filter_pairs.add((i + shape_count, j + shape_count))
        for group, shapes in builder.shape_collision_group_map.items():
            if separate_collision_group:
                extend_group = self.last_collision_group + 1
            else:
                extend_group = group + self.last_collision_group if group > -1 else -1

            if extend_group not in self.shape_collision_group_map:
                self.shape_collision_group_map[extend_group] = []

            self.shape_collision_group_map[extend_group].extend([s + shape_count for s in shapes])

        # update last collision group counter
        if separate_collision_group:
            self.last_collision_group += 1
        elif builder.last_collision_group > -1:
            self.last_collision_group += builder.last_collision_group

        more_builder_attrs = [
            "body_inertia",
            "body_mass",
            "body_inv_inertia",
            "body_inv_mass",
            "body_com",
            "body_qd",
            "body_name",
            "joint_type",
            "joint_enabled",
            "joint_X_c",
            "joint_armature",
            "joint_axis",
            "joint_axis_dim",
            "joint_axis_mode",
            "joint_name",
            "joint_qd",
            "joint_act",
            "joint_limit_lower",
            "joint_limit_upper",
            "joint_limit_ke",
            "joint_limit_kd",
            "joint_target_ke",
            "joint_target_kd",
            "joint_linear_compliance",
            "joint_angular_compliance",
            "shape_visible",
            "shape_geo_type",
            "shape_geo_scale",
            "shape_geo_src",
            "shape_geo_is_solid",
            "shape_geo_thickness",
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_kf",
            "shape_material_ka",
            "shape_material_mu",
            "shape_material_restitution",
            "shape_collision_radius",
            "shape_ground_collision",
            "shape_shape_collision",
            "particle_qd",
            "particle_mass",
            "particle_radius",
            "particle_flags",
            "edge_rest_angle",
            "edge_rest_length",
            "edge_bending_properties",
            "spring_rest_length",
            "spring_stiffness",
            "spring_damping",
            "spring_control",
            "tri_poses",
            "tri_activations",
            "tri_materials",
            "tri_areas",
            "tet_poses",
            "tet_activations",
            "tet_materials",
            # added
            "rod_q",
            "rod_qd",
            "rod_seg_inv_mass_q",
            "rod_seg_rest_lengths",
            "rod_rest_darbouxs",
            "rod_stiffness_bend",
            "rod_stiffness_stretch",
            "body_is_robot", 
            "gravity_factor"
        ]

        for attr in more_builder_attrs:
            getattr(self, attr).extend(getattr(builder, attr))

        self.joint_dof_count += builder.joint_dof_count
        self.joint_coord_count += builder.joint_coord_count
        self.joint_axis_total_count += builder.joint_axis_total_count

        self.up_vector = builder.up_vector
        self.gravity_factor = builder.gravity_factor
        self._ground_params = builder._ground_params

        if update_num_env_count:
            self.num_envs += 1
            

    def finalize(self, device=None, requires_grad=False) -> Model:
        # ensure the env count is set correctly
        self.num_envs = max(1, self.num_envs)

        # add ground plane if not already created
        if not self._ground_created:
            self._create_ground_plane()

        # construct particle inv masses
        ms = np.array(self.particle_mass, dtype=np.float32)
        # static particles (with zero mass) have zero inverse mass
        particle_inv_mass = np.divide(1.0, ms, out=np.zeros_like(ms), where=ms != 0.0)
        

        with wp.ScopedDevice(device):
            m = Model(device)
            m.requires_grad = requires_grad

            m.ground_plane_params = self._ground_params["plane"]

            m.num_envs = self.num_envs
            
            # ---------------------
            # particles

            # state (initial)
            m.particle_q = wp.array(self.particle_q, dtype=wp.vec3, requires_grad=requires_grad)
            m.particle_qd = wp.array(self.particle_qd, dtype=wp.vec3, requires_grad=requires_grad)
            m.particle_mass = wp.array(self.particle_mass, dtype=wp.float32, requires_grad=requires_grad)
            m.particle_inv_mass = wp.array(particle_inv_mass, dtype=wp.float32, requires_grad=requires_grad)
            m.particle_radius = wp.array(self.particle_radius, dtype=wp.float32, requires_grad=requires_grad)
            m.particle_flags = wp.array([flag_to_int(f) for f in self.particle_flags], dtype=wp.uint32)
            m.particle_max_radius = np.max(self.particle_radius) if len(self.particle_radius) > 0 else 0.0
            m.particle_max_velocity = self.particle_max_velocity

            particle_colors = np.empty(self.particle_count, dtype=int)
            for color in range(len(self.particle_color_groups)):
                particle_colors[self.particle_color_groups[color]] = color
            m.particle_colors = wp.array(particle_colors, dtype=int)
            m.particle_color_groups = [wp.array(group, dtype=int) for group in self.particle_color_groups]
            m.particle_obj_ids = wp.array(self.particle_obj_ids, dtype=wp.int32)
            
            # hash-grid for particle interactions
            m.particle_grid = wp.HashGrid(128, 128, 128)

            # --------------------------- elastic rod --------------------------- # 
            if self.rod_seg_count > 0:
                m.rod_q = wp.array(self.rod_q, dtype=wp.quat, device=device, requires_grad=requires_grad)
                m.rod_qd = wp.array(self.rod_qd, dtype=wp.vec3, device=device, requires_grad=requires_grad)
                m.rod_seg_inv_mass_q = wp.array(self.rod_seg_inv_mass_q, dtype=wp.float32, device=device, requires_grad=requires_grad)
                m.rod_seg_rest_lengths = wp.array(self.rod_seg_rest_lengths, dtype=wp.float32, device=device, requires_grad=requires_grad)
                m.rod_seg_particle_indices = wp.array(self.rod_seg_particle_indices, dtype=wp.int32, device=device)
                m.rod_rest_darbouxs = wp.array(self.rod_rest_darbouxs, dtype=wp.quat, device=device, requires_grad=requires_grad)
                m.rod_darboux_seg_indices = wp.array(self.rod_darboux_seg_indices, dtype=wp.int32, device=device)
                m.rod_stiffness_bend = wp.array(self.rod_stiffness_bend, dtype=wp.vec3, device=device, requires_grad=requires_grad)
                m.rod_stiffness_stretch = wp.array(self.rod_stiffness_stretch, dtype=wp.vec3, device=device, requires_grad=requires_grad)
                m.rod_stiffness_stretch_indices = wp.array(self.rod_stiffness_stretch_indices, dtype=wp.int32, device=device)
                m.rod_stiffness_bend_indices = wp.array(self.rod_stiffness_bend_indices, dtype=wp.int32, device=device)
                
            # ---------------------
            # collision geometry

            m.shape_transform = wp.array(self.shape_transform, dtype=wp.transform, requires_grad=requires_grad)
            m.shape_body = wp.array(self.shape_body, dtype=wp.int32)
            m.shape_visible = wp.array(self.shape_visible, dtype=wp.bool)
            m.body_shapes = self.body_shapes

            # build list of ids for geometry sources (meshes, sdfs)
            geo_sources = []
            finalized_meshes = {}  # do not duplicate meshes
            for geo in self.shape_geo_src:
                geo_hash = hash(geo)  # avoid repeated hash computations
                if geo:
                    if geo_hash not in finalized_meshes:
                        finalized_meshes[geo_hash] = geo.finalize(device=device)
                    geo_sources.append(finalized_meshes[geo_hash])
                else:
                    # add null pointer
                    geo_sources.append(0)

            m.shape_geo.type = wp.array(self.shape_geo_type, dtype=wp.int32)
            m.shape_geo.source = wp.array(geo_sources, dtype=wp.uint64)
            m.shape_geo.scale = wp.array(self.shape_geo_scale, dtype=wp.vec3, requires_grad=requires_grad)
            m.shape_geo.is_solid = wp.array(self.shape_geo_is_solid, dtype=wp.uint8)
            m.shape_geo.thickness = wp.array(self.shape_geo_thickness, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_geo_src = self.shape_geo_src  # used for rendering
            # store refs to geometry
            m.geo_meshes = self.geo_meshes
            m.geo_sdfs = self.geo_sdfs

            m.shape_materials.ke = wp.array(self.shape_material_ke, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_materials.kd = wp.array(self.shape_material_kd, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_materials.kf = wp.array(self.shape_material_kf, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_materials.ka = wp.array(self.shape_material_ka, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_materials.mu = wp.array(self.shape_material_mu, dtype=wp.float32, requires_grad=requires_grad)
            m.shape_materials.restitution = wp.array(
                self.shape_material_restitution, dtype=wp.float32, requires_grad=requires_grad
            )

            m.shape_collision_filter_pairs = self.shape_collision_filter_pairs
            m.shape_collision_group = self.shape_collision_group
            m.shape_collision_group_map = self.shape_collision_group_map
            m.shape_collision_radius = wp.array(
                self.shape_collision_radius, dtype=wp.float32, requires_grad=requires_grad
            )
            m.shape_ground_collision = self.shape_ground_collision
            m.shape_shape_collision = self.shape_shape_collision

            # ---------------------
            # springs

            m.spring_indices = wp.array(self.spring_indices, dtype=wp.int32)
            m.spring_rest_length = wp.array(self.spring_rest_length, dtype=wp.float32, requires_grad=requires_grad)
            m.spring_stiffness = wp.array(self.spring_stiffness, dtype=wp.float32, requires_grad=requires_grad)
            m.spring_damping = wp.array(self.spring_damping, dtype=wp.float32, requires_grad=requires_grad)
            m.spring_control = wp.array(self.spring_control, dtype=wp.float32, requires_grad=requires_grad)

            # ---------------------
            # triangles

            m.tri_indices = wp.array(self.tri_indices, dtype=wp.int32)
            m.tri_poses = wp.array(self.tri_poses, dtype=wp.mat22, requires_grad=requires_grad)
            m.tri_activations = wp.array(self.tri_activations, dtype=wp.float32, requires_grad=requires_grad)
            m.tri_materials = wp.array(self.tri_materials, dtype=wp.float32, requires_grad=requires_grad)
            m.tri_areas = wp.array(self.tri_areas, dtype=wp.float32, requires_grad=requires_grad)

            # ---------------------
            # edges

            m.edge_indices = wp.array(self.edge_indices, dtype=wp.int32)
            m.edge_rest_angle = wp.array(self.edge_rest_angle, dtype=wp.float32, requires_grad=requires_grad)
            m.edge_rest_length = wp.array(self.edge_rest_length, dtype=wp.float32, requires_grad=requires_grad)
            m.edge_bending_properties = wp.array(
                self.edge_bending_properties, dtype=wp.float32, requires_grad=requires_grad
            )

            # ---------------------
            # tetrahedra

            m.tet_indices = wp.array(self.tet_indices, dtype=wp.int32)
            m.tet_poses = wp.array(self.tet_poses, dtype=wp.mat33, requires_grad=requires_grad)
            m.tet_activations = wp.array(self.tet_activations, dtype=wp.float32, requires_grad=requires_grad)
            m.tet_materials = wp.array(self.tet_materials, dtype=wp.float32, requires_grad=requires_grad)

            # -----------------------
            # muscles

            # close the muscle waypoint indices
            muscle_start = copy.copy(self.muscle_start)
            muscle_start.append(len(self.muscle_bodies))

            m.muscle_start = wp.array(muscle_start, dtype=wp.int32)
            m.muscle_params = wp.array(self.muscle_params, dtype=wp.float32, requires_grad=requires_grad)
            m.muscle_bodies = wp.array(self.muscle_bodies, dtype=wp.int32)
            m.muscle_points = wp.array(self.muscle_points, dtype=wp.vec3, requires_grad=requires_grad)
            m.muscle_activations = wp.array(self.muscle_activations, dtype=wp.float32, requires_grad=requires_grad)

            # --------------------------------------
            # rigid bodies

            m.body_q = wp.array(self.body_q, dtype=wp.transform, requires_grad=requires_grad)
            m.body_qd = wp.array(self.body_qd, dtype=wp.spatial_vector, requires_grad=requires_grad)
            m.body_inertia = wp.array(self.body_inertia, dtype=wp.mat33, requires_grad=requires_grad)
            m.body_inv_inertia = wp.array(self.body_inv_inertia, dtype=wp.mat33, requires_grad=requires_grad)
            m.body_mass = wp.array(self.body_mass, dtype=wp.float32, requires_grad=requires_grad)
            m.body_inv_mass = wp.array(self.body_inv_mass, dtype=wp.float32, requires_grad=requires_grad)
            m.body_com = wp.array(self.body_com, dtype=wp.vec3, requires_grad=requires_grad)
            m.body_name = self.body_name
            
            m.body_is_robot = wp.array(self.body_is_robot, dtype=wp.bool)
            m.gravity_factor = wp.array(self.gravity_factor, dtype=wp.float32, requires_grad=False)

            # joints
            m.joint_type = wp.array(self.joint_type, dtype=wp.int32)
            m.joint_parent = wp.array(self.joint_parent, dtype=wp.int32)
            m.joint_child = wp.array(self.joint_child, dtype=wp.int32)
            m.joint_X_p = wp.array(self.joint_X_p, dtype=wp.transform, requires_grad=requires_grad)
            m.joint_X_c = wp.array(self.joint_X_c, dtype=wp.transform, requires_grad=requires_grad)
            m.joint_axis_start = wp.array(self.joint_axis_start, dtype=wp.int32)
            m.joint_axis_dim = wp.array(np.array(self.joint_axis_dim), dtype=wp.int32, ndim=2)
            m.joint_axis = wp.array(self.joint_axis, dtype=wp.vec3, requires_grad=requires_grad)
            m.joint_q = wp.array(self.joint_q, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_qd = wp.array(self.joint_qd, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_name = self.joint_name
            # compute joint ancestors
            child_to_joint = {}
            for i, child in enumerate(self.joint_child):
                child_to_joint[child] = i
            parent_joint = []
            for parent in self.joint_parent:
                parent_joint.append(child_to_joint.get(parent, -1))
            m.joint_ancestor = wp.array(parent_joint, dtype=wp.int32)

            # dynamics properties
            m.joint_armature = wp.array(self.joint_armature, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_target_ke = wp.array(self.joint_target_ke, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_target_kd = wp.array(self.joint_target_kd, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_axis_mode = wp.array(self.joint_axis_mode, dtype=wp.int32)
            m.joint_act = wp.array(self.joint_act, dtype=wp.float32, requires_grad=requires_grad)

            m.joint_limit_lower = wp.array(self.joint_limit_lower, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_limit_upper = wp.array(self.joint_limit_upper, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_limit_ke = wp.array(self.joint_limit_ke, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_limit_kd = wp.array(self.joint_limit_kd, dtype=wp.float32, requires_grad=requires_grad)
            m.joint_linear_compliance = wp.array(
                self.joint_linear_compliance, dtype=wp.float32, requires_grad=requires_grad
            )
            m.joint_angular_compliance = wp.array(
                self.joint_angular_compliance, dtype=wp.float32, requires_grad=requires_grad
            )
            m.joint_enabled = wp.array(self.joint_enabled, dtype=wp.int32)

            # 'close' the start index arrays with a sentinel value
            joint_q_start = copy.copy(self.joint_q_start)
            joint_q_start.append(self.joint_coord_count)
            joint_qd_start = copy.copy(self.joint_qd_start)
            joint_qd_start.append(self.joint_dof_count)
            articulation_start = copy.copy(self.articulation_start)
            articulation_start.append(self.joint_count)

            m.joint_q_start = wp.array(joint_q_start, dtype=wp.int32)
            m.joint_qd_start = wp.array(joint_qd_start, dtype=wp.int32)
            m.articulation_start = wp.array(articulation_start, dtype=wp.int32)

            # counts
            m.joint_count = self.joint_count
            m.joint_axis_count = self.joint_axis_count
            m.joint_dof_count = self.joint_dof_count
            m.joint_coord_count = self.joint_coord_count
            m.particle_count = len(self.particle_q)
            m.body_count = len(self.body_q)
            m.shape_count = len(self.shape_geo_type)
            m.tri_count = len(self.tri_poses)
            m.tet_count = len(self.tet_poses)
            m.edge_count = len(self.edge_rest_angle)
            m.spring_count = len(self.spring_rest_length)
            m.muscle_count = len(self.muscle_start)
            m.articulation_count = len(self.articulation_start)
            
            # added for rope
            m.rod_count = self.rod_count
            m.rod_seg_count = self.rod_seg_count
            m.rod_darboux_count = self.rod_darboux_count
            
            # contacts
            if m.particle_count:
                m.allocate_soft_contacts(self.soft_contact_max, requires_grad=requires_grad)
            m.find_shape_contact_pairs()
            if self.num_rigid_contacts_per_env is None:
                contact_count, limited_contact_count = m.count_contact_points()
            else:
                contact_count = limited_contact_count = self.num_rigid_contacts_per_env * self.num_envs
            if contact_count:
                if wp.config.verbose:
                    print(f"Allocating {contact_count} rigid contacts.")
                m.allocate_rigid_contacts(
                    count=contact_count, limited_contact_count=limited_contact_count, requires_grad=requires_grad
                )
            m.rigid_mesh_contact_max = self.rigid_mesh_contact_max
            m.rigid_contact_margin = self.rigid_contact_margin
            m.rigid_contact_torsional_friction = self.rigid_contact_torsional_friction
            m.rigid_contact_rolling_friction = self.rigid_contact_rolling_friction

            # enable ground plane
            m.ground_plane = wp.array(self._ground_params["plane"], dtype=wp.float32, requires_grad=requires_grad)
            m.gravity = np.array(self.up_vector, dtype=wp.float32) * self.gravity
            m.up_axis = self.up_axis
            m.up_vector = np.array(self.up_vector, dtype=wp.float32)

            m.enable_tri_collisions = False
            
        return m 