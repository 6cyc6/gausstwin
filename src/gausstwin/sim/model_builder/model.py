import warp as wp
import warp.sim
import numpy as np

from .state import State
from typing import List, Tuple
from warp.sim.inertia import compute_mesh_inertia


Vec3 = List[float]
Vec4 = List[float]
Quat = List[float]
Mat33 = List[float]
Transform = Tuple[Vec3, Quat]


# Shape geometry types
GEO_SPHERE = wp.constant(0)
GEO_BOX = wp.constant(1)
GEO_CAPSULE = wp.constant(2)
GEO_CYLINDER = wp.constant(3)
GEO_CONE = wp.constant(4)
GEO_MESH = wp.constant(5)
GEO_SDF = wp.constant(6)
GEO_PLANE = wp.constant(7)
GEO_NONE = wp.constant(8)

# Types of joints linking rigid bodies
JOINT_PRISMATIC = wp.constant(0)
JOINT_REVOLUTE = wp.constant(1)
JOINT_BALL = wp.constant(2)
JOINT_FIXED = wp.constant(3)
JOINT_FREE = wp.constant(4)
JOINT_COMPOUND = wp.constant(5)
JOINT_UNIVERSAL = wp.constant(6)
JOINT_DISTANCE = wp.constant(7)
JOINT_D6 = wp.constant(8)

# Joint axis control mode types
JOINT_MODE_FORCE = wp.constant(0)
JOINT_MODE_TARGET_POSITION = wp.constant(1)
JOINT_MODE_TARGET_VELOCITY = wp.constant(2)


class Mesh:
    """Describes a triangle collision mesh for simulation

    Example mesh creation from a triangle OBJ mesh file:
    ====================================================

    See :func:`load_mesh` which is provided as a utility function.

    .. code-block:: python

        import numpy as np
        import warp as wp
        import warp.sim
        import openmesh

        m = openmesh.read_trimesh("mesh.obj")
        mesh_points = np.array(m.points())
        mesh_indices = np.array(m.face_vertex_indices(), dtype=np.int32).flatten()
        mesh = wp.sim.Mesh(mesh_points, mesh_indices)

    Attributes:

        vertices (List[Vec3]): Mesh 3D vertices points
        indices (List[int]): Mesh indices as a flattened list of vertex indices describing triangles
        I (Mat33): 3x3 inertia matrix of the mesh assuming density of 1.0 (around the center of mass)
        mass (float): The total mass of the body assuming density of 1.0
        com (Vec3): The center of mass of the body
    """

    def __init__(self, vertices: list[Vec3], indices: list[int], compute_inertia=True, is_solid=True):
        """Construct a Mesh object from a triangle mesh

        The mesh center of mass and inertia tensor will automatically be
        calculated using a density of 1.0. This computation is only valid
        if the mesh is closed (two-manifold).

        Args:
            vertices: List of vertices in the mesh
            indices: List of triangle indices, 3 per-element
            compute_inertia: If True, the mass, inertia tensor and center of mass will be computed assuming density of 1.0
            is_solid: If True, the mesh is assumed to be a solid during inertia computation, otherwise it is assumed to be a hollow surface
        """

        self.vertices = np.array(vertices).reshape(-1, 3)
        self.indices = np.array(indices, dtype=np.int32).flatten()
        self.is_solid = is_solid
        self.has_inertia = compute_inertia

        if compute_inertia:
            self.mass, self.com, self.I, _ = compute_mesh_inertia(1.0, vertices, indices, is_solid=is_solid)
        else:
            self.I = wp.mat33(np.eye(3))
            self.mass = 1.0
            self.com = wp.vec3()


class Model(warp.sim.Model):
    def __init__(self, device):
        super().__init__(device)
        # for elastic rod
        self.rod_q = None
        """Rod segment quaternions for state initialization, shape [rod_seg_count, 4], float."""
        self.rod_qd = None
        """Rod segment velocities (local frame) for state initialization, shape [rod_seg_count, 3], float."""
        self.rod_seg_inv_mass_q = None
        """Rod segment inverse mass, shape [rod_seg_count], float."""
        self.rod_seg_rest_lengths = None
        """Rod segment rest length, shape [rod_seg_count], float."""
        self.rod_seg_particle_indices = None
        """Rod segment particle indices, shape [rod_seg_count*2], int."""
        self.rod_rest_darbouxs = None
        """Rod rest Darboux vectors, shape [rod_darboux_count, 4], float."""
        self.rod_darboux_seg_indices = None
        """Rod Darboux segment indices, shape [rod_darboux_count*2], int."""
        self.rod_stiffness_bend = None
        """Rod segment bending stiffness, shape [rod_count], wp.vec3."""
        self.rod_stiffness_stretch = None
        """Rod segment stretching stiffness, shape [rod_count], wp.vec3."""
        self.rod_stiffness_stretch_indices = None
        """Rod segment stiffness indices, shape [rod_seg_count], int."""
        self.rod_stiffness_bend_indices = None
        """Rod segment stiffness indices, shape [rod_darboux_count], int."""
        
        # for rigid bodies
        self.body_is_robot = None
        """Boolean array indicating if a body is part of the robot, shape [body_count], bool."""
        self.gravity_factor = None
        """Per-body gravity factor, shape [body_count], float."""
        
        # for particle
        self.particle_obj_ids = None
        """Particle object IDs for collision filtering, shape [particle_count], int."""
        
        # count
        self.rod_count = 0
        """Total number of rods in the system."""
        self.rod_seg_count = 0
        """Total number of rod segments in the system."""
        self.rod_darboux_count = 0
        """Total number of rod Darboux vectors in the system."""

        self.particle_max_omega = 1e2
    
    
    def state(self, requires_grad=None) -> State:
        """Returns a state object for the model

        The returned state will be initialized with the initial configuration given in
        the model description.

        Args:
            requires_grad (bool): Manual overwrite whether the state variables should have `requires_grad` enabled (defaults to `None` to use the model's setting :attr:`requires_grad`)

        Returns:
            State: The state object
        """

        s = State()
        if requires_grad is None:
            requires_grad = self.requires_grad
        
        # particles
        if self.particle_count:
            s.particle_q = wp.clone(self.particle_q, requires_grad=requires_grad)
            s.particle_qd = wp.clone(self.particle_qd, requires_grad=requires_grad)
            s.particle_f = wp.zeros_like(self.particle_qd, requires_grad=requires_grad)

        # rigid body
        if self.body_count:
            s.body_q = wp.clone(self.body_q, requires_grad=requires_grad)
            s.body_qd = wp.clone(self.body_qd, requires_grad=requires_grad)
            s.body_f = wp.zeros_like(self.body_qd, requires_grad=requires_grad)
            
        # elastic rod
        if self.rod_seg_count:
            s.rod_q = wp.clone(self.rod_q, requires_grad=requires_grad)
            s.rod_qd = wp.clone(self.rod_qd, requires_grad=requires_grad)

        return s
        
