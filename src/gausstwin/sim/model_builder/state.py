import warp as wp
import warp.sim

from typing import Optional

class State(warp.sim.State):
    """Time-varying state data for a :class:`Model`.

    Time-varying state data includes particle positions, velocities, rigid body states, and
    anything that is output from the integrator as derived data, e.g.: forces.

    The exact attributes depend on the contents of the model. State objects should
    generally be created using the :func:`Model.state()` function.
    """
    def __init__(self):
        super().__init__()
        # ======================= particles ======================= # 
        self.particle_q: wp.array | None = None
        """Array of 3D particle positions with shape ``(particle_count,)`` and type :class:`vec3`."""

        self.particle_qd: wp.array | None = None
        """Array of 3D particle velocities with shape ``(particle_count,)`` and type :class:`vec3`."""

        self.particle_f: wp.array | None = None
        """Array of 3D particle forces with shape ``(particle_count,)`` and type :class:`vec3`."""
        
        # ======================= rigid bodies ======================= # 
        self.body_q: wp.array | None = None
        """Array of body coordinates (7-dof transforms) in maximal coordinates with shape ``(body_count,)`` and type :class:`transform`."""

        self.body_qd: wp.array | None = None
        """Array of body velocities in maximal coordinates (first three entries represent angular velocity,
        last three entries represent linear velocity) with shape ``(body_count,)`` and type :class:`spatial_vector`.
        """

        self.body_f: wp.array | None = None

        # ======================= elastic rod ======================= # 
        self.rod_q: Optional[wp.array] = None
        self.rod_qd: Optional[wp.array] = None
        

    @property
    def rod_seg_count(self) -> int:
        """The number of rod segments in the state."""
        if self.rod_q is None:
            return 0
        return len(self.rod_q)
