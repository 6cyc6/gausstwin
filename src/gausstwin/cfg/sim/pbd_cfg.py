from dataclasses import dataclass, field


@dataclass
class PBDConfig:
    """
    Configuration class for the Position-Based Dynamic simulation.
    """
    # ======================================= General Config ======================================= #
    # solver
    substeps: int = 20
    solver_iterations: int = 1
    fps: int = 60
    
    # visualization
    render: bool = False
    stage_path: str = "result.usda"
    
    # ground config
    ground_mu: float = 0.5
    ground_restitution: float = 0.0
    
    # gripper config
    gripper_m: float = 0.015
    gripper_mu: float = 0.3
    gripper_density_factor: float = 0.8
    obj_radii: float = 0.005
    
    # rigid object config
    rigid_mu: float = 0.4
    rigid_density_factor: float = 1.0
    rigid_restitution: float = 0.0

    rigid2_mu: float = 0.4
    rigid2_density_factor: float = 1.0
    rigid2_restitution: float = 0.0

    # rope config
    rope_m: float = 0.1
    rope_mass_q_factor: float = 1.0
    rope_stiffness_bend: list = field(default_factory=list)
    rope_stiffness_stretch: list = field(default_factory=list)

    # torsional and rolling friction factors
    torsional_friction: float = 0.1
    rolling_friction: float = 0.01

    