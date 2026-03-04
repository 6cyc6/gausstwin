import json
import yaml

from pydantic import BaseModel


def default_learning_rates() -> dict[str, float]:
    return {
        "means": 1.6e-5,
        "quats": 1.0e-3,
        "colors": 2.5e-3,
        "scales": 5.0e-3,
        "opacities": 5.0e-2,
    }


class GSConfig(BaseModel):
    """
    Configuration class for the GS model.
    """
    # ======================================= General Config ======================================= #
    vis: bool = False
    ps_vis: bool = False

    # ======================================= GS Optimization Config ======================================= #
    # general
    device: str = "cuda:0"
    # camera
    near_plane: float = 0.01
    far_plane: float = 10.0

    # initialization
    scene_scale: float = 1.0
    init_opacity: float = 0.1
    num_pts: int = 2000
    disk: bool = False   # if use disk Gaussians (False: ellipsoid Gaussians)

    # optimization
    max_iter: int = 2000
    batch_size: int = 8 
    ssim_lambda: float = 0.2
    learning_rates: dict[str, float] = None

    # Strategy Config
    refine_start_iter: int = 800
    refine_stop_iter: int = 1800
    refine_every: int = 100
    prune_opa: float = 0.1
    prune_scale3d: float = 0.1
    grow_scale3d: float = 0.01

    def model_post_init(self, __context) -> None:
        if self.learning_rates is None:
            self.learning_rates = default_learning_rates()

    @staticmethod
    def from_json(file_path: str) -> "GSConfig":
        with open(file_path, "r") as f:
            data = json.load(f)
        return GSConfig.model_validate(data)

    @staticmethod
    def from_yaml(file_path: str) -> "GSConfig":
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return GSConfig.model_validate(data)


# ========================= RIGID BODY CONFIG ========================= #
class RigidBodyConfig(BaseModel):
    """
    Configuration class for the Rigid Object model.
    """
    # ================================== Visualization Config ===================================== #
    vis_init_particles: bool = False
    vis_init_gaussians: bool = False

    # ==================================== General Config ========================================== #
    data_dir: str = "data/real_world"
    exp_name: str = "test_exp"  # Experiment name for saving results

    # ===================================== Camera View Config ======================================= #
    # views
    use_wrist_1: bool = False
    use_wrist_2: bool = False
    wrist_idx_1: int = 0
    wrist_idx_2: int = 0

    fix_idx_1: int = 0
    fix_idx_2: int = 0
    fix_idx_3: int = 0
    fix_idx_4: int = 0

    # ============================== EfficientTAM and SAM2 Config ===================================== #
    sam2_checkpoint: str = ""  # Path to the SAM2 model checkpoint, absolute path
    sam2_config: str = ""      # Path to the SAM2 model config, relative path (relative to the root dir of the sam2 repo)
    tam_checkpoint: str = ""   # Path to the TAM model checkpoint, absolute path
    tam_config: str = ""       # Path to the TAM model config, relative path (relative to the root dir of the efficienttam repo)

    load_prompt: bool = False
    save_prompt: bool = False

    # ======================================= Rigid Object Config ======================================= #
    obj_list: list[str] = []
    n_objs: int = 2
    particle_radius: float = 0.005
    num_views: int = 4

    # =================================== Particle Init Config ==================================== #
    dense_z: bool = False
    cover_radius: float = 0.005
    depth_threshold: float = 0.005
    n_mask_tol: int = 0
    n_depth_tol: int = 0

    # ======================================= 3DGS Config ======================================= #
    save_ground_gaussians: bool = False
    # general
    device: str = "cuda:0"
    near_plane: float = 0.01
    far_plane: float = 10.0

    # initialization
    scene_scale: float = 1.0
    init_opacity: float = 0.1
    num_pts: int = 80000
    num_gaussians: int = 10000
    disk: bool = True   # if use disk Gaussians (False: ellipsoid Gaussians)
    batch_size: int = 5

    # particle initialization
    particle_max_iter: int = 500
    particle_collision_max_iter: int = 5
    particle_batch: bool = True  # if use batch optimization

    opacity_threshold: float = 0.3  # Opacity threshold for optimization
    min_scale: tuple[float, float, float] = (0.001, 0.001, 0.001)
    max_scale: tuple[float, float, float] = (0.005, 0.005, 0.005)

    # gaussians initialization
    gaussian_max_iter: int = 1050
    gaussian_batch: bool = False

    # optimization
    ssim_lambda: float = 0.2
    use_rgbd: bool = False  # if True, use RGBD for 3DGS training (only for non-ground objects)
    depth_lambda: float = 0.005  # weight for depth regularization
    learning_rates_particles: dict[str, float] = None
    learning_rates_gaussians: dict[str, float] = None

    # Gaussian Control Strategy Config
    refine_start_iter: int = 800
    refine_stop_iter: int = 1800
    refine_every: int = 100
    reset_every: int = 500
    prune_opa: float = 0.1
    prune_scale3d: float = 0.1
    grow_scale3d: float = 0.01

    def model_post_init(self, __context) -> None:
        if self.learning_rates_gaussians is None:
            self.learning_rates_gaussians = {
                "means": 1.0e-4,
                "quats": 1.0e-3,
                "colors": 2.5e-3,
                "scales": 5.0e-3,
                "opacities": 1.0e-2,
            }
        if self.learning_rates_particles is None:
            self.learning_rates_particles = default_learning_rates()

    @staticmethod
    def from_json(file_path: str) -> "RigidBodyConfig":
        with open(file_path, "r") as f:
            data = json.load(f)
        return RigidBodyConfig.model_validate(data)

    @staticmethod
    def from_yaml(file_path: str) -> "RigidBodyConfig":
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return RigidBodyConfig.model_validate(data)


# ========================= ROPE CONFIG ========================= #
class RopeConfig(BaseModel):
    """
    Configuration class for the Rope model.
    """
    # ================================== Visualization Config ===================================== #
    vis_init_particles: bool = False
    vis_init_gaussians: bool = False

    # ==================================== General Config ========================================== #
    data_dir: str = "data/real_world"
    exp_name: str = "test_exp"  # Experiment name for saving results

    # ===================================== Camera View Config ======================================= #
    # views
    use_wrist_1: bool = False
    use_wrist_2: bool = False
    wrist_idx_1: int = 0
    wrist_idx_2: int = 0

    fix_idx_1: int = 0
    fix_idx_2: int = 0
    fix_idx_3: int = 0
    fix_idx_4: int = 0

    # ============================== EfficientTAM and SAM2 Config ===================================== #
    sam2_checkpoint: str = ""  # Path to the SAM2 model checkpoint, absolute path
    sam2_config: str = ""      # Path to the SAM2 model config, relative path (relative to the root dir of the sam2 repo)
    tam_checkpoint: str = ""   # Path to the TAM model checkpoint, absolute path
    tam_config: str = ""       # Path to the TAM model config, relative path (relative to the root dir of the tam repo)

    load_prompt: bool = False
    save_prompt: bool = False

    # ======================================= Object Config ======================================= #
    obj_list: list[str] = []
    n_objs: int = 1
    particle_radius: float = 0.005
    num_views: int = 5

    # rope
    n_segments: int = 24
    remove_ends: bool = False

    # =================================== Particle Init Config ==================================== #
    dense_z: bool = False
    cover_radius: float = 0.005
    depth_threshold: float = 0.005
    n_mask_tol: int = 0
    n_depth_tol: int = 0

    # ======================================= 3DGS Config ======================================= #
    save_ground_gaussians: bool = False
    # general
    device: str = "cuda:0"
    near_plane: float = 0.01
    far_plane: float = 10.0

    # initialization
    scene_scale: float = 1.0
    init_opacity: float = 0.1
    num_pts: int = 80000
    num_gaussians: int = 10000
    disk: bool = True   # if use disk Gaussians (False: ellipsoid Gaussians)
    batch_size: int = 5

    # particle initialization
    particle_max_iter: int = 500
    particle_collision_max_iter: int = 5
    particle_batch: bool = True  # if use batch optimization

    opacity_threshold: float = 0.3  # Opacity threshold for optimization
    min_scale: tuple[float, float, float] = (0.001, 0.001, 0.001)
    max_scale: tuple[float, float, float] = (0.005, 0.005, 0.005)

    # gaussians initialization
    gaussian_max_iter: int = 1050
    gaussian_batch: bool = False

    # optimization
    ssim_lambda: float = 0.2
    use_rgbd: bool = False  # if True, use RGBD for 3DGS training (only for non-ground objects)
    depth_lambda: float = 0.005  # weight for depth regularization
    learning_rates_particles: dict[str, float] = None
    learning_rates_gaussians: dict[str, float] = None

    # Gaussian Control Strategy Config
    refine_start_iter: int = 800
    refine_stop_iter: int = 1800
    refine_every: int = 100
    reset_every: int = 1000
    prune_opa: float = 0.1
    prune_scale3d: float = 0.1
    grow_scale3d: float = 0.01

    def model_post_init(self, __context) -> None:
        if self.learning_rates_gaussians is None:
            self.learning_rates_gaussians = {
                "means": 1.0e-4,
                "quats": 1.0e-3,
                "colors": 2.5e-3,
                "scales": 5.0e-3,
                "opacities": 1.0e-2,
            }
        if self.learning_rates_particles is None:
            self.learning_rates_particles = default_learning_rates()

    @staticmethod
    def from_json(file_path: str) -> "RopeConfig":
        with open(file_path, "r") as f:
            data = json.load(f)
        return RopeConfig.model_validate(data)

    @staticmethod
    def from_yaml(file_path: str) -> "RopeConfig":
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return RopeConfig.model_validate(data)
    