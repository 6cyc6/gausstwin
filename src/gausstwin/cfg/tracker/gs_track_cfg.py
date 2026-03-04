import yaml, json
from dataclasses import dataclass, field
from gausstwin.cfg.sim.pbd_cfg import PBDConfig


def default_learning_rates():
    return {
        "means": 1.6e-5,
        "quats": 1.0e-3,
        "colors": 0.0,
        "scales": 0.0,
        "opacities": 0.0,
    } 


@dataclass
class EfficientTAM:
    tam_checkpoint: str = ""
    tam_config: str = "configs/efficienttam/efficienttam_s_512x512.yaml"
    sam2_checkpoint: str = ""
    sam2_config: str = ""
    n_cams: int = 3

    fix_idx_1: int = 0
    fix_idx_2: int = 0
    fix_idx_3: int = 0
    fix_idx_4: int = 0


@dataclass
class GSTrackUnifiedConfig(PBDConfig, EfficientTAM):
    """
    Configuration class for the GS model.
    """
    # ======================================= General Config ======================================= #
    # visualization
    vis: bool = False
    eval: bool = False
    
    # experiment configs
    exp_name: str = "exp_test"
    robot_name: str = "fr3"
    data_dir: str = "/home/galois/Downloads/output_data/t_shape_0"
    obj_list: list[str] = field(default_factory=list)
    device: str = "cuda:0"
    
    # video crop
    disp: int = 0
    length: int = 200
    max_idx: int = 500
    # optic track id
    gt_id: int = 1052
    gt_id_2: int = 1053  # for second object in multi-object tracking
    
    # ======================================= GS Optimization Config ======================================= #
    num_views: int = 3
    mode: int = 4  # 0: GS means + quats, 1: GS means + quats + object position, 2: obj pos + rot, 3: obj pos + rot + GS means + quats, 4: obj pos + rot + GS means 
    
    # general
    near_plane: float = 0.01
    far_plane: float = 10.0
    
    # initialization
    scene_scale: float = 1.0
    
    # visual force
    use_seg_mask: bool = True
    batch: bool = True
    
    # learning rates
    vf_iterations: int = 5
    vf_lr_disps: float = 0.001
    vf_lr_rots: float = 0.001
    vf_lr_means: float = 0.001
    vf_lr_quats: float = 0.0001
    vf_lr_colors: float = 0.0005
    vf_lr_opacities: float = 0.0005
    vf_lr_scales: float = 0.0
    
    # visual force
    kp_f: float = 1.0
    kp_m: float = 1.0
    gain_1: float = 1.0
    gain_2: float = 1.0

    
    @staticmethod
    def from_json(file_path: str) -> "GSTrackUnifiedConfig":
        """ Load config from a JSON file """
        with open(file_path, "r") as f:
            data = json.load(f)
        return GSTrackUnifiedConfig(**data) 
    
    
    @staticmethod
    def from_yaml(file_path: str) -> "GSTrackUnifiedConfig":
        """ Load config from a YAML file """
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return GSTrackUnifiedConfig(**data) # type: ignore
    
    
@dataclass
class GSTrackRealWorldMultiConfig(PBDConfig, EfficientTAM):
    """
    Configuration class for the GS model.
    """
    # ======================================= General Config ======================================= #
    log_info: bool = False
    vis: bool = False
    ps_vis: bool = False
    
    exp_name: str = "exp_test"
    robot_name: str = "fr3"
    data_dir: str = "/home/galois/Downloads/output_data/t_shape_0"
    obj_list: list[str] = field(default_factory=list)
    
    num_views: int = 3
    mode: int = 2  # 0: GS means + quats, 1: GS means + quats + object position, 2: obj pos + rot, 3: obj pos + rot + GS means + quats, 4: obj pos + rot + GS means 
    set_target: bool = False
    
    # video crop
    disp: int = 0
    length: int = 200
    max_idx: int = 500
    # optic track id
    gt_id: int = 1052
    gt_id_2: int = 1053  # for second object in multi-object tracking

    # object 2
    obj2_mu: float = 0.4
    obj2_m: float = 0.2
    obj2_radii: float = 0.005
    obj2_density_factor: float = 1.0
    obj2_restitution: float = 0.0

    # ======================================= GS Optimization Config ======================================= #
    # general
    device: str = "cuda:0"
    near_plane: float = 0.01
    far_plane: float = 10.0
    
    # initialization
    scene_scale: float = 1.0
    
    # visual force
    use_seg_mask: bool = True
    batch: bool = True
    vf_iterations: int = 5
    vf_lr_disps: float = 0.001
    vf_lr_rots: float = 0.001
    vf_lr_means: float = 0.001
    vf_lr_quats: float = 0.001
    vf_lr_colors: float = 0.0
    vf_lr_opacities: float = 0.0
    vf_lr_scales: float = 0.0
    
    vf_kp: float = 15.0
    kp_f: float = 0.5
    kp_m: float = 1.0

    # gain for multi-object tracking
    gain_1: float = 1.0
    gain_2: float = 1.0
    
    @staticmethod
    def from_json(file_path: str) -> "GSTrackRealWorldMultiConfig":
        """ Load config from a JSON file """
        with open(file_path, "r") as f:
            data = json.load(f)
        return GSTrackRealWorldMultiConfig(**data) 
    
    
    @staticmethod
    def from_yaml(file_path: str) -> "GSTrackRealWorldMultiConfig":
        """ Load config from a YAML file """
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return GSTrackRealWorldMultiConfig(**data) # type: ignore
    