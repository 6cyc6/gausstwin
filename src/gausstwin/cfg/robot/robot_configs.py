import torch
from dataclasses import dataclass
from typing import Dict, List, Optional

# Import existing robot configurations
from .fr3_cfg import (
    FR3_LINK_LIST, FR3_INIT_GS_NUM_PTS_LIST,
    VIS_FR3_SPH_PT_LIST, FR3_RADII_LIST, 
    FR3_DEFAULT_LINK_POS, FR3_DEFAULT_LINK_QUAT
)
from .fr3_v3_cfg import (
    FR3_V3_LINK_LIST, FR3_V3_INIT_GS_NUM_PTS_LIST,
    VIS_FR3_V3_SPH_PT_LIST, FR3_V3_RADII_LIST,
    FR3_V3_DEFAULT_LINK_POS, FR3_V3_DEFAULT_LINK_QUAT
)


@dataclass
class RobotConfig:
    """Configuration for a single robot model."""
    link_list: List[str]
    init_gs_num_pt_list: List[int]
    radii_list: List[List[float]]
    sph_pt_list: List[torch.Tensor]
    default_positions: Optional[torch.Tensor] = None
    default_orientations: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """Validate configuration consistency."""
        n_links = len(self.link_list)
        if len(self.init_gs_num_pt_list) != n_links:
            raise ValueError(f"init_gs_num_pt_list length ({len(self.init_gs_num_pt_list)}) doesn't match link_list length ({n_links})")
        if len(self.radii_list) != n_links:
            raise ValueError(f"radii_list length ({len(self.radii_list)}) doesn't match link_list length ({n_links})")
        if len(self.sph_pt_list) != n_links:
            raise ValueError(f"sph_pt_list length ({len(self.sph_pt_list)}) doesn't match link_list length ({n_links})")
        
        if self.default_positions is not None and self.default_positions.shape[0] != n_links:
            raise ValueError(f"default_positions length ({self.default_positions.shape[0]}) doesn't match link_list length ({n_links})")
        if self.default_orientations is not None and self.default_orientations.shape[0] != n_links:
            raise ValueError(f"default_orientations length ({self.default_orientations.shape[0]}) doesn't match link_list length ({n_links})")


# robot configurations dictionary
ROBOT_CONFIGS: Dict[str, RobotConfig] = {
    "fr3": RobotConfig(
        link_list=FR3_LINK_LIST,
        init_gs_num_pt_list=FR3_INIT_GS_NUM_PTS_LIST,
        radii_list=FR3_RADII_LIST,
        sph_pt_list=VIS_FR3_SPH_PT_LIST,
        default_positions=FR3_DEFAULT_LINK_POS,
        default_orientations=FR3_DEFAULT_LINK_QUAT,
    ),
    
    "fr3_v3": RobotConfig(
        link_list=FR3_V3_LINK_LIST,
        init_gs_num_pt_list=FR3_V3_INIT_GS_NUM_PTS_LIST,
        radii_list=FR3_V3_RADII_LIST,
        sph_pt_list=VIS_FR3_V3_SPH_PT_LIST,
        default_positions=FR3_V3_DEFAULT_LINK_POS,
        default_orientations=FR3_V3_DEFAULT_LINK_QUAT,
    ),
    
}


def get_robot_config(robot_name: str) -> RobotConfig:
    if robot_name not in ROBOT_CONFIGS:
        available_robots = list(ROBOT_CONFIGS.keys())
        raise KeyError(f"Robot '{robot_name}' not found. Available robots: {available_robots}")
    
    return ROBOT_CONFIGS[robot_name]


def list_available_robots() -> List[str]:
    return list(ROBOT_CONFIGS.keys())


def update_robot_config(robot_name: str, config: RobotConfig) -> None:
    if robot_name not in ROBOT_CONFIGS:
        available_robots = list(ROBOT_CONFIGS.keys())
        raise KeyError(f"Robot '{robot_name}' not found. Available robots: {available_robots}")
    
    ROBOT_CONFIGS[robot_name] = config


def get_robot_link_count(robot_name: str) -> int:
    config = get_robot_config(robot_name)
    return len(config.link_list)


def get_robot_total_points(robot_name: str) -> int:
    config = get_robot_config(robot_name)
    return sum(config.init_gs_num_pt_list)


def get_link_list(robot_name: str) -> List[str]:
    """Get link list for backward compatibility."""
    return get_robot_config(robot_name).link_list


def get_init_gs_num_pt_list(robot_name: str) -> List[int]:
    """Get initial GS point list for backward compatibility."""
    return get_robot_config(robot_name).init_gs_num_pt_list


def get_radii_list(robot_name: str) -> List[List[float]]:
    """Get radii list for backward compatibility."""
    return get_robot_config(robot_name).radii_list


def get_sph_pt_list(robot_name: str) -> List[torch.Tensor]:
    """Get spherical point list for backward compatibility."""
    return get_robot_config(robot_name).sph_pt_list
