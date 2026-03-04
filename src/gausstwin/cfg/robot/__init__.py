"""
Robot configuration module.
"""

from .robot_configs import (
    RobotConfig,
    ROBOT_CONFIGS,
    get_robot_config,
    list_available_robots,
    update_robot_config,
    get_robot_link_count,
    get_robot_total_points,
    get_link_list,
    get_init_gs_num_pt_list,
    get_radii_list,
    get_sph_pt_list,
)

__all__ = [
    "RobotConfig",
    "ROBOT_CONFIGS", 
    "get_robot_config",
    "list_available_robots",
    "update_robot_config", 
    "get_robot_link_count",
    "get_robot_total_points",
    "get_link_list",
    "get_init_gs_num_pt_list", 
    "get_radii_list",
    "get_sph_pt_list",
]