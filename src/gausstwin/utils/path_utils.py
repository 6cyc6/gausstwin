import os

ROOT_DIR = os.environ["ROOT_DIR"]
GS_TWIN_DIR = os.environ["GS_TWIN_DIR"]
FRANKA_DIR = os.environ["FRANKA_DIR"]

def safe_mkdir(dir_path):
    """
    Create a directory if it does not exist.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_gs_cam_cfg_dir():
    return os.path.join(GS_TWIN_DIR, "cfg/camera")


def get_gs_fig_dir():
    return os.path.join(GS_TWIN_DIR, "fig")


def get_cfg_dir():
    return os.path.join(GS_TWIN_DIR, "cfg")


# =============== For Loading Configs =============== #
def get_track_cfg_path(cfg_name):
    return os.path.join(ROOT_DIR, "config/tracking", cfg_name)


def get_builder_cfg_path(cfg_name):
    return os.path.join(ROOT_DIR, "config/builder", cfg_name)


def get_plan_cfg_path(cfg_name):
    return os.path.join(ROOT_DIR, "config/planning", cfg_name)


def get_save_dir():
    return os.path.join(GS_TWIN_DIR, "save")


# =============== Simulation Config =============== #
def get_cfg_sim_path(cfg_name):
    return os.path.join(GS_TWIN_DIR, "cfg/config/sim", cfg_name)


def get_save_sim_dir():
    return os.path.join(GS_TWIN_DIR, "save/sim")


def get_fig_dir():
    return os.path.join(GS_TWIN_DIR, "fig")


def get_franka_dir():
    return FRANKA_DIR
