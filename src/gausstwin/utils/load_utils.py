import os
import json

from gausstwin.utils.path_utils import get_save_dir, get_save_sim_dir


def get_robot_json(file_name: str="panda"):
    """
    Load the robot model from json.
    """
    save_dir = get_save_dir()
    save_path = f"{save_dir}/{file_name}.json"
    
    try:
        with open(save_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {save_path} not found.")
    
    return data


def get_rigid_body_json(obj_name: str="Cube", exp_name: str="test_exp", sim=False):
    """
    Load the rigid object model from json.
    """
    if sim:
        save_dir = os.path.join(get_save_sim_dir(), exp_name)
    else:
        save_dir = os.path.join(get_save_dir(), exp_name)
    save_path = f"{save_dir}/{obj_name}.json"
    
    try:
        with open(save_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {save_path} not found.")
    
    return data


def get_rope_json(obj_name: str="Rope", exp_name: str="test_exp", sim=False):
    """
    Load the rope model from json.
    """
    if sim:
        save_dir = os.path.join(get_save_sim_dir(), exp_name)
    else:
        save_dir = os.path.join(get_save_dir(), exp_name)
    save_path = f"{save_dir}/{obj_name}.json"
    
    try:
        with open(save_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {save_path} not found.")
    
    return data


def get_objs_prompt_json(exp_name: str="test_exp", sim=False):
    """
    Load the rigid object model from json.
    """
    if sim:
        save_dir = os.path.join(get_save_sim_dir(), exp_name)
    else:
        save_dir = os.path.join(get_save_dir(), exp_name)
    save_path = f"{save_dir}/prompt_objs.json"

    try:
        with open(save_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {save_path} not found.")

    return data


def get_ground_json(exp_name: str="test_exp", sim=False):
    """
    Load the ground model from json.
    """
    if sim:
        save_dir = os.path.join(get_save_sim_dir(), exp_name)
    else:
        save_dir = os.path.join(get_save_dir(), exp_name)
    save_path = f"{save_dir}/ground.json"
    
    try:
        with open(save_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {save_path} not found.")
    
    return data