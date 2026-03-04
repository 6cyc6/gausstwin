import os
import json
import torch
import logging
import numpy as np

from typing import Literal, Dict, Optional
from pydantic import BaseModel
from dataclasses import dataclass
from gausstwin.utils.path_utils import safe_mkdir, get_save_dir, get_save_sim_dir


class Gaussians(BaseModel):
    """
    Class representing a Gaussian distribution.
    """
    means: list[list[float]]   # (n, 3)
    quats: list[list[float]]   # (n, 4) (w, x, y, z)
    scales: list[list[float]]  # (n, 3)
    opacities: list[float]     # (n,)
    colors: list[list[float]]  # (n, 3)

    def __len__(self):
        return len(self.means)
    
    def mask(self, mask: np.ndarray):
        return Gaussians(
            means=np.asarray(self.means)[mask].tolist(),
            quats=np.asarray(self.quats)[mask].tolist(),
            scales=np.asarray(self.scales)[mask].tolist(),
            opacities=np.asarray(self.opacities)[mask].tolist(),
            colors=np.asarray(self.colors)[mask].tolist()
        )
        

class Particles(BaseModel):
    """
    Particles.
    """
    means: list[list[float]]  # (n, 3)
    radii: list[float]        # (n,)
    sdf_mag: Optional[list[float]] = None  # (n,)
    sdf_is_boundary: Optional[list[bool]] = None # (n,)
    
    def __len__(self):
        return len(self.means)
    
    def mask_slice(self, mask: np.ndarray):
        return Particles(
            means=np.asarray(self.means)[mask].tolist(),
            radii=np.asarray(self.radii)[mask].tolist(),
        )


class SegPrompt(BaseModel):
    """
    Segmentation prompt class for specifying segmentation information.
    """
    bbox: list[float]  # [x_min, y_min, x_max, y_max] top left and bottom right corners
    uvs: list[list[float]]  # List of [u, v] coordinates for segmentation points
    labels: list[int]  # List of labels for each point in uvs (1 for object, 0 for non-object)
    

class Ground(BaseModel):
    """
    Class representing the ground plane.
    """
    normal: list[float]  # (3,)
    h: float   # (1,)
    gaussians: Gaussians | None = None
    
    def save(self, exp_name: str = "test_exp", sim=False):
        # save the model
        if sim:
            save_dir = os.path.join(get_save_sim_dir(), exp_name)
        else:       
            save_dir = os.path.join(get_save_dir(), exp_name)
        safe_mkdir(save_dir)
        save_path = f"{save_dir}/ground.json"
        
        with open(save_path, "w") as f:
            f.write(self.model_dump_json(indent=4))
        logging.info(f"Ground plane is saved to {save_path}")
        

class RigidBody(BaseModel):
    """
    Rigid body class for representing a rigid body in 3D space.
    """
    name: str
    gaussians: Gaussians | None = None
    particles: Particles | None = None
    pose: list[list[float]] | None = None  # (4, 4)
    gs_particle_indices: list[int] | None = None
    prompts: Optional[Dict[str, SegPrompt]] = None  # key: view name, e.g., 'view_0'
    
    def save(self, exp_name: str = "test_exp", sim=False):
        # save the model
        if sim:
            save_dir = os.path.join(get_save_sim_dir(), exp_name)
        else:
            save_dir = os.path.join(get_save_dir(), exp_name)
        safe_mkdir(save_dir)
        save_path = f"{save_dir}/{self.name}.json"
        
        with open(save_path, "w") as f:
            f.write(self.model_dump_json(indent=4))
        logging.info(f"Rigid body {self.name} is saved to {save_path}")


class Rope(BaseModel):
    """
    Rigid body class for representing a rigid body in 3D space.
    """
    name: str
    gaussians: Gaussians | None = None
    particles: Particles | None = None
    poses: list[list[list[float]]] | None = None  # (N_particles, 4, 4)
    gs_particle_indices: list[int] | None = None
    prompts: Optional[Dict[str, SegPrompt]] = None  # key: view name, e.g., 'view_0'
    
    def save(self, exp_name: str = "test_exp", sim=False):
        # save the model
        if sim:
            save_dir = os.path.join(get_save_sim_dir(), exp_name)
        else:
            save_dir = os.path.join(get_save_dir(), exp_name)
        safe_mkdir(save_dir)
        save_path = f"{save_dir}/{self.name}.json"
        
        with open(save_path, "w") as f:
            f.write(self.model_dump_json(indent=4))
        logging.info(f"Rigid body {self.name} is saved to {save_path}")
        

class Robot(BaseModel):
    """
    Robot class for representing a robot in 3D space.
    """
    robot_name: Literal["panda", "talos", "fr3", "fr3_v3", "fr3_sim"]
    
    links: Dict[str, RigidBody] = {}


    def add_link(self, link_name: str, rigid_body: RigidBody):
        self.links[link_name] = rigid_body


    def get_link(self, link_name: str) -> Optional[RigidBody]:
        return self.links.get(link_name)

    
    def save(self, file_name: str="panda"):
        """
        Save the robot model to a file.
        """
        save_dir = get_save_dir()
        safe_mkdir(save_dir)
        save_path = f"{save_dir}/{file_name}.json"
        
        with open(save_path, "w") as f:
            f.write(self.model_dump_json(indent=4))
        logging.info(f"Robot model saved to {save_path}")
    
    
    # def load(self, file_name: str="panda"):
    #     """
    #     Load the robot model from a file.
    #     """
    #     save_dir = get_save_dir()
    #     save_path = f"{save_dir}/{file_name}.json"
        
    #     try:
    #         with open(save_path, "r") as f:
    #             data = json.load(f)
    #     except FileNotFoundError:
    #         raise FileNotFoundError(f"File {save_path} not found.")
    

class RigidBodyRealWorld(RigidBody):
    prompts: Dict[str, SegPrompt] = {} # key: view name, e.g., 'view_0'
    
    def save(self, exp_name: str = "test_exp"):
        # save the model
        save_dir = os.path.join(get_save_dir(), exp_name)
        safe_mkdir(save_dir)
        save_path = f"{save_dir}/{self.name}.json"
        
        with open(save_path, "w") as f:
            f.write(self.model_dump_json(indent=4))
        logging.info(f"Rigid body {self.name} is saved to {save_path}")


class RigidBodySim(RigidBody):
    prompts: Dict[str, SegPrompt] = {} # key: view name, e.g., 'view_0'
    
    def save(self, exp_name: str = "test_exp"):
        # save the model
        save_dir = os.path.join(get_save_sim_dir(), exp_name)
        safe_mkdir(save_dir)
        save_path = f"{save_dir}/{self.name}.json"
        
        with open(save_path, "w") as f:
            f.write(self.model_dump_json(indent=4))
        logging.info(f"Rigid body {self.name} is saved to {save_path}")


class RopeSim(Rope):
    def save(self, exp_name: str = "test_exp"):
        # save the model
        save_dir = os.path.join(get_save_sim_dir(), exp_name)
        safe_mkdir(save_dir)
        save_path = f"{save_dir}/{self.name}.json"
        
        with open(save_path, "w") as f:
            f.write(self.model_dump_json(indent=4))
        logging.info(f"Rope {self.name} is saved to {save_path}")


class RopeSimEmbodied(Rope):
    prompts: Dict[str, SegPrompt] = {}  # key: view name, e.g., 'view_0'
    gs_segment_indices: list[int] | None = None
    particle_segment_indices: list[int] | None = None
    
    def save(self, exp_name: str = "test_exp"):
        # save the model
        save_dir = os.path.join(get_save_sim_dir(), exp_name)
        safe_mkdir(save_dir)
        save_path = f"{save_dir}/{self.name}.json"
        
        with open(save_path, "w") as f:
            f.write(self.model_dump_json(indent=4))
        logging.info(f"Rope {self.name} is saved to {save_path}")
        
        