import torch
import logging
import warp as wp
import numpy as np

from gausstwin.sim.model_builder.model import Model
from gausstwin.cfg.domain import Rope, Robot, Ground
from gausstwin.sim.model_builder.builder import ModelBuilder
from gausstwin.gaussians.gaussians import GaussianUnifiedModel
from gausstwin.cfg.tracker.gs_track_cfg import GSTrackUnifiedConfig
from gausstwin.cfg.robot.fr3_cfg import FR3_DEFAULT_LINK_POS, FR3_DEFAULT_LINK_QUAT
from gausstwin.utils.math_utils import convert_quat
from gausstwin.utils.model_builder_utils import mass_inertia_spheres
from gausstwin.utils.load_utils import get_robot_json, get_rope_json, get_ground_json


class RopeBuilder(ModelBuilder):
    def __init__(self, exp_name="test_exp"):
        super().__init__()
        
        wp.init()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.exp_name = exp_name
        # logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # gaussians
        self.gaussian_means = []
        self.gaussian_quats = []
        self.gaussian_scales = []
        self.gaussian_opacities = []
        self.gaussian_colors = []
        # Gaussian indices
        self.gs_idx = 0              # gaussian index
        self.gs_robot_indices = []   # Gaussians - robot
        self.gs_rope_indices = []    # Gaussians - rope
        # Gaussians - bodies
        self.gs_body_ids = []             # body id for gaussians belong to bodies (num_body_gaussians,)
        self.gs_robot_body_idx = 0        # robot body index
        self.gs_robot_body_ids = []       # robot body id for gaussians belong to robot (num_robot_gaussians,)
        self.gs_robot_body_local_ids = [] # local id (num_robot_gaussians,)
        self.gs_rope_part_idx = 0         # rope particle index
        self.gs_rope_part_ids = []        # rope particle id for gaussians belong to rope (num_rope_gaussians,)
        self.gs_rope_quat_ids = []        # rope quaternion(segment) id (num_rope_gaussians,)
        # Gaussians - particles
        self.gaussian_particle_ids = [] 
        self.gaussian_particle_id_weights = [] 
        
        # rope in track idx list
        self.rope_part_in_track_idx_list = []
        # default robot link positions and quaternions
        self.n_robot_links = 0
        self.robot_positions = FR3_DEFAULT_LINK_POS.to(self.device)
        self.robot_positions[:, 2] -= 1.024
        self.robot_quats = FR3_DEFAULT_LINK_QUAT.to(self.device)
      
    
    def set_ground(
        self,
        mu: 0.5,
        restitution: 0.0,
    ):
        """ Add the ground plane to the model. """
        data = get_ground_json(exp_name=self.exp_name)
        ground = Ground(**data)
        logging.info(f"Ground plane is set")
        
        # get ground parameters
        self.ground_normal = ground.normal
        self.ground_h = ground.h # height, negative value
        self.ground_disp = torch.tensor([0.0, 0.0, self.ground_h], device=self.device, dtype=torch.float32) # (3,)
        
        self._ground_params["mu"] = mu
        self._ground_params["restitution"] = restitution


    def add_robot_rigid_body(
        self,
        name: str = "fr3_v3",
        obj_radii: float = 0.005,
        gripper_mu: float = 0.3,
        gripper_density_factor: float = 0.8,
    ):
        """ Add rigid bodies for the robot to the model. """
        data = get_robot_json(name)
        robot = Robot(**data)
        links_dict = robot.links
        logging.info(f"------------------ Loading robot {name} ---------------------")
        
        link_idx = 0
        for link_name, data in links_dict.items():
            logging.info(f"Loading robot link: {link_name}")
            # -------------------------------- load gaussians ---------------------------------- #
            # load gaussians
            gaussians = data.gaussians
            n_gaussians = len(gaussians.means)
            self.gaussian_means.extend(gaussians.means)
            self.gaussian_quats.extend(gaussians.quats) 
            self.gaussian_scales.extend(gaussians.scales) 
            self.gaussian_opacities.extend(gaussians.opacities)
            self.gaussian_colors.extend(gaussians.colors)

            # save indices
            self.gs_robot_indices.extend([self.gs_idx + idx for idx in range(n_gaussians)])
            self.gs_robot_body_ids.extend([self.body_count] * n_gaussians) 
            self.gs_robot_body_local_ids.extend([self.gs_robot_body_idx] * n_gaussians) 
            self.gs_body_ids.extend([self.body_count] * n_gaussians)
            
            # -------------------------------- build PBD model ---------------------------------- #
            # load particles
            particles = data.particles
            link_pos = self.robot_positions[link_idx] # (3,)
            link_quat = convert_quat(self.robot_quats[link_idx], to="xyzw")

            if "finger" in link_name:
                m, I_m, com = mass_inertia_spheres(particles.means, particles.radii, gripper_density_factor) # compute mass and inertia
                logging.info(f"Gripper mass: {m}")
                b = self.add_body(
                    origin=wp.transform(link_pos.cpu().numpy().tolist(), link_quat.cpu().numpy().tolist()), # link pose 
                    m=m,                        # mass
                    com=wp.vec3(com[0], com[1], com[2]), # center of mass 
                    I_m=wp.mat33(I_m.tolist()), # inertia
                    is_robot=True,
                )
                # add collision shapes (particles)
                for pt_idx in range(len(particles.means)): 
                    self.add_shape_sphere(
                        body=b,
                        pos=particles.means[pt_idx],
                        radius=particles.radii[pt_idx] - obj_radii, # consider object particle radii
                        mu=gripper_mu,
                        density=0.0, # set density to 0.0 to avoid double counting
                    )
            else:
                m, I_m, com = mass_inertia_spheres(particles.means, particles.radii, 2.7) 
                b = self.add_body(
                    origin=wp.transform(link_pos.cpu().numpy().tolist(), link_quat.cpu().numpy().tolist()), # link pose 
                    m=m,                        # mass
                    com=wp.vec3(com[0], com[1], com[2]), # center of mass 
                    I_m=wp.mat33(I_m.tolist()), # inertia
                    is_robot=True,
                )
                for pt_idx in range(len(particles.means)):
                    self.add_shape_sphere(
                        body=b,
                        pos=particles.means[pt_idx],
                        radius=particles.radii[pt_idx] - obj_radii,
                        density=0.0, # set density to 0.0 to avoid double counting
                    )
            
            # update index
            self.gs_idx += n_gaussians 
            self.gs_robot_body_idx += 1
            link_idx += 1

        self.n_robot_links = link_idx # number of robot links

    
    def add_rope(
        self,
        name: str = "Rope",
        m: float = 0.1,
        mass_q_factor: float = 1.0,
        stiffness_bend: list = [0.01, 0.01, 0.02],
        stiffness_stretch: list = [0.85, 0.85, 0.95],
    ):
        """Add a Rope to the model."""
        # load rope
        data = get_rope_json(name, self.exp_name)
        obj = Rope(**data)
        n_parts = len(obj.particles.means) 
        logging.info(f"--------------- Loading tracking object: {name} with {n_parts} parts ---------------")
        
        # save pose
        pose = obj.poses
        pose_mat = torch.tensor(pose, device=self.device, dtype=torch.float32) # (n_parts, 4, 4)

        # ------------------------------- load gaussians ---------------------------------- #
        gaussians = obj.gaussians
        n_gaussians = len(gaussians.means) 
        self.gaussian_means.extend(gaussians.means) 
        self.gaussian_quats.extend(gaussians.quats)
        self.gaussian_scales.extend(gaussians.scales) 
        self.gaussian_opacities.extend(gaussians.opacities) 
        self.gaussian_colors.extend(gaussians.colors) 
        
        # save indices
        self.gs_rope_indices.extend([self.gs_idx + idx for idx in range(n_gaussians)])
        self.gs_rope_part_ids.extend([self.gs_rope_part_idx + idx for idx in obj.gs_particle_indices]) 
        self.gs_rope_quat_ids.extend([self.gs_rope_part_idx + idx if idx != n_parts - 1 else self.gs_rope_part_idx + idx - 1
                                      for idx in obj.gs_particle_indices]) 
        self.rope_part_in_track_idx_list.extend([self.gs_rope_part_idx + idx for idx in range(n_parts)]) 
        
        # -------------------------- load gaussians - particles ---------------------------- #
        start_particle_idx = self.particle_count
        # save gaussian particle ids
        self.gaussian_particle_ids.extend((np.array(obj.gs_particle_indices) + start_particle_idx).tolist())
        
        # compute weights for gaussians based on particle counts
        gaussian_particle_local_indices = torch.tensor(obj.gs_particle_indices).to(self.device)
        particle_counts = torch.bincount(gaussian_particle_local_indices) 
        gaussian_counts = particle_counts[obj.gs_particle_indices]
        weights = 1.0 / gaussian_counts.float() # (N, )
        # save weights
        self.gaussian_particle_id_weights.extend(weights.cpu().numpy().tolist()) 
        
        # ----------------------------- add rope to the PBD model ------------------------------- #
        self.add_rod(
            pos=obj.particles.means,
            mass=[m]*n_parts,
            radius=obj.particles.radii[0],
            mass_q_factor=mass_q_factor,
            stiffness_bend=stiffness_bend,
            stiffness_stretch=stiffness_stretch,
            dh=0.01 + self.ground_h,
        )
        
        # update index
        self.gs_idx += n_gaussians 
        self.gs_rope_part_idx += n_parts
        
        
    def build_gaussian_model(self, device: str = "cuda"):
        gaussian_model = GaussianUnifiedModel(
            means=torch.tensor(self.gaussian_means, device=device, dtype=torch.float32),
            quats=torch.tensor(self.gaussian_quats, device=device, dtype=torch.float32),
            scales=torch.tensor(self.gaussian_scales, device=device, dtype=torch.float32),
            opacities=torch.tensor(self.gaussian_opacities, device=device, dtype=torch.float32),
            colors=torch.tensor(self.gaussian_colors, device=device, dtype=torch.float32),
            
            # Gaussian indices
            robot_indices=torch.tensor(self.gs_robot_indices, device=device, dtype=torch.int32),
            rigid_indices=torch.empty(0, device=device, dtype=torch.int32),  
            rope_indices=torch.tensor(self.gs_rope_indices, device=device, dtype=torch.int32),
            
            # Global body/part ids
            body_ids=torch.tensor(self.gs_body_ids, device=device, dtype=torch.int32),
            robot_body_ids=torch.tensor(self.gs_robot_body_ids, device=device, dtype=torch.int32),
            rigid_body_ids=torch.empty(0, device=device, dtype=torch.int32),  
            rope_part_ids=torch.tensor(self.gs_rope_part_ids, device=device, dtype=torch.int32),
            rope_quat_ids=torch.tensor(self.gs_rope_quat_ids, device=device, dtype=torch.int32),
            
            # Local body/part ids
            robot_body_local_ids=torch.tensor(self.gs_robot_body_local_ids, device=device, dtype=torch.int32),
            rigid_body_local_ids=torch.empty(0, device=device, dtype=torch.int32), 
        )
        
        return gaussian_model


    def add_builder(
        self,
        builder,
        xform=None,
    ):
        super().add_builder(builder, xform=xform)
        

    def finalize(self, cfg: GSTrackUnifiedConfig, device=None, requires_grad=False) -> Model:
        """ Finalize the model. """
        # finalize the PBD model
        self.requires_grad = requires_grad
        if device is None:
            device = wp.get_preferred_device()
        model = super().finalize(device, requires_grad)

        model.rigid_contact_torsional_friction = cfg.torsional_friction  
        model.rigid_contact_rolling_friction = cfg.rolling_friction 
        
        # build gaussians
        self.gaussian_model = self.build_gaussian_model(device=self.device)
        self.gaussian_state = self.gaussian_model.state()

        return model
    