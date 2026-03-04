import torch
import logging
import warp as wp
import numpy as np

from gausstwin.sim.model_builder.model import Model
from gausstwin.cfg.domain import RigidBody, Robot, Ground
from gausstwin.sim.model_builder.builder import ModelBuilder
from gausstwin.gaussians.gaussians import GaussianUnifiedModel
from gausstwin.cfg.tracker.gs_track_cfg import GSTrackUnifiedConfig
from gausstwin.cfg.robot.fr3_cfg import FR3_DEFAULT_LINK_POS, FR3_DEFAULT_LINK_QUAT
from gausstwin.utils.model_builder_utils import mass_inertia_spheres
from gausstwin.utils.math_utils import convert_quat, quat_from_matrix
from gausstwin.utils.load_utils import get_robot_json, get_rigid_body_json, get_ground_json


class RigidBuilder(ModelBuilder):
    def __init__(self, exp_name="test_exp"):
        super().__init__()
        
        wp.init()  # init warp runtime
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
        self.gs_idx = 0            # gaussian index
        self.gs_robot_indices = [] # Gaussians - robot
        self.gs_rigid_indices = [] # Gaussians - rigid bodies
        # Gaussians - bodies
        self.gs_body_ids = []             # body id for gaussians belong to bodies (num_body_gaussians,)
        self.gs_robot_body_idx = 0        # robot body index
        self.gs_robot_body_ids = []       # robot body id for gaussians belong to robot (num_robot_gaussians,)
        self.gs_robot_body_local_ids = [] # local id (num_robot_gaussians,)
        self.gs_rigid_body_idx = 0        # rigid body index
        self.gs_rigid_body_ids = []       # rigid body id for gaussians belong to rigid bodies (num_rigid_gaussians,)
        self.gs_rigid_body_local_ids = [] # local id (num_rigid_gaussians,)
        # Gaussians - particles
        self.gaussian_particle_ids = [] 
        self.gaussian_particle_id_weights = [] 
        
        # bodies in track idx list
        self.body_in_track_idx_list = []
        self.body_in_track_idx_local_list = []
        # default robot link positions and quaternions
        self.n_robot_links = 0
        self.robot_positions = FR3_DEFAULT_LINK_POS.to("cuda:0")
        self.robot_positions[:, 2] -= 1.024
        self.robot_quats = FR3_DEFAULT_LINK_QUAT.to("cuda:0")
      
    
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
        
        # set ground physical parameters
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
                    com=wp.vec3(com[0], com[1], com[2]), # center of mass 
                    I_m=wp.mat33(I_m.tolist()), # inertia
                    m=m,  # mass
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
                    com=wp.vec3(com[0], com[1], com[2]), # center of mass 
                    I_m=wp.mat33(I_m.tolist()), # inertia
                    m=m,  # mass
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

    
    def add_rigid_object(
        self,
        name: str,
        obj_radii=0.005,
        mu=0.2,
        density_factor=1.0,
        dz=0.01,
    ):
        """Add a rigid body object to the model."""
        # load rigid body
        data = get_rigid_body_json(name, self.exp_name)
        obj = RigidBody(**data)
        logging.info(f"Loading tracking object: {name}")
        # get pose
        pose = obj.pose
        pose_mat = torch.tensor(pose, device=self.device, dtype=torch.float32) # (4, 4)
        obj_quat = quat_from_matrix(pose_mat[:3, :3]) # (4, ), wxyz
        obj_pos = pose_mat[:3, 3] # (3, )

        # ------------------------------- load gaussians ---------------------------------- #
        gaussians = obj.gaussians
        n_gaussians = len(gaussians.means) 
        self.gaussian_means.extend(gaussians.means) 
        self.gaussian_quats.extend(gaussians.quats)
        self.gaussian_scales.extend(gaussians.scales) 
        self.gaussian_opacities.extend(gaussians.opacities) 
        self.gaussian_colors.extend(gaussians.colors) 
        
        # save indices
        self.gs_rigid_indices.extend([self.gs_idx + idx for idx in range(n_gaussians)])
        self.gs_rigid_body_ids.extend([self.body_count] * n_gaussians) 
        self.gs_rigid_body_local_ids.extend([self.gs_rigid_body_idx] * n_gaussians) 
        self.gs_body_ids.extend([self.body_count] * n_gaussians) 
        self.body_in_track_idx_list.append(self.body_count) # add body id to the bodies in track idx list
        self.body_in_track_idx_local_list.append(self.gs_rigid_body_idx) # add local body id to the bodies in track idx list

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

        # ----------------------------- add rigid body to the PBD model ------------------------------- #
        # load particles
        particles = obj.particles
        # load rigid body and collision shapes to PBD
        obj_pos[2] += dz + self.ground_h # lift the object above the ground
        obj_quat = convert_quat(obj_quat, to="xyzw")
        m, I_m, _ = mass_inertia_spheres(particles.means, particles.radii, density_factor) # compute mass and inertia
        logging.info(f"{name} mass: {m}")
        b = self.add_body(
            origin=wp.transform(obj_pos.cpu().numpy().tolist(), obj_quat.cpu().numpy().tolist()), # object pose
            com=wp.vec3(0.0, 0.0, 0.0), # center of mass 
            I_m=wp.mat33(I_m.tolist()), # inertia
            m=m, # mass
            is_robot=False, 
        )
        for pt_idx in range(len(particles.means)): # type: ignore
            self.add_shape_sphere(
                body=b,
                pos=particles.means[pt_idx],
                # radius=particles.radii[pt_idx],
                radius=obj_radii,
                density=0.0, # set density to 0.0 to avoid double counting
                mu=mu,
            )
        
        # update index
        self.gs_idx += n_gaussians 
        self.gs_rigid_body_idx += 1
        
    
    def build_gaussian_model(self, device: str = "cuda"):
        gaussian_model = GaussianUnifiedModel(
            means=torch.tensor(self.gaussian_means, device=device, dtype=torch.float32),
            quats=torch.tensor(self.gaussian_quats, device=device, dtype=torch.float32),
            scales=torch.tensor(self.gaussian_scales, device=device, dtype=torch.float32),
            opacities=torch.tensor(self.gaussian_opacities, device=device, dtype=torch.float32),
            colors=torch.tensor(self.gaussian_colors, device=device, dtype=torch.float32),
            
            # Gaussian indices
            robot_indices=torch.tensor(self.gs_robot_indices, device=device, dtype=torch.int32),
            rigid_indices=torch.tensor(self.gs_rigid_indices, device=device, dtype=torch.int32),
            rope_indices=torch.empty(0, device=device, dtype=torch.int32),  # Empty for no rope
            
            # Global body/part ids
            body_ids=torch.tensor(self.gs_body_ids, device=device, dtype=torch.int32),
            robot_body_ids=torch.tensor(self.gs_robot_body_ids, device=device, dtype=torch.int32),
            rigid_body_ids=torch.tensor(self.gs_rigid_body_ids, device=device, dtype=torch.int32),
            rope_part_ids=torch.empty(0, device=device, dtype=torch.int32),  # Empty for no rope
            rope_quat_ids=torch.empty(0, device=device, dtype=torch.int32),  # Empty for no rope
            
            # Local body/part ids
            robot_body_local_ids=torch.tensor(self.gs_robot_body_local_ids, device=device, dtype=torch.int32),
            rigid_body_local_ids=torch.tensor(self.gs_rigid_body_local_ids, device=device, dtype=torch.int32),
            # rope_part_local_ids=torch.empty(0, device=device, dtype=torch.int32),  # Empty for no rope
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
        
        # set visual force gains
        self.kp_gains = torch.tensor([cfg.gain_1, cfg.gain_2], device=self.device, dtype=torch.float32)
        
        return model
