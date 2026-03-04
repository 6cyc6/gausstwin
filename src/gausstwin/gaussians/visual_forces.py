import torch
import warp as wp

from typing import Tuple
from gausstwin.gaussians.adam import Adam
from gausstwin.utils.math_utils import quat_mul, matrix_from_quat, quat_rotate
from gausstwin.gaussians.gaussians import GaussianState, GaussianUnifiedModel


# ========================== Rigid Body Visual Forces ========================== #
class VisualForcesRigidBody:
    def __init__(
        self,
        gaussian_model: GaussianUnifiedModel,
        gaussian_state: GaussianState,
        bodies_in_track: list[int],
        mode: int = 2, # 0: GS means + quats, 1: GS means + quats + object position, 2: object position + orientation, 3: object position + orientation + GS means + quats
    ):
        '''
        num_gaussians 
        body_ids:                    tensor(n_gaussians, ) which body does the gaussian belong to (warp) [0 0 0 1 1 1 ... 11 11 12 12]
        _gaussians_not_tracked_mask  tensor(n_gaussians, ) part of gaussians that do not generate visual force (robot bodies, background)
        _gaussians_in_track_mask     tensor(n_gaussians, ) part of gaussians that contribute to visual force (objects in track)
        ================================
        num_gaussians_in_track
        gs_obj_body_ids         tensor(num_gaussians_in_track, ) which body does the gaussian belong to (global) [11 11 12 12]
        gs_obj_body_local_ids   tensor(num_gaussians_in_track, ) which body does the gaussian belong to (local) [0 0 1 1]
        ================================
        num_obj_in_track
        bodies_in_track:      tensor(num_obj_in_track, ) list of body_idices (warp) in track [11, 12]
        ================================
        '''
        device = gaussian_model.means.device
        self.device = device
        self.mode = mode
        self.gaussian_state = gaussian_state
        
        num_gaussians = gaussian_model.means.shape[0] # n_gaussians
        num_obj_in_track = len(bodies_in_track) # num_obj_in_track
        self.num_obj_in_track = num_obj_in_track
        # ------------------ save initial means and quaternions
        self.init_means = gaussian_model.means.clone().to(device)  # (n_gaussians, 3)
        self.init_quats = gaussian_model.quats.clone().to(device)  # (n_gaussians, 4)
        # ------------------ for optimization
        # means and quaternions of Gaussians
        self.means = torch.zeros((num_gaussians, 3), dtype=torch.float32, device=device) # (n_gaussians, 3)
        self.quats = torch.zeros((num_gaussians, 4), dtype=torch.float32, device=device) # (n_gaussians, 4) wxyz
        # individual displacement of Gaussians
        self.disp_pts = torch.zeros((num_gaussians, 3), dtype=torch.float32, device=device) # (n_gaussians, 3)
        # coherent transformation of Gaussians (w.r.t. the object)
        self.displacement = torch.zeros((num_obj_in_track, 3), dtype=torch.float32, device=device) # (num_obj_in_track, 3)
        self.rotation = torch.zeros((num_obj_in_track, 4), dtype=torch.float32, device=device)     # (num_obj_in_track, 4) wxyz
        self.rotation[:, 0] = 1.0  
        
        # initialize training parameters
        self._init_parameters()

        # get required indices
        # robot
        self.robot_indices = gaussian_model.robot_indices   # (num_robot_gaussians, ) indices of robot gaussians
        self.robot_body_ids = gaussian_model.robot_body_ids # (num_robot_gaussians, ) which rigid body does each Gaussian belong to
        # object
        self.obj_indices = gaussian_model.rigid_indices               # (num_obj_gaussians, ) indices of object gaussians
        self.obj_body_ids = gaussian_model.rigid_body_ids             # (num_obj_gaussians, ) which body does each Gaussian belong to
        self.obj_body_local_ids = gaussian_model.rigid_body_local_ids # (num_obj_gaussians, ) which local body does each Gaussian belong to
        # get masks
        self._gaussians_in_track_mask = torch.zeros(num_gaussians, dtype=torch.bool, device=device)    # (n_gaussians, ) mask for gaussians in track
        self._gaussians_not_tracked_mask = torch.zeros(num_gaussians, dtype=torch.bool, device=device) # (n_gaussians, ) mask for gaussians not in track
        self._gaussians_in_track_mask[self.obj_indices] = True
        self._gaussians_not_tracked_mask[self.robot_indices] = True
        # save tracked means and quaternions
        self.tracked_means = self.init_means[self._gaussians_in_track_mask] # (num_obj_in_track, 3)             
        self.tracked_quats = self.init_quats[self._gaussians_in_track_mask] # (num_obj_in_track, 4)
        self.tracked_translations = torch.zeros((self.num_obj_in_track, 3), dtype=torch.float32, device=device)  # (num_obj_in_track, 3)
        # save labels for mask rendering: 0 for object, 1 for background
        self.labels = torch.ones((num_gaussians, 1), dtype=torch.float32, device=device)
        self.labels[self._gaussians_in_track_mask] = 0.0
        # ------------------ forces and moments
        self.forces_gaussians = torch.zeros((num_gaussians, 3), dtype=torch.float32, device=device)     # (n_gaussians, 3)
        self.moments_gaussians = torch.zeros((num_gaussians, 3), dtype=torch.float32, device=device)    # (n_gaussians, 3)
        self.forces_bodies = torch.zeros((self.num_obj_in_track, 3), dtype=torch.float32, device=device)  # (num_obj_in_track, 3)
        self.moments_bodies = torch.zeros((self.num_obj_in_track, 3), dtype=torch.float32, device=device) # (num_obj_in_track, 3)


    def _init_parameters(self):
        # determine and initialize training parameters
        if self.mode == 0:
            self.means.requires_grad = True
            self.quats.requires_grad = True
            # optimizer
            self.optimizer = Adam(
                [                
                    wp.from_torch(self.means, dtype=wp.vec3),
                    wp.from_torch(self.quats, dtype=wp.vec4),
                ],
                lrs=[0.005, 0.005], # type: ignore
            )
            self.lrs = [0.005, 0.005]
        elif self.mode == 1:
            self.means.requires_grad = True
            self.quats.requires_grad = True
            self.displacement.requires_grad = True
            # optimizer
            self.optimizer = Adam(
                [                
                    wp.from_torch(self.means, dtype=wp.vec3),
                    wp.from_torch(self.quats, dtype=wp.vec4),
                    wp.from_torch(self.displacement, dtype=wp.vec3),
                ],
                lrs=[0.005, 0.005, 0.005], # type: ignore
            )
            self.lrs = [0.005, 0.005, 0.005]
        elif self.mode == 2:
            self.displacement.requires_grad = True
            self.rotation.requires_grad = True
            # optimizer
            self.optimizer = Adam(
                [                
                    wp.from_torch(self.displacement, dtype=wp.vec3),
                    wp.from_torch(self.rotation, dtype=wp.vec4),
                ],
                lrs=[0.005, 0.005], # type: ignore
            )
            self.lrs = [0.005, 0.005]
        elif self.mode == 3:
            self.means.requires_grad = True
            self.quats.requires_grad = True
            self.displacement.requires_grad = True
            self.rotation.requires_grad = True
            # optimizer
            self.optimizer = Adam(
                [                
                    wp.from_torch(self.means, dtype=wp.vec3),
                    wp.from_torch(self.quats, dtype=wp.vec4),
                    wp.from_torch(self.displacement, dtype=wp.vec3),
                    wp.from_torch(self.rotation, dtype=wp.vec4),
                ],
                lrs=[0.005, 0.005, 0.005, 0.005], # type: ignore
            )
            self.lrs = [0.005, 0.005, 0.005, 0.005]
        else:
            self.disp_pts.requires_grad = True
            self.displacement.requires_grad = True
            self.rotation.requires_grad = True
            # optimizer
            self.optimizer = Adam(
                [                
                    wp.from_torch(self.disp_pts, dtype=wp.vec3),
                    wp.from_torch(self.displacement, dtype=wp.vec3),
                    wp.from_torch(self.rotation, dtype=wp.vec4),
                ],
                lrs=[0.005, 0.005, 0.005], # type: ignore
            )
            self.lrs = [0.005, 0.005, 0.005]
        
        
    def set_learnings_rates(self, lrs):
        if self.mode == 0:
            self.optimizer.lrs = lrs[:2]
            self.lrs = lrs[:2]
        elif self.mode == 1:
            self.optimizer.lrs = lrs[:3]
            self.lrs = lrs[:3]
        elif self.mode == 2:
            self.optimizer.lrs = lrs[2:4]
            self.lrs = lrs[2:4]
        elif self.mode == 3:
            self.optimizer.lrs = lrs
            self.lrs = lrs
        else:
            self.optimizer.lrs = [lrs[0], lrs[2], lrs[3]] # type: ignore
            self.lrs = [lrs[0], lrs[2], lrs[3]] # type: ignore
    
    
    def reset_learning_rates(self):
        self.optimizer.lrs[:] = self.lrs # type: ignore


    def zero_grad(self):
        if self.mode == 0:
            self.means.grad.zero_() 
            self.quats.grad.zero_() 
        elif self.mode == 1:
            self.means.grad.zero_()
            self.quats.grad.zero_() 
            self.displacement.grad.zero_()
        elif self.mode == 2:
            self.displacement.grad.zero_() 
            self.rotation.grad.zero_()
        elif self.mode == 3:
            self.means.grad.zero_() 
            self.quats.grad.zero_() 
            self.displacement.grad.zero_() 
            self.rotation.grad.zero_() 
        else:
            self.disp_pts.grad.zero_() 
            self.displacement.grad.zero_() 
            self.rotation.grad.zero_() 
        

    def step(self):
        if self.mode == 0:
            self.means.grad[self._gaussians_not_tracked_mask, :] = 0 
            self.quats.grad[self._gaussians_not_tracked_mask, :] = 0 
            self.optimizer.step(
                grad=[
                    wp.from_torch(self.means.grad, dtype=wp.vec3),
                    wp.from_torch(self.quats.grad, dtype=wp.vec4),
                ]
            )
        elif self.mode == 1:
            self.means.grad[self._gaussians_not_tracked_mask, :] = 0 
            self.quats.grad[self._gaussians_not_tracked_mask, :] = 0 
            self.optimizer.step(
                grad=[
                    wp.from_torch(self.means.grad, dtype=wp.vec3),
                    wp.from_torch(self.quats.grad, dtype=wp.vec4),
                    wp.from_torch(self.displacement.grad, dtype=wp.vec3),
                ]
            )
        elif self.mode == 2:
            self.optimizer.step(
                grad=[
                    wp.from_torch(self.displacement.grad, dtype=wp.vec3),
                    wp.from_torch(self.rotation.grad, dtype=wp.vec4),
                ]
            )
        elif self.mode == 3:
            self.means.grad[self._gaussians_not_tracked_mask, :] = 0 
            self.quats.grad[self._gaussians_not_tracked_mask, :] = 0 
            self.optimizer.step(
                grad=[
                    wp.from_torch(self.means.grad, dtype=wp.vec3),
                    wp.from_torch(self.quats.grad, dtype=wp.vec4),
                    wp.from_torch(self.displacement.grad, dtype=wp.vec3),
                    wp.from_torch(self.rotation.grad, dtype=wp.vec4),
                ]
            )
        else:
            self.disp_pts.grad[self._gaussians_not_tracked_mask, :] = 0 
            self.optimizer.step(
                grad=[
                    wp.from_torch(self.disp_pts.grad, dtype=wp.vec3),
                    wp.from_torch(self.displacement.grad, dtype=wp.vec3),
                    wp.from_torch(self.rotation.grad, dtype=wp.vec4),
                ]
            )
        
    
    def apply_transformation(self):
        # get rotation and translation of the objects
        obj_rotations = self.rotation # (num_obj_in_track, 4)
        obj_displacements = self.displacement # (num_obj_in_track, 3)     
        
        # get the guassians for tracking (of the objects)
        means = torch.clone(self.means) # (n, 3)
        quats = torch.clone(self.quats) # (n, 4)
        
        # get the guassians for tracking (of the objects)
        tracked_means = means[self._gaussians_in_track_mask] # (n_track, 3)             
        tracked_quats = quats[self._gaussians_in_track_mask] # (n_track, 4)

        # center the gaussian means of the objects
        tracked_means -= self.tracked_translations[self.obj_body_local_ids]
        
        # apply transformation to the gaussian means
        # ---------------- quat -> rotation matrix -> apply transformation
        # obj_rot_mat = matrix_from_quat(obj_rotations) # (n_obj, 3, 3)
        # R = obj_rot_mat[self._tracked_gaussians_opt_body_ids] # (n_track, 3, 3)
        # # rotated_means = torch.bmm(R, tracked_means.unsqueeze(-1)).squeeze(-1)  # (n_track, 3)
        # rotated_means = compute_transformed_means(R, tracked_means)  # (n_track, 3)
        # trans_means = rotated_means + obj_displacements[self._tracked_gaussians_opt_body_ids] + self.tracked_translations[self._tracked_gaussians_opt_body_ids]
        
        # ---------------- directly using quaternion to transform the vector
        obj_quat = obj_rotations[self.obj_body_local_ids] 
        rotated_means = quat_rotate(obj_quat, tracked_means)
        trans_means = rotated_means + obj_displacements[self.obj_body_local_ids] + self.tracked_translations[self.obj_body_local_ids]
        
        # apply transformation to the gaussian quaternions
        trans_quats = quat_mul(obj_rotations[self.obj_body_local_ids], tracked_quats) # (n_track, 4)   
        # trans_quats = normalize(trans_quats) # (n_track, 4) # this will be done while rendering
        
        # put transfromed means and quaternions back to the original means and quaternions
        means[self._gaussians_in_track_mask] = trans_means
        quats[self._gaussians_in_track_mask] = trans_quats
        
        if self.mode == 4:
            means += self.disp_pts # add displacement if mode is 4
        
        return means, quats
    
    
    def get_target_positions(self):
        with torch.no_grad():
            means = self.means.clone() # (num_gaussians, 3)
                
            # get rotation and translation of the objects
            obj_rotations = self.rotation # (num_obj_in_track, 4)
            obj_displacements = self.displacement # (num_obj_in_track, 3)     
            
            # get the guassians for tracking (of the objects)
            tracked_means = means[self._gaussians_in_track_mask] # (num_obj_in_track, 3)             
            tracked_means = tracked_means - self.tracked_translations[self.obj_body_local_ids]

            # apply transformation to the gaussian means
            obj_rot_mat = matrix_from_quat(obj_rotations) # (num_obj_in_track, 3, 3)
            R = obj_rot_mat[self.obj_body_local_ids] # (num_obj_in_track, 3, 3)
            rotated_means = torch.bmm(R, tracked_means.unsqueeze(-1)).squeeze(-1)  # (num_obj_in_track, 3)
            trans_means = rotated_means + obj_displacements[self.obj_body_local_ids] + self.tracked_translations[self.obj_body_local_ids]
            
            means[self._gaussians_in_track_mask] = trans_means
            
            if self.mode == 4:
                means += self.disp_pts
        
        return means
    

# ========================== Rope Visual Forces ========================== #
class VisualForcesRope:
    def __init__(
        self,
        gaussian_model: GaussianUnifiedModel,
        gaussian_state: GaussianState,
        parts_in_track: list[int],
        mode: int = 4, # 0: GS means + quats, 1: GS means + quats + object position, 2: object position + orientation, 3: object position + orientation + GS means + quats
    ):
        '''
        num_gaussians 
        _gaussians_not_tracked_mask  tensor(n_gaussians, ) part of gaussians that do not generate visual force (robot bodies, background)
        _gaussians_in_track_mask     tensor(n_gaussians, ) part of gaussians that contribute to visual force (objects in track)
        ================================
        num_robot_gaussians
        gs_robot_indices:     tensor(num_robot_gaussians, ) part of gaussians that belong to the robot
        ================================
        num_rope_parts_in_track
        gs_rope_indices:      tensor(num_rope_parts_in_track, ) part of gaussians that belong to the rope
        ================================
        '''
        device = gaussian_model.means.device
        self.device = device
        self.mode = mode
        self.gaussian_state = gaussian_state
        
        num_gaussians = gaussian_model.means.shape[0] # n_gaussians
        num_rope_parts_in_track = len(parts_in_track) # num_rope_parts_in_track
        self.num_rope_parts_in_track = num_rope_parts_in_track
        # ------------------ save initial means and quaternions
        self.init_means = gaussian_model.means.clone().to(device)  # (n_gaussians, 3)
        self.init_quats = gaussian_model.quats.clone().to(device)  # (n_gaussians, 4)
        # ------------------ for optimization
        self.means = torch.zeros((num_gaussians, 3), dtype=torch.float32, device=device) # (n_gaussians, 3)
        self.quats = torch.zeros((num_gaussians, 4), dtype=torch.float32, device=device) # (n_gaussians, 4) wxyz
        # individual displacement of Gaussians
        self.disp_pts = torch.zeros((num_gaussians, 3), dtype=torch.float32, device=device) # (n_gaussians, 3)
        # coherent transformation of Gaussians (with the rope segments)
        self.displacement = torch.zeros((num_rope_parts_in_track, 3), dtype=torch.float32, device=device) # (num_rope_parts_in_track, 3)
        self.rotation = torch.zeros((num_rope_parts_in_track, 4), dtype=torch.float32, device=device) # (num_rope_parts_in_track, 4) wxyz
        self.rotation[:, 0] = 1.0  
        
        # initialize training parameters
        self._init_parameters()

        # get required indices
        # robot
        self.robot_indices = gaussian_model.robot_indices   # (num_robot_gaussians, ) indices of robot gaussians
        self.robot_body_ids = gaussian_model.robot_body_ids # (num_robot_gaussians, ) which rigid body does each Gaussian belong to
        # object (rope)
        self.rope_indices = gaussian_model.rope_indices               # (num_rope_gaussians, ) indices of rope gaussians
        self.rope_part_ids = gaussian_model.rope_part_ids             # (num_rope_gaussians, ) which rope part does each Gaussian belong to
        self.rope_quat_ids = gaussian_model.rope_quat_ids             # (num_rope_gaussians, ) which rope part does each Gaussian belong to
        # get masks
        self._gaussians_in_track_mask = torch.zeros(num_gaussians, dtype=torch.bool, device=device) # (n_gaussians, ) mask for gaussians in track
        self._gaussians_not_tracked_mask = torch.zeros(num_gaussians, dtype=torch.bool, device=device) # (n_gaussians, ) mask for gaussians not in track
        self._gaussians_in_track_mask[self.rope_indices] = True
        self._gaussians_not_tracked_mask[self.robot_indices] = True
        # save tracked means and quaternions
        self.tracked_means = self.init_means[self._gaussians_in_track_mask] # (n_track, 3)             
        self.tracked_quats = self.init_quats[self._gaussians_in_track_mask] # (n_track, 4)
        self.tracked_translations = torch.zeros((self.num_rope_parts_in_track, 3), dtype=torch.float32, device=device)  # (num_rope_parts_in_track, 3)
        # save labels for mask rendering: 0 for object, 1 for background
        self.labels = torch.ones((num_gaussians, 1), dtype=torch.float32, device=device)
        self.labels[self._gaussians_in_track_mask] = 0.0
        # ------------------ visual forces
        # self.forces_particles = torch.zeros((self.num_rope_parts_in_track, 3), dtype=torch.float32, device=device) # (num_rope_parts_in_track, 3)
        self.forces_particles = torch.zeros((self.num_rope_parts_in_track, 3), dtype=torch.float32, device=device) # (num_rope_parts_in_track, 3)
        self.forces_particles_wp = wp.from_torch(self.forces_particles, dtype=wp.vec3) # warp array for visual forces on particles

    
    def _init_parameters(self):
        # determine and initialize training parameters
        if self.mode == 0:
            self.means.requires_grad = True
            self.quats.requires_grad = True
            # optimizer
            self.optimizer = Adam(
                [                
                    wp.from_torch(self.means, dtype=wp.vec3),
                    wp.from_torch(self.quats, dtype=wp.vec4),
                ],
                lrs=[0.005, 0.005], # type: ignore
            )
            self.lrs = [0.005, 0.005]
        elif self.mode == 1:
            self.means.requires_grad = True
            self.quats.requires_grad = True
            self.displacement.requires_grad = True
            # optimizer
            self.optimizer = Adam(
                [                
                    wp.from_torch(self.means, dtype=wp.vec3),
                    wp.from_torch(self.quats, dtype=wp.vec4),
                    wp.from_torch(self.displacement, dtype=wp.vec3),
                ],
                lrs=[0.005, 0.005, 0.005], # type: ignore
            )
            self.lrs = [0.005, 0.005, 0.005]
        elif self.mode == 2:
            self.displacement.requires_grad = True
            self.rotation.requires_grad = True
            # optimizer
            self.optimizer = Adam(
                [                
                    wp.from_torch(self.displacement, dtype=wp.vec3),
                    wp.from_torch(self.rotation, dtype=wp.vec4),
                ],
                lrs=[0.005, 0.005], # type: ignore
            )
            self.lrs = [0.005, 0.005]
        elif self.mode == 3:
            self.means.requires_grad = True
            self.quats.requires_grad = True
            self.displacement.requires_grad = True
            self.rotation.requires_grad = True
            # optimizer
            self.optimizer = Adam(
                [                
                    wp.from_torch(self.means, dtype=wp.vec3),
                    wp.from_torch(self.quats, dtype=wp.vec4),
                    wp.from_torch(self.displacement, dtype=wp.vec3),
                    wp.from_torch(self.rotation, dtype=wp.vec4),
                ],
                lrs=[0.005, 0.005, 0.005, 0.005], # type: ignore
            )
            self.lrs = [0.005, 0.005, 0.005, 0.005]
        else:
            self.disp_pts.requires_grad = True
            self.displacement.requires_grad = True
            self.rotation.requires_grad = True
            # optimizer
            self.optimizer = Adam(
                [                
                    wp.from_torch(self.disp_pts, dtype=wp.vec3),
                    wp.from_torch(self.displacement, dtype=wp.vec3),
                    wp.from_torch(self.rotation, dtype=wp.vec4),
                ],
                lrs=[0.005, 0.005, 0.005], # type: ignore
            )
            self.lrs = [0.005, 0.005, 0.005]
            

    def set_learnings_rates(self, lrs):
        if self.mode == 0:
            self.optimizer.lrs = lrs[:2]
            self.lrs = lrs[:2]
        elif self.mode == 1:
            self.optimizer.lrs = lrs[:3]
            self.lrs = lrs[:3]
        elif self.mode == 2:
            self.optimizer.lrs = lrs[2:4]
            self.lrs = lrs[2:4]
        elif self.mode == 3:
            self.optimizer.lrs = lrs
            self.lrs = lrs
        else:
            self.optimizer.lrs = [lrs[0], lrs[2], lrs[3]] # type: ignore
            self.lrs = [lrs[0], lrs[2], lrs[3]] # type: ignore
    
    
    def reset_learning_rates(self):
        self.optimizer.lrs[:] = self.lrs # type: ignore


    def zero_grad(self):
        if self.mode == 0:
            self.means.grad.zero_() 
            self.quats.grad.zero_() 
        elif self.mode == 1:
            self.means.grad.zero_()
            self.quats.grad.zero_() 
            self.displacement.grad.zero_()
        elif self.mode == 2:
            self.displacement.grad.zero_() 
            self.rotation.grad.zero_()
        elif self.mode == 3:
            self.means.grad.zero_() 
            self.quats.grad.zero_() 
            self.displacement.grad.zero_() 
            self.rotation.grad.zero_() 
        else:
            self.disp_pts.grad.zero_() 
            self.displacement.grad.zero_() 
            self.rotation.grad.zero_() 
        

    def step(self):
        if self.mode == 0:
            self.means.grad[self._gaussians_not_tracked_mask, :] = 0 
            self.quats.grad[self._gaussians_not_tracked_mask, :] = 0 
            self.optimizer.step(
                grad=[
                    wp.from_torch(self.means.grad, dtype=wp.vec3),
                    wp.from_torch(self.quats.grad, dtype=wp.vec4),
                ]
            )
        elif self.mode == 1:
            self.means.grad[self._gaussians_not_tracked_mask, :] = 0 
            self.quats.grad[self._gaussians_not_tracked_mask, :] = 0 
            self.optimizer.step(
                grad=[
                    wp.from_torch(self.means.grad, dtype=wp.vec3),
                    wp.from_torch(self.quats.grad, dtype=wp.vec4),
                    wp.from_torch(self.displacement.grad, dtype=wp.vec3),
                ]
            )
        elif self.mode == 2:
            self.optimizer.step(
                grad=[
                    wp.from_torch(self.displacement.grad, dtype=wp.vec3),
                    wp.from_torch(self.rotation.grad, dtype=wp.vec4),
                ]
            )
        elif self.mode == 3:
            self.means.grad[self._gaussians_not_tracked_mask, :] = 0 
            self.quats.grad[self._gaussians_not_tracked_mask, :] = 0 
            self.optimizer.step(
                grad=[
                    wp.from_torch(self.means.grad, dtype=wp.vec3),
                    wp.from_torch(self.quats.grad, dtype=wp.vec4),
                    wp.from_torch(self.displacement.grad, dtype=wp.vec3),
                    wp.from_torch(self.rotation.grad, dtype=wp.vec4),
                ]
            )
        else:
            self.disp_pts.grad[self._gaussians_not_tracked_mask, :] = 0 
            self.optimizer.step(
                grad=[
                    wp.from_torch(self.disp_pts.grad, dtype=wp.vec3),
                    wp.from_torch(self.displacement.grad, dtype=wp.vec3),
                    wp.from_torch(self.rotation.grad, dtype=wp.vec4),
                ]
            )
    
    def step_new(self):
        self.disp_pts.grad[self._gaussians_not_tracked_mask, :] = 0
        self.optimizer.step(
            grad=[
                wp.from_torch(self.disp_pts.grad, dtype=wp.vec3),
                wp.from_torch(self.displacement.grad, dtype=wp.vec3),
                wp.from_torch(self.rotation.grad, dtype=wp.vec4),
            ]
        )
        
    
    def apply_transformation(self):
        # get rotation and translation of the objects
        obj_rotations = self.rotation # (n_obj, 4)
        obj_displacements = self.displacement # (n_obj, 3)     
        
        # get the guassians for tracking (of the objects)
        means = torch.clone(self.means) # (n, 3)
        quats = torch.clone(self.quats) # (n, 4)
        
        # get the guassians for tracking (of the objects)
        tracked_means = means[self._gaussians_in_track_mask] # (n_track, 3)             
        tracked_quats = quats[self._gaussians_in_track_mask] # (n_track, 4)

        # center the gaussian means of the objects
        # tracked_means = tracked_means - self.tracked_translations[self._tracked_gaussians_opt_body_ids]
        tracked_means -= self.tracked_translations[self.rope_part_ids]
        
        # apply transformation to the gaussian means
        # ---------------- quat -> rotation matrix -> apply transformation
        # obj_rot_mat = matrix_from_quat(obj_rotations) # (n_obj, 3, 3)
        # R = obj_rot_mat[self._tracked_gaussians_opt_body_ids] # (n_track, 3, 3)
        # # rotated_means = torch.bmm(R, tracked_means.unsqueeze(-1)).squeeze(-1)  # (n_track, 3)
        # rotated_means = compute_transformed_means(R, tracked_means)  # (n_track, 3)
        # trans_means = rotated_means + obj_displacements[self._tracked_gaussians_opt_body_ids] + self.tracked_translations[self._tracked_gaussians_opt_body_ids]
        
        # ---------------- directly using quaternion to transform the vector
        obj_quat = obj_rotations[self.rope_part_ids] 
        rotated_means = quat_rotate(obj_quat, tracked_means)
        trans_means = rotated_means + obj_displacements[self.rope_part_ids] + self.tracked_translations[self.rope_part_ids]
        
        # apply transformation to the gaussian quaternions
        trans_quats = quat_mul(obj_rotations[self.rope_quat_ids], tracked_quats) # (n_track, 4)   
        # trans_quats = normalize(trans_quats) # (n_track, 4) # this will be done while rendering
        
        # put transfromed means and quaternions back to the original means and quaternions
        means[self._gaussians_in_track_mask] = trans_means
        quats[self._gaussians_in_track_mask] = trans_quats
        
        if self.mode == 4:
            # means = means + self.disp_pts # add displacement if mode is 4
            means += self.disp_pts # add displacement if mode is 4
        
        return means, quats
    
    
    def get_target_positions(self):
        with torch.no_grad():
            means = self.means.clone() # (n, 3)
                
            # get rotation and translation of the objects
            obj_rotations = self.rotation # (n_obj, 4)
            obj_displacements = self.displacement # (n_obj, 3)     
            
            # get the guassians for tracking (of the objects)
            tracked_means = means[self._gaussians_in_track_mask] # (n_track, 3)             
            tracked_means = tracked_means - self.tracked_translations[self.rope_part_ids]

            # apply transformation to the gaussian means
            obj_rot_mat = matrix_from_quat(obj_rotations) # (n_obj, 3, 3)
            R = obj_rot_mat[self.rope_quat_ids] # (n_track, 3, 3)
            rotated_means = torch.bmm(R, tracked_means.unsqueeze(-1)).squeeze(-1)  # (n_track, 3)
            trans_means = rotated_means + obj_displacements[self.rope_part_ids] + self.tracked_translations[self.rope_part_ids]

            means[self._gaussians_in_track_mask] = trans_means
            
            if self.mode == 4:
                means += self.disp_pts
        
        return means
    

# ================================ differentiable batch transformation ================================ # 
@wp.kernel
def apply_transform_kernel(
    means_in: wp.array(dtype=wp.vec3f),          # GS means                       # type: ignore
    quats_in: wp.array(dtype=wp.vec4f),          # GS quats                       # type: ignore
    obj_rotations: wp.array(dtype=wp.quatf),     # (opt) Object rotation          # type: ignore
    obj_displacements: wp.array(dtype=wp.vec3f), # (opt) Object displacement      # type: ignore
    disp_pts: wp.array(dtype=wp.vec3f),          # (opt) Point-wise displacement  # type: ignore
    obj_translations: wp.array(dtype=wp.vec3f),  # Current object position        # type: ignore
    gs_obj_indices: wp.array(dtype=wp.int32),    # object gs indices              # type: ignore
    gs_obj_body_ids: wp.array(dtype=wp.int32),   # local obj body ids for obj gs  # type: ignore
    means_out: wp.array(dtype=wp.vec3f),         # GS out means                   # type: ignore
    quats_out: wp.array(dtype=wp.vec4f)          # GS out quats                   # type: ignore
):
    tid = wp.tid()
    gsid = gs_obj_indices[tid]
    bid = gs_obj_body_ids[tid]

    # get gaussian means and quaternions
    means = means_in[gsid]
    q = quats_in[gsid]
    quats = wp.quatf(q[1], q[2], q[3], q[0]) # wxyz -> xyzw
    # get object info
    trans = obj_translations[bid]
    disp = obj_displacements[bid]
    rot = obj_rotations[bid]
    quat_rot = wp.quatf(rot[1], rot[2], rot[3], rot[0]) # wxyz -> xyzw
    # quat_rot = wp.normalize(quat_rot)
    # center the points (world -> local)
    mean_centered = means - trans
    # apply rotation 
    mean_rot = wp.quat_rotate(quat_rot, mean_centered)
    # apply coherent displacement, translate back (local -> world)
    mean_rotated = mean_rot + disp + trans
    # apply individual displacement
    mean_final = mean_rotated + disp_pts[gsid]

    # apply rotation on quaternions
    quat_rotated = quat_rot * quats
    quat_norm = wp.normalize(quat_rotated)
    quat_final = wp.vec4f(quat_norm[3], quat_norm[0], quat_norm[1], quat_norm[2]) # xyzw -> wxyz 
    
    means_out[gsid] = mean_final
    quats_out[gsid] = quat_final


# ------------ forward, register
@torch.library.custom_op("wp::warp_batch_transform", mutates_args=())
def warp_batch_transform(
    means: torch.Tensor, 
    quats: torch.Tensor, 
    tracked_translations: torch.Tensor, 
    obj_rotations: torch.Tensor, 
    obj_displacements: torch.Tensor, 
    disp_pts: torch.Tensor, 
    gs_in_track_indices: torch.Tensor, 
    gs_in_track_body_indices: torch.Tensor, 
    n_gs_track: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    wp_mean = wp.from_torch(means, dtype=wp.vec3f, requires_grad=False)
    wp_quats = wp.from_torch(quats, dtype=wp.vec4f, requires_grad=False)
    wp_tracked_translations = wp.from_torch(tracked_translations, dtype=wp.vec3f, requires_grad=False)
    wp_obj_rotations = wp.from_torch(obj_rotations, dtype=wp.quatf, requires_grad=True)
    wp_obj_displacements = wp.from_torch(obj_displacements, dtype=wp.vec3f, requires_grad=True)
    wp_disp_pts = wp.from_torch(disp_pts, dtype=wp.vec3f, requires_grad=True)

    wp_gs_in_track_indices = wp.from_torch(gs_in_track_indices, dtype=wp.int32, requires_grad=False)
    wp_gs_in_track_body_indices = wp.from_torch(gs_in_track_body_indices, dtype=wp.int32, requires_grad=False)

    # wp_trans_means = wp.zeros_like(wp_mean, requires_grad=False)
    # wp_trans_quats = wp.zeros_like(wp_quats, requires_grad=False)
    wp_trans_means = wp.clone(wp_mean, requires_grad=False)
    wp_trans_quats = wp.clone(wp_quats, requires_grad=False)

    wp.launch(
        kernel=apply_transform_kernel, 
        dim=n_gs_track, 
        inputs=[
            wp_mean,
            wp_quats,
            wp_obj_rotations,
            wp_obj_displacements,
            wp_disp_pts,
            wp_tracked_translations,
            # gs_in_track_indices,
            # gs_in_track_body_indices,
            wp_gs_in_track_indices,
            wp_gs_in_track_body_indices
        ], 
        outputs=[wp_trans_means, wp_trans_quats]
    )
    
    # wp_mean = wp.from_torch(means, dtype=wp.vec3f)
    # wp_quats = wp.from_torch(quats, dtype=wp.vec4f)
    # wp_trans_means = wp.zeros_like(wp_mean)
    # wp_trans_quats = wp.zeros_like(wp_quats)

    # wp.launch(
    #     kernel=apply_transform_kernel, 
    #     dim=n_gs_track, 
    #     inputs=[
    #         wp_mean,
    #         wp_quats,
    #         obj_rotations,
    #         obj_displacements,
    #         disp_pts,
    #         tracked_translations,
    #         gs_in_track_indices,
    #         gs_in_track_body_indices,
    #     ], 
    #     outputs=[wp_trans_means, wp_trans_quats]
    # )

    return wp.to_torch(wp_trans_means), wp.to_torch(wp_trans_quats)


@warp_batch_transform.register_fake
def _(
    means: torch.Tensor, 
    quats: torch.Tensor, 
    tracked_translations: torch.Tensor, 
    obj_rotations: torch.Tensor, 
    obj_displacements: torch.Tensor, 
    disp_pts: torch.Tensor, 
    gs_in_track_indices: torch.Tensor, 
    gs_in_track_body_indices: torch.Tensor, 
    n_gs_track: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Return empty tensors with the same shape/dtype as inputs
    return (
        torch.empty_like(means),
        torch.empty_like(quats)
    )

# ------------ backward, register
@torch.library.custom_op("wp::warp_batch_transform_backward", mutates_args=())
def warp_batch_transform_backward(
    means: torch.Tensor, 
    quats: torch.Tensor, 
    tracked_translations: torch.Tensor, 
    obj_rotations: torch.Tensor, 
    obj_displacements: torch.Tensor, 
    disp_pts: torch.Tensor, 
    gs_in_track_indices: torch.Tensor, 
    gs_in_track_body_indices: torch.Tensor, 
    n_gs_track: int,
    out_means: torch.Tensor,     
    out_quats: torch.Tensor,
    adj_out_means: torch.Tensor,   
    adj_out_quats: torch.Tensor     
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Convert inputs and upstream grads to Warp arrays
    wp_means = wp.from_torch(means, dtype=wp.vec3f, requires_grad=False)
    wp_quats = wp.from_torch(quats, dtype=wp.vec4f, requires_grad=False)
    wp_tracked_translations = wp.from_torch(tracked_translations, dtype=wp.vec3f, requires_grad=False)

    wp_obj_rotations = wp.from_torch(obj_rotations, dtype=wp.quatf, requires_grad=True)
    wp_obj_displacements = wp.from_torch(obj_displacements, dtype=wp.vec3f, requires_grad=True)
    wp_disp_pts = wp.from_torch(disp_pts, dtype=wp.vec3f, requires_grad=True)

    wp_gs_in_track_indices = wp.from_torch(gs_in_track_indices, dtype=wp.int32, requires_grad=False)
    wp_gs_in_track_body_indices = wp.from_torch(gs_in_track_body_indices, dtype=wp.int32, requires_grad=False)
    gs_in_track_indices.requires_grad = False
    gs_in_track_body_indices.requires_grad = False

    wp_out_means = wp.from_torch(out_means, dtype=wp.vec3f, requires_grad=False)
    wp_out_quats = wp.from_torch(out_quats, dtype=wp.vec4f, requires_grad=False)

    wp_adj_out_means = wp.from_torch(adj_out_means, dtype=wp.vec3f, requires_grad=False)
    wp_adj_out_quats = wp.from_torch(adj_out_quats, dtype=wp.vec4f, requires_grad=False)

    wp.launch(
        kernel=apply_transform_kernel, 
        dim=n_gs_track, 
        inputs=[
            wp_means,
            wp_quats,
            wp_obj_rotations,
            wp_obj_displacements,
            wp_disp_pts,
            wp_tracked_translations,
            # gs_in_track_indices,
            # gs_in_track_body_indices,
            wp_gs_in_track_indices,
            wp_gs_in_track_body_indices
        ], 
        outputs=[wp_out_means, wp_out_quats],
        # adj_inputs=[
        #     wp_obj_rotations.grad,
        #     wp_obj_displacements.grad,
        #     wp_disp_pts.grad
        # ],
        adj_inputs=[
            None,                       # means -> no adjoint expected here
            None,                       # quats -> no adjoint
            wp_obj_rotations.grad,      # obj_rotations -> matches quatf
            wp_obj_displacements.grad,  # obj_displacements -> matches vec3
            wp_disp_pts.grad,           # disp_pts -> matches vec3
            None,                       # tracked_translations
            None,                       # gs_in_track_indices
            None,                       # gs_in_track_body_indices
        ],
        adj_outputs=[wp_adj_out_means, wp_adj_out_quats],
        adjoint=True,
    )
    
    return wp.to_torch(wp_obj_rotations.grad), wp.to_torch(wp_obj_displacements.grad), wp.to_torch(wp_disp_pts.grad)


@warp_batch_transform_backward.register_fake
def _(means, quats, tracked_translations, obj_rotations, obj_displacements, disp_pts, gs_in_track_indices, gs_in_track_body_indices, n_gs_track, out_means, out_quats, adj_out_means, adj_out_quats):
    return torch.empty_like(obj_rotations), torch.empty_like(obj_displacements), torch.empty_like(disp_pts)


# ！===================================== This version would increasingly consume new memory
# def backward(ctx, adj_out_means, adj_out_quats):
#     ctx.obj_rotations.grad, ctx.obj_displacements.grad, ctx.disp_pts.grad  = warp_batch_transform_backward(
#         ctx.means, ctx.quats, ctx.tracked_translations, ctx.obj_rotations, ctx.obj_displacements, ctx.disp_pts,
#         ctx.gs_in_track_indices, ctx.gs_in_track_body_indices, ctx.n_gs_track,
#         ctx.out_means, ctx.out_quats,
#         adj_out_means, adj_out_quats
#     )
#     # return ctx.obj_rotations.grad, ctx.obj_displacements.grad, ctx.disp_pts.grad, None
#     return None, None, None, ctx.obj_rotations.grad, ctx.obj_displacements.grad, ctx.disp_pts.grad, None, None, None


# def setup_context(ctx, inputs, output):
#     ctx.means, ctx.quats, ctx.tracked_translations, ctx.obj_rotations, ctx.obj_displacements, ctx.disp_pts, ctx.gs_in_track_indices, ctx.gs_in_track_body_indices, ctx.n_gs_track = inputs
#     ctx.out_means, ctx.out_quats = output


# ======================================= New version
def backward(ctx, adj_out_means, adj_out_quats):
    (
        means, quats, tracked_translations,
        # obj_rotations, obj_displacements, disp_pts,
        gs_in_track_indices, gs_in_track_body_indices,
        out_means, out_quats
    ) = ctx.saved_tensors

    # Recompute forward outputs without tracking, so we don't save outputs in ctx
    # with torch.no_grad():
    #     out_means, out_quats = warp_batch_transform(
    #         means, quats,
    #         tracked_translations,
    #         obj_rotations, obj_displacements, disp_pts,
    #         gs_in_track_indices, gs_in_track_body_indices,
    #         ctx.n_gs_track
    #     )

    ctx.obj_rotations.grad, ctx.obj_displacements.grad, ctx.disp_pts.grad = warp_batch_transform_backward(
        means, quats, tracked_translations,
        # obj_rotations, obj_displacements, disp_pts,
        ctx.obj_rotations, ctx.obj_displacements, ctx.disp_pts,
        gs_in_track_indices, gs_in_track_body_indices,
        ctx.n_gs_track,
        out_means, out_quats,
        adj_out_means, adj_out_quats
    )

    # Return grads for inputs in order; None for non-differentiable args
    return (
        None,               # means
        None,               # quats
        None,               # tracked_translations
        # grad_rots,          # obj_rotations
        # grad_disps,         # obj_displacements
        # grad_disp_pts,      # disp_pts
        ctx.obj_rotations.grad, 
        ctx.obj_displacements.grad, 
        ctx.disp_pts.grad,
        None,               # gs_in_track_indices
        None,               # gs_in_track_body_indices
        None,               # n_gs_track
    )


def setup_context(ctx, inputs, output):
    (
        means, quats, tracked_translations,
        # obj_rotations, obj_displacements, disp_pts,
        ctx.obj_rotations, ctx.obj_displacements, ctx.disp_pts,
        gs_in_track_indices, gs_in_track_body_indices,
        n_gs_track
    ) = inputs
    out_means, out_quats = output

    # Save only what's needed to recompute forward inside backward,
    # and to feed the adjoint. We avoid saving forward outputs.
    ctx.save_for_backward(
        means, quats, tracked_translations,
        # obj_rotations, obj_displacements, disp_pts,
        gs_in_track_indices, gs_in_track_body_indices,
        out_means, out_quats
    )
    ctx.n_gs_track = int(n_gs_track)

warp_batch_transform.register_autograd(backward, setup_context=setup_context)


# =================================== warp kernel functions =================================== # 
@wp.kernel
def apply_transform_means_kernel(
    means_in: wp.array(dtype=wp.vec3f),     # all means # type: ignore
    obj_rotations: wp.array(dtype=wp.quatf), # type: ignore
    obj_displacements: wp.array(dtype=wp.vec3f), # type: ignore
    tracked_translations: wp.array(dtype=wp.vec3f), # type: ignore
    gauss_idx: wp.array(dtype=int),         # indices of Gaussians being tracked # type: ignore
    opt_body_ids: wp.array(dtype=int),      # per tracked Gaussian → body id # type: ignore
    disp_pts: wp.array(dtype=wp.vec3f),     # optional point-wise displacement # type: ignore
    # means_out: wp.array(dtype=wp.vec3f), # type: ignore
    # quats_out: wp.array(dtype=wp.quatf) # type: ignore
):
    """ Apply the optimized pose on Gaussian means """
    tid = wp.tid()
    idx = gauss_idx[tid]
    body_id = opt_body_ids[tid]

    means = means_in[idx]

    trans = tracked_translations[body_id]
    disp = obj_displacements[body_id]
    rot = obj_rotations[body_id]
    quat_rot = wp.quatf(rot[1], rot[2], rot[3], rot[0]) # wxyz -> xyzw
    # center the points
    mean_centered = means - trans
    # apply rotation on means
    mean_rot = wp.quat_rotate(quat_rot, mean_centered)
    mean_rotated = mean_rot + disp + trans
    # apply individual displacement
    mean_final = mean_rotated + disp_pts[idx]

    means_in[idx] = mean_final


@wp.kernel
def get_rigid_body_target_pose_kernel(
    body_q: wp.array(dtype=wp.transform),         # body state # type: ignore
    obj_rotations: wp.array(dtype=wp.quatf),      # type: ignore
    obj_displacements: wp.array(dtype=wp.vec3f),  # type: ignore
    body_indices: wp.array(dtype=int),            # body id # type: ignore
    body_target_q: wp.array(dtype=wp.transform),  # body state target # type: ignore
):
    """ Apply the optimized pose on the current rigid body pose to get the target pose """
    tid = wp.tid()
    bid = body_indices[tid]

    # get rigid body pose
    X_wb = body_q[bid]
    t_body = wp.transform_get_translation(X_wb)
    q_body = wp.transform_get_rotation(X_wb)

    # get optimized transformation
    t_opt = obj_displacements[tid]
    rot = obj_rotations[tid]
    q_opt = wp.quatf(rot[1], rot[2], rot[3], rot[0]) # wxyz -> xyzw

    # apply the transformation on the body pose to get target pose
    t_target = t_body + t_opt
    q_target = q_opt * q_body

    body_target_q[bid] = wp.transform(t_target, q_target)


# ==================================== Target Visual Forces ==================================== #
@wp.kernel
def set_force_moment_kernel(
    kp_f: float,  
    kp_m: float,
    body_q: wp.array(dtype=wp.transform),          # body state # type: ignore
    body_target_q: wp.array(dtype=wp.transform),   # body state target # type: ignore
    body_indices: wp.array(dtype=int),             # body id # type: ignore
    body_f_out: wp.array(dtype=wp.spatial_vector), # Output array for forces # type: ignore
):
    """ Set the correction force according to the target pose """
    tid = wp.tid()
    bid = body_indices[tid]
    
    # current pose
    body_state = body_q[bid]
    body_t = wp.transform_get_translation(body_state)
    body_quat = wp.transform_get_rotation(body_state)
    # target pose
    body_target_state = body_target_q[bid]
    body_target_t = wp.transform_get_translation(body_target_state)
    body_target_quat = wp.transform_get_rotation(body_target_state)
    
    # correction force
    displacement = body_target_t - body_t
    force = kp_f * displacement
    # correction moment
    rotation_diff = wp.quat_inverse(body_target_quat) * body_quat
    if rotation_diff[3] < 0:  # If the scalar (real) part is negative, flip the quaternion
        rotation_diff = -rotation_diff
    axis = wp.vec3()
    angle = wp.float32(0.0)  # type: ignore
    wp.quat_to_axis_angle(rotation_diff, axis, angle)
    moment = axis * angle * kp_m
    
    body_f_out[bid] = wp.spatial_vector(force, moment)
    