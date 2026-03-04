import os
import cv2
import time
import torch
import logging
import warp as wp
import torch.nn.functional as F

from gsplat import rasterization
from typing_extensions import override

from gausstwin.camera.cam import Camera
from gausstwin.sim.model_builder.model import Model
from gausstwin.sim.pbd_sim.simulator import Simulator
from gausstwin.tracker.rope.model_rope_builder import RopeBuilder
from gausstwin.cfg.tracker.gs_track_cfg import GSTrackUnifiedConfig
from gausstwin.utils.math_utils import convert_quat
from gausstwin.gaussians.appearance_optimizer import AppearanceOptimizer
from gausstwin.gaussians.visual_forces import VisualForcesRope, warp_batch_transform, apply_transform_means_kernel
from gausstwin.utils.warp_utils import transform_rigid_body_gaussians_part_kernel, transform_rope_gaussians_part_kernel, set_visual_rope_force_kernel, \
velocity_damping_particle_kernel, velocity_damping_rod_kernel


class RopeSimulator(Simulator):
    def __init__(
        self, 
        builder: RopeBuilder, 
        cfg: GSTrackUnifiedConfig, 
        model: Model, 
        device: str = "cuda:0",
    ):
        super().__init__(builder=builder, cfg=cfg, model=model, device=device, capture_graph=False)
        # save info
        self.cfg = cfg
        self.ground_h = -builder.ground_h
        self.gaussian_model = builder.gaussian_model
        self.gaussian_state = builder.gaussian_state
        self.n_robot_links = builder.n_robot_links
        
        # visual forces
        self.visual_forces = VisualForcesRope(
            gaussian_model=self.gaussian_model, 
            gaussian_state=self.gaussian_state, 
            parts_in_track=builder.rope_part_in_track_idx_list,
            mode=cfg.mode,
        )
        self.appearance_optimizer = AppearanceOptimizer(self.gaussian_state)
        
        # save indices
        self.rope_part_in_track_idx_list = builder.rope_part_in_track_idx_list
        
        # object positions and quaternions (world frame)
        self.gaussian_particle_indices = builder.gaussian_particle_ids 
        self.gaussian_particle_indices_wp = wp.array(self.gaussian_particle_indices, dtype=wp.int32, device=self.device) 
        self.gaussian_particle_id_weights_wp = wp.array(builder.gaussian_particle_id_weights, dtype=wp.float32, device=self.device) 
        
        # set learning rates
        self.init_optimizer()
        # load cameras
        self.load_camera_info()
        # capture graph
        self.capture()
    
    
    # ================================== Load Camera Info ======================================= #
    def load_camera_info(self):
        """
        Build the camera model.
        """
        cam_cfg_list = []
        for i in range(self.cfg.num_views):
            cam_idx = i + 1
            f_name = os.path.join(self.cfg.data_dir, f"trajectory_{self.cfg.exp_name}", f"camera_{cam_idx}.json")
            cam_cfg_list.append(f_name)
        
        self.cam = Camera(cam_cfg_list)
        logging.info(f"{len(cam_cfg_list)} Cameras are loaded")
    
    # ================================== Initialize 3DGS Optimizers ======================================= #
    def init_optimizer(self):
        self.visual_forces.set_learnings_rates([
            self.cfg.vf_lr_means, 
            self.cfg.vf_lr_quats, 
            self.cfg.vf_lr_disps, 
            self.cfg.vf_lr_rots,
        ])
        self.appearance_optimizer.set_learnings_rates([
            self.cfg.vf_lr_colors, 
            self.cfg.vf_lr_opacities, 
            self.cfg.vf_lr_scales
        ])
    
    # =============================== Capture Graph ====================================== #
    @override
    def capture(self):
        # capture graph, avoid launch overhead
        if wp.get_device().is_cuda:
            # graph for pbd sim step
            with wp.ScopedCapture() as capture_sim:
                self.simulate()
            self.graph_sim = capture_sim.graph
            
            # graph for 3DGS state update
            with wp.ScopedCapture() as capture_trans_gs:
                self.transform_gs()
            self.graph_trans_gs = capture_trans_gs.graph
        else:
            self.graph_sim = None
            self.graph_trans_gs = None
    
    
    @override
    def simulate(self):
        # Assign forces
        wp.copy(self.state_0.particle_f, self.visual_forces.forces_particles_wp)
        wp.copy(self.state_1.particle_f, self.visual_forces.forces_particles_wp)
            
        # Run the PBD simulation
        # detect collision
        wp.sim.collide(self.model, self.state_0) 
        # run substeps
        for _ in range(self.substeps):
            self.solver.simulate(
                self.model,
                self.state_0,
                self.state_1,
                self.dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
        
        # Damping
        wp.launch(kernel=velocity_damping_particle_kernel, dim=self.model.particle_count, inputs=[0.8, self.state_0.particle_qd, self.state_1.particle_qd])
        wp.launch(kernel=velocity_damping_rod_kernel, dim=self.model.rod_seg_count, inputs=[0.8, self.state_0.rod_qd, self.state_1.rod_qd])
        
    # ========================================================================================== #
    # ================================== Gausstwin Step ======================================== #
    # ========================================================================================== #
    def step(
        self, 
        robot_positions,
        robot_quats,
        robot_lin_vel=None,
        robot_ang_vel=None,
        gt_pixels=None,
        gt_masks=None, 
        gt_depths=None,
        vis_pts=None,
        clear_forces: bool = False,
        render: bool = False,
    ):
        """
        Perform one step simulation and visual correction. For one step case.
        """
        # 1. Update the robot state
        self.update_robot_state(robot_positions, robot_quats, robot_lin_vel, robot_ang_vel)
        
        if clear_forces:
            self.visual_forces.forces_particles_wp.zero_()

        # 2. Run PBD simulation
        self.step_pbd(vis_pts, render)
        
        if gt_pixels is not None:
            # 3. Transform the gaussians 
            self.transform_gaussians()
            
            # 4. Compute the visual forces
            self.compute_visual_forces(gt_pixels, gt_masks, gt_depths)
    
    
    def step_simulation(
        self, 
        vis_pts=None,
        clear_forces: bool = False,
        render: bool = False
    ):
        """
        Perform one step simulation and visual correction. For two step case.
        """
        if clear_forces:
            self.visual_forces.forces_particles_wp.zero_()
        self.step_pbd(vis_pts, render, half_dt=True)
        
        
    def visual_correction(
        self, 
        gt_pixels,
        gt_masks, 
    ):
        """
        Compute visual forces. For two step case.
        """
        # transform the gaussians 
        self.transform_gaussians()
        # compute visual forces
        self.compute_visual_forces(gt_pixels, gt_masks)
        
        
    def step_init(
        self, 
        robot_positions,
        robot_quats,
        robot_lin_vel=None,
        robot_ang_vel=None,
        gt_pixels=None,
        gt_masks=None, 
        clear_forces: bool = False,
    ):
        """
        For initialization 
        """
        # 1. Update the robot state
        self.update_robot_state(robot_positions, robot_quats, robot_lin_vel, robot_ang_vel)
        
        if clear_forces:
            self.visual_forces.forces_particles_wp.zero_()

        # 2. Run PBD simulation
        self.step_pbd(vis_pts=None)
        
        if gt_pixels is not None:
            # 3. Transform the gaussians 
            self.transform_gaussians()
            
            # 4. Compute the visual forces
            self.compute_visual_forces(gt_pixels, gt_masks, n_iter=10)

    
    # ========================================================================================== #
    # ================================ Update Robot State ====================================== #
    # ========================================================================================== #
    def update_robot_state(self, positions: torch.tensor, quats: torch.tensor, linear_vel: torch.tensor = None, angular_vel: torch.tensor = None):
        """ Update the robot state with the new positions and quaternions. """
        quats = convert_quat(quats, 'xyzw') # convert to xyzw format

        # update the positions and quaternions for robot links
        body_q_0 = wp.to_torch(self.state_0.body_q) 
        body_q_0[:self.n_robot_links, :3] = positions
        body_q_0[:self.n_robot_links, 3:] = quats
        
        wp.copy(self.state_1.body_q, self.state_0.body_q)
        
        # update the velocities and angular velocities for robot links
        if linear_vel is not None and angular_vel is not None:
            body_qd_0 = wp.to_torch(self.state_0.body_qd)
            body_qd_0[:self.n_robot_links, :3] = angular_vel
            body_qd_0[:self.n_robot_links, 3:] = linear_vel
            
            wp.copy(self.state_1.body_qd, self.state_0.body_qd)
            
                
    # ========================================================================================== #
    # =============================== PBD Simulation Step ====================================== #
    # ========================================================================================== #
    @override
    def step_pbd(self, vis_pts=None, render: bool=False, half_dt: bool=False):
        # step simulation
        if self.graph_sim is not None:
            wp.capture_launch(self.graph_sim)
        else:
            self.simulate()
        
        # update time
        if half_dt:
            self.sim_time += self.frame_dt / 2.0
        else:
            self.sim_time += self.frame_dt
        
        # visualization
        if self.cfg.render and render:
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            if vis_pts is not None:
                self.renderer.render_points("vis_pts", points=vis_pts, colors=(0.0, 1.0, 0.0), radius=0.005)
            self.renderer.end_frame()
        

    # ===================================================================================== #
    # ================================= Transform 3DGS ==================================== #
    # ===================================================================================== #
    def transform_gs(self):
        # transform the GS of the Robot
        wp.launch(
            kernel=transform_rigid_body_gaussians_part_kernel, 
            dim=self.gaussian_model.num_robot_gaussians, 
            inputs=[
                self.gaussian_model.means, self.gaussian_model.quats, 
                self.gaussian_model.robot_indices, self.gaussian_model.robot_body_ids,
                self.state_0.body_q,
                self.ground_h,
            ], 
            outputs=[self.gaussian_state.means, self.gaussian_state.quats]
        )
        
        # transform the GS of the Rope
        wp.launch(
            kernel=transform_rope_gaussians_part_kernel, 
            dim=self.gaussian_model.num_rope_gaussians, 
            inputs=[
                self.gaussian_model.means, self.gaussian_model.quats, 
                self.gaussian_model.rope_indices, 
                self.gaussian_model.rope_part_ids, self.gaussian_model.rope_quat_ids,
                self.state_0.particle_q,
                self.state_0.rod_q,
                self.ground_h,
            ], 
            outputs=[self.gaussian_state.means, self.gaussian_state.quats]
        )
    
    
    def transform_gaussians(self):
        if self.graph_trans_gs is not None:
            wp.capture_launch(self.graph_trans_gs)
        else:
            self.transform_gs()
    
    # ======================================================================================== #
    # =============================== Visual Correction ====================================== #
    # ======================================================================================== #
    def compute_visual_forces(self, gt_pixels_list, gt_masks_list, n_iter=None):
        # t_s = time.time()
        n_views = self.cam.viewmats.shape[0]
        if n_iter is not None:
            max_iter = n_iter
        else:
            max_iter = self.cfg.vf_iterations
    
        with torch.no_grad():
            # update the 3D Gaussian states
            self.visual_forces.means.copy_(self.gaussian_state.means)
            self.visual_forces.quats.copy_(self.gaussian_state.quats)
            # self.visual_forces.optimizer.reset_internal_state()
            self.visual_forces.reset_learning_rates()
            
            # reset the parameters for optimization
            if self.cfg.mode == 2 or self.cfg.mode == 3 or self.cfg.mode == 4:
                self.visual_forces.rotation.zero_()
                self.visual_forces.rotation[:, 0] = 1.0 # set the rotation to identity
                self.visual_forces.displacement.zero_()
                self.visual_forces.disp_pts.zero_()
                
                # set the center of the rope
                rope_t = wp.to_torch(wp.clone(self.state_0.particle_q))[:, :3]
                self.visual_forces.tracked_translations = rope_t
        
        # process the mask
        backgrounds = torch.rand((max_iter, 3)).float().cuda() # (max_iter, 3)
        # backgrounds = torch.zeros((max_iter, 3)).float().cuda() # (max_iter, 3)
        gt_pixels = torch.stack(gt_pixels_list, dim=0) # (N, H, W, 3)

        gt_masks = torch.stack(gt_masks_list, dim=0) # (N, H, W)
        gt_masks = gt_masks.unsqueeze(-1).bool().expand(-1, -1, -1, 3) # (N, H, W, 3)   
        
        backgrounds_exp = backgrounds.view(max_iter, 1, 1, 1, 3)  # (T, 1, 1, 1, 3)
        gt_pixels_exp = gt_pixels.unsqueeze(0).expand(max_iter, -1, -1, -1, -1)
        gt_masks_exp = gt_masks.unsqueeze(0).expand(max_iter, -1, -1, -1, -1)
        pixels_seq = torch.where(gt_masks_exp, backgrounds_exp, gt_pixels_exp)  # (T, N, H, W, 3)

        # run the loop
        for step in range(max_iter):
            background = backgrounds[step]
            pixels = pixels_seq[step]
            
            # get transformed 3D gaussians
            means = torch.clone(self.visual_forces.means) # (n, 3)
            quats = torch.clone(self.visual_forces.quats) # (n, 4)
            trans_means, trans_quats = warp_batch_transform(
                means, quats,
                self.visual_forces.tracked_translations,
                self.visual_forces.rotation, self.visual_forces.displacement, self.visual_forces.disp_pts,
                self.gaussian_model.rope_indices, self.gaussian_model.rope_part_ids,
                self.gaussian_model.num_rope_gaussians, 
            )
        
            # render images
            render_images, _, _ = rasterization(
                means=trans_means,
                quats=trans_quats,
                scales=self.gaussian_state.scales,
                colors=self.gaussian_state.colors,
                opacities=self.gaussian_state.opacities,
                viewmats=self.cam.viewmats,
                Ks=self.cam.Ks,
                width=self.cam.W,
                height=self.cam.H,
                backgrounds=background.reshape(1, 3).repeat(n_views, 1),
                packed=False,
                camera_model="pinhole",
                render_mode="RGB",
            )

            # mse loss
            loss = torch.nn.functional.mse_loss(render_images, pixels)
            # run optimizers
            self.visual_forces.zero_grad()
            # self.appearance_optimizer.zero_grad()
            loss.backward()
            self.visual_forces.step()
            # self.appearance_optimizer.step()
            
            # normalize quaternions
            with torch.no_grad():
                self.visual_forces.quats = torch.nn.functional.normalize(self.visual_forces.quats)
                
            # visualize
            if self.cfg.vis:
                if step == 0 or step == max_iter - 1:
                    rgb = render_images.detach().cpu().numpy()
                    w_u = 100
                    w_v = 100
                    disp_u = rgb.shape[2]
                    for cam in range(rgb.shape[0]):
                        # Create independent named windows
                        cv2.namedWindow(f"Render_{cam} Step: {step}", cv2.WINDOW_AUTOSIZE)
                        cv2.namedWindow(f"GT_{cam} Step: {step}", cv2.WINDOW_AUTOSIZE)
                        # Move them to different positions on the screen
                        cv2.moveWindow(f"Render_{cam} Step: {step}", w_u, w_v)
                        cv2.moveWindow(f"GT_{cam} Step: {step}", w_u + disp_u, w_v)
                        image_array = rgb[cam]
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                        cv2.imshow(f"Render_{cam} Step: {step}", image_array)
                        
                        # pixel = gt_pixels[cam].detach().cpu().numpy()
                        pixel = pixels[cam].detach().cpu().numpy()
                        image_array = cv2.cvtColor(pixel, cv2.COLOR_RGB2BGR)
                        cv2.imshow(f"GT_{cam} Step: {step}", image_array)
                        
                        w_v += rgb.shape[1]
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        
        # logging.info(f"After Optimization, Time eplased: {time.time() - t_s:.4f}")
        
        # ============================ Compute Visual Forces ============================ # 
        target_means = torch.clone(self.visual_forces.means) # (n, 3)
        wp.launch(
            kernel=apply_transform_means_kernel,
            dim=self.gaussian_model.num_rope_gaussians, 
            inputs=[
                target_means,
                self.visual_forces.rotation.detach(), self.visual_forces.displacement.detach(),
                self.visual_forces.tracked_translations,
                self.gaussian_model.rope_indices, self.gaussian_model.rope_part_ids,
                self.visual_forces.disp_pts.detach(),
            ],
        )

        prev_means = self.gaussian_state.means.detach()[self.gaussian_model.rope_indices]
        # compute and apply the visual forces
        with torch.no_grad():
            self.visual_forces.forces_particles_wp.zero_()
            wp.launch(
                kernel=set_visual_rope_force_kernel,
                dim=self.gaussian_model.num_rope_gaussians,
                inputs=[
                    self.cfg.kp_f, 
                    prev_means,
                    target_means[self.gaussian_model.rope_indices], 
                    self.gaussian_model.rope_part_ids,
                    self.gaussian_particle_id_weights_wp,
                ], 
                outputs=[self.visual_forces.forces_particles_wp],
            )
    
    # ======================================================================================== #
    # =============================== Visualization Render =================================== #
    # ======================================================================================== #
    def vis_render(self):
        self.transform_gaussians()

        with torch.no_grad():
            render_images, _, _ = rasterization(
                means=self.gaussian_state.means,
                quats=self.gaussian_state.quats,
                scales=self.gaussian_state.scales,
                colors=self.gaussian_state.colors,
                opacities=self.gaussian_state.opacities,
                viewmats=self.cam.viewmats,
                Ks=self.cam.Ks,
                width=self.cam.W,
                height=self.cam.H,
                packed=False,
                camera_model="pinhole",
                render_mode="RGB",
            )

        return render_images
    