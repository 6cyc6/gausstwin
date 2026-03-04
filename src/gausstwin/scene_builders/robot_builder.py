import os
import cv2
import json
import time
import torch
import logging
import polyscope as ps
import torch.nn.functional as F

from tqdm import tqdm
from fused_ssim import fused_ssim
from torch.utils.data import DataLoader
from gsplat import DefaultStrategy, rasterization

from gausstwin.cfg.builder.builder_cfg import GSConfig
from gausstwin.cfg.robot.robot_configs import get_robot_config
from gausstwin.cfg.domain import Robot, RigidBody, Gaussians, Particles
from gausstwin.utils.dataset import Dataset
from gausstwin.utils.path_utils import get_gs_fig_dir
from gausstwin.utils.ps_utils import polyscope_vis_pcl
from gausstwin.utils.cam_utils import get_pcl_from_rgbd_mask, downsample_pcl
from gausstwin.utils.gs_utils import create_splats_with_optimizers, vis_render_gs


class RobotBuilder:
    def __init__(self, cfg: GSConfig, name: str):
        self.cfg = cfg
        self.name = name
        self.fig_dir = get_gs_fig_dir()
        self.device = cfg.device
        # get robot configuration
        self.robot_config = get_robot_config(name)
        self.link_name_list = self.robot_config.link_list
        self.num_pts_list = self.robot_config.init_gs_num_pt_list
        self.link_radii_list = self.robot_config.radii_list
        self.link_particle_list = self.robot_config.sph_pt_list

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # setup camera
        self.build_camera()
    
    
    def build_robot(self):
        """
        Build the robot model.
        """
        self.robot = Robot(robot_name=self.name) # type: ignore

        for i in tqdm(range(len(self.link_name_list))):
            rigid_body = RigidBody(
                name=self.link_name_list[i],
            )
            logging.info(f"Building {self.link_name_list[i]} ...")
            link_name = self.link_name_list[i]
            
            # load dataset
            dataset = Dataset(
                data_dir=os.path.join(self.fig_dir, f"{self.name}/{link_name}"),
                device=self.cfg.device,
            )
            
            pts_list, pts_rgb_list = self.merge_point_clouds(dataset)
            
            # stack the point clouds and RGB values
            pts_all = torch.cat(pts_list)  # (N, 3)
            pts_rgb_all = torch.cat(pts_rgb_list[:])  # (N, 3)
            
            # downsample the points
            pts, pts_rgb = downsample_pcl(pts_all, pts_rgb_all, self.num_pts_list[i])
            
            # visualize with polyscope
            if self.cfg.ps_vis:
                polyscope_vis_pcl(pts, pts_rgb, radius=0.005)
            
            # prepare splats and optimizers
            self.splats, self.optimizers = create_splats_with_optimizers(
                pcl=pts, 
                rgb=pts_rgb, 
                cfg=self.cfg, 
                norm=False,
            )
            
            # run optimization
            self.optimize_gs(dataset)
                
            # save the results
            rigid_body.gaussians = Gaussians(
                means=self.splats["means"].tolist(),
                quats=torch.nn.functional.normalize(self.splats["quats"]).tolist(),
                scales=torch.exp(self.splats["scales"]).tolist(),
                opacities=torch.sigmoid(self.splats["opacities"]).tolist(),
                colors=torch.sigmoid(self.splats["colors"]).tolist(),
            )
            rigid_body.particles = Particles(
                means=self.link_particle_list[i].tolist(),
                radii=self.link_radii_list[i],
            )
            
            self.robot.add_link(link_name, rigid_body)
        self.robot.save("fr3_v3")
        
            
    def build_camera(self):
        """
        Build the camera model.
        """
        # Load the camera intrinsics
        cam_intrinsics_path = os.path.join(self.fig_dir, f"{self.name}/link0/camera_intrinsics.json")
        with open(cam_intrinsics_path, "r") as f:
            cam_data = json.load(f)

        self.cam_K = torch.tensor(cam_data["k"], dtype=torch.float32, device=self.cfg.device)
        self.H = cam_data["h"]
        self.W = cam_data["w"]

        logging.info(f"Camera intrinsics loaded from: {cam_intrinsics_path}")
        logging.info(f"Camera K: \n{self.cam_K}")
        logging.info(f"Image size: {self.W} x {self.H}")
        

    def merge_point_clouds(self, dataset: Dataset):
        """
        Load the image and point cloud data for a specific link.
        """
        init_data = dataset.get_init_data(n_views=8)
        images = init_data['colors'].to(self.device)
        depths = init_data['depths'].to(self.device)
        masks = init_data['masks'].to(self.device)
        poses = init_data['poses'].to(self.device)
        
        pcl_list = []
        pcl_colors_list = []
        print("Generating point clouds from RGBD data...")
        for i in range(len(images)):
            pose = poses[i]
            pcl, pcl_colors = get_pcl_from_rgbd_mask(self.cam_K, pose, images[i],depths[i], masks[i])

            pcl_list.append(pcl)
            pcl_colors_list.append(pcl_colors)
        
        return pcl_list, pcl_colors_list


    def optimize_gs(self, dataset: Dataset):
        """
        Optimize the Gaussian splatting model.
        """
        # prepare dataloader
        dataloader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        
        # optimization strategy
        strategy = DefaultStrategy(
            refine_start_iter=self.cfg.refine_start_iter, 
            refine_stop_iter=self.cfg.refine_stop_iter, 
            refine_every=self.cfg.refine_every,
            prune_opa=self.cfg.prune_opa, 
        )
        
        # get camera intrinsics
        Ks = self.cam_K.unsqueeze(0).repeat(self.cfg.batch_size, 1, 1)
        
        # prepare random backgrounds for optimization
        backgrounds = torch.rand((self.cfg.max_iter, 3)).float().cuda()

        # initialize the strategy state
        strategy_state = strategy.initialize_state(scene_scale=self.cfg.scene_scale)
        
        start_time = time.time()
        # start optimization loop
        for step in tqdm(range(self.cfg.max_iter), leave=False):
            # load batch data
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
            # Get data from batch
            gt_colors = batch['colors'].to(self.device)  # (B, H, W, 3)
            gt_depths = batch['depths'].to(self.device)  # (B, H, W)
            masks = batch['masks'].to(self.device)    # (B, H, W)
            poses = batch['poses'].to(self.device)    # (B, 4, 4)
            
            background = backgrounds[step]
            indicies = masks == 0
            gt_colors[indicies, :] = background
            
            # set camera pose
            viewmats = poses # [C, 4, 4]

            # Forward pass
            renders, render_alpha, info = rasterization(
                means=self.splats["means"],
                quats=F.normalize(self.splats["quats"]),
                scales=torch.exp(self.splats["scales"]),
                opacities=torch.sigmoid(self.splats["opacities"]),
                colors=torch.sigmoid(self.splats["colors"]),
                viewmats=viewmats,  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=self.W,
                height=self.H,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                backgrounds=background.reshape(1, 3).repeat(self.cfg.batch_size, 1),
                packed=False,
                render_mode="RGB+D",
            )
            
            # pre-backward step
            strategy.step_pre_backward(self.splats, self.optimizers, strategy_state, step, info)

            if renders.shape[-1] == 4:
                render_images, render_depths = renders[..., 0:3], renders[..., 3:4]
            else:
                render_images, render_depths = renders, None
                
            # compute losses
            l1_loss = F.l1_loss(render_images, gt_colors)
            ssim_loss = 1.0 - fused_ssim(
                render_images.permute(0, 3, 1, 2), gt_colors.permute(0, 3, 1, 2), padding="valid"
            )

            loss = 0.8 * l1_loss + 0.2 * ssim_loss

            if render_depths is not None:
                depth_loss = F.l1_loss(render_depths, gt_depths.unsqueeze(-1))
                loss += 0.005 * depth_loss
            else:
                depth_loss = torch.tensor(0.0, device=self.device)

            # backward pass
            loss.backward()
            
            # step optimizers 
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Post-backward step
            strategy.step_post_backward(self.splats, self.optimizers, strategy_state, step, info)
            
            # visualization
            if self.cfg.vis and (step % 500 == 0):
                vis_render_gs(Ks=Ks, viewmats=viewmats, splats=self.splats, gt_img=gt_colors)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        n_gaussians = self.splats["means"].shape[0]
        logging.info(f"{n_gaussians} Gaussian self.splats for the link.")
        logging.info(f"Time elapsed: {elapsed_time:.4f} seconds")
        