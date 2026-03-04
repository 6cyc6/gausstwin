import os
import json
import time
import torch
import imageio
import logging
import warp as wp
import numpy as np
import open3d as o3d
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from torch_cluster import knn
from fused_ssim import fused_ssim
from scipy.spatial.transform import Rotation as R
from gsplat import DefaultStrategy, rasterization
from sam2.build_sam import build_sam2_camera_predictor 

from gausstwin.camera.cam import Camera
from gausstwin.segmentation.prompt import PromptTool
from gausstwin.cfg.builder.builder_cfg import RigidBodyConfig
from gausstwin.cfg.domain import RigidBody, Gaussians, Particles, Ground, SegPrompt
from gausstwin.utils.ps_utils import polyscope_vis_pcl
from gausstwin.utils.path_utils import get_save_dir, safe_mkdir
from gausstwin.utils.vis_utils import to_torch, show_track_masks_cv2
from gausstwin.utils.init_utils import compute_bounding_box, fill_bounding_box_with_spheres, is_point_in_mask
from gausstwin.utils.gs_utils import ransac_plane, vis_render_gs_cam, assign_nearest_neighbor_properties, down_sample_pcl, create_rigid_object_splats_with_optimizers


class RigidBodyBuilder:
    def __init__(self, cfg: RigidBodyConfig, device: str = "cuda:0"):
        self.cfg = cfg
        self.data_dir = os.path.join(cfg.data_dir, f"trajectory_{cfg.exp_name}")
        self.device = device
        
        # init warp
        wp.init()
        
        # if use wrist cameras
        self.n_views = self.cfg.num_views
        self.n_batch = self.cfg.batch_size

        # logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        
        
    def build(
        self
    ):
        """
        Build the rigid body in the scene.
        """
        # ============================== Step 1: Load Info ====================================== #
        self.load_info()
        
        # ============================= Step 2: Process the scene =============================== #
        self.process_scene()
        
        # =================== Step 3: Get the ground plane function (Ground) ==================== #
        self.fit_ground_plane()
        
        # =================== Step 4: Get the bounding box of the pointcloud ==================== #
        self.extract_spheres()
        
        # ================== Step 5: Run gs optimization to get particles ======================= #
        # self.build_particles() # This step is used in embodied Gaussians

        # ================== Step 6: Run gs optimization to get particles ======================= #
        self.build_gaussians()
        
        # ====================== Step 7: Assign gaussians to particles ========================== #
        self.assign_gaussians_to_particles()
        
        # ========================== Step 8: Convert to body frame ============================== #
        self.convert_to_body_frame()
        
        # ========================== Step 9: Save the rigid bodies ============================== #
        self.save_rigid_bodies_and_ground()
        
    
    # ========================================================================================== #
    # ==================================== Load information ==================================== #
    # ========================================================================================== #
    def load_info(self):
        """
        Load information.
        """
        # load camera configuration
        self.load_camera_info()
        # load SAM2
        self.load_seg_model()
        # load scene information
        self.load_scene_info()
    
    
    def load_camera_info(self):
        """
        Build the camera model.
        """
        # Load the camera configuration
        cam_cfg_list = []
        for i in range(self.n_views):
            cam_idx = i + 1
            f_name = os.path.join(self.data_dir, f"camera_{cam_idx}.json")
            cam_cfg_list.append(f_name)
        
        self.cam = Camera(cam_cfg_list)
        logging.info(f"{self.cam.viewmats.shape[0]} Cameras are loaded")
    
    
    def load_seg_model(self):
        """
        Load SAM2 models.
        """
        # Load SAM2 model
        self.sam2_predictor = build_sam2_camera_predictor(self.cfg.sam2_config, self.cfg.sam2_checkpoint, device=self.device)
        
        
    def _load_rgb_depth(self, rgb_path: str, depth_path: str):
        """Load RGB and depth images from given paths."""
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"View file {rgb_path} does not exist.")
        # load rgb
        img = Image.open(rgb_path)
        img_np = np.array(img.convert("RGB"))
        img_torch = torch.from_numpy(img_np).to(self.device)
        # load depth
        depth_image = imageio.imread(depth_path)
        depth = depth_image.astype(np.float32) / 1000.0
        depth_torch = torch.from_numpy(depth).unsqueeze(-1).to(self.device).float()
        return img_np, img_torch, depth_torch


    def load_scene_info(self):
        """
        Load the scene information.
        """
        self.depth_list = []     # [(H, W, 1), (H, W, 1), ...]
        self.rgb_list = []       # [(H, W, 3), (H, W, 3), ...]
        self.rgb_np_list = []    # [(H, W, 3), (H, W, 3), ...]
        self.img_path_list = []

        # frame indices for each fixed camera
        fix_indices = [self.cfg.fix_idx_1, self.cfg.fix_idx_2, self.cfg.fix_idx_3, self.cfg.fix_idx_4]

        # load images from fixed cameras
        for i in range(self.cfg.num_views):
            cam_idx = i + 1
            fix_idx = fix_indices[i]
            rgb_path = os.path.join(self.data_dir, f"camera/static_{cam_idx}/rgb/{fix_idx}.png")
            depth_path = os.path.join(self.data_dir, f"camera/static_{cam_idx}/depth/{fix_idx}.png")

            img_np, img_torch, depth_torch = self._load_rgb_depth(rgb_path, depth_path)
            self.img_path_list.append(rgb_path)
            self.rgb_np_list.append(img_np)
            self.rgb_list.append(img_torch)
            self.depth_list.append(depth_torch)

        self.depths = to_torch(self.depth_list, device=self.device) # type: ignore (N, H, W, 1)
        self.rgbs = to_torch(self.rgb_list, device=self.device)     # type: ignore (N, H, W, 3)

        self.gt_rgb_list = [rgb / 255.0 for rgb in self.rgbs]       # [tensor(H, W, 3), ...]
        self.gt_rgbs = self.rgbs / 255.0                            # (N, H, W, 3); ground truth rgb images # 0-255 -> 0-1
        self.gt_depths = self.depths                                # (N, H, W, 1); ground truth depth images

        logging.info(f"{len(self.gt_rgb_list)} Views are loaded")
        
    
    # ========================================================================================== #
    # ==================================== Process Scene ======================================= #
    # ========================================================================================== #
    def _process_mask_logits(self, mask_logits):
        """Convert mask logits to numpy uint8 and torch bool tensor."""
        mask_np = (mask_logits > 0).squeeze().cpu().numpy().astype(np.uint8)
        mask_torch = torch.from_numpy(mask_np).to(self.device).bool()
        return mask_np, mask_torch

    def process_scene(self):
        """
        Process the scene to extract objects and their point clouds.
        """
        # init buffers
        all_objects = ["Ground"] + self.cfg.obj_list  # {"object_name": {"points": [], "labels": [], "bbox": []}}
        self.prompt_dict = {}
        self.obj_mask_dict = {obj: [] for obj in all_objects} # {"object_name": [torch.Tensor(H, W), ...]}; object masks for each view

        # load prompts from file if configured
        saved_prompts = None
        if self.cfg.load_prompt:
            save_path = f"{get_save_dir()}/{self.cfg.exp_name}/prompt.json"
            with open(save_path, "r") as f:
                saved_prompts = json.load(f)

        # ---------------------------- Ground ---------------------------- #
        prompt_tool = PromptTool(image_path=self.img_path_list[0], obj_name="Ground")
        point_coords, point_labels, bbox = prompt_tool.run()

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam2_predictor.load_first_frame(self.rgb_np_list[0])
            _, _, out_mask_logits = self.sam2_predictor.add_new_prompt(
                frame_idx=0, obj_id=-1, bbox=bbox, points=point_coords, labels=point_labels
            )
            mask_np, mask_torch = self._process_mask_logits(out_mask_logits)
            show_track_masks_cv2(self.rgb_np_list[0], mask_np, point_coords=point_coords,
                                 input_labels=point_labels, box_coords=bbox, borders=True, window_name="Mask for View 0")
            self.obj_mask_dict["Ground"].append(mask_torch)

            # track ground across remaining views
            for i in range(1, self.n_views):
                _, out_mask_logits = self.sam2_predictor.track(self.rgb_np_list[i])
                mask_np, mask_torch = self._process_mask_logits(out_mask_logits)
                show_track_masks_cv2(self.rgb_np_list[i], mask_np, borders=True, window_name=f"Mask for View {i}")
                self.obj_mask_dict["Ground"].append(mask_torch)

        # ---------------------------- Objects ---------------------------- #
        for obj_idx, obj in enumerate(self.cfg.obj_list):
            logging.info(f"Adding prompts for the object: {obj} ...")
            self.prompt_dict[obj] = {}

            for view_idx in range(self.n_views):
                # get prompt (from saved file or interactive)
                if saved_prompts:
                    prompt_data = saved_prompts[obj][f"view_{view_idx}"]
                    point_coords, point_labels, bbox = prompt_data["points"], prompt_data["labels"], prompt_data["bbox"]
                else:
                    prompt_tool = PromptTool(image_path=self.img_path_list[view_idx], obj_name=obj)
                    point_coords, point_labels, bbox = prompt_tool.run()

                # run segmentation
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    self.sam2_predictor.load_first_frame(self.rgb_np_list[view_idx])
                    _, _, out_mask_logits = self.sam2_predictor.add_new_prompt(
                        frame_idx=0, obj_id=obj_idx, bbox=bbox, points=point_coords, labels=point_labels
                    )
                    mask_np, mask_torch = self._process_mask_logits(out_mask_logits)
                    show_track_masks_cv2(self.rgb_np_list[view_idx], mask_np, point_coords=point_coords,
                                         input_labels=point_labels, box_coords=bbox, borders=True,
                                         window_name=f"Mask for View {view_idx} - {obj}")

                # save prompt and mask
                self.prompt_dict[obj][f"view_{view_idx}"] = {
                    "points": [list(pt) for pt in point_coords],
                    "labels": point_labels,
                    "bbox": bbox
                }
                self.obj_mask_dict[obj].append(mask_torch)

        # ---------------------------- Extract point clouds ---------------------------- #
        self.obj_pcl_dict = {obj: [] for obj in all_objects} # {"obj_name": torch.tensor(N, 3)}; object point clouds
        self.obj_pcl_rgb_dict = {obj: [] for obj in all_objects} # {"obj_name": [torch.Tensor(N, 3)]}, object point cloud colors
        self.gt_mask_list = []  # [torch.Tensor(H, W, 1), ...]; masks for each viewpoint, background: 1.0, object: 0.0, for gs optimization
        self.gt_mask_ground_list = [] # [torch.Tensor(H, W, 1), ...]; masks for each viewpoint, background: 1.0, ground: 0.0

        for i in range(self.n_views):
            mask_img = torch.ones((self.cam.H, self.cam.W, 1), dtype=torch.float, device=self.device) # (H, W, 1); background: 1.0, object: 0.0
            mask_ground_img = torch.ones((self.cam.H, self.cam.W, 1), dtype=torch.float, device=self.device)  # (H, W, 1); background: 1.0, ground: 0.0

            for obj in all_objects:
                obj_mask = self.obj_mask_dict[obj][i] # (H, W)
                self.obj_pcl_dict[obj].append(self.cam.image_seg_to_world(depth=self.depths[i], mask=obj_mask, cam_idx=i))
                self.obj_pcl_rgb_dict[obj].append(self.rgbs[i][obj_mask].float())

                if obj != "Ground":
                    mask_img[obj_mask > 0.0] = 0.0
                else:
                    mask_ground_img[obj_mask > 0.0] = 0.0

            self.gt_mask_list.append(mask_img)
            self.gt_mask_ground_list.append(mask_ground_img)

        logging.info("Segmentation info is loaded.")

        # save prompts if needed
        if self.cfg.save_prompt:
            save_dir = f"{get_save_dir()}/{self.cfg.exp_name}"
            safe_mkdir(save_dir)
            with open(f"{save_dir}/prompt.json", "w") as f:
                json.dump(self.prompt_dict, f, indent=4)


    # ========================================================================================== #
    # ================================== Fit Ground Plane ====================================== #
    # ========================================================================================== #
    def fit_ground_plane(self):
        """
        Fit the ground plane using RANSAC.
        """
        # run RANSAC to get the Ground plane
        pts_Grounds = torch.cat(self.obj_pcl_dict["Ground"], dim=0)
        pts_Grounds = self.clean_point_cloud(pts_Grounds, n_pts=50000)  # clean the point cloud
        plane, _ = ransac_plane(pts_Grounds, threshold=0.001, iterations=1000)
        normal, d = plane
        # always d negative
        if d > 0:
            normal = -normal
            d = -d
        self.plane_normal = F.normalize(normal, p=2, dim=0)
        self.plane_h = d
        self.plane_disp = torch.tensor([0.0, 0.0, -d], device=self.device)
        
        # train the gaussians for the ground
        if self.cfg.save_ground_gaussians:
            logging.info(f"Building Gaussian Splattings for the ground ...")
            self.build_gaussians(ground=True)
            means = self.splats_gaussians["means"].detach().clone()
            quats = F.normalize(self.splats_gaussians["quats"]).detach().clone()
            colors = torch.sigmoid(self.splats_gaussians["colors"]).detach().clone()
            opacities = torch.sigmoid(self.splats_gaussians["opacities"]).detach().clone()
            scales = torch.exp(self.splats_gaussians["scales"]).detach().clone()
            # save the gaussians
            self.ground_gaussians = Gaussians(
                means=means.tolist(),
                quats=quats.tolist(),
                colors=colors.tolist(),
                opacities=opacities.tolist(),
                scales=scales.tolist(),
            )
        self.ground_pts = pts_Grounds
        logging.info(f"Ground plane is fitted. \nThe plane normal is {self.plane_normal}. \nThe height is {-self.plane_h}")
    
    
    # ========================================================================================== #
    # =================================== Extract Spheres ====================================== #
    # ========================================================================================== #
    def extract_spheres(self):
        """
        Extract spheres 
        """
        # Get BBox and init point centers
        self.obj_sphere_centers = {}
        self.obj_sphere_means = {}
        self.obj_radius = {}
        self.obj_particles_dict = {}
        for obj in self.cfg.obj_list:
            # get object point cloud
            pts_obj_list = self.obj_pcl_dict[obj]
            pts_obj = torch.cat(pts_obj_list, dim=0)
            # remove outliers
            pts_obj = self.clean_point_cloud(pts_obj)
            # get bbox and sphere centers
            min_coord, max_coord = compute_bounding_box(pts_obj)
            sphere_centers = fill_bounding_box_with_spheres(min_coord, max_coord, self.cfg.cover_radius)
            min_coord = min_coord.to(self.device)
            max_coord = max_coord.to(self.device)
            sphere_centers = sphere_centers.to(self.device)
            
            # filter out points that do not belong to the object
            pts_inlier = self.get_inlier_pts(sphere_centers, self.obj_mask_dict[obj], return_mask=False, threshold=self.cfg.depth_threshold)
            # filter out points under the ground plane
            mask_remove = pts_inlier[:, 2] < -self.plane_h 
            pts_inlier = pts_inlier[~mask_remove]
            # save the sphere centers and radius
            self.obj_sphere_centers[obj] = pts_inlier
            self.obj_sphere_means[obj] = torch.mean(self.obj_sphere_centers[obj], dim=0)
            self.obj_radius[obj] = torch.max(torch.norm(self.obj_sphere_centers[obj] - self.obj_sphere_means[obj], dim=1))

            # save the particles
            self.obj_particles_dict[obj] = pts_inlier.clone().detach()
            
        # visualize the sphere centers
        if self.cfg.vis_init_particles:
            vis_pts_list = []   
            for obj in self.cfg.obj_list:
                pts_obj = self.obj_sphere_centers[obj]
                vis_pts_list.append(pts_obj.cpu().numpy())
                
            pcl_ground = torch.cat(self.obj_pcl_dict["Ground"], dim=0)
            pcl_ground = down_sample_pcl(pcl_ground, num_points=10000)
            obj_mean = torch.mean(pts_obj, dim=0)
            d = torch.norm(pcl_ground - obj_mean, dim=1) 
            vis_pts_list.append(pcl_ground[d <= 0.3])
            polyscope_vis_pcl(vis_pts_list, radius=0.004)
        logging.info(f"Sphere centers are extracted.")
    

    # ========================================================================================== #
    # ============================== Train 3DGS for Particles ================================== #
    # ========================================================================================== #
    def build_particles(self):
        """
        Build the particles for the rigid body.
        """
        # get object point clouds, create splats and optimizers
        pts = []
        rgbs = []
        labels = []
        for i, obj in enumerate(self.cfg.obj_list):
            pts.append(self.obj_sphere_centers[obj])
            # assign color of each Gaussian to the color of its nearest neighbor in the segmented object point cloud
            rgb = assign_nearest_neighbor_properties(
                self.obj_sphere_centers[obj], 
                torch.cat(self.obj_pcl_dict[obj][:]), 
                torch.cat(self.obj_pcl_rgb_dict[obj][:])
            )
            rgbs.append(rgb)
            labels.extend([i] * rgb.shape[0])

        self.splats_particles, self.optimizers = create_rigid_object_splats_with_optimizers(
            pcl=torch.cat(pts, dim=0),
            rgb=torch.cat(rgbs, dim=0),
            cfg=self.cfg,
            gaussians=False,
        )
        # self.particle_labels = torch.tensor(labels, device=self.device)
        self.splats_particles["labels"] = torch.tensor(labels, device=self.device, dtype=torch.float32, requires_grad=False)
        self.optimizers["labels"] = torch.optim.Adam(
            [{"params": self.splats_particles["labels"], "lr": 0.0, "name": "labels"}],
            eps=1e-10,
            betas=(0.9, 0.999),
        )

        # initialize the density control strategy
        self.strategy = DefaultStrategy(
            refine_start_iter=self.cfg.refine_start_iter, 
            refine_stop_iter=self.cfg.refine_stop_iter, 
            refine_every=self.cfg.refine_every,
            reset_every=self.cfg.reset_every,
            prune_opa=self.cfg.prune_opa,
        )

        # run 3DGS optimization
        self.train_gs_particles()
    
    
    # ========================================================================================== #
    # ============================== Train 3DGS for Gaussians ================================== #
    # ========================================================================================== #
    def build_gaussians(self, ground=False):
        """
        Train 3DGS for the rigid body gaussians.
        """
        # get initial gaussians
        pts = []
        rgbs = []
        labels = []
        
        # set parameters
        if ground:
            obj_list = ["Ground"]
            n_gaussians = 15000
            refine_start_iter = 200
            refine_stop_iter = 800
        else:
            obj_list = self.cfg.obj_list
            n_gaussians = self.cfg.num_gaussians
            refine_start_iter = self.cfg.refine_start_iter
            refine_stop_iter = self.cfg.refine_stop_iter
            
        for i, obj in enumerate(obj_list):
            pts_obj = torch.cat(self.obj_pcl_dict[obj][:])
            rgb_obj = torch.cat(self.obj_pcl_rgb_dict[obj][:])
            # down sample the point cloud
            pts_obj, rgb_obj = down_sample_pcl(pts_obj, rgb_obj, num_points=n_gaussians)
            pts.append(pts_obj)
            rgbs.append(rgb_obj)
            labels.extend([i] * rgb_obj.shape[0])
        
        # initialize gaussians, optimizers and startegy
        self.splats_gaussians, self.optimizers = create_rigid_object_splats_with_optimizers(
            pcl=torch.cat(pts, dim=0),
            rgb=torch.cat(rgbs, dim=0),
            cfg=self.cfg,
            gaussians=True,
        )
        
        self.strategy = DefaultStrategy(
            refine_start_iter=refine_start_iter, 
            refine_stop_iter=refine_stop_iter, 
            refine_every=self.cfg.refine_every,
            reset_every=self.cfg.reset_every,
            prune_opa=self.cfg.prune_opa,
        )
        
        self.train_gs_gaussians(ground=ground)
    
    
    # ========================================================================================== #
    # ============================== Bond 3DGS with Particles ================================== #
    # ========================================================================================== #
    def assign_gaussians_to_particles(self):
        """
        Assign gaussians to particles based on nearest neighbor search.
        Assign Bonds for each object.
        """
        # Get all particles and their object labels
        all_particles = []
        particle_object_labels = []
        
        for i, obj in enumerate(self.cfg.obj_list):
            particles = self.obj_particles_dict[obj]
            all_particles.append(particles)
            particle_object_labels.extend([i] * particles.shape[0])
        
        all_particles = torch.cat(all_particles, dim=0)  # (N_particles, 3)
        particle_object_labels = torch.tensor(particle_object_labels, device=self.device)  # (N_particles,)
        
        # Get gaussian positions
        pts_gaussians = self.splats_gaussians["means"].detach().clone()  # (N_gaussians, 3)
        
        # Find closest particle for each gaussian
        # knn returns (distances, indices) where indices are the particle indices
        _, closest_particle_indices = knn(all_particles, pts_gaussians, k=1, num_workers=10)
        closest_particle_indices = closest_particle_indices.squeeze()  # (N_gaussians,)
        
        # Get object assignments based on closest particles
        gaussian_object_assignments = particle_object_labels[closest_particle_indices]  # (N_gaussians,)
        
        # Initialize dictionaries to store gaussians for each object
        self.obj_gaussians_means_dict = {}
        self.obj_gaussians_quats_dict = {}
        self.obj_gaussians_colors_dict = {}
        self.obj_gaussians_opacities_dict = {}
        self.obj_gaussians_scales_dict = {}
        self.obj_gaussians_assign_idx_dict = {}
        
        # Separate gaussians by object
        for i, obj in enumerate(self.cfg.obj_list):
            obj_gaussian_mask = (gaussian_object_assignments == i)
            obj_gaussian_indices = torch.where(obj_gaussian_mask)[0]
            
            # Store gaussians for this object
            self.obj_gaussians_means_dict[obj] = self.splats_gaussians["means"][obj_gaussian_indices].detach()
            self.obj_gaussians_quats_dict[obj] = F.normalize(self.splats_gaussians["quats"][obj_gaussian_indices]).detach()
            self.obj_gaussians_colors_dict[obj] = torch.sigmoid(self.splats_gaussians["colors"][obj_gaussian_indices]).detach()
            self.obj_gaussians_opacities_dict[obj] = torch.sigmoid(self.splats_gaussians["opacities"][obj_gaussian_indices]).detach()
            self.obj_gaussians_scales_dict[obj] = torch.exp(self.splats_gaussians["scales"][obj_gaussian_indices]).detach()
            
            # Now assign gaussians to specific particles within this object
            obj_gaussians_means = self.obj_gaussians_means_dict[obj]
            obj_particles = self.obj_particles_dict[obj]
            
            if obj_gaussians_means.shape[0] > 0 and obj_particles.shape[0] > 0:
                # Find closest particle within this object for each gaussian
                _, gaussian_to_particle_assignments = knn(obj_particles, obj_gaussians_means, k=1, num_workers=10)
                self.obj_gaussians_assign_idx_dict[obj] = gaussian_to_particle_assignments.squeeze().to(dtype=torch.int16)
            else:
                # Handle edge case where no gaussians or particles exist for this object
                self.obj_gaussians_assign_idx_dict[obj] = torch.empty(0, dtype=torch.int16, device=self.device)
            
            logging.info(f"Object: {obj}")
            logging.info(f"Number of Gaussians: {self.obj_gaussians_means_dict[obj].shape[0]}")
            logging.info(f"Number of Particles: {self.obj_particles_dict[obj].shape[0]}")

    
    # ========================================================================================== #
    # ================================ Convert to body frame =================================== #
    # ========================================================================================== #
    def convert_to_body_frame(self):
        """
        Convert the particles and gaussians to the body frame of each rigid body.
        """
        self.obj_particles_pose_dict = {}
        for i, obj in enumerate(self.cfg.obj_list):
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(self.obj_particles_dict[obj].cpu().numpy())
            obb: o3d.geometry.OrientedBoundingBox = pc.get_minimal_oriented_bounding_box()
            mat_obj = np.eye(4, dtype=np.float32)
            mat_obj[:3, :3] = obb.R
            # mat_obj[:3, 3] = obb.get_center()
            mat_obj[:3, 3] = np.mean(self.obj_particles_dict[obj].cpu().numpy(), axis=0)  
            mat_obj_inv = np.linalg.inv(mat_obj)
            mat_obj_inv_torch = torch.tensor(mat_obj_inv, device=self.device)

            particles_means = self.obj_particles_dict[obj]
            gaussians_means = self.obj_gaussians_means_dict[obj]
            gaussians_quats = self.obj_gaussians_quats_dict[obj]

            # transform particle means, sdf direction and gaussians
            particles_means = (mat_obj_inv_torch[:3, :3] @ particles_means.T + mat_obj_inv_torch[:3, 3:4]).T
            gaussians_means = (mat_obj_inv_torch[:3, :3] @ gaussians_means.T + mat_obj_inv_torch[:3, 3:4]).T

            self.obj_particles_dict[obj] = particles_means
            self.obj_gaussians_means_dict[obj] = gaussians_means
            
            # transform quaternions
            quat_np = gaussians_quats.cpu().numpy()
            quat_np_mat = R.from_quat(quat_np, scalar_first=True).as_matrix()  # type:ignore # (N, 3, 3)
            quat_np_mat = mat_obj_inv[:3, :3] @ quat_np_mat  
            quat_np = R.from_matrix(quat_np_mat).as_quat(scalar_first=True) # type:ignore # (N, 4)
            
            self.obj_gaussians_quats_dict[obj] = torch.from_numpy(quat_np).to(self.device).type_as(gaussians_quats)
            
            # save the pose
            self.obj_particles_pose_dict[obj] = mat_obj
            
            
    # ========================================================================================== #
    # ============================== Save all the information ================================== #
    # ========================================================================================== #
    def save_rigid_bodies_and_ground(self):
        """
        Save the rigid body and ground.
        """
        # save the rigid body
        for i, obj in enumerate(self.cfg.obj_list):
            # save the particles
            particles = Particles(
                means=self.obj_particles_dict[obj].tolist(),
                radii=[self.cfg.particle_radius] * self.obj_particles_dict[obj].shape[0],
            )
            # save the gaussians
            gaussians = Gaussians(
                means=self.obj_gaussians_means_dict[obj].tolist(),
                quats=self.obj_gaussians_quats_dict[obj].tolist(),
                colors=self.obj_gaussians_colors_dict[obj].tolist(),
                opacities=self.obj_gaussians_opacities_dict[obj].tolist(),
                scales=self.obj_gaussians_scales_dict[obj].tolist(),
            )
            # save the prompts
            prompts = {}
            prompt_obj = self.prompt_dict[obj]
            for view_idx, prompt_data in prompt_obj.items():
                prompts[view_idx] = SegPrompt(
                    uvs=prompt_data["points"],
                    labels=prompt_data["labels"],
                    bbox=prompt_data["bbox"]
                )
            # save the rigid body
            rigid_body = RigidBody(
                name=obj,
                particles=particles,
                gaussians=gaussians,
                pose=self.obj_particles_pose_dict[obj].tolist(), 
                gs_particle_indices=self.obj_gaussians_assign_idx_dict[obj].tolist(),
                prompts=prompts,
            )
            rigid_body.save(exp_name=self.cfg.exp_name)
        logging.info(f"Rigid bodies are saved.")
        
        # save the ground plane
        ground = Ground(
            normal=self.plane_normal.tolist(),
            h=self.plane_h.item(),
            gaussians=self.ground_gaussians
        )
        ground.save(exp_name=self.cfg.exp_name)
        logging.info(f"Ground plane is saved.")
    
            
    # ------------------------------------ auxiliary functions ------------------------------------ #
    def clean_point_cloud(self, pts, n_pts=30000):
        """
        Clean the point cloud by removing outliers.
        """
        # down sample the point cloud to n_pts
        N = pts.shape[0]
        if N <= n_pts:
            indices = torch.arange(N, device=pts.device)
        else:
            indices = torch.randperm(N, device=pts.device)[:n_pts]
        pts = pts[indices]  # (n_pts, 3)
        
        # clean the point cloud using open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.cpu().numpy())
        pcd_1, _ = pcd.remove_radius_outlier(nb_points=35, radius=0.03)
        pcd_cleaned, _ = pcd_1.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        return torch.tensor(np.asarray(pcd_cleaned.points), device=self.device, dtype=torch.float32)
    
    
    def get_inlier_pts(self, pts, masks, return_mask=False, threshold=0.008):
        """
        Get the inlier points from the point cloud based on the depth.
        """
        mask_obj_img = torch.zeros(pts.shape[0], dtype=torch.int, device=self.device)
        mask_obj_depth = torch.zeros(pts.shape[0], dtype=torch.int, device=self.device)
        for cam_idx in range(self.n_views):
            # check for sphere centers
            us, vs = self.cam.world_to_image(pts, cam_idx)
            us = torch.round(us).long()
            vs = torch.round(vs).long()
            # check whether the points are in the object mask
            result = is_point_in_mask(masks[cam_idx].squeeze(-1), us, vs)
            mask_obj_img += result
            # check whether the points are in front of the depth value
            us = torch.clamp(us, 0, self.cam.W - 1)  # Clamp us to [0, width-1]
            vs = torch.clamp(vs, 0, self.cam.H - 1)  # Clamp vs to [0, height-1]
            depth_value = self.depths[cam_idx][vs, us].squeeze(-1) # get the depth value 
            cam_coords = self.cam.world_to_cam(pts, cam_idx)  # (N, 3)
            z_pts = cam_coords[:, 2]  # (N, )
            mask_obj_depth += (z_pts >= depth_value - threshold).int()  # mask_obj: 0 for inliers, 1 for outliers

        mask_obj_img = mask_obj_img >= (self.n_views - self.cfg.n_mask_tol)  # Convert to boolean mask
        mask_obj_depth = mask_obj_depth >= (self.n_views - self.cfg.n_depth_tol)  # Convert to boolean mask
        mask_obj = mask_obj_img & mask_obj_depth  # (N, )
        inlier_pts = pts[mask_obj]  # (N, 3)
        
        if return_mask:
            return inlier_pts, mask_obj
        else:
            return inlier_pts
    
    # ------------------------------------ Training GS ------------------------------------ #
    def train_gs_particles(self):
        """
        Train the GS particles. Only used in embodied physics.
        """
        start_time = time.time()
        
        init_pts = self.splats_particles["means"].detach().clone()
        inv_min_scale = torch.log(torch.tensor(self.cfg.min_scale, device=self.device))
        inv_max_scale = torch.log(torch.tensor(self.cfg.max_scale, device=self.device))
        
        Ks = self.cam.Ks
        viewmats = self.cam.viewmats
        gt_pixels = self.gt_rgbs
        gt_masks = torch.stack(self.gt_mask_list) # type: ignore
        
        if self.cfg.particle_batch:
            batch_size = self.n_batch
        else:
            batch_size = 1
        n_views = self.cam.viewmats.shape[0]
        max_iter = self.cfg.particle_max_iter
        backgrounds = torch.rand((max_iter, 3)).float().cuda()

        # Initialize the strategy state
        strategy_state = self.strategy.initialize_state(scene_scale=self.cfg.scene_scale)
        
        # run training
        for step in tqdm(range(max_iter)):
            if not self.cfg.particle_batch:
                idx = torch.tensor([step % n_views])
                viewmats = self.cam.viewmats[idx, ...]
                Ks = self.cam.Ks[idx, ...]
                pixels = gt_pixels[idx, ...]
                masks = gt_masks[idx, ...]
            else:
                pixels = gt_pixels
                masks = gt_masks
            
            background = backgrounds[step]
            indicies = (masks == 1.0).squeeze(-1)
            pixels[indicies, :] = background
                
            # Forward pass
            render_image, _, info = rasterization(
                means=self.splats_particles["means"],
                quats=F.normalize(self.splats_particles["quats"]),
                scales=torch.exp(self.splats_particles["scales"]),
                opacities=torch.sigmoid(self.splats_particles["opacities"]),
                colors=torch.sigmoid(self.splats_particles["colors"]),
                viewmats=viewmats,  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=self.cam.W,
                height=self.cam.H,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                backgrounds=background.reshape(1, 3).repeat(batch_size, 1),
                packed=False,
                render_mode="RGB",
            )
            
            # Loss calculation
            l1loss = F.l1_loss(render_image, pixels)
            ssimloss = 1.0 - fused_ssim(
                render_image.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )

            loss = l1loss * (1.0 - self.cfg.ssim_lambda) \
                   + ssimloss * self.cfg.ssim_lambda 
            
            # Backward pass
            loss.backward()
            
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # clamp the scales
            self.splats_particles["scales"].detach().clamp_(inv_min_scale, inv_max_scale)

            # Post-backward step
            # self.strategy.step_post_backward(self.splats_particles, self.optimizers, strategy_state, step, info)
            
            # solve collisions
            self.solve_collisions()
            
            # # visualization
            if self.cfg.vis_init_particles and step % 500 == 0:
                vis_render_gs_cam(cam=self.cam, splats=self.splats_particles)
            
        
        if self.cfg.vis_init_particles:
            vis_render_gs_cam(cam=self.cam, splats=self.splats_particles)
            # polyscope_vis_pcl([self.splats_particles["means"].detach().cpu().numpy(), torch.cat(self.obj_pcl_dict["Ground"], dim=0).cpu().numpy()])
        
        # final check if the particles are within the object mask
        pts = self.splats_particles["means"].detach()
        keep_mask = torch.zeros(self.splats_particles["means"].shape[0], dtype=torch.bool, device=self.device)
        for obj in self.cfg.obj_list:
            _, obj_mask = self.get_inlier_pts(pts, self.obj_mask_dict[obj], return_mask=True, threshold=self.cfg.depth_threshold)
            keep_mask += obj_mask
        # self.particle_labels[~keep_mask] = -1.0 # assign -1 to the points outside the object mask

        labels = self.splats_particles["labels"].detach().clone()
        labels[~keep_mask] = -1.0
        
        # remove particles under the table
        mask_remove = pts[:, 2] < -self.plane_h - self.cfg.particle_radius
        labels = self.splats_particles["labels"].detach().clone()
        labels[mask_remove] = -1.0
        self.splats_particles["labels"] = labels

        self.obj_particles_dict = {}
        for i, obj in enumerate(self.cfg.obj_list):
            # self.obj_particles_dict[obj] = pts[self.particle_labels == i]
            self.obj_particles_dict[obj] = pts[self.splats_particles["labels"] == i]
            
        if self.cfg.vis_init_particles:
            vis_list = []
            for obj in self.cfg.obj_list:
                vis_list.append(self.obj_particles_dict[obj].detach().cpu().numpy())
            vis_list.append(self.ground_pts.cpu().numpy())
            # vis_list.append(init_pts.cpu().numpy())
            if self.cfg.ps_vis:
                polyscope_vis_pcl(vis_list, radius=0.004)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # polyscope_vis_pcl([self.splats_particles["means"].detach().cpu().numpy(), torch.cat(self.obj_pcl_dict["Ground"], dim=0).cpu().numpy()])
        # vis_pcl_obj = self.splats_particles["means"].detach()
        vis_pcl_obj = self.splats_particles["means"].detach()[self.splats_particles["labels"] >= 0]
        obj_mean = torch.mean(vis_pcl_obj, dim=0)
        pcl_ground = torch.cat(self.obj_pcl_dict["Ground"], dim=0)
        pcl_ground = down_sample_pcl(pcl_ground, num_points=10000)
        d = torch.norm(pcl_ground - obj_mean, dim=1) 
        vis_pcl_ground = pcl_ground[d <= 0.3]
        
        # polyscope_vis_pcl([self.splats_particles["means"].detach().cpu().numpy(), vis_pcl_ground.cpu().numpy()]) # type: ignore
        polyscope_vis_pcl([vis_pcl_obj.cpu().numpy(), vis_pcl_ground.cpu().numpy()], radius=0.004) # type: ignore
        
        logging.info(f"Time elapsed: {elapsed_time:.4f} seconds")
        # logging.info(f"Particles are trained. {torch.sum(self.particle_labels >= 0)} particles in total.")
        logging.info(f"Particles are trained. {torch.sum(self.splats_particles['labels'] >= 0)} particles in total.")

    
    def train_gs_gaussians(self, ground=False):
        """
        Train the GS for the gaussians.
        """
        start_time = time.time()
        
        # Initialize the strategy state
        strategy_state = self.strategy.initialize_state(scene_scale=self.cfg.scene_scale)
        
        Ks = self.cam.Ks
        viewmats = self.cam.viewmats
        gt_pixels = torch.clone(self.gt_rgbs)
        gt_depths = torch.clone(self.gt_depths)
        if self.cfg.use_rgbd:
            render_mode = "RGB+D"
        else:
            render_mode = "RGB"
        if ground:
            gt_masks = torch.stack(self.gt_mask_ground_list) # type: ignore
            max_iter = 1000
        else:
            gt_masks = torch.stack(self.gt_mask_list) # type: ignore
            max_iter = self.cfg.gaussian_max_iter
        gt_depths[gt_masks == 1.0] = 0.0  # set background depth to 0.0
            
        if self.cfg.gaussian_batch:
            batch_size = self.cam.viewmats.shape[0]
        else:
            batch_size = 1
        n_views = self.cam.viewmats.shape[0]
        backgrounds = torch.rand((max_iter, 3)).float().cuda()
        
        # run training
        for step in tqdm(range(max_iter)):
            if not self.cfg.gaussian_batch:
                idx = torch.tensor([step % n_views])
                viewmats = self.cam.viewmats[idx, ...]
                Ks = self.cam.Ks[idx, ...]
                pixels = gt_pixels[idx, ...]
                masks = gt_masks[idx, ...]
                depths = gt_depths[idx, ...]
            else:
                pixels = gt_pixels
                masks = gt_masks
                depths = gt_depths
            
            background = backgrounds[step]
            indicies = (masks == 1.0).squeeze(-1)
            pixels[indicies, :] = background
                
            # Forward pass
            renders, render_alpha, info = rasterization(
                means=self.splats_gaussians["means"],
                quats=F.normalize(self.splats_gaussians["quats"]),
                scales=torch.exp(self.splats_gaussians["scales"]),
                opacities=torch.sigmoid(self.splats_gaussians["opacities"]),
                colors=torch.sigmoid(self.splats_gaussians["colors"]),
                viewmats=viewmats,  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=self.cam.W,
                height=self.cam.H,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                backgrounds=background.reshape(1, 3).repeat(batch_size, 1),
                packed=False,
                render_mode=render_mode,
            )
            
            # Pre-backward step
            self.strategy.step_pre_backward(self.splats_gaussians, self.optimizers, strategy_state, step, info)
            
            if renders.shape[-1] == 4:
                render_images, render_depths = renders[..., 0:3], renders[..., 3:4]
            else:
                render_images, render_depths = renders, None
            
            # Loss calculation
            l1loss = F.mse_loss(render_images, pixels)
            ssimloss = 1.0 - fused_ssim(
                render_images.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )

            loss = l1loss * (1.0 - self.cfg.ssim_lambda) \
                   + ssimloss * self.cfg.ssim_lambda 
            
            if self.cfg.use_rgbd and render_depths is not None:
                depth_loss = F.l1_loss(render_depths, depths)
                loss += depth_loss * self.cfg.depth_lambda
            
            # Backward pass
            loss.backward()
            
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
            # Post-backward step
            self.strategy.step_post_backward(self.splats_gaussians, self.optimizers, strategy_state, step, info)
            
            # visualization
            if self.cfg.vis_init_gaussians and step % 500 == 0:
                vis_render_gs_cam(cam=self.cam, splats=self.splats_gaussians)
            
        if self.cfg.vis_init_gaussians:
            vis_render_gs_cam(cam=self.cam, splats=self.splats_gaussians)
            # polyscope_vis_pcl([self.splats_particles["means"].detach().cpu().numpy(), torch.cat(self.obj_pcl_dict["Ground"], dim=0).cpu().numpy()])
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        n_pts = self.splats_gaussians["means"].shape[0]
        logging.info(f"{n_pts} Gaussians in total.")
        logging.info(f"Time elapsed: {elapsed_time:.4f} seconds")
        
        
    def solve_collisions(self):
        """
        Solve the collisions for the particles.
        """
        # solve collisions
        with torch.no_grad():
            pts = self.splats_particles["means"].detach()
            pts_warp = wp.from_torch(pts, dtype=wp.vec3)
            ground_n_wp = wp.from_torch(self.plane_normal)
            
            radii = self.cfg.particle_radius
            for _ in range(self.cfg.particle_collision_max_iter):
                grid = wp.HashGrid(128, 128, 128)
                grid.build(pts_warp, radii * 5.0)
                wp.launch(
                    kernel=solve_particle_ground_contacts,
                    dim=pts_warp.shape[0],
                    inputs=[
                        pts_warp,
                        radii,
                        ground_n_wp,
                        self.plane_h,
                        0.2,
                    ],
                )
                wp.launch(
                    dim=pts_warp.shape[0],
                    kernel=solve_particle_particle_contacts,
                    inputs=[
                        grid.id,
                        pts_warp,
                        radii,
                        0.001,
                        0.2,
                    ],
                )
            self.splats_particles["means"] = wp.to_torch(pts_warp)
        
        
# -------------------------- Warp functions for GS optimization -------------------------- #
@wp.kernel
def solve_particle_ground_contacts(
    particle_x: wp.array(dtype=wp.vec3), # type: ignore
    particle_radius: float,
    ground_n: wp.array(dtype=float), # type: ignore
    gound_h: float,
    relaxation: float = 0.2,
):
    tid = wp.tid()
    x = particle_x[tid]
    n = wp.vec3(ground_n[0], ground_n[1], ground_n[2])
    
    c = wp.min(wp.dot(n, x) + gound_h - particle_radius, 0.0)
    
    if c > 0.0:
        return

    delta = n * c

    wp.atomic_add(particle_x, tid, -delta * relaxation)


@wp.kernel
def solve_particle_particle_contacts(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3), # type: ignore
    particle_radius: float,
    k_cohesion: float,
    relaxation: float = 0.2,
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid) # type: ignore
    if i == -1:
        # hash grid has not been built yet
        return

    x = particle_x[i]
    
    # particle contact
    query = wp.hash_grid_query(grid, x, particle_radius + k_cohesion) # type: ignore
    index = int(0)

    delta = wp.vec3(0.0)

    while wp.hash_grid_query_next(query, index): # type: ignore
        # compute distance to point
        n = x - particle_x[index]
        d = wp.length(n) + 1e-10
        err = d - particle_radius * 2.0
        
        if err <= k_cohesion:
            n = n / d
            delta += n * err * 0.5
    
    wp.atomic_add(particle_x, i, -delta * relaxation)
    