import json
import torch

from typing import List
from gausstwin.utils.vis_utils import to_torch
from gausstwin.utils.math_utils import make_pose, matrix_from_quat, unproject_depth, transform_points, quat_from_matrix, unmake_pose


class Camera:
    def __init__(
        self, 
        cam_cfg_json_list: List,
        device: str = "cuda:0"
    ):
        """
        Camera Class
        Ks: intrinsics
        positions: positions of camera in world frame
        quats: orientations of camera in world frame
        """
        self.device = device
        
        # load camera configuration
        self.load_camera_cfg(cam_cfg_json_list)
        
        # get cam2world
        self.cam2worlds = self.get_cam_to_world()
        
        # get world2cam
        self.worlds2cam = torch.linalg.inv(self.cam2worlds)
        
        # view matrices
        self.viewmats = torch.linalg.inv(self.cam2worlds)  # [C, 4, 4]
    
    
    def load_camera_cfg(self, cfg_json_list: List):
        """
        Load camera configuration from a list of json config file.
        """
        pos_list = []
        quat_list = []
        k_list = []
        for cfg_json in cfg_json_list:
            with open(cfg_json, "r") as f:
                cam_cfg = json.load(f)
            pos = torch.tensor(cam_cfg["pos"], device=self.device)        # Position (3,)
            quat = torch.tensor(cam_cfg["quat"], device=self.device)      # Orientation (4,)
            k = torch.tensor(cam_cfg["k"], device=self.device)            # Intrinsic matrix (3, 3)
            h = cam_cfg["h"]                                              # Image height
            w = cam_cfg["w"]                                              # Image width

            pos_list.append(pos)
            quat_list.append(quat)
            k_list.append(k)
        
        # convert to tensors
        self.positions = to_torch(pos_list, device=self.device)
        self.quats = to_torch(quat_list, device=self.device)
        self.Ks = to_torch(k_list, device=self.device)
        self.Ks_inv = torch.linalg.inv(self.Ks)
        
        self.n_cam = len(k_list)
        self.H = h
        self.W = w
    
        
    def get_cam_to_world(self) -> torch.Tensor:
        """
        Get the transformation matrix from camera to world frame.
        """

        return make_pose(self.positions, matrix_from_quat(self.quats))
    
    
    def world_to_image(self, pts, cam_idx):
        """
        Transformation world coordinates to image coordinates.
        """
        R = self.viewmats[cam_idx][:3, :3]
        t = self.viewmats[cam_idx][:3, 3]
        pts_cam = torch.matmul(pts, R.T) + t  # (N x 3) world to camera

        # # Multiply by the intrinsic matrix K (3x3)
        pts_img = torch.matmul(pts_cam, self.Ks[cam_idx].T)  # (N x 3) -> image coordinates
        
        # pts = torch.cat([pts, torch.ones(pts.shape[0], 1, dtype=torch.float32, device=self.device)], dim=1)
        # pts_img = self.Ks[cam_idx] @ self.R_ts[cam_idx][:3, :] @ pts.T
        # pts_img = pts_img.T
        
        # Divide by the Z component to get normalized pixel coordinates
        u = pts_img[:, 0] / pts_img[:, 2]
        v = pts_img[:, 1] / pts_img[:, 2]
        
        return u, v
    
    
    def image_to_world(self, depth, cam_idx):
        """
        Transformation image coordinates to world coordinates.
        """
        depth_cloud = unproject_depth(depth, self.Ks[cam_idx])
        # convert 3D points to world frame
        depth_cloud = transform_points(depth_cloud, self.positions[cam_idx], self.quats[cam_idx])
        
        return depth_cloud
    
    
    def world_to_cam(self, pts, cam_idx):
        """
        Transformation world coordinates to camera coordinates.
        """
        R = self.viewmats[cam_idx][:3, :3]
        t = self.viewmats[cam_idx][:3, 3]
        pts_cam = torch.matmul(pts, R.T) + t  # (N x 3) world to camera
        
        return pts_cam

    
    def batch_image_to_world(self, depths, concat=False):
        pts_world_list = [self.image_to_world(depths[i], i) for i in range(self.n_cam)]
        
        if concat:
            pts_world = torch.concat(pts_world_list, dim=0)
            
            # Randomly sample 20000 points if we have more than that
            if len(pts_world) > 20000:
                indices = torch.randperm(len(pts_world))[:20000]
                pts_world = pts_world[indices]
            return pts_world
        else:
            pts_list = []
            pts_center = torch.tensor([0.58, 0.0, 0.25]).to(self.device)
            # return pts_world
            for i in range(len(pts_world_list)):
                pts = pts_world_list[i]
                pts_means = torch.mean(pts, dim=0)
                pts = pts[torch.linalg.norm(pts - pts_means, dim=1) < 2.0]  # filter out points too far away
                pts = pts[torch.linalg.norm(pts - pts_center, dim=1) < 1.0]  # filter out points too far away
                if len(pts) > 20000:
                    indices = torch.randperm(len(pts))[:20000]
                    pts = pts[indices]
                pts_list.append(pts)
            return pts_list
    
    
    def image_seg_to_world(self, depth, mask, cam_idx):
        """
        Transformation segmented image coordinates to world coordinates.
        """
        # get uvs from mask
        # uv_coords = mask_to_uv(mask)
        mask = mask.bool()
        v, u = torch.nonzero(mask, as_tuple=True)  # Extract pixel coordinates

        uv_coords = torch.stack((u, v), dim=1)
        depth_values = depth[uv_coords[:, 1], uv_coords[:, 0], 0]  # (N,)
        
        # Create homogeneous image coordinates
        N = uv_coords.shape[0]
        ones = torch.ones(N, 1, device=uv_coords.device)
        pixel_coords = torch.cat((uv_coords.float(), ones), dim=1)  # (N,3)
        
        # get K inverse
        K_inv = self.Ks_inv[cam_idx]
        # get camera coordinates
        cam_coords = (K_inv @ pixel_coords.T).T * depth_values.unsqueeze(1)  # (N,3)

        # # get world coordinates from camera coordinates
        R = self.cam2worlds[cam_idx][:3, :3]
        t = self.cam2worlds[cam_idx][:3, 3]
        
        pts_world = (R @ cam_coords.T).T + t
        
        return pts_world

    
    def batch_image_seg_to_world_filtered(self, depths, masks, ground_h):
        """
        Convert segmented pixels from multiple cameras to world coordinates in batch.
        depth: (N_cam, H, W)
        mask:  (N_cam, H, W)
        """
        pts_world_list = [self.image_seg_to_world(depths[i], 1.0 - masks[i], i) for i in range(self.n_cam)]
        pts_world_filtered_list = [pts_world_list[i][torch.nonzero(pts_world_list[i][:, 2] > ground_h, as_tuple=True)[0]]
                                   for i in range(self.n_cam)]
        
        pts_all = torch.concat(pts_world_filtered_list, dim=0)
        
        # # Randomly sample 20000 points if we have more than that
        # if len(pts_all) > 20000:
        #     indices = torch.randperm(len(pts_all))[:20000]
        #     pts_all = pts_all[indices]
        
        pts_mean = torch.mean(pts_all, dim=0)
        
        # filter
        dists = torch.norm(pts_all - pts_mean, dim=1)  # (N,)
        threshold = 0.2  # adjust depending on your scale
        pts_filtered = pts_all[dists < threshold]
        pts_mean = torch.mean(pts_filtered, dim=0)

        return pts_filtered, pts_mean


class Camera_Sim(Camera):
    def __init__(
        self, 
        cam_cfg_json_list: List,
        device: str = "cuda:0"
    ):
        super().__init__(cam_cfg_json_list, device=device)
        self.set_parameters()


    def set_parameters(self):
        # Store original matrices
        self.Ks_orig = self.Ks.clone()
        self.cam2worlds_orig = self.cam2worlds.clone()
        
        # STEP 1: ONLY handle horizontal flip through intrinsics
        self.Ks[:, 0, 0] = -self.Ks[:, 0, 0]  # Negate fx (horizontal flip)
        self.Ks[:, 0, 2] = self.W - 1 - self.Ks[:, 0, 2]  # Flip cx
        self.Ks_inv = torch.linalg.inv(self.Ks)
        
        # STEP 2: Handle depth sign convention through extrinsics
        # This matrix flips X,Y,Z (like multiplying by -I in 3D)
        flip_xyz = torch.diag(torch.tensor([-1., -1., -1., 1.], device=self.device))
        flip_xyz = flip_xyz.unsqueeze(0).repeat(self.n_cam, 1, 1)

        # Apply once to all camera extrinsics
        self.cam2worlds = self.cam2worlds_orig @ flip_xyz

        # Update dependent matrices
        self.worlds2cam = torch.linalg.inv(self.cam2worlds)
        self.viewmats = self.worlds2cam

        # Update positions and quaternions for compatibility
        positions, rotations = unmake_pose(self.cam2worlds)
        quats = quat_from_matrix(rotations)
        self.positions = positions
        self.quats = quats


    def image_to_world_sim(self, depth, cam_idx):
        """
        Transformation image coordinates to world coordinates.
        """
        depth_cloud = unproject_depth(depth, self.Ks[cam_idx])
        # convert 3D points to world frame
        # depth_cloud = transform_points(depth_cloud, self.positions[cam_idx], self.quats[cam_idx])
        R = self.cam2worlds[cam_idx][:3, :3]
        t = self.cam2worlds[cam_idx][:3, 3]
        
        # Transform to world coordinates
        depth_cloud = (R @ depth_cloud.T).T + t
        
        return depth_cloud
    
