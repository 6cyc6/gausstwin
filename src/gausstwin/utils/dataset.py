import os
import cv2
import json
import glob
import torch
import numpy as np

from gausstwin.utils.math_utils import make_pose, matrix_from_quat


class Dataset(torch.utils.data.Dataset):
    def __init__(self, 
            data_dir, 
            device: str = "cuda:0"
        ):
        self.data_dir = data_dir
        self.device = device
        
        # load dataset
        self.load_data()


    def load_data(self):
        color_paths = sorted(glob.glob(os.path.join(self.data_dir, 'rgb', '*.png')))
        depth_paths = sorted(glob.glob(os.path.join(self.data_dir, 'depth', '*.png')))
        mask_paths = sorted(glob.glob(os.path.join(self.data_dir, 'mask', '*.png')))

        # Load all images into torch tensors at initialization
        colors, depths, masks = [], [], []
        
        for i, (color_path, depth_path, mask_path) in enumerate(zip(color_paths, depth_paths, mask_paths)):
            # Load color image
            color = cv2.imread(color_path)
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            colors.append(torch.from_numpy(color).float() / 255.0)  # Normalize to [0, 1]
            
            # Load mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_bool = mask > 127  # binary mask
            masks.append(torch.from_numpy(mask_bool.astype(np.uint8)))
            
            # Load depth image and apply mask
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000.0  # Assuming depth is stored in millimeters
            depth = depth * mask_bool  # Set depth to 0 outside the mask
            depths.append(torch.from_numpy(depth).float())
            
        # Stack into tensors: (N, H, W, C) for colors, (N, H, W) for depth/mask
        self.colors = torch.stack(colors, dim=0)
        self.depths = torch.stack(depths, dim=0)
        self.masks = torch.stack(masks, dim=0)
        
        print(f"Loaded images shape - Colors: {self.colors.shape}, Depths: {self.depths.shape}, Masks: {self.masks.shape}")

        # Load camera poses
        cam_poses_path = os.path.join(self.data_dir, "camera_poses.json")
        with open(cam_poses_path, "r") as f:
            cam_poses = json.load(f)

        # cam_poses is a list of {"pos": [x,y,z], "quat": [x,y,z,w]}
        self.pos = torch.tensor(
            [p["pos"] for p in cam_poses], dtype=torch.float32, device=self.device
        )
        self.quat = torch.tensor(
            [p["quat"] for p in cam_poses], dtype=torch.float32, device=self.device
        )
        
        print(f"Loaded poses - Pos: {self.pos.shape}, Quat: {self.quat.shape}")
        print("-------------------- Dataset loading complete --------------------")

    
    def get_init_data(self, n_views=8):
        # Get equally spaced indices across the dataset
        total_frames = len(self.colors)
        indices = torch.linspace(0, total_frames - 1, n_views).long()
        
        # get camera poses in world frame
        pos = self.pos[indices]
        quat = self.quat[indices]
        # pose_mat = make_pose(pos, matrix_from_quat(quat)) # get cam->world
        pose_mat = torch.linalg.inv(make_pose(pos, matrix_from_quat(quat))) # get world->cam
        
        return {
            "colors": self.colors[indices],
            "depths": self.depths[indices],
            "masks": self.masks[indices],
            "poses": pose_mat,
        }


    def __len__(self):
        return len(self.colors)
    

    def __getitem__(self, idx):
        pos = self.pos[idx]
        quat = self.quat[idx]
        # pose_mat = make_pose(pos, matrix_from_quat(quat)) # get cam->world
        pose_mat = torch.linalg.inv(make_pose(pos, matrix_from_quat(quat))) # get world->cam
            
        return {
            "colors": self.colors[idx],
            "depths": self.depths[idx],
            "masks": self.masks[idx],
            "poses": pose_mat,
        }
        