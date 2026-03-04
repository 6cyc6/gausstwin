import os
import cv2
import json
import click
import torch
import numpy as np

from scipy.spatial.transform import Rotation as R
from gausstwin.utils.path_utils import get_builder_cfg_path
from gausstwin.cfg.builder.builder_cfg import RigidBodyConfig, RopeConfig


@click.command()
@click.option(
    "--obj",
    type=click.Choice(['rigid', 'rope'], case_sensitive=False),
    help="Object type: 'rigid' or 'rope'",
    show_default=True
)
@click.option(
    "--vis", 
    type=bool,
    default=True,
    help="Whether to visualize the images",
    show_default=True
)
def save_cam_cfg(obj, vis):
    # load path and configuration
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Select config file based on object type
    if obj.lower() == "rigid":
        cfg_path = get_builder_cfg_path("rigid_body_builder.yaml")
        cfg = RigidBodyConfig.from_yaml(cfg_path)
    elif obj.lower() == "rope":
        cfg_path = get_builder_cfg_path("rope_builder.yaml")
        cfg = RopeConfig.from_yaml(cfg_path)
    else:
        raise ValueError(f"Invalid object type: {obj}. Must be 'rigid' or 'rope'")
    
    
    save_dir = os.path.join(cfg.data_dir, f"trajectory_{cfg.exp_name}")
    
    print(f"Processing {obj} object configuration...")
    print(f"Using config file: {cfg_path}")
    print(f"Save directory: {save_dir}")

    # ----------------------------- for fixed cameras ----------------------------- #
    for i in range(4):
        # get camera index
        cam_id = i + 1
        if i == 0:
            fix_idx = cfg.fix_idx_1
        elif i == 1:
            fix_idx = cfg.fix_idx_2
        elif i == 2:
            fix_idx = cfg.fix_idx_3
        else:
            fix_idx = cfg.fix_idx_4
        # load path
        cam_ext_path = f"{save_dir}/camera/static_{cam_id}/extrinsic/extrinsic.npy"
        cam_int_path = f"{save_dir}/camera/static_{cam_id}/intrinsic.npy"
        img_path = f"{save_dir}/camera/static_{cam_id}/rgb/{fix_idx}.png"
        
        # vis image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if vis:
            cv2.imshow('Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        h, w, _ = img.shape
        print(f"Image shape: {img.shape}, Height: {h}, Width: {w}")

        # load parameters
        # get extrinsics
        cam_ext_mat = np.load(cam_ext_path)
        cam_ext_mat = np.linalg.inv(cam_ext_mat)  # Invert the extrinsic matrix to get camera to world transformation
        # Extract rotation and translation
        rotation_matrix = cam_ext_mat[:3, :3]
        translation = cam_ext_mat[:3, 3]
        # Convert rotation matrix to quaternion
        quat = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w]
        
        # get intrinsics
        cam_int_mat = np.load(cam_int_path)

        camera_cfg = {
            "pos": translation.tolist(),
            "quat": np.roll(quat, 1).tolist(), # convert to [w, x, y, z]
            "k": cam_int_mat.tolist(),
            "h": h,
            "w": w,
        }
        
        print(cam_ext_mat)
        
        # save to json
        with open(f"{save_dir}/camera_{cam_id}.json", "w") as f:
            json.dump(camera_cfg, f, indent=4)
        
        print(f"Camera {cam_id} configuration saved.")


if __name__ == "__main__":
    save_cam_cfg()
    