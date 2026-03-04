import os
import click
import torch
import imageio
import logging
import numpy as np
import polyscope as ps

from PIL import Image
from tqdm import tqdm
from gausstwin.cfg.domain import Ground
from gausstwin.camera.cam import Camera
from gausstwin.utils.load_utils import get_ground_json
from gausstwin.utils.path_utils import get_track_cfg_path
from gausstwin.segmentation.tracker_TAM import TAMTracker
from gausstwin.cfg.tracker.gs_track_cfg import GSTrackUnifiedConfig


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def setup_segmentation_tracker(cfg, data_dir):
    # Create config for first 3 cameras (EfficientTAM)
    cfg_efficient = type(cfg)(**{k: v for k, v in cfg.__dict__.items()})
    cfg_efficient.n_cams = 3
    
    # Setup EfficientTAM for first 3 cameras
    logging.info("Setting up EfficientTAM tracker for cameras 1-3...")
    efficient_tracker = TAMTracker(cfg_efficient, use_sam2=False)
    img_path_list_efficient = [os.path.join(data_dir, f"camera/static_{cam_idx + 1}/rgb/0.png") for cam_idx in range(3)]
    rgb_np_list_efficient = [np.array(Image.open(img_path).convert("RGB")) for img_path in img_path_list_efficient]
    efficient_tracker.load_first_frames(rgb_np_list_efficient)
    
    if len(cfg.obj_list) > 1:
        efficient_tracker.add_new_prompts_multi(vis=cfg.vis)
    else:
        for obj in cfg.obj_list:
            efficient_tracker.add_new_prompts(obj, vis=cfg.vis)
    
    return efficient_tracker


@click.command()
@click.option(
    "--obj",
    type=click.Choice(["rigid", "rope"]),
    default="rigid",
    help="Config type to load: 'rigid' for track_rigid.yaml, 'rope' for track_rope.yaml.",
)
def main(obj):
    cfg_path = get_track_cfg_path(f"track_{obj}.yaml")
    cfg = GSTrackUnifiedConfig.from_yaml(cfg_path)
    data_dir = os.path.join(cfg.data_dir, f"trajectory_{cfg.exp_name}")

    # Load ground height
    ground_data = get_ground_json(exp_name=cfg.exp_name)
    ground = Ground(**ground_data)
    ground_h = -ground.h + 0.006

    # Output directory for point clouds
    pcl_output_dir = os.path.join(data_dir, "pcl_output")
    os.makedirs(pcl_output_dir, exist_ok=True)
    
    # Output directory for evaluation masks (camera 4 only)
    eval_mask_dir = os.path.join(data_dir, "mask")
    os.makedirs(eval_mask_dir, exist_ok=True)

    logging.info(f"-------------------- Experiment: {cfg.exp_name} --------------------")
    logging.info(f"--------------------- Objects: {cfg.obj_list} ---------------------")
    logging.info(f"Ground height: {ground_h}")
    logging.info(f"Data directory: {data_dir}")
    logging.info(f"Output directory: {pcl_output_dir}")
    logging.info(f"Evaluation mask directory: {eval_mask_dir}")

    # Setup camera for first 3 views only (for point cloud)
    cam_cfg_list = []
    for i in range(3):  # Only first 3 cameras for point cloud
        cam_idx = i + 1
        f_name = os.path.join(data_dir, f"camera_{cam_idx}.json")
        cam_cfg_list.append(f_name)
    cam = Camera(cam_cfg_list)
    logging.info(f"{cam.viewmats.shape[0]} Cameras loaded for point cloud generation")

    # Setup segmentation trackers
    efficient_tracker = setup_segmentation_tracker(cfg, data_dir)

    # Parameters
    disp = cfg.disp
    max_idx = cfg.max_idx
    length = cfg.length
    n_points_per_frame = 2000

    logging.info(f"Processing {length} frames, disp={disp}, max_idx={max_idx}")

    # Collect all point clouds
    pcl_list_all = []

    for i in tqdm(range(length)):
        if i >= max_idx:
            idx = max_idx + disp
        else:
            idx = i + disp

        with torch.no_grad():
            # Load RGB images for first 3 cameras (EfficientTAM for point cloud)
            img_path_list_pcl = [os.path.join(data_dir, f"camera/static_{cam_idx + 1}/rgb/{idx}.png") for cam_idx in range(3)]
            rgb_np_list_pcl = [np.array(Image.open(img_path).convert("RGB")) for img_path in img_path_list_pcl]
            
            # Load depth for first 3 cameras only (for point cloud)
            depth_path_list = [os.path.join(data_dir, f"camera/static_{cam_idx + 1}/depth/{idx}.png") for cam_idx in range(3)]
            depth_np_list = [imageio.imread(depth_path).astype(np.float32) / 1000.0 for depth_path in depth_path_list]
            gt_depth_list = [torch.from_numpy(depth).unsqueeze(-1).to(device).float() for depth in depth_np_list]  # (3, H, W, 1)

            # Run EfficientTAM segmentation on first 3 cameras (for point cloud)
            gt_masks_pcl = efficient_tracker.track(rgb_np_list_pcl)
            gt_masks_list_pcl = [gt_masks_pcl[j, :, :].to(device) for j in range(3)]  # (3, H, W)

            # Get point cloud using first 3 cameras only
            if i <= max_idx:
                obj_pcl, _ = cam.batch_image_seg_to_world_filtered(gt_depth_list, gt_masks_list_pcl, ground_h)

                N = obj_pcl.shape[0]

                # Debug: visualize first frame
                if i == 0:
                    logging.info(f"First frame pcl has {N} points")
                    if N > 0:
                        ps.init()
                        ps.set_up_dir("z_up")
                        ps.set_ground_plane_mode("shadow_only")
                        ps.register_point_cloud("obj_pcl", obj_pcl.cpu().numpy(), radius=0.002)
                        ps.show()

                # Downsample to n_points_per_frame
                if N > n_points_per_frame:
                    indices = torch.randperm(N)[:n_points_per_frame]
                    obj_pcl = obj_pcl[indices]

                # Pad if needed
                obj_pcl_np = obj_pcl.cpu().numpy()
                if obj_pcl_np.shape[0] < n_points_per_frame:
                    pad_size = n_points_per_frame - obj_pcl_np.shape[0]
                    padding = np.zeros((pad_size, 3), dtype=np.float32)
                    obj_pcl_np = np.concatenate([obj_pcl_np, padding], axis=0)

                pcl_list_all.append(obj_pcl_np.astype(np.float32))
            else:
                # For frames beyond max_idx, use zeros
                pcl_list_all.append(np.zeros((n_points_per_frame, 3), dtype=np.float32))

            # Clean up
            del gt_depth_list, gt_masks_pcl, gt_masks_list_pcl

    # Stack all frames into single array: (T, N, 3)
    all_pcl = np.stack(pcl_list_all, axis=0)

    # Save to single NPZ file
    output_path = os.path.join(pcl_output_dir, "pcl_sequence.npz")
    np.savez(output_path, pcl=all_pcl)
    logging.info(f"Saved point cloud sequence to {output_path}")
    logging.info(f"Shape: {all_pcl.shape} (frames, points, xyz)")

    logging.info("Done!")


if __name__ == "__main__":
    main()