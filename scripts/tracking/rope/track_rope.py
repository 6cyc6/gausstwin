import os
import time
import torch
import lycon
import click
import random
import logging
import warp as wp
import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from gausstwin.utils.franka_utils import FrankaV3
from gausstwin.utils.path_utils import get_track_cfg_path, safe_mkdir
from gausstwin.segmentation.tracker_TAM import TAMTracker
from gausstwin.tracker.rope.rope_sim import RopeSimulator
from gausstwin.tracker.rope.model_rope_builder import RopeBuilder
from gausstwin.cfg.tracker.gs_track_cfg import GSTrackUnifiedConfig


def run_tracking(seed=None):
    """
    Run rope tracking.
    
    Args:
        seed: Random seed for reproducibility. If None, no seed is set.
    """
    # Set random seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Note: warp does not have a global seed setter
        logging.info(f"Random seed set to: {seed}")

    # load path and configuration
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg_path = get_track_cfg_path("track_rope.yaml")
    cfg = GSTrackUnifiedConfig.from_yaml(cfg_path)
    data_dir = os.path.join(cfg.data_dir, f"trajectory_{cfg.exp_name}")

    logging.info(f"-------------------- Experiment: {cfg.exp_name} --------------------")
    logging.info(f"--------------------- Objects: {cfg.obj_list} ---------------------")
    
    # ========================================= setup ============================================= #
    # ----------------- setup robot ------------------ #
    robot = FrankaV3()

    # --------- setup segmentation tracker ----------- #
    seg_tracker = TAMTracker(cfg)
    # load prompts
    img_path_list = []
    for cam_idx in range(cfg.n_cams):
        if cam_idx == 0:
            fix_idx = cfg.fix_idx_1
        elif cam_idx == 1:
            fix_idx = cfg.fix_idx_2
        elif cam_idx == 2:
            fix_idx = cfg.fix_idx_3
        else:
            fix_idx = cfg.fix_idx_4
        img_path_list.append(os.path.join(data_dir, f"camera/static_{cam_idx + 1}/rgb/{fix_idx}.png"))
    rgb_np_list = [np.array(Image.open(img_path).convert("RGB")) for img_path in img_path_list]
    seg_tracker.load_first_frames(rgb_np_list)
    if len(cfg.obj_list) > 1:
        seg_tracker.add_new_prompts_multi(vis=cfg.vis)
    else:
        for obj in cfg.obj_list:
            seg_tracker.add_new_prompts(obj, vis=cfg.vis)

    # ---------------- setup GSTwin ------------------ #
    builder = RopeBuilder(exp_name=cfg.exp_name)
    # add ground
    builder.set_ground(
        mu=cfg.ground_mu,
        restitution=cfg.ground_restitution,
    )
    # add robot 
    builder.add_robot_rigid_body(
        name="fr3_v3", 
        obj_radii=cfg.obj_radii,
        gripper_mu=cfg.gripper_mu,
        gripper_density_factor=cfg.gripper_density_factor
    )
    # add rope 
    for obj in cfg.obj_list:
        builder.add_rope( 
            m=cfg.rope_m,
            mass_q_factor=cfg.rope_mass_q_factor,
            stiffness_bend=cfg.rope_stiffness_bend,
            stiffness_stretch=cfg.rope_stiffness_stretch,
        )
    # build model
    model = builder.finalize(cfg)  
    # get the simulator
    sim = RopeSimulator(builder=builder, cfg=cfg, model=model, device=device)

    # ---------------- setup video ------------------ #
    disp = cfg.disp
    max_idx = cfg.max_idx
    length = cfg.length

    pcl_path = os.path.join(data_dir, "pcl_output/pcl_sequence.npz")
    pcl_data = np.load(pcl_path)
    pcl_list_all = pcl_data["pcl"]  # (T, 2000, 3)

    # Helper function to load RGB images using lycon
    def load_rgb(frame_idx: int) -> list[np.ndarray]:
        """Load RGB images for all cameras at a given frame index."""
        rgb_list = []
        for cam_idx in range(cfg.n_cams):
            rgb_path = os.path.join(data_dir, f"camera/static_{cam_idx + 1}/rgb/{frame_idx}.png")
            rgb = lycon.load(rgb_path)  # (H, W, 3) uint8
            rgb_list.append(rgb)
        return rgb_list

    # ========================================= Initialize ============================================= #
    # let the object settle on the ground before tracking
    for i in range(10):
        sim.step_pbd()

    idx = disp
    robot_cfg = np.load(os.path.join(data_dir, f"robot/joint_pos_seq/{idx}.npy"))
    robot_cfg_qd = np.load(os.path.join(data_dir, f"robot/joint_vel_seq/{idx}.npy"))

    robot_cfg_sampled = robot_cfg[::5]
    robot_cfg_qd_sampled = robot_cfg_qd[::5]

    positions, quats, linear_vel, angular_vel = robot.forward_kinematics(q=robot_cfg_sampled[0], dq=robot_cfg_qd_sampled[0], h=builder.ground_h)
    positions = torch.from_numpy(positions).to(device)
    quats = torch.from_numpy(quats).to(device)
    linear_vel = torch.from_numpy(linear_vel).to(device)
    angular_vel = torch.from_numpy(angular_vel).to(device)

    rgb_np_list = load_rgb(disp)
    gt_rgbs_list = [torch.from_numpy(rgb_np).to(device).float() / 255.0 for rgb_np in rgb_np_list]
    gt_masks = seg_tracker.track(rgb_np_list)
    gt_masks_list = [gt_masks[i, :, :].to(device) for i in range(cfg.n_cams)]
    for _ in range(10):
        sim.step_init(positions, quats, linear_vel, angular_vel, gt_rgbs_list, gt_masks_list)

    # ========================================= Tracking Loop ============================================= #
    vis_dir = os.path.join(data_dir, "vis")
    safe_mkdir(vis_dir)
    iter_times = []
    pbar = tqdm(range(length), desc="Tracking Progress", unit="frame")
    for i in pbar:
        t_s = time.time()
        if i >= max_idx:
            idx = max_idx + disp
        else:
            idx = i + disp

        # --------------- load input data --------------- #
        # load RGB images
        rgb_np_list = load_rgb(idx)
        
        # load point cloud
        obj_pcl = None
        if sim.cfg.render and i < len(pcl_list_all):
            obj_pcl = pcl_list_all[i]
            obj_pcl[:, -1] += builder.ground_h
        
        # load robot states
        robot_cfg = np.load(os.path.join(data_dir, f"robot/joint_pos_seq/{idx}.npy")) # (rougly 33, 7)
        robot_cfg_qd = np.load(os.path.join(data_dir, f"robot/joint_vel_seq/{idx}.npy")) # (rougly 33, 7)
        
        torch.cuda.synchronize()
        t_load_end = time.time()
        load_time = t_load_end - t_s
        
        # --------------- Img segmentation --------------- #
        # transfer image from cpu to gpu
        gt_rgbs_list = [torch.from_numpy(rgb_np).to(device).float() / 255.0 for rgb_np in rgb_np_list]  # (N, H, W, 3)

        # run segmentation
        if cfg.use_seg_mask:
            gt_masks = seg_tracker.track(rgb_np_list)
            gt_masks_list = [gt_masks[i, :, :].to(device) for i in range(cfg.n_cams)]  # (N, H, W)
        else:
            gt_masks_list = None

        torch.cuda.synchronize()
        t_seg_end = time.time()
        seg_time = t_seg_end - t_load_end

        # ------------------ step simulation ------------------- #
        # two steps
        robot_cfg_sampled = robot_cfg[[15, -1]]
        robot_cfg_qd_sampled = robot_cfg_qd[[15, -1]]

        for j in range(robot_cfg_sampled.shape[0]):
            positions, quats, linear_vel, angular_vel = robot.forward_kinematics(q=robot_cfg_sampled[j], dq=robot_cfg_qd_sampled[j], h=builder.ground_h)
            # positions, quats, linear_vel, angular_vel = robot.forward_kinematics(q=robot_cfg_sampled[j])
            positions = torch.from_numpy(positions).to(device)
            quats = torch.from_numpy(quats).to(device)
            linear_vel = torch.from_numpy(linear_vel).to(device)
            angular_vel = torch.from_numpy(angular_vel).to(device)
            
            # two steps
            if j == 0:
                # set robot to the intermediate state and step simulation (prediction step)
                sim.update_robot_state(positions, quats, linear_vel, angular_vel)
                sim.step_simulation(clear_forces=True)
            else:
                # set robot to the last state, apply visual correction, and step simulation (correction step)
                sim.update_robot_state(positions, quats, linear_vel, angular_vel)
                sim.visual_correction(gt_rgbs_list, gt_masks_list)
                sim.step_simulation(clear_forces=False, vis_pts=obj_pcl, render=True)
        
        # --------------------------- alternative: single step --------------------------- #
        # # single step
        # robot_cfg_sampled = robot_cfg[[-1]]
        # robot_cfg_qd_sampled = robot_cfg_qd[[-1]]
        
        # # run forward kinematics
        # positions, quats, linear_vel, angular_vel = robot.forward_kinematics(q=robot_cfg_sampled[-1], dq=robot_cfg_qd_sampled[-1], h=builder.ground_h)
        # # positions, quats, linear_vel, angular_vel = robot.forward_kinematics(q=robot_cfg_sampled[j])
        # positions = torch.from_numpy(positions).to(device)
        # quats = torch.from_numpy(quats).to(device)
        # linear_vel = torch.from_numpy(linear_vel).to(device)
        # angular_vel = torch.from_numpy(angular_vel).to(device)

        # # step simulation and visual correction
        # sim.step(positions, quats, linear_vel, angular_vel, gt_rgbs_list, gt_masks_list, clear_forces=False, vis_pts=obj_pcl, render=True)

        # ===================================== end of algorithm iteration ===================================== #
        torch.cuda.synchronize()
        t_opt_end = time.time()
        opt_time = t_opt_end - t_seg_end
        
        # ---------------- Save Information ---------------- #
        iter_time = t_opt_end - t_load_end
        iter_times.append(iter_time)
        avg_time = np.mean(iter_times)
        
        pbar.set_postfix({
            'Load': f'{load_time:.3f}s',
            'Seg': f'{seg_time:.3f}s',
            'Opt': f'{opt_time:.3f}s',
            'Iter': f'{iter_time:.3f}s',
            'Avg': f'{avg_time:.3f}s'
        })
        
        # # ------------------- Rendering for visualization ------------------- #
        # # render
        # render_imgs = sim.vis_render()
        # for i in range(cfg.n_cams):
        #     render_img = render_imgs[i]
        #     render_img = (render_img.cpu().numpy() * 255).astype(np.uint8)
        #     lycon.save(os.path.join(vis_dir, f"vis_{idx}_{i}.png"), render_img)
        
    # save video
    if cfg.render:
        sim.renderer.save()
    

@click.command()
@click.option('--seed', default=None, type=int, help='Random seed for reproducibility')
def main(seed):
    """Run rope tracking."""
    # logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_tracking(seed=seed)


if __name__ == "__main__":
    main()
    