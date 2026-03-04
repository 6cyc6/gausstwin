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
from gausstwin.cfg.tracker.gs_track_cfg import GSTrackUnifiedConfig
from gausstwin.tracker.rigid_body.rigid_sim import RigidMultiSimulator
from gausstwin.tracker.rigid_body.model_rigid_builder import RigidBuilder


def run_tracking(seed=None):
    """
    Run rigid body tracking.
    
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
    cfg_path = get_track_cfg_path("track_rigid.yaml")
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
        obj = cfg.obj_list[0]
        seg_tracker.add_new_prompts(obj, vis=cfg.vis)

    # ---------------- setup GSTwin ------------------ #
    builder = RigidBuilder(exp_name=cfg.exp_name)
    # add ground
    builder.set_ground(
        mu=cfg.ground_mu,
        restitution=cfg.ground_restitution,
    )
    # add robot 
    builder.add_robot_rigid_body(
        name="fr3_v3", 
        obj_radii=0.006,
        gripper_mu=cfg.gripper_mu,
        gripper_density_factor=cfg.gripper_density_factor
    )

    # add objects 
    obj_idx = 0
    for obj in cfg.obj_list:
        if obj_idx == 0:
            builder.add_rigid_object(
                name=obj, 
                obj_radii=cfg.obj_radii,
                mu=cfg.rigid_mu,
                density_factor=cfg.rigid_density_factor,
            )
        elif obj_idx == 1:
            builder.add_rigid_object(
                name=obj, 
                obj_radii=cfg.obj_radii,
                mu=cfg.rigid2_mu,
                density_factor=cfg.rigid2_density_factor,
            )
        obj_idx += 1

    # build model
    model = builder.finalize(cfg)  
    
    # get the simulator
    sim = RigidMultiSimulator(builder=builder, cfg=cfg, model=model, device=device)

    # ---------------- setup video ------------------ #
    disp = cfg.disp
    max_idx = cfg.max_idx
    length = cfg.length
    
    pcl_path = os.path.join(data_dir, "pcl_output/pcl_sequence.npz")
    pcl_data = np.load(pcl_path)
    pcl_list_all = pcl_data["pcl"]  # (T, 2000, 3)

    # Helper function to load RGBD images using lycon
    def load_rgb(frame_idx: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Load RGB and depth images for all cameras at a given frame index."""
        rgb_list = []
        for cam_idx in range(cfg.n_cams):
            rgb_path = os.path.join(data_dir, f"camera/static_{cam_idx + 1}/rgb/{frame_idx}.png")
            depth_path = os.path.join(data_dir, f"camera/static_{cam_idx + 1}/depth/{frame_idx}.png")
            # Load RGB using lycon
            rgb = lycon.load(rgb_path)  # (H, W, 3) uint8
            rgb_list.append(rgb)
        return rgb_list

    # ========================================= Initialize ============================================= #
    for i in range(10):
        sim.step_pbd()

    idx = disp
    robot_cfg = np.load(os.path.join(data_dir, f"robot/joint_pos_seq/{idx}.npy")) # (rougly 100, 7)
    robot_cfg_qd = np.load(os.path.join(data_dir, f"robot/joint_vel_seq/{idx}.npy")) # (rougly 100, 7)

    robot_cfg_sampled = robot_cfg[::5]
    robot_cfg_qd_sampled = robot_cfg_qd[::5]
    # zeros = np.zeros((robot_cfg_sampled.shape[0], 2))
    # robot_cfg_sampled = np.hstack([robot_cfg_sampled, zeros])
    # robot_cfg_qd_sampled = np.hstack([robot_cfg_qd_sampled, zeros])

    positions, quats, linear_vel, angular_vel = robot.forward_kinematics(q=robot_cfg_sampled[0], dq=robot_cfg_qd_sampled[0], h=builder.ground_h)
    positions = torch.from_numpy(positions).to(device)
    quats = torch.from_numpy(quats).to(device)
    linear_vel = torch.from_numpy(linear_vel).to(device) 
    angular_vel = torch.from_numpy(angular_vel).to(device)

    rgb_np_list = load_rgb(disp)
    gt_rgbs_list = [torch.from_numpy(rgb_np).to(device).float() / 255.0 for rgb_np in rgb_np_list]  # (N, H, W, 3)
    gt_masks = seg_tracker.track(rgb_np_list)
    gt_masks_list = [gt_masks[i, :, :].to(device) for i in range(cfg.n_cams)]  # (N, H, W)
    for _ in range(10):
        sim.step_init(positions, quats, linear_vel, angular_vel, gt_rgbs_list, gt_masks_list)
    
    # ========================================= Tracking Loop ============================================= #
    vis_dir = os.path.join(data_dir, "vis")
    safe_mkdir(vis_dir)
    if cfg.eval:
        pose_save_1 = []
        pose_save_2 = []
        save_dir = os.path.join(data_dir, "result")
        safe_mkdir(save_dir)
    iter_times = []
    pbar = tqdm(range(length), desc="Tracking Progress", unit="frame")
    for i in pbar:
        t_s = time.time()
        if i >= max_idx: 
            idx = max_idx + disp
        else:
            idx = i + disp
        
        # --------------- load input data --------------- #
        # Load RGB images 
        rgb_np_list = load_rgb(idx)
        
        # # Get point cloud
        obj_pcl = None
        if sim.cfg.render and i < len(pcl_list_all):
            obj_pcl = pcl_list_all[i]
            obj_pcl[:, -1] += builder.ground_h
            # obj_pcl[:, 2] += 0.005
        
        # load robot states
        # robot_cfg = np.load(os.path.join(data_dir, f"robot/joint_pos_seq/{idx}.npy")) # (rougly 33, 7)
        # robot_cfg_qd = np.load(os.path.join(data_dir, f"robot/joint_vel_seq/{idx}.npy")) # (rougly 33, 7)
        
        if idx == 0:
            robot_cfg = np.load(os.path.join(data_dir, f"robot/joint_pos_seq/{idx}.npy")) # (rougly 33, 7)
            robot_cfg_qd = np.load(os.path.join(data_dir, f"robot/joint_vel_seq/{idx}.npy")) # (rougly 33, 7)
        else:
            robot_cfg = np.load(os.path.join(data_dir, f"robot/joint_pos_seq/{idx - 1}.npy")) # (rougly 33, 7)
            robot_cfg_qd = np.load(os.path.join(data_dir, f"robot/joint_vel_seq/{idx - 1}.npy")) # (rougly 33, 7)
        
        torch.cuda.synchronize()
        t_load_end = time.time()
        load_time = t_load_end - t_s
        
        # --------------- Img segmentation --------------- #
        # transfer image from cpu to gpu
        gt_rgbs_list = [torch.from_numpy(rgb_np).to(device).float() / 255.0 for rgb_np in rgb_np_list]  # (N, H, W, 3)

        # Run segmentation
        if cfg.use_seg_mask:
            gt_masks = seg_tracker.track(rgb_np_list)
            gt_masks_list = [gt_masks[i, :, :].to(device) for i in range(cfg.n_cams)]  # (N, H, W)
        else:
            gt_masks_list = None

        torch.cuda.synchronize()
        t_seg_end = time.time()
        seg_time = t_seg_end - t_load_end

        # ------------------ step simulation
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
            
            # sim.step_new(positions, quats, linear_vel, angular_vel, gt_rgbs_list, gt_masks_list, clear_forces=False)
            
            # # two steps
            # if j == 0:
            #     sim.step(positions, quats, linear_vel, angular_vel, clear_forces=True)
            # elif j == 1:
            #     sim.step(positions, quats, linear_vel, angular_vel, gt_rgbs_list, gt_masks_list, clear_forces=False, vis_pts=obj_pcl, render=True)
            if j == 0:
                # set robot to the intermediate state and step simulation (prediction step)
                sim.update_robot_state(positions, quats, linear_vel, angular_vel)
                sim.step_simulation(clear_forces=True)
            else:
                # set robot to the last state, apply visual correction, and step simulation (correction step)
                sim.update_robot_state(positions, quats, linear_vel, angular_vel)
                sim.visual_correction(gt_rgbs_list, gt_masks_list)
                sim.step_simulation(clear_forces=False, vis_pts=obj_pcl, render=True)
        
        # # single step
        # robot_cfg_sampled = robot_cfg[[-1]]
        # robot_cfg_qd_sampled = robot_cfg_qd[[-1]]
        
        # positions, quats, linear_vel, angular_vel = robot.forward_kinematics(q=robot_cfg_sampled[-1], dq=robot_cfg_qd_sampled[-1], h=builder.ground_h)
        # # positions, quats, linear_vel, angular_vel = robot.forward_kinematics(q=robot_cfg_sampled[j])
        # positions = torch.from_numpy(positions).to(device)
        # quats = torch.from_numpy(quats).to(device)
        # linear_vel = torch.from_numpy(linear_vel).to(device)
        # angular_vel = torch.from_numpy(angular_vel).to(device)

        # sim.step(positions, quats, linear_vel, angular_vel, gt_rgbs_list, gt_masks_list, clear_forces=False, vis_pts=obj_pcl, render=cfg.render)

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
        
        # save information for evaluation
        if cfg.eval:
            if len(cfg.obj_list) == 2:
                body_q = wp.to_torch(sim.state_0.body_q).clone()
                pose_save_1.append(body_q[-2, :].cpu().numpy())
                pose_save_2.append(body_q[-1, :].cpu().numpy())
            else:
                pose_save_1.append(wp.to_torch(sim.state_0.body_q)[-1, :].clone().cpu().numpy())
            
            # ------------------- Rendering and saving images ------------------- #
            # render_obj_img, render_robot_img, render_all_img = sim.eval_render()
            
            # # Convert to numpy and ensure correct shape (H, W, C)
            # render_obj_img_np = (render_obj_img.cpu().numpy() * 255).astype(np.uint8)[0]
            # render_robot_img_np = (render_robot_img.cpu().numpy() * 255).astype(np.uint8)[0]
            # render_all_img_np = (render_all_img.cpu().numpy() * 255).astype(np.uint8)[0]
            
            # lycon.save(os.path.join(save_dir, f"obj_{idx}.png"), render_obj_img_np)  
            # lycon.save(os.path.join(save_dir, f"robot_{idx}.png"), render_robot_img_np)
            # lycon.save(os.path.join(save_dir, f"all_{idx}.png"), render_all_img_np)
            
        # # ------------------- Rendering for visualization ------------------- #
        # render_imgs = sim.vis_render()
        # for i in range(render_imgs.shape[0]):
        #     render_img = render_imgs[i]
        #     render_img_np = (render_img.cpu().numpy() * 255).astype(np.uint8)
        #     lycon.save(os.path.join(vis_dir, f"vis_{idx}_{i}.png"), render_img_np)
            
    # save video
    if cfg.render:
        sim.renderer.save()
        
    # save optimized poses
    if cfg.eval:
        np.savez(os.path.join(save_dir, "tracking_result"), pose_1=pose_save_1, pose_2=pose_save_2)
        logging.info("Saving optimized poses.")
        logging.info(f"Average algorithm time per iteration: {np.mean(iter_times):.4f}s over {len(iter_times)} iterations")
    
    
@click.command()
@click.option('--seed', default=42, type=int, help='Random seed for reproducibility')
def main(seed):
    """Run rigid body tracking with optional random seed."""
    run_tracking(seed=seed)


if __name__ == "__main__":
    # logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
    