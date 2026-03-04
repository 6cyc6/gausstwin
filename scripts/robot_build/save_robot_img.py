import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--robot", type=str, default="fr3_v3", help="Robot type to spawn.")
parser.add_argument("--link", type=str, default="link0", help="Link name to spawn.")
parser.add_argument("--h", type=int, default=480, help="Height of the camera.")
parser.add_argument("--w", type=int, default=848, help="Width of the camera.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports after app launch ---
import os
import json
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
import torch
import numpy as np
from PIL import Image
from gausstwin.utils.path_utils import get_gs_fig_dir


def quat_mul(q1, q2):
    """Multiply two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return torch.tensor([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], device=q1.device, dtype=q1.dtype)


def quat_rotate(q, v):
    """Rotate vector v by quaternion q (w, x, y, z)."""
    q_v = torch.tensor([0.0, v[0], v[1], v[2]], device=q.device, dtype=q.dtype)
    q_conj = torch.tensor([q[0], -q[1], -q[2], -q[3]], device=q.device, dtype=q.dtype)
    result = quat_mul(quat_mul(q, q_v), q_conj)
    return result[1:4]


FRANKA_DIR = os.environ["FRANKA_DIR"]
LINK = args_cli.link
H = args_cli.h
W = args_cli.w
ROBOT = args_cli.robot

@configclass
class FrankaImgSave(InteractiveSceneCfg):
    # light
    demo_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light", 
        spawn=sim_utils.DomeLightCfg(intensity=600.0, color=(0.75, 0.75, 0.75)),
    )

    # robot
    panda: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Franka",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{FRANKA_DIR}/usd/{LINK}.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=10.0,
            ),
            semantic_tags=[("class", "panda")]
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),  
        )
    )

    # camera
    tiled_camera_0: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera_0",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.8, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb", "depth", "instance_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 10.0)
        ),
        width=W, # 848
        height=H, # 480 
    )
    

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    import math

    camera = scene["tiled_camera_0"]
    robot = scene["panda"]
    
    id_2_semantics = camera.data.info['instance_segmentation_fast']['idToSemantics']
    for id, sem in id_2_semantics.items():
        if sem["class"] == "panda":
            print(f"ID: {id}, Semantics: {sem}")
            panda_id = id

    sim_dt = sim.get_physics_dt()
    count = 0
    save_dir = os.path.join(get_gs_fig_dir(), ROBOT, LINK)
    os.makedirs(os.path.join(save_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "mask"), exist_ok=True)

    total_views = 200
    views_per_rotation = 100
    start_count = 10
    view_idx = 0
    all_poses = []
    intrinsics = None

    while simulation_app.is_running():
        if count % 10000000000000 == 0:
            scene.reset()
            count = 0

        # Rotate the object starting from count 10
        if count >= start_count and view_idx < total_views:
            if view_idx < views_per_rotation:
                # Rotate around Y axis
                angle = 2.0 * math.pi * view_idx / views_per_rotation
                quat = (math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0)
            else:
                # Rotate around Z axis
                angle = 2.0 * math.pi * (view_idx - views_per_rotation) / views_per_rotation
                quat = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))

            # Set object orientation (w, x, y, z)
            root_state = robot.data.default_root_state.clone()
            root_state[0, 3:7] = torch.tensor(quat, device=sim.device)
            robot.write_root_state_to_sim(root_state)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        # Save camera data after stepping
        if count >= start_count and view_idx < total_views:
            camera._update_outdated_buffers()

            seg = camera.data.output["instance_segmentation_fast"][0].clone()
            
            mask = (
                (seg[..., 0] == panda_id[0]) &
                (seg[..., 1] == panda_id[1]) &
                (seg[..., 2] == panda_id[2]) &
                (seg[..., 3] == panda_id[3])
            ).float()  # (H, W)

            depth = camera.data.output["depth"][0]
            depth[depth == float("inf")] = 0  # (H, W, 1)
            rgb = camera.data.output["rgb"][0]  # (H, W, 4)

            # Save RGB as PNG
            rgb_np = rgb.cpu().numpy().astype(np.uint8)
            if rgb_np.shape[-1] == 4:
                rgb_np = rgb_np[..., :3]
            Image.fromarray(rgb_np, mode="RGB").save(os.path.join(save_dir, "rgb", f"{view_idx:06d}.png"))

            # Save depth as 16-bit PNG (in mm)
            depth_np = depth.squeeze(-1).cpu().numpy()
            depth_np = (depth_np * 1000).astype(np.uint16)
            Image.fromarray(depth_np).save(os.path.join(save_dir, "depth", f"{view_idx:06d}.png"))

            # Save mask as PNG
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(mask_np).save(os.path.join(save_dir, "mask", f"{view_idx:06d}.png"))

            # Collect camera pose relative to object
            # Camera is fixed in world, object rotates. We need cam pose in object frame.
            # Object pose: position=0, orientation=quat (applied above)
            # Relative camera pose = inv(object_pose) * camera_world_pose
            obj_quat = torch.tensor(quat, device=sim.device, dtype=torch.float32)  # (w, x, y, z)
            cam_pos_world = camera.data.pos_w[0]  # (3,)
            cam_quat_world = camera.data.quat_w_ros[0]  # (x, y, z, w) ROS convention

            # Convert cam quat from ROS (x,y,z,w) to (w,x,y,z)
            cam_quat_wxyz = torch.tensor([
                cam_quat_world[3].item(), cam_quat_world[0].item(),
                cam_quat_world[1].item(), cam_quat_world[2].item()
            ], device=sim.device)

            # Inverse of object rotation (conjugate since it's unit quaternion)
            obj_quat_inv = torch.tensor([obj_quat[0], -obj_quat[1], -obj_quat[2], -obj_quat[3]], device=sim.device)

            # Rotate camera position into object frame
            cam_pos_obj = quat_rotate(obj_quat_inv, cam_pos_world)

            # Compose quaternions: q_rel = q_obj_inv * q_cam
            cam_quat_obj = quat_mul(obj_quat_inv, cam_quat_wxyz)

            all_poses.append({
                "pos": cam_pos_obj.cpu().tolist(),
                "quat": cam_quat_obj.cpu().tolist(),  # (w, x, y, z)
            })

            # Save intrinsics once
            if intrinsics is None:
                intrinsics = {
                    "k": camera.data.intrinsic_matrices[0].tolist(),
                    "h": H,
                    "w": W,
                }

            print(f"[INFO]: Saved view {view_idx + 1}/{total_views}")
            view_idx += 1

            if view_idx >= total_views:
                # Write camera intrinsics
                with open(f"{save_dir}/camera_intrinsics.json", "w") as f:
                    json.dump(intrinsics, f, indent=4)
                # Write all camera poses
                with open(f"{save_dir}/camera_poses.json", "w") as f:
                    json.dump(all_poses, f, indent=4)
                print("[INFO]: Finished saving all views.")
                break

        count += 1
        

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cuda", dt=1/100)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view((5.0, 0.0, 1.8), (0.0, 0.0, 1.5))
    # Design scene
    scene_cfg = FrankaImgSave(num_envs=1, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)
        

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
