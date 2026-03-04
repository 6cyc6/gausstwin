import cv2
import math
import click
import torch
import warp as wp
import numpy as np
import torch.nn.functional as F

from gsplat import rasterization
from gausstwin.cfg.domain import Robot
from gausstwin.utils.load_utils import get_robot_json
from gausstwin.utils.warp_utils import transform_gaussians_kernel
from gausstwin.cfg.robot.robot_configs import list_available_robots
from gausstwin.cfg.robot.fr3_cfg import FR3_DEFAULT_LINK_POS, FR3_DEFAULT_LINK_QUAT


@click.command()
@click.option(
    "--robot",
    type=click.Choice(list_available_robots(), case_sensitive=False),
    default="fr3_v3",
    help="Robot type to build",
    show_default=True
)
def check_robot(robot: str):
    """Build a robot model."""
    wp.init()
    device = "cuda:0"

    name = robot.lower()
    click.echo(f"Loading robot: {name}")

    # check robot model
    data = get_robot_json(name)
    robot_model = Robot(**data)
    links_dict = robot_model.links

    # gaussians
    gaussian_means = []
    gaussian_quats = []
    gaussian_scales = []
    gaussian_opacities = []
    gaussian_colors = []

    gs_link_indices = []  # which link each gaussian belongs to

    # get default poses
    robot_positions = FR3_DEFAULT_LINK_POS.to(device)
    robot_positions[:, 2] -= 1.5  # adjust ground offset
    robot_quats = FR3_DEFAULT_LINK_QUAT.to(device)

    for link_idx, (_, link_data) in enumerate(links_dict.items()):
        # load gaussians
        gaussians = link_data.gaussians
        n_gaussians = len(gaussians.means)
        gaussian_means.extend(gaussians.means)
        gaussian_quats.extend(gaussians.quats)
        gaussian_scales.extend(gaussians.scales)
        gaussian_opacities.extend(gaussians.opacities)
        gaussian_colors.extend(gaussians.colors)

        # save link indices for each gaussian
        gs_link_indices.extend([link_idx] * n_gaussians)

    total_gaussians = len(gaussian_means)
    click.echo(f"Loaded robot '{name}' with {total_gaussians} Gaussians across {len(links_dict)} links")

    # Convert to tensors
    means_tensor = torch.tensor(gaussian_means, dtype=torch.float32, device=device)  # (N, 3)
    quats_tensor = torch.tensor(gaussian_quats, dtype=torch.float32, device=device)  # (N, 4) wxyz
    scales_tensor = torch.tensor(gaussian_scales, dtype=torch.float32, device=device)  # (N, 3)
    opacities_tensor = torch.tensor(gaussian_opacities, dtype=torch.float32, device=device)  # (N,)
    colors_tensor = torch.tensor(gaussian_colors, dtype=torch.float32, device=device)  # (N, 3)
    indices_tensor = torch.tensor(gs_link_indices, dtype=torch.int32, device=device)  # (N,)

    # Prepare output arrays
    transformed_means = torch.zeros_like(means_tensor)
    transformed_quats = torch.zeros((total_gaussians, 4), dtype=torch.float32, device=device)

    # Convert to warp arrays
    means_wp = wp.from_torch(means_tensor, dtype=wp.vec3)
    quats_wp = wp.from_torch(quats_tensor, dtype=wp.quatf)  # wxyz format
    indices_wp = wp.from_torch(indices_tensor, dtype=wp.int32)
    rb_pos_wp = wp.from_torch(robot_positions, dtype=wp.vec3)
    # Convert robot quats from wxyz to xyzw for the kernel
    robot_quats_xyzw = torch.stack([
        robot_quats[:, 1], robot_quats[:, 2], robot_quats[:, 3], robot_quats[:, 0]
    ], dim=-1)
    rb_quats_wp = wp.from_torch(robot_quats_xyzw, dtype=wp.quatf)  # xyzw format

    transformed_means_wp = wp.from_torch(transformed_means, dtype=wp.vec3)
    transformed_quats_wp = wp.from_torch(transformed_quats, dtype=wp.vec4f)

    # Apply transform using warp kernel
    wp.launch(
        kernel=transform_gaussians_kernel,
        dim=total_gaussians,
        inputs=[
            means_wp,
            quats_wp,
            indices_wp,
            rb_pos_wp,
            rb_quats_wp,
            transformed_means_wp,
            transformed_quats_wp,
        ],
    )
    wp.synchronize()

    # Get transformed data back to torch
    transformed_means = wp.to_torch(transformed_means_wp)
    transformed_quats = wp.to_torch(transformed_quats_wp)

    click.echo("Applied default pose transforms to gaussians")

    # -------------------------------- Render ---------------------------------- #
    # Setup 4 cameras at different perspectives
    H, W = 480, 848
    fx, fy = 500.0, 500.0
    cx, cy = W / 2, H / 2

    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)

    # Target point (center of robot)
    cam_target = torch.tensor([0.0, 0.0, -0.3], device=device)
    cam_up = torch.tensor([0.0, 0.0, 1.0], device=device)
    cam_dist = 1.2  # distance from target
    cam_height = 0.3

    # 4 camera positions: front, right, back, left
    angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2]  # 0, 90, 180, 270 degrees
    cam_names = ["Front", "Right", "Back", "Left"]

    viewmats_list = []
    for angle in angles:
        cam_pos = torch.tensor([
            cam_dist * math.cos(angle),
            cam_dist * math.sin(angle),
            cam_height
        ], device=device)

        # Compute view matrix
        forward = F.normalize(cam_target - cam_pos, dim=0)
        right = F.normalize(torch.cross(forward, cam_up), dim=0)
        up = torch.cross(right, forward)

        R = torch.stack([right, -up, forward], dim=0)
        t = -R @ cam_pos

        viewmat = torch.eye(4, device=device)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t
        viewmats_list.append(viewmat)

    # Stack all viewmats
    viewmats = torch.stack(viewmats_list, dim=0)  # (4, 4, 4)
    Ks = K.unsqueeze(0).repeat(4, 1, 1)  # (4, 3, 3)

    with torch.no_grad():
        render_images, _, _ = rasterization(
            means=transformed_means,
            quats=F.normalize(transformed_quats, dim=-1),
            scales=scales_tensor,
            opacities=opacities_tensor,
            colors=colors_tensor,
            viewmats=viewmats,
            Ks=Ks,
            width=W,
            height=H,
            near_plane=0.01,
            far_plane=10.0,
            packed=False,
            render_mode="RGB",
        )

    # Display all 4 rendered images in a 2x2 grid
    grid_images = []
    for i in range(4):
        rgb = render_images[i].cpu().numpy()
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # Add label
        cv2.putText(rgb_bgr, cam_names[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        grid_images.append(rgb_bgr)

    # Create 2x2 grid
    top_row = np.hstack([grid_images[0], grid_images[1]])
    bottom_row = np.hstack([grid_images[2], grid_images[3]])
    grid = np.vstack([top_row, bottom_row])

    click.echo(f"Rendered 4 views, grid shape: {grid.shape}")
    cv2.imshow("Robot Gaussians - 4 Views", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    check_robot()
