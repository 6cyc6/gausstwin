import torch


def downsample_pcl(pcl: torch.Tensor, colors: torch.Tensor, num_points: int = 10000) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Downsample point cloud and corresponding colors to a target number of points.

    Args:
        pcl: Point cloud tensor (N, 3)
        colors: Color tensor (N, 3)
        num_points: Target number of points to sample

    Returns:
        downsampled_pcl: Downsampled point cloud (num_points, 3)
        downsampled_colors: Downsampled colors (num_points, 3)
    """
    if pcl.shape[0] <= num_points:
        return pcl, colors

    # Randomly sample num_points
    indices = torch.randperm(pcl.shape[0], device=pcl.device)[:num_points]
    downsampled_pcl = pcl[indices]
    downsampled_colors = colors[indices]

    return downsampled_pcl, downsampled_colors


def get_pcl_from_rgbd_mask(
    cam_k: torch.Tensor, 
    cam_T: torch.Tensor, 
    images: torch.Tensor, 
    depth: torch.Tensor, 
    mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transformation segmented image coordinates to world coordinates.

    Args:
        images: RGB image tensor (H, W, 3)
        depth: Depth map tensor (H, W, 1)
        mask: Binary mask tensor (H, W)

    Returns:
        pts_world: World coordinates of the points (N, 3)
        rgb_values: RGB values of the points (N, 3)
    """
    # get uvs from mask
    mask = mask.bool()
    v, u = torch.nonzero(mask, as_tuple=True)  # Extract pixel coordinates

    uv_coords = torch.stack((u, v), dim=1)
    depth_values = depth[uv_coords[:, 1], uv_coords[:, 0]]  # (N,)

    # Extract RGB values at the corresponding pixel coordinates
    pts_rgb = images[uv_coords[:, 1], uv_coords[:, 0]]  # (N, 3)

    # Create homogeneous image coordinates
    N = uv_coords.shape[0]
    ones = torch.ones(N, 1, device=uv_coords.device)
    pixel_coords = torch.cat((uv_coords.float(), ones), dim=1)  # (N, 3)

    # get K inverse from Camera object
    K_inv = torch.inverse(cam_k)
    
    # get camera coordinates
    cam_coords = (K_inv @ pixel_coords.T).T * depth_values.unsqueeze(1)  # (N, 3)

    # get world coordinates from camera coordinates
    # cam.T is world-to-camera, so inverse gives camera-to-world
    cam2world = torch.inverse(cam_T)
    R = cam2world[:3, :3]
    t = cam2world[:3, 3]

    pts_world = (R @ cam_coords.T).T + t

    return pts_world, pts_rgb
