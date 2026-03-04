import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_bounding_box(points):
    """Compute the axis-aligned bounding box for a set of 3D points."""
    min_coords, _ = torch.min(points, dim=0)
    max_coords, _ = torch.max(points, dim=0)
    return min_coords, max_coords


def fill_bounding_box_with_spheres(min_coords, max_coords, radius=0.006, up_z=False):
    """Generate sphere centers within the bounding box."""
    # Define the grid spacing (diameter of a sphere)
    step = 2 * radius
    
    # Generate grid points within the bounding box
    x_range = torch.arange(min_coords[0], max_coords[0] + step, step)
    y_range = torch.arange(min_coords[1], max_coords[1] + step, step)
    if up_z:
        step /= 1.5
    z_range = torch.arange(min_coords[2], max_coords[2] + step, step)
    
    # Create all possible sphere centers
    mesh = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
    sphere_centers = torch.stack(mesh, dim=-1).reshape(-1, 3)
    
    return sphere_centers


def sample_points_within_bounding_box(min_coord, max_coord, num_points=1000, device="cuda:0"):
    """Sample random points within the bounding box."""
    length = max_coord - min_coord
    min_coord = min_coord - length * 0.1
    max_coord = max_coord + length * 0.1
    
    return min_coord + (max_coord - min_coord) * torch.rand(num_points, 3).to(device)


def get_points_in_obj_mask(cam, pts, masks, get_mask=False, device="cuda:0"):
    # mask_obj = torch.ones(pts.shape[0], dtype=torch.bool, device=device)
    # for idx_cam in range(cam.Ks.shape[0]):
    #     # check for sphere centers
    #     us, vs = cam.world_to_image(pts, idx_cam)
    #     us = torch.round(us).long()
    #     vs = torch.round(vs).long()
        
    #     result = is_point_in_mask(masks[idx_cam].squeeze(-1), us, vs)
    #     mask_obj *= result
    
    mask_obj = torch.zeros(pts.shape[0], dtype=torch.int, device=device)
    for idx_cam in range(cam.Ks.shape[0]):
        # check for sphere centers
        us, vs = cam.world_to_image(pts, idx_cam)
        us = torch.round(us).long()
        vs = torch.round(vs).long()
        
        result = is_point_in_mask(masks[idx_cam].squeeze(-1), us, vs)
        mask_obj += result
    
    n_cams = cam.Ks.shape[0]
    mask_obj = mask_obj >= (n_cams - 1) # Convert to boolean mask
    if not get_mask:
        return pts[mask_obj]
    else:
        return pts[mask_obj], mask_obj


def is_point_in_mask(mask, us, vs):
    """
    Check if a batch of UV coordinates is within the object mask using PyTorch.

    Args:
        mask (torch.Tensor): Mask of shape (H, W, 1) where values > 0 indicate the object.
        uv (torch.Tensor): Batch of UV coordinates of shape (N, 2).

    Returns:
        torch.Tensor: Boolean tensor of shape (N,) indicating whether each point is within the mask.
    """
    # Round the UV coordinates to integers
    # uv_rounded = torch.round(uv).long()  # (N, 2)
    # u, v = uv_rounded[:, 0], uv_rounded[:, 1]  # Split U and V coordinates
    # us = torch.round(us).long()  
    # vs = torch.round(vs).long()  
    
    # Get image height and width
    height, width = mask.shape
    mask = mask.to(us.device)

    # print(us.shape)
    # print(vs.shape)
    # Check bounds
    in_bounds = (us >= 0) & (us < width) & (vs >= 0) & (vs < height)

    # Create the result tensor initialized to False
    result = torch.zeros(us.shape[0], dtype=torch.bool, device=us.device)

    print("image size:", width, height)
    print("us min/max:", us.min().item(), us.max().item())
    print("vs min/max:", vs.min().item(), vs.max().item())
    # Select only in-bound coordinates and check mask values
    if in_bounds.any():
        valid_indices = in_bounds.nonzero(as_tuple=True)[0]

        result[valid_indices] = mask[vs[valid_indices], us[valid_indices]] > 0

    # if valid_indices.numel() > 0:
    #     print(mask[0, vs[valid_indices], us[valid_indices]])
    #     result[valid_indices] = mask[0, vs[valid_indices], us[valid_indices]] > 0

    return result


def set_equal_aspect(ax):
    """Set equal aspect ratio for 3D plot axes."""
    extents = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    centers = np.mean(extents, axis=1)
    max_size = np.max(np.abs(extents[:, 1] - extents[:, 0]))
    r = max_size / 2
    for ctr, axis in zip(centers, [ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        axis([ctr - r, ctr + r])


def visualize_bounding_box_and_spheres(min_coords, max_coords, sphere_centers, radius):
    """Visualize the bounding box and spheres in 3D."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot bounding box edges
    r = [min_coords[0].item(), max_coords[0].item()]
    s = [min_coords[1].item(), max_coords[1].item()]
    t = [min_coords[2].item(), max_coords[2].item()]
    
    for i in range(2):
        for j in range(2):
            ax.plot([r[i], r[i]], [s[j], s[j]], [t[0], t[1]], 'k', linewidth=2)
            ax.plot([r[i], r[i]], [s[0], s[1]], [t[j], t[j]], 'k', linewidth=2)
            ax.plot([r[0], r[1]], [s[i], s[i]], [t[j], t[j]], 'k', linewidth=2)
    
    # Plot spheres with correct radius
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 5)
    for center in sphere_centers:
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0].item()
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1].item()
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2].item()
        ax.plot_surface(x, y, z, color='r', alpha=0.25)
        
    set_equal_aspect(ax)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Bounding Box with Spheres')
    plt.show()