import torch
import polyscope as ps


def init_polyscope(up_dir="z_up", ground_plane_mode="none"):
    """
    Initializes Polyscope with specified up direction and ground plane mode.

    Args:
        up_dir (str): The up direction for the scene. Default is "z_up".
        ground_plane_mode (str): The ground plane mode. Default is "none".
    """
    ps.init()
    ps.set_up_dir(up_dir)
    ps.set_ground_plane_mode(ground_plane_mode)
    

def polyscope_vis_pcl(pts, colors=None, names=None, radius=0.001):
    """
    Visualizes a point cloud with optional colors using Polyscope.

    Args:
        pts (torch.Tensor or list of torch.Tensor): 
            - Either a single (N,3) tensor or a list of (Ni,3) tensors.
        colors (torch.Tensor or list of torch.Tensor, optional): 
            - Same structure as `pts`, containing RGB values. 
            - If `None`, colors are not applied.
    """

    # Initialize Polyscope
    init_polyscope(up_dir="z_up", ground_plane_mode="none")
    
    def process_tensor(tensor):
        """Ensure input is a tensor and convert to numpy, handling lists."""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        elif isinstance(tensor, list):
            return [t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in tensor]
        elif tensor is None:
            return None  # Allow None for colors
        else:
            raise TypeError("Input must be a torch.Tensor, a list of torch.Tensor, or None")

    # Ensure inputs are tensors or lists of tensors
    pts = process_tensor(pts)
    if colors is not None:
        colors = process_tensor(colors)

    # If input is a list of multiple point sets
    if isinstance(pts, list):
        for i, p in enumerate(pts):
            c = colors[i] if colors is not None else None  # Apply colors only if provided
            ps_cloud = ps.register_point_cloud(f"PointCloud_{i}", p, radius=radius)
            if colors is not None:
                ps_cloud.add_color_quantity("colors", c, enabled=True)  # Add colors if provided
    else:
        # Single point cloud case
        ps_cloud = ps.register_point_cloud("PointCloud", pts, radius=radius)
        if colors is not None:
            ps_cloud.add_color_quantity("colors", colors, enabled=True)  # Add colors if provided

    ps.show()
    ps.remove_all_structures()
    
