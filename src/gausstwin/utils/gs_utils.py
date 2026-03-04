import cv2
import torch
import math
import trimesh
import random
import warp as wp
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, Tuple
from jaxtyping import Float, Int
from torch_cluster import knn
from gsplat import rasterization
from gausstwin.utils import math_utils
from gausstwin.cfg.builder.builder_cfg import GSConfig, RigidBodyConfig, RopeConfig


# ----------------------------------- Auxiliary Functions for GS------------------------------------ #
def k_nearest_torch(x: torch.Tensor, k: int) -> Tuple[Float[Tensor, "*batch k"], Int[Tensor, "*batch k"]]:
    """
    k-nearest neighbors using PyTorch's knn function.
    """
    edge_index = knn(x, x, k=k+1, num_workers=10)
    query_indices = edge_index[0]  # Indices of query points
    neighbor_indices = edge_index[1]  # Indices of nearest neighbors
    # Remove self-matches (Each point is closest to itself)
    mask = [i for i in range(query_indices.shape[0]) if i % (k + 1) != 0]
    query_indices = query_indices[mask]
    neighbor_indices = neighbor_indices[mask]
    # Compute Euclidean distances
    distances = torch.norm(x[query_indices] - x[neighbor_indices], dim=1)
    distances = distances.view(-1, 3)
    
    return distances, neighbor_indices


def assign_nearest_neighbor_properties(pts, pts_ref, p_ref):
    """
    Return the properties of the nearest neighbor.
    """
    _, nearest_indices = knn(pts_ref, pts, k=1, num_workers=10)  # k=1 for nearest neighbor

    # Gather properties from the nearest reference points
    assigned_properties = p_ref[nearest_indices]

    return assigned_properties


def down_sample_pcl(pts, rgbs=None, num_points=100000):
    """
    Down sample the point cloud.
    """
    num_pick = torch.min(torch.tensor([pts.shape[0], num_points]))
    indices = torch.randperm(pts.shape[0])[:num_pick]
    
    if rgbs is not None:
        return pts[indices], rgbs[indices]
    else:
        return pts[indices]
    

def position_from_depth_pt(
        depth: torch.Tensor, 
        intrinsic_matrix: torch.Tensor,
        position: torch.Tensor,
        orientation: torch.Tensor,
    ) -> torch.Tensor:
    """
    Compute the 3D position from depth and intrinsics.
    """
    depth_cloud = math_utils.unproject_depth(depth, intrinsic_matrix)
    # convert 3D points to world frame
    depth_cloud = math_utils.transform_points(depth_cloud, position, orientation)
    
    return depth_cloud


def fit_plane(points):
    """
    Fit a plane from three points.
    Returns plane normal [a, b, c] and offset d.
    """
    p1, p2, p3 = points
    # Vectors
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Cross product to get normal
    normal = torch.linalg.cross(v1, v2)
    normal = normal / torch.norm(normal)
    
    # Plane equation: ax + by + cz + d = 0
    d = -torch.dot(normal, p1)
    
    return normal, d


def compute_distance(points, normal, d):
    """
    Compute distance from points to the plane.
    """
    distances = torch.abs(torch.matmul(points, normal) + d) / torch.norm(normal)
    return distances


def ransac_plane(points, threshold=0.003, iterations=100):
    best_inliers = 0
    best_plane = None

    for _ in range(iterations):
        # Randomly select 3 points
        indices = random.sample(range(len(points)), 3)
        sample = points[indices]

        # Fit plane
        normal, d = fit_plane(sample)

        # Compute distances
        distances = compute_distance(points, normal, d)

        # Count inliers
        inliers = (distances < threshold).sum().item()

        # Update best model
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (normal, d)

    return best_plane, best_inliers


def compute_signed_distance_and_direction(mesh, points):
    """Computes signed distance and direction to the surface for given points."""
    proximity_query = trimesh.proximity.ProximityQuery(mesh)

    # Compute signed distances
    sdf = proximity_query.signed_distance(points)

    # Compute nearest surface points
    surface_points, distances, _ = proximity_query.on_surface(points)

    # Compute direction to the surface
    direction = surface_points - points
    magnitude = np.linalg.norm(direction, axis=1)  # Compute magnitude
    direction /= magnitude[:, np.newaxis]  # Normalize

    return sdf, magnitude, direction, points, surface_points


# ----------------------------------- Model Setup ------------------------------------ #
def create_splats_with_optimizers(
    pcl: torch.Tensor,
    rgb: torch.Tensor,
    cfg: GSConfig,
    norm: bool = True,
    device: str = "cuda:0"
):
    """
    Build the splats and optimizers for the Gaussian splatting model.
    """
    # GS params
    # means (N, 3)
    points = pcl
    # rgb (N, 3)
    if norm:
        rgbs = rgb / 255.0    
    else:
        rgbs = rgb
    colors = torch.logit(rgbs)

    distances, _ = k_nearest_torch(points, 3)
    avg_dist = torch.mean(distances, dim=-1, keepdim=True)
    scales = torch.log(avg_dist.repeat(1, 3))    

    N = points.shape[0]
    quats = torch.rand((N, 4)) 
    quats = F.normalize(quats)
    opacities = torch.logit(torch.full((N,), cfg.init_opacity)) 

    lrs = cfg.learning_rates
    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points, requires_grad=True), lrs["means"] * cfg.scene_scale),  
        ("scales", torch.nn.Parameter(scales, requires_grad=True), lrs["scales"]), # log (exp)
        ("colors", torch.nn.Parameter(colors, requires_grad=True), lrs["colors"]), # logit (sigmoid)
        ("quats", torch.nn.Parameter(quats, requires_grad=True), lrs["quats"]), 
        ("opacities", torch.nn.Parameter(opacities, requires_grad=True), lrs["opacities"]), # logit (sigmoid)
    ]
    
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to("cuda")

    world_size = 1.0
    batch_size = cfg.batch_size
    BS = batch_size * world_size
    
    optimizer_class = torch.optim.Adam
    optimizers: Dict[str, torch.optim.Optimizer] = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }

    return splats, optimizers


def create_rigid_object_splats_with_optimizers(
    pcl: torch.Tensor,
    rgb: torch.Tensor,
    cfg: RigidBodyConfig | RopeConfig,
    norm: bool = True,
    gaussians: bool = False,
    device: str = "cuda:0"
):
    """
    Build the splats and optimizers for the Gaussian splatting model.
    """
    # GS params
    # means (N, 3)
    points = pcl
    # rgb (N, 3)
    if norm:
        rgbs = rgb / 255.0    
    else:
        rgbs = rgb
    colors = torch.logit(rgbs)

    distances, _ = k_nearest_torch(points, 3)
    avg_dist = torch.mean(distances, dim=-1, keepdim=True)
    scales = avg_dist.repeat(1, 3)
    if cfg.disk:
        scales[:, 2] = 0.001
    scales = torch.log(scales)

    N = points.shape[0]
    quats = torch.rand((N, 4)) 
    quats = F.normalize(quats)
    opacities = torch.logit(torch.full((N,), cfg.init_opacity)) 

    if gaussians:
        lrs = cfg.learning_rates_gaussians  
    else:
        lrs = cfg.learning_rates_particles
        
    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points, requires_grad=True), lrs["means"] * cfg.scene_scale),  
        ("scales", torch.nn.Parameter(scales, requires_grad=True), lrs["scales"]), # log (exp)
        ("colors", torch.nn.Parameter(colors, requires_grad=True), lrs["colors"]), # logit (sigmoid)
        ("quats", torch.nn.Parameter(quats, requires_grad=True), lrs["quats"]), 
        ("opacities", torch.nn.Parameter(opacities, requires_grad=True), lrs["opacities"]), # logit (sigmoid)
    ]
    
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to("cuda")

    world_size = 1.0
    if cfg.particle_batch:
        batch_size = cfg.batch_size
    else:
        batch_size = 1
    BS = batch_size * world_size
    
    optimizer_class = torch.optim.Adam
    optimizers: Dict[str, torch.optim.Optimizer] = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }

    return splats, optimizers


# ----------------------------------- visualize GS ------------------------------------ #
def vis_render_gs(Ks: torch.Tensor, viewmats: torch.Tensor, splats, gt_img: torch.Tensor, near_plane=0.01, far_plane=5.0):
    """
    Visualize the Gaussian splatting model.
    """
    _, H, W, _ = gt_img.shape
    with torch.no_grad():
        render_image, _, _ = rasterization(
            means=splats["means"],
            quats=torch.nn.functional.normalize(splats["quats"]),
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=torch.sigmoid(splats["colors"]),
            viewmats=viewmats,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=W,
            height=H,
            near_plane=near_plane,
            far_plane=far_plane,
            packed=False,
            render_mode="RGB",
        )
        
    rgb = render_image.cpu().numpy()
    for cam in range(rgb.shape[0]):
        image_array = rgb[cam]
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        cv2.imshow(f"color_{cam}", image_array)

        gt_image = gt_img[cam].cpu().numpy()
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR)
        cv2.imshow(f"gt_{cam}", gt_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def vis_render_gs_cam(cam, splats, gt_img=None, near_plane=0.01, far_plane=5.0):
    """
    Visualize the Gaussian splatting model.
    """
    with torch.no_grad():
        render_image, _, _ = rasterization(
            means=splats["means"],
            quats=torch.nn.functional.normalize(splats["quats"]),
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=torch.sigmoid(splats["colors"]),
            viewmats=cam.viewmats,  # [C, 4, 4]
            Ks=cam.Ks,  # [C, 3, 3]
            width=cam.W,
            height=cam.H,
            near_plane=near_plane,
            far_plane=far_plane,
            packed=False,
            render_mode="RGB",
        )
    rgb = render_image.cpu().numpy()
    for cam in range(rgb.shape[0]):
        image_array = rgb[cam]
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        cv2.imshow(f"color_{cam}", image_array)
        if gt_img is not None:
            gt_image = gt_img[cam].cpu().numpy()
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR)
            cv2.imshow(f"gt_{cam}", gt_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

