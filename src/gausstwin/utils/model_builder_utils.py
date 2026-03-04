import torch
import numpy as np 
from typing import Tuple
from .math_utils import matrix_from_quat


def mass_inertia_spheres(centers, radii, density_factor):
    """Compute mass and inertia of spheres."""
    centers = np.asarray(centers, dtype=float)
    radii   = np.asarray(radii, dtype=float)
    I3 = np.eye(3)

    vols = (4.0 / 3.0) * np.pi * radii**3 # volumes of spheres
    m = density_factor * 1000.0 * vols # (N,)
    M = m.sum()

    # inertia of each sphere around its own center
    I_cm_i = (2.0/5.0) * (m * radii**2)[:,None,None] * I3 # (N,3,3)

    # # inertia w.r.t. the origin
    # I_origin = np.zeros((3,3))
    # for mi, ci, Icm in zip(m, centers, I_cm_i):
    #     r2 = np.dot(ci, ci)
    #     I_origin += Icm + mi * (r2 * I3 - np.outer(ci, ci))

    # # center of mass
    C = (m[:,None] * centers).sum(axis=0) / M

    # inertia w.r.t. the COM
    I_C = np.zeros((3,3))
    for mi, ci, Icm in zip(m, centers, I_cm_i):
        d = ci - C
        d2 = np.dot(d, d)
        I_C += Icm + mi * (d2 * I3 - np.outer(d, d))

    return M, I_C, C


def transform_particles(
    pose: torch.Tensor, 
    pts: torch.Tensor, 
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Transform particles by translation and rotation.
    R = pose[:3, :3]    
    t = pose[:3, 3]      

    # Transform points (apply R and t)
    pts_transformed = (R @ pts.T).T + t  # (N, 3)

    return pts_transformed


def transform_particles_with_sdf(
    pose: torch.Tensor, 
    pts: torch.Tensor, 
    sdf_dir: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Transform particles by translation and rotation.
    R = pose[:3, :3]    
    t = pose[:3, 3]      

    # Transform points (apply R and t)
    pts_transformed = (R @ pts.T).T + t  # (N, 3)

    # Rotate direction vectors only (no translation)
    sdf_dir_transformed = (R @ sdf_dir.T).T  
    
    return pts_transformed, sdf_dir_transformed


def compute_balanced_weights(
    particle_pts, 
    gaussian_particle_indices,
):
    N = gaussian_particle_indices.shape[0]
    device = particle_pts.device
    assert gaussian_particle_indices.max() < particle_pts.shape[0], "Index out of bounds"

    # Compute center of object
    center = particle_pts.mean(dim=0, keepdim=True)  # (1, 3)
    r_pt = particle_pts[gaussian_particle_indices] - center  # (N, 3)
    height_threshold = 0.95 * torch.max(r_pt[:, 2], dim=0).values  # (3,)
    is_top = r_pt[:, 2] > height_threshold  # (N,)

    # apply unit force
    force_direction = torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(N, 3)  # (N, 3)
    torques = torch.cross(r_pt, force_direction, dim=1)  # (N, 3)

     # Split torques
    torque_top = torques[is_top].sum(dim=0)        # (3,)
    torque_rest = torques[~is_top].sum(dim=0)      # (3,)

    dot = torch.dot(torque_top, torque_rest)
    norm_sq = torch.dot(torque_top, torque_top)
    d = -dot / (norm_sq + 1e-8)

    # Clamp to [0, 1]
    d_clamped = torch.clamp(d, 0.0, 1.0)

    # 6. Construct weights
    weights = torch.ones_like(gaussian_particle_indices, device=device)  # (N, 3)
    weights[is_top] = d_clamped.item()

    return weights  # shape (N,)

