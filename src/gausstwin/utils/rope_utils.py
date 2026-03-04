import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from collections import deque
from gsplat import rasterization
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
from gausstwin.utils.ps_utils import polyscope_vis_pcl


def extract_rope_centerline_from_gaussians(
    splats, 
    n_segments, 
    table_h, 
    obj_sphere_centers=None,
    device="cuda:0", 
    vis=False
) -> tuple[torch.Tensor, float, bool]:
    """Extract 3D rope centerline from trained Gaussian splats.

    Renders a top-down view, extracts the binary mask, skeletonizes it,
    resamples to equally-spaced 2D points, and back-projects to 3D.

    Args:
        splats: dict of Gaussian splat parameters (means, quats, scales, opacities, colors).
        n_segments: number of segments for centerline resampling.
        table_h: height of the table (ground plane).
        obj_sphere_centers: optional tensor of object sphere centers for reference.
        vis: if True, display the binary mask, skeleton and centerline.

    Returns:
        particles_3d: (n_segments+1, 3) tensor of 3D centerline points.
        particle_radius: estimated radius of the rope particles.
        is_loop: whether the rope forms a closed loop.
    """
    # get the gaussians
    means = splats["means"].detach()
    quats = F.normalize(splats["quats"]).detach()
    scales = torch.exp(splats["scales"]).detach()
    opacities = torch.sigmoid(splats["opacities"]).detach()
    colors = torch.sigmoid(splats["colors"]).detach()

    # set up a camera looking straight down at the rope
    render_H, render_W = 512, 512
    fov = 60
    focal = 0.5 * render_W / np.tan(0.5 * np.radians(fov))
    K = torch.tensor([
        [focal, 0, render_W / 2],
        [0, focal, render_H / 2],
        [0, 0, 1]
    ], device=device, dtype=torch.float32)
    Ks = K.unsqueeze(0)  # (1, 3, 3)

    rope_center = means.mean(dim=0)
    camera_height = 0.5
    camera_pos = rope_center + torch.tensor([0, 0, camera_height], device=device)

    # Camera extrinsics: looking straight down
    cam2world = torch.eye(4, device=device, dtype=torch.float32)
    cam2world[:3, :3] = torch.tensor([
        [1, 0, 0],     # camera X = world X
        [0, -1, 0],    # camera Y = world -Y
        [0, 0, -1],    # camera Z = world -Z (looking down)
    ], device=device, dtype=torch.float32)
    cam2world[:3, 3] = camera_pos

    viewmat = torch.linalg.inv(cam2world)
    viewmats = viewmat.unsqueeze(0)  # (1, 4, 4)

    # render top-down view
    with torch.no_grad():
        render_image, render_alpha, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=render_W,
            height=render_H,
            near_plane=0.01,
            far_plane=2.0,
            packed=False,
            render_mode="RGB",
            backgrounds=torch.zeros(1, 3, device=device),
        )

    rendered_alpha = render_alpha[0].cpu().numpy()  # (H, W, 1)

    # extract binary mask
    alpha_threshold = 0.2
    binary_mask = (rendered_alpha[:, :, 0] > alpha_threshold).astype(np.uint8)

    # fill holes using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # extract centerline
    extractor = RopeCenterlineExtractor()
    skeleton = extractor.extract_skeleton(binary_mask)
    ordered_points = extractor.order_skeleton_points(skeleton)

    resampled_2d = extractor.segment_centerline(
        ordered_points, num_segments=n_segments, use_spline=True, smoothing=0
    )

    if vis:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # binary mask
        axes[0].imshow(binary_mask, cmap='gray')
        axes[0].set_title('Binary Mask')
        axes[0].axis('off')

        # skeleton
        axes[1].imshow(skeleton, cmap='gray')
        axes[1].set_title(f'Skeleton ({len(ordered_points)} pts)')
        axes[1].axis('off')

        # centerline over mask
        axes[2].imshow(binary_mask, cmap='gray', alpha=0.5)
        if len(ordered_points) > 1:
            axes[2].plot(ordered_points[:, 0], ordered_points[:, 1], 'r-', linewidth=1, alpha=0.5)
        cl_colors = plt.cm.viridis(np.linspace(0, 1, len(resampled_2d)))
        axes[2].scatter(resampled_2d[:, 0], resampled_2d[:, 1], c=cl_colors, s=50, zorder=5)
        axes[2].plot(resampled_2d[0, 0], resampled_2d[0, 1], 'go', markersize=12, label='Start', zorder=6)
        axes[2].plot(resampled_2d[-1, 0], resampled_2d[-1, 1], 'ro', markersize=12, label='End', zorder=6)
        dists_2d = np.sqrt(np.sum(np.diff(resampled_2d, axis=0)**2, axis=1))
        axes[2].set_title(f'{n_segments} segments (spacing: {dists_2d.mean():.1f}px)')
        axes[2].legend()
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    # back-project 2D centerline points to 3D
    rope_z = rope_center[2].item()
    cx, cy = render_W / 2.0, render_H / 2.0
    particles_3d = np.zeros((len(resampled_2d), 3), dtype=np.float32)
    for i, (u, v) in enumerate(resampled_2d):
        x_cam = (u - cx) / focal * camera_height
        y_cam = (v - cy) / focal * camera_height
        # cam2world: x_world = x_cam, y_world = -y_cam (relative to camera_pos)
        particles_3d[i, 0] = x_cam + camera_pos[0].item()
        particles_3d[i, 1] = -y_cam + camera_pos[1].item()
        particles_3d[i, 2] = rope_z

    particles_3d = torch.tensor(particles_3d, device=device, dtype=torch.float32)
    
    # get rope radius
    if obj_sphere_centers is not None:
        particle_radius = (torch.max(obj_sphere_centers[:, 2]).item() - torch.min(obj_sphere_centers[:, 2]).item()) / 2.0
        particles_3d[i, 2] = table_h + particle_radius
    else:
        particle_radius = rope_z - table_h
        
    if vis:
        pcl_list = [particles_3d]
        if obj_sphere_centers is not None:
            pcl_list.append(obj_sphere_centers)
        polyscope_vis_pcl(pcl_list)

    return particles_3d, particle_radius, extractor.is_loop


class RopeCenterlineExtractor:
    """Extract and segment rope centerline from binary mask images."""

    def __init__(self):
        self.skeleton = None
        self.ordered_points = None
        self.segments = None
        self.is_loop = False

    def extract_skeleton(self, binary_image):
        """Extract skeleton using Zhang-Suen thinning algorithm."""
        binary = binary_image > 0
        skeleton = skeletonize(binary, method='lee')
        self.skeleton = skeleton
        return skeleton

    def order_skeleton_points(self, skeleton, start_point=None):
        """Order skeleton points by finding the longest path between endpoints."""
        points = np.argwhere(skeleton)
        if len(points) == 0:
            return np.array([])

        points = points[:, [1, 0]]  # Swap to (x, y)
        adjacency = self._build_adjacency_graph(skeleton)
        endpoints = self._find_endpoints(adjacency)

        is_loop = len(endpoints) == 0
        self.is_loop = is_loop

        if is_loop:
            ordered_indices = self._traverse_skeleton(adjacency, 0, is_loop=True)
        elif len(endpoints) >= 2:
            ordered_indices = self._find_longest_path(adjacency, endpoints)
        else:
            start_idx = endpoints[0] if endpoints else 0
            ordered_indices = self._traverse_skeleton(adjacency, start_idx, is_loop=False)

        ordered_points = points[ordered_indices]
        self.ordered_points = ordered_points
        return ordered_points

    def _find_longest_path(self, adjacency, endpoints):
        """Find the longest path between any two endpoints using BFS."""
        longest_path = []

        for start in endpoints:
            distances = {start: 0}
            parents = {start: None}
            queue = deque([start])

            while queue:
                node = queue.popleft()
                for neighbor in adjacency[node]:
                    if neighbor not in distances:
                        distances[neighbor] = distances[node] + 1
                        parents[neighbor] = node
                        queue.append(neighbor)

            for end in endpoints:
                if end != start and end in distances:
                    path = []
                    current = end
                    while current is not None:
                        path.append(current)
                        current = parents[current]
                    path = path[::-1]

                    if len(path) > len(longest_path):
                        longest_path = path

        return longest_path

    def _build_adjacency_graph(self, skeleton):
        """Build adjacency graph from skeleton."""
        points = np.argwhere(skeleton)
        n_points = len(points)
        adjacency = {i: [] for i in range(n_points)}
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        point_map = {}
        for idx, (r, c) in enumerate(points):
            point_map[(r, c)] = idx

        for idx, (r, c) in enumerate(points):
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if (nr, nc) in point_map:
                    neighbor_idx = point_map[(nr, nc)]
                    if neighbor_idx not in adjacency[idx]:
                        adjacency[idx].append(neighbor_idx)

        return adjacency

    def _find_endpoints(self, adjacency):
        """Find endpoints in the skeleton graph (degree 1 nodes)."""
        return [node for node, neighbors in adjacency.items() if len(neighbors) == 1]

    def _traverse_skeleton(self, adjacency, start_idx, is_loop=False):
        """Traverse skeleton graph to order points."""
        visited = set()
        ordered = []

        if is_loop:
            queue = deque([start_idx])

            while queue:
                node = queue.popleft()

                if node in visited:
                    continue

                visited.add(node)
                ordered.append(node)

                unvisited_neighbors = [n for n in adjacency[node] if n not in visited]

                if len(ordered) > 1 and unvisited_neighbors:
                    prev_node = ordered[-2]
                    for neighbor in unvisited_neighbors:
                        if neighbor != prev_node:
                            queue.append(neighbor)
                            break
                    else:
                        queue.extend(unvisited_neighbors)
                else:
                    queue.extend(unvisited_neighbors)
        else:
            stack = [start_idx]

            while stack:
                node = stack.pop()

                if node in visited:
                    continue

                visited.add(node)
                ordered.append(node)

                unvisited_neighbors = [n for n in adjacency[node] if n not in visited]

                if unvisited_neighbors:
                    def priority(n):
                        if n in visited:
                            return 999
                        degree = len(adjacency[n])
                        if degree == 2:
                            return 0
                        elif degree == 1:
                            return 1
                        else:
                            return 2

                    unvisited_neighbors.sort(key=priority)
                    stack.extend(reversed(unvisited_neighbors))

        return ordered

    def segment_centerline(self, ordered_points, num_segments, use_spline=True, smoothing=None):
        """Segment the centerline into equally spaced points."""
        if use_spline and len(ordered_points) >= 4:
            x = ordered_points[:, 0]
            y = ordered_points[:, 1]
            tck, _ = splprep([x, y], s=smoothing, k=3)
            u_new = np.linspace(0, 1, num_segments + 1)
            x_new, y_new = splev(u_new, tck)
            segments = np.column_stack([x_new, y_new])
        else:
            dists = np.sqrt(np.sum(np.diff(ordered_points, axis=0)**2, axis=1))
            cumulative_dist = np.concatenate([[0], np.cumsum(dists)])
            total_length = cumulative_dist[-1]
            target_dists = np.linspace(0, total_length, num_segments + 1)

            segments = np.zeros((num_segments + 1, 2))
            for i, target_dist in enumerate(target_dists):
                idx = np.searchsorted(cumulative_dist, target_dist)
                if idx == 0:
                    segments[i] = ordered_points[0]
                elif idx >= len(ordered_points):
                    segments[i] = ordered_points[-1]
                else:
                    t = (target_dist - cumulative_dist[idx-1]) / (cumulative_dist[idx] - cumulative_dist[idx-1] + 1e-8)
                    segments[i] = (1 - t) * ordered_points[idx-1] + t * ordered_points[idx]

        self.segments = segments
        return segments
    