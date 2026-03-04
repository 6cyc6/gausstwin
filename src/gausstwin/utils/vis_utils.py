import cv2
import torch
import warp as wp
import numpy as np
import polyscope as ps
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gausstwin.utils.math_utils as math_utils

from PIL import Image
from typing import Union
from scipy.linalg import eigh
from collections.abc import Sequence
from scipy.spatial.transform import Rotation as R


TensorData = Union[np.ndarray, torch.Tensor, wp.array]


def convert_to_torch(
    array: TensorData,
    dtype: torch.dtype = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Converts a given array into a torch tensor.

    The function tries to convert the array to a torch tensor. If the array is a numpy/warp arrays, or python
    list/tuples, it is converted to a torch tensor. If the array is already a torch tensor, it is returned
    directly.

    If ``device`` is None, then the function deduces the current device of the data. For numpy arrays,
    this defaults to "cpu", for torch tensors it is "cpu" or "cuda", and for warp arrays it is "cuda".

    Note:
        Since PyTorch does not support unsigned integer types, unsigned integer arrays are converted to
        signed integer arrays. This is done by casting the array to the corresponding signed integer type.

    Args:
        array: The input array. It can be a numpy array, warp array, python list/tuple, or torch tensor.
        dtype: Target data-type for the tensor.
        device: The target device for the tensor. Defaults to None.

    Returns:
        The converted array as torch tensor.
    """
    # Convert array to tensor
    # if the datatype is not currently supported by torch we need to improvise
    # supported types are: https://pytorch.org/docs/stable/tensors.html
    if isinstance(array, torch.Tensor):
        tensor = array
    elif isinstance(array, np.ndarray):
        if array.dtype == np.uint32:
            array = array.astype(np.int32)
        # need to deal with object arrays (np.void) separately
        tensor = torch.from_numpy(array)
    elif isinstance(array, wp.array):
        if array.dtype == wp.uint32:
            array = array.view(wp.int32)
        tensor = wp.to_torch(array)
    else:
        tensor = torch.Tensor(array)
    # Convert tensor to the right device
    if device is not None and str(tensor.device) != str(device):
        tensor = tensor.to(device)
    # Convert dtype of tensor if requested
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.type(dtype)

    return tensor


def to_torch(inputs, device="cuda:0"):
    """
    Convert the numpy arrays to torch tensors.
    """
    if isinstance(inputs[0], np.ndarray):
        return convert_to_torch(np.array(inputs), dtype=torch.float, device=device)
    else:
        if isinstance(inputs, list):
            return torch.stack(inputs, dim=0).to(device)
        else:
            return inputs.to(device)


def visualize_image(image_array):
    """
    Display the image using matplotlib.
    
    :param image_array: NumPy array representation of the image.
    """
    
    if image_array.shape[0] == 3:
        image_array = image_array.transpose(1, 2, 0)
    plt.imshow(image_array)
    plt.axis('off')  # Hide axes
    plt.show()


def vis_cv2_image(image_array, name="Image"):
    """
    Display the image using OpenCV.
    
    :param image_array: NumPy array representation of the image.
    """
    
    if image_array.shape[-1] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, image_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_image_and_bbox(image_array, bbox):
    """
    Display image and bounding box
    """
    fig, ax = plt.subplots(1, figsize=(8, 6))

    # Show the image
    ax.imshow(image_array)
    
    # Create a rectangle patch (bounding box)
    for _, (u, v, w, h) in enumerate(bbox):
        if isinstance(u, torch.Tensor):
            u = u.cpu().numpy()
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        if isinstance(w, torch.Tensor):
            w = w.cpu().numpy()
        if isinstance(h, torch.Tensor):
            h = h.cpu().numpy()
        rect = patches.Rectangle(
            (u - w , v - h), w * 2, h * 2,  # type: ignore (x, y, width, height)
            linewidth=2, edgecolor='red', facecolor='none'  # Red border, no fill
        )
        # Add the rectangle to the plot
        ax.add_patch(rect)
        
    plt.show()
    

def depth_to_pointcloud(depth_image, intrinsic_matrix, position, quaternion):
    """
    Convert a depth image to a 3D point cloud.
    
    :param depth_image: NumPy array of the depth image.
    :param intrinsic_matrix: 3x3 camera intrinsic matrix.
    :param position: 3D position of the camera.
    :param quaternion: Quaternion (w, x, y, z) representing orientation.
    :return: Open3D point cloud object.
    """
    height, width, _ = depth_image.shape
    depth_image[:, :, 0] = np.where(depth_image[:, :, 0] > 4.0, 0.0, depth_image[:, :, 0])
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))
    x_indices = x_indices.flatten()
    y_indices = y_indices.flatten()
    
    z = depth_image.flatten()
    x = (x_indices - cx) * z / fx
    y = (y_indices - cy) * z / fy
    
    points = np.vstack((x, y, z)).T
    
    quaternion = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    # Apply camera rotation and translation
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    transformed_points = (rotation_matrix @ points.T).T + position
    
    return transformed_points


def uv_to_world_pts(depths, us, vs, intrinsics, translation, quaternion, device="cuda"):
    """
    Convert a depth image to a world-frame point cloud using quaternion rotation.

    Args:
        depth (torch.Tensor): Depth of a point.
        us (torch.Tensor): U coordinates.
        vs (torch.Tensor): V coordinates.
        intrinsics (torch.Tensor): Camera intrinsics matrix (3, 3).
        translation (torch.Tensor): Translation vector (3,).
        quaternion (torch.Tensor): Quaternion (w, x, y, z).
        device (str): "cuda" or "cpu".

    Returns:
        torch.Tensor: World-frame point cloud (N, 3).
    """
    # Invert intrinsic matrix
    K_inv = torch.inverse(intrinsics)

    # # Create pixel grids
    # # height, width, _ = depth.shape
    # u = torch.arange(0, width, device=device).float().expand(height, width)
    # v = torch.arange(0, height, device=device).float().unsqueeze(1).expand(height, width)

    pixels = torch.tensor([[us, vs, 1]], device=device).float()  
    
    # Depth values
    Z = depths.reshape(-1)  # Flatten to (H*W,)
    # valid_mask = torch.logical_and(Z > 0, Z < 4.0) # Remove zero-depth points
    # pixels = pixels[valid_mask]
    # Z = Z[valid_mask]
    invalid_mask = torch.logical_or(Z <= 0, Z > 4.0)
    Z[invalid_mask] = 0.0

    # Convert to camera frame: X_c = K_inv @ [u, v, 1]^T * Z
    points_camera = torch.mm(pixels, K_inv.T) * Z[:, None]

    # Convert quaternion to rotation matrix
    R = math_utils.matrix_from_quat(quaternion).to(device)

    # Transform to world frame: p_world = R * p_camera + t
    points_world = torch.mm(points_camera, R.T) + translation

    return points_world


def uvs_to_world_pts(depths, uvs, intrinsics, translation, quaternion, device="cuda"):
    """
    Convert a depth image to a world-frame point cloud using quaternion rotation.

    Args:
        depth (torch.Tensor): Depth of a point.
        us (torch.Tensor): U coordinates.
        vs (torch.Tensor): V coordinates.
        intrinsics (torch.Tensor): Camera intrinsics matrix (3, 3).
        translation (torch.Tensor): Translation vector (3,).
        quaternion (torch.Tensor): Quaternion (w, x, y, z).
        device (str): "cuda" or "cpu".

    Returns:
        torch.Tensor: World-frame point cloud (N, 3).
    """
    # remove invalid pts
    Z = depths.reshape(-1)  # Flatten to (H*W,)
    valid_mask = torch.logical_and(Z > 0, Z < 4.0)
    Z = Z[valid_mask]
    uvs = uvs[valid_mask]
    
    B = uvs.shape[0]  # Number of points

    uv_h = torch.cat([uvs, torch.ones(B, 1, device=device)], dim=1).unsqueeze(2)
    
    # Convert to camera frame: X_c = K_inv @ [u, v, 1]^T * Z
    # points_camera = torch.mm(pixels, K_inv.T) * Z[:, None]
    points_camera = torch.linalg.solve(intrinsics, uv_h) * Z.view(B, 1, 1)

    # Convert quaternion to rotation matrix
    R = math_utils.matrix_from_quat(quaternion).to(device)

    # Transform to world frame: p_world = R * p_camera + t
    points_world = torch.mm(points_camera.squeeze(-1), R.T) + translation

    return points_world



def cam_to_world(position, quaternion):
    """
    Convert camera position and orientation to world coordinates.
    
    :param position: 3D position of the camera.
    :param quaternion: Quaternion (w, x, y, z) representing orientation.
    :return: Transformation matrix.
    """
    quaternion = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position
    
    return transformation_matrix


def transform_points(
    points: TensorData,
    position: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    device: torch.device | str | None = None,
) -> np.ndarray | torch.Tensor:
    r"""Transform input points in a given frame to a target frame.

    This function transform points from a source frame to a target frame. The transformation is defined by the
    position ``t`` and orientation ``R`` of the target frame in the source frame.

    .. math::
        p_{target} = R_{target} \times p_{source} + t_{target}

    If either the inputs `position` and `orientation` are None, the corresponding transformation is not applied.

    Args:
        points: a tensor of shape (p, 3) or (n, p, 3) comprising of 3d points in source frame.
        position: The position of source frame in target frame. Defaults to None.
        orientation: The orientation (w, x, y, z) of source frame in target frame.
            Defaults to None.
        device: The device for torch where the computation
            should be executed. Defaults to None, i.e. takes the device that matches the depth image.

    Returns:
        A tensor of shape (N, 3) comprising of 3D points in target frame.
        If the input is a numpy array, the output is a numpy array. Otherwise, it is a torch tensor.
    """
    # check if numpy
    is_numpy = isinstance(points, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert to torch
    points = convert_to_torch(points, dtype=torch.float32, device=device)
    # update the device with the device of the depth image
    # note: this is needed since warp does not provide the device directly
    device = points.device
    # apply rotation
    if orientation is not None:
        orientation = convert_to_torch(orientation, dtype=torch.float32, device=device)
    # apply translation
    if position is not None:
        position = convert_to_torch(position, dtype=torch.float32, device=device)
    # apply transformation
    points = math_utils.transform_points(points, position, orientation)

    # return everything according to input type
    if is_numpy:
        return points.detach().cpu().numpy()
    else:
        return points


def create_pointcloud_from_depth(
    intrinsic_matrix: np.ndarray | torch.Tensor | wp.array,
    depth: np.ndarray | torch.Tensor | wp.array,
    keep_invalid: bool = False,
    position: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    device: torch.device | str | None = None,
) -> np.ndarray | torch.Tensor:
    r"""Creates pointcloud from input depth image and camera intrinsic matrix.

    This function creates a pointcloud from a depth image and camera intrinsic matrix. The pointcloud is
    computed using the following equation:

    .. math::
        p_{camera} = K^{-1} \times [u, v, 1]^T \times d

    where :math:`K` is the camera intrinsic matrix, :math:`u` and :math:`v` are the pixel coordinates and
    :math:`d` is the depth value at the pixel.

    Additionally, the pointcloud can be transformed from the camera frame to a target frame by providing
    the position ``t`` and orientation ``R`` of the camera in the target frame:

    .. math::
        p_{target} = R_{target} \times p_{camera} + t_{target}

    Args:
        intrinsic_matrix: A (3, 3) array providing camera's calibration matrix.
        depth: An array of shape (H, W) with values encoding the depth measurement.
        keep_invalid: Whether to keep invalid points in the cloud or not. Invalid points
            correspond to pixels with depth values 0.0 or NaN. Defaults to False.
        position: The position of the camera in a target frame. Defaults to None.
        orientation: The orientation (w, x, y, z) of the camera in a target frame. Defaults to None.
        device: The device for torch where the computation should be executed.
            Defaults to None, i.e. takes the device that matches the depth image.

    Returns:
        An array/tensor of shape (N, 3) comprising of 3D coordinates of points.
        The returned datatype is torch if input depth is of type torch.tensor or wp.array. Otherwise, a np.ndarray
        is returned.
    """
    # We use PyTorch here for matrix multiplication since it is compiled with Intel MKL while numpy
    # by default uses OpenBLAS. With PyTorch (CPU), we could process a depth image of size (480, 640)
    # in 0.0051 secs, while with numpy it took 0.0292 secs.

    # convert to numpy matrix
    is_numpy = isinstance(depth, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert depth to torch tensor
    depth = convert_to_torch(depth, dtype=torch.float32, device=device)
    # update the device with the device of the depth image
    # note: this is needed since warp does not provide the device directly
    device = depth.device
    # convert inputs to torch tensors
    intrinsic_matrix = convert_to_torch(intrinsic_matrix, dtype=torch.float32, device=device)
    if position is not None:
        position = convert_to_torch(position, dtype=torch.float32, device=device)
    if orientation is not None:
        orientation = convert_to_torch(orientation, dtype=torch.float32, device=device)
    # compute pointcloud
    depth_cloud = math_utils.unproject_depth(depth, intrinsic_matrix)
    # convert 3D points to world frame
    depth_cloud = math_utils.transform_points(depth_cloud, position, orientation)

    # keep only valid entries if flag is set
    if not keep_invalid:
        pts_idx_to_keep = torch.all(torch.logical_and(~torch.isnan(depth_cloud), ~torch.isinf(depth_cloud)), dim=1)
        depth_cloud = depth_cloud[pts_idx_to_keep, ...]

    # return everything according to input type
    if is_numpy:
        return depth_cloud.detach().cpu().numpy()
    else:
        return depth_cloud


def create_pointcloud_from_rgbd(
    intrinsic_matrix: torch.Tensor | np.ndarray | wp.array,
    depth: torch.Tensor | np.ndarray | wp.array,
    rgb: torch.Tensor | wp.array | np.ndarray | tuple[float, float, float] = None,
    normalize_rgb: bool = False,
    position: Sequence[float] | None = None,
    orientation: Sequence[float] | None = None,
    device: torch.device | str | None = None,
    num_channels: int = 3,
    remove: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray]:
    """Creates pointcloud from input depth image and camera transformation matrix.

    This function provides the same functionality as :meth:`create_pointcloud_from_depth` but also allows
    to provide the RGB values for each point.

    The ``rgb`` attribute is used to resolve the corresponding point's color:

    - If a ``np.array``/``wp.array``/``torch.tensor`` of shape (H, W, 3), then the corresponding channels encode RGB values.
    - If a tuple, then the point cloud has a single color specified by the values (r, g, b).
    - If None, then default color is white, i.e. (0, 0, 0).

    If the input ``normalize_rgb`` is set to :obj:`True`, then the RGB values are normalized to be in the range [0, 1].

    Args:
        intrinsic_matrix: A (3, 3) array/tensor providing camera's calibration matrix.
        depth: An array/tensor of shape (H, W) with values encoding the depth measurement.
        rgb: Color for generated point cloud. Defaults to None.
        normalize_rgb: Whether to normalize input rgb. Defaults to False.
        position: The position of the camera in a target frame. Defaults to None.
        orientation: The orientation `(w, x, y, z)` of the camera in a target frame. Defaults to None.
        device: The device for torch where the computation should be executed. Defaults to None, in which case
            it takes the device that matches the depth image.
        num_channels: Number of channels in RGB pointcloud. Defaults to 3.

    Returns:
        A tuple of (N, 3) arrays or tensors containing the 3D coordinates of points and their RGB color respectively.
        The returned datatype is torch if input depth is of type torch.tensor or wp.array. Otherwise, a np.ndarray
        is returned.

    Raises:
        ValueError:  When rgb image is a numpy array but not of shape (H, W, 3) or (H, W, 4).
    """
    # check valid inputs
    if rgb is not None and not isinstance(rgb, tuple):
        if len(rgb.shape) == 3:
            if rgb.shape[2] not in [3, 4]:
                raise ValueError(f"Input rgb image of invalid shape: {rgb.shape} != (H, W, 3) or (H, W, 4).")
        else:
            raise ValueError(f"Input rgb image not three-dimensional. Received shape: {rgb.shape}.")
    if num_channels not in [3, 4]:
        raise ValueError(f"Invalid number of channels: {num_channels} != 3 or 4.")

    # check if input depth is numpy array
    is_numpy = isinstance(depth, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert depth to torch tensor
    if is_numpy:
        depth = torch.from_numpy(depth).to(device=device)
    # retrieve XYZ pointcloud
    points_xyz = create_pointcloud_from_depth(intrinsic_matrix, depth, True, position, orientation, device=device)
    
    # get image height and width
    im_height, im_width = depth.shape[:2]
    # total number of points
    num_points = im_height * im_width
    # extract color value
    if rgb is not None:
        if isinstance(rgb, (np.ndarray, torch.Tensor, wp.array)):
            # copy numpy array to preserve
            rgb = convert_to_torch(rgb, device=device, dtype=torch.float32)
            rgb = rgb[:, :, :3]
            # convert the matrix to (W, H, 3) from (H, W, 3) since depth processing
            # is done in the order (u, v) where u: (0, W-1) and v: (0 - H-1)
            points_rgb = rgb.permute(1, 0, 2).reshape(-1, 3)
        elif isinstance(rgb, (tuple, list)):
            # same color for all points
            points_rgb = torch.Tensor((rgb,) * num_points, device=device, dtype=torch.uint8)
        else:
            # default color is white
            points_rgb = torch.Tensor(((0, 0, 0),) * num_points, device=device, dtype=torch.uint8)
    else:
        points_rgb = torch.Tensor(((0, 0, 0),) * num_points, device=device, dtype=torch.uint8)
    # normalize color values
    if normalize_rgb:
        points_rgb = points_rgb.float() / 255

    # remove invalid points
    pts_idx_to_keep = torch.all(torch.logical_and(~torch.isnan(points_xyz), ~torch.isinf(points_xyz)), dim=1)
    if remove:
        points_rgb = points_rgb[pts_idx_to_keep, ...]
        points_xyz = points_xyz[pts_idx_to_keep, ...]
    else:
        points_xyz[~pts_idx_to_keep, ...] = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)

    # add additional channels if required
    if num_channels == 4:
        points_rgb = torch.nn.functional.pad(points_rgb, (0, 1), mode="constant", value=1.0)

    # return everything according to input type
    if is_numpy:
        return points_xyz.cpu().numpy(), points_rgb.cpu().numpy()
    else:
        return points_xyz, points_rgb
    
    
def mask_to_uv(mask):
    """
    Extracts (u, v) coordinates from a binary mask.
    """
    # H, W = mask.shape  # Get height and width
    v, u = torch.nonzero(mask, as_tuple=True)  # Extract pixel coordinates

    return torch.stack((u, v), dim=1)


# ----------------------------- Util Functions for Visualization ----------------------------- #
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def create_ellipsoid(mean, cov, num_points=20):
    """
    Create Gaussian ellipsoid mesh for a given mean and covariance.
    """
    phi = np.linspace(0, np.pi, num_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi, theta = np.meshgrid(phi, theta)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    sphere_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # Eigen decomposition to get Gaussian axes
    eigenvalues, eigenvectors = eigh(cov)
    scaling = np.sqrt(eigenvalues) 

    # Scale and rotate sphere
    scaled_points = sphere_points @ np.diag(scaling)
    rotated_points = scaled_points @ eigenvectors.T
    ellipsoid_points = rotated_points + mean

    # Triangulate sphere surface
    faces = []
    for i in range(num_points - 1):
        for j in range(num_points - 1):
            p1 = i * num_points + j
            p2 = p1 + 1
            p3 = (i + 1) * num_points + j
            p4 = p3 + 1

            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])

    return ellipsoid_points, np.array(faces)


def show_masks_cv2(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True, window_name="Mask Display"):
    image = image.copy()

    # Convert RGB to BGR if needed (optional, depending on your input image)
    if image.shape[2] == 3 and np.max(image) <= 255:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        overlay = image.copy()

        # Color mask overlay
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[mask > 0] = (0, 255, 0)  # green
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

        # Draw mask border (contour)
        if borders:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

        # Draw points
        if point_coords is not None and input_labels is not None:
            for (x, y), label in zip(point_coords, input_labels):
                color = (0, 255, 0) if label == 1 else (0, 0, 255)  # green or red
                cv2.circle(overlay, (int(x), int(y)), 6, color, -1)

        # Draw bounding box
        if box_coords is not None:
            if isinstance(box_coords[0], (list, tuple, np.ndarray)):
                # Multiple boxes
                for box in box_coords:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
            else:
                # Single box
                x1, y1, x2, y2 = map(int, box_coords)
                print(f"Drawing box: {x1}, {y1}, {x2}, {y2}")
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Add score text
        if len(scores) > 1:
            cv2.putText(
                overlay, f"Mask {i+1}, Score: {score:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA
            )

        cv2.imshow(f"{window_name} {i+1}", overlay)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    

def show_track_masks_cv2(image, mask, point_coords=None, box_coords=None, input_labels=None, borders=True, window_name="Mask Display"):
    image = image.copy()

    # Convert RGB to BGR if needed (optional, depending on your input image)
    if image.shape[2] == 3 and np.max(image) <= 255:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    overlay = image.copy()

    # Color mask overlay
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask > 0] = (0, 255, 0)  # green
    overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

    # Draw mask border (contour)
    if borders:
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    # Draw points
    if point_coords is not None and input_labels is not None:
        for (x, y), label in zip(point_coords, input_labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)  # green or red
            cv2.circle(overlay, (int(x), int(y)), 6, color, -1)

    # Draw bounding box
    if box_coords is not None:
        x1, y1, x2, y2 = map(int, box_coords)
        print(f"Drawing box: {x1}, {y1}, {x2}, {y2}")
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    cv2.imshow(f"{window_name}", overlay)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    
    
def show_img_cv2(image, window_name="Image"):
    """
    Display an image using OpenCV.
    
    :param image: NumPy array representation of the image.
    :param window_name: Name of the window to display the image.
    """
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def show_img_and_bbox_cv2(image, bbox, window_name="Image"):
    """
    Display an image using OpenCV.
    
    :param image: NumPy array representation of the image.
    :param window_name: Name of the window to display the image.
    """
    # Draw rectangle
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    