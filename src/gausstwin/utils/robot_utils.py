import torch
import torch.nn.functional as F


def batch_transform(points_list, translations, quaternions, quat_list=None):
    """
    Applies a batch of rigid transformations to a variable-length set of 3D points in parallel.

    Args:
        points_list (list of torch.Tensor): List of tensors (N_i, 3) for each batch element.
        translations (torch.Tensor): Shape (B, 3) - Batch of translation vectors.
        quaternions (torch.Tensor): Shape (B, 4) - Batch of rotation quaternions (w, x, y, z).
    """
    B = len(points_list)
    if B == 0:  # Handle empty input
        return []

    # Normalize quaternions for valid rotations
    quaternions = F.normalize(quaternions, dim=1)

    # Compute rotation matrices for all batches (B, 3, 3)
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    R = torch.stack([
        torch.stack([1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)], dim=-1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)], dim=-1)
    ], dim=1)  # (B, 3, 3)

    # Concatenate all points and prepare batch indices
    all_points = torch.cat(points_list, dim=0)  # (total_N, 3)
    batch_indices = torch.cat([torch.full((p.shape[0],), i, dtype=torch.long) for i, p in enumerate(points_list)])

    # Apply rotation
    rotated_points = torch.bmm(R[batch_indices], all_points.unsqueeze(-1)).squeeze(-1)  # (total_N, 3)
    # Apply translations
    transformed_points = rotated_points + translations[batch_indices]

    # # Split back into list of tensors for each batch
    # transformed_points_list = []
    # start_idx = 0
    # for i in range(B):
    #     end_idx = start_idx + points_list[i].shape[0]
    #     transformed_points_list.append(transformed_points[start_idx:end_idx])
    #     start_idx = end_idx
    
    if quat_list is not None:
        def quat_mul(q1, q2):
            """Multiply two quaternions: q = q1 * q2"""
            w1, x1, y1, z1 = q1.unbind(-1)
            w2, x2, y2, z2 = q2.unbind(-1)
            return torch.stack([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
            ], dim=-1)

        all_quats = torch.cat(quat_list, dim=0)  # (total_N, 4)
        rotated_quats = quat_mul(quaternions[batch_indices], all_quats)  # (total_N, 4)

        # # Split back into list of tensors
        # lens = [p.shape[0] for p in points_list]
        # transformed_quats = torch.split(rotated_quats, lens)
        return transformed_points, rotated_quats

    return transformed_points
