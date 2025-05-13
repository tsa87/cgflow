import torch
import torch.nn.functional as F

import cgflow.util.algorithms as smolA


def aligned_coord_loss(pred_coords, coords, reduction="none"):
    # pred_coords: Predicted coordinates tensor (batch_size, N, 3)
    # coords: Target coordinates tensor (batch_size, N, 3)

    aligned_preds = torch.zeros_like(pred_coords)

    # Apply Kabsch alignment for each sample in the batch
    for i in range(pred_coords.size(0)):
        aligned_preds[i] = smolA.kabsch_alignment(pred_coords[i], coords[i])

    # Compute the MSE loss on the aligned coordinates
    coord_loss = F.mse_loss(aligned_preds, coords, reduction)

    return coord_loss


def pairwise_dist_loss(pred_coords, coords, mask=None):
    """
    Computes the pairwise distance loss between predicted and target coordinates,
    optionally applying a mask.

    Args:
      pred_coords (torch.Tensor): Predicted coordinates tensor (batch_size, N, 3)
      coords (torch.Tensor): Target coordinates tensor (batch_size, N, 3)
      mask (torch.Tensor, optional): Mask tensor (batch_size, N, N).
          Elements with value 1 indicate positions to include in the loss.
      reduction (str): 'mean', 'sum', or 'none'

    Returns:
      torch.Tensor: Loss value after applying the mask and reduction.
    """
    # Compute pairwise distance matrices for both predicted and target coordinates.
    pred_dists = smolA.pairwise_distance_matrix(pred_coords)
    target_dists = smolA.pairwise_distance_matrix(coords)

    # Compute element-wise squared error.
    loss = (pred_dists - target_dists) ** 2

    if mask is not None:
        # Ensure the mask is float so multiplication works as expected.
        mask = mask.float()
        loss = loss * mask
        return loss
    else:
        return loss
