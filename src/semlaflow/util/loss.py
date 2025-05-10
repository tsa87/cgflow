import torch
import torch.nn.functional as F
import semlaflow.util.algorithms as smolA


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


def pairwise_dist_loss(pred_coords, coords, mask=None, reduction="none"):
    # pred_coords: Predicted coordinates tensor (batch_size, N, 3)
    # coords: Target coordinates tensor (batch_size, N, 3)
    # mask: Mask tensor (batch_size, N, N)

    # Compute pairwise distances for predicted and target coordinates
    # (batch_size, N, N)
    pred_dists = smolA.pairwise_distance_matrix(pred_coords)
    target_dists = smolA.pairwise_distance_matrix(coords)

    # # Compute the MSE loss on the pairwise distances
    dist_loss = F.mse_loss(pred_dists, target_dists, reduction)

    if mask is not None:
        dist_loss = dist_loss * mask
    return dist_loss
