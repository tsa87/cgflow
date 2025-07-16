import torch
from scipy.spatial.transform.rotation import Rotation

_T = torch.Tensor
TupleRot = tuple[float, float, float]

# *** Tensor Util Functions ***


def pad_and_stack(data, ptr, length=None, device="cuda"):
    """Pad and stack a list of data tensors.
    Args:
        data: Tensor of shape (num_data, dim)
        ptr: Tensor of shape (batch_size + 1) with the pointers to the start of each batch
    """
    dim_size = data.shape[1]
    batch_size = ptr.shape[0] - 1

    if length is None:
        length = torch.max(ptr[1:] - ptr[:-1])

    result = torch.zeros((batch_size, length, dim_size))
    for i in range(batch_size):
        start, end = ptr[i], ptr[i + 1]
        result[i, : end - start, :] = data[start:end]

    return result.to(device)


def pad_tensors(tensors: list[_T], pad_dim: int = 0) -> _T:
    """Pad a list of tensors with zeros

    All dimensions other than pad_dim must have the same shape. A single tensor is returned with the batch dimension
    first, where the batch dimension is the length of the tensors list.

    Args:
        tensors (list[torch.Tensor]): List of tensors
        pad_dim (int): Dimension on tensors to pad. All other dimensions must be the same size.

    Returns:
        torch.Tensor: Batched, padded tensor, if pad_dim is 0 then shape [B, L, *] where L is length of longest tensor.
    """

    if pad_dim != 0:
        # TODO
        raise NotImplementedError()

    padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    return padded


def pairwise_concat(t: _T) -> _T:
    """Concatenates two representations from all possible pairings in dimension 1

    Computes all possible pairs of indices into dimension 1 and concatenates whatever representation they have in
    higher dimensions. Note that all higher dimensions will be flattened. The output will have its shape for
    dimension 1 duplicated in dimension 2.

    Example:
    Input shape [100, 16, 128]
    Output shape [100, 16, 16, 256]
    """

    idx_pairs = torch.cartesian_prod(*((torch.arange(t.shape[1]),) * 2))
    output = t[:, idx_pairs].view(t.shape[0], t.shape[1], t.shape[1], -1)
    return output


def segment_sum(data, segment_ids, num_segments):
    """Computes the sum of data elements that are in each segment

    The inputs must have shapes that look like the following:
    data [batch_size, seq_length, num_features]
    segment_ids [batch_size, seq_length], must contain integers

    Then the output will have the following shape:
    output [batch_size, num_segments, num_features]
    """

    err_msg = "data and segment_ids must have the same shape in the first two dimensions"
    assert data.shape[0:2] == segment_ids.shape[0:2], err_msg

    result_shape = (data.shape[0], num_segments, data.shape[2])
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, -1, data.shape[2])
    result.scatter_add_(1, segment_ids, data)
    return result


# *** Functions for handling edges ***


def compute_local_mask_batched(adj, max_hops=2):
    """
    Compute a local mask based on bond connectivity for batched data.

    Args:
        adj (torch.Tensor): Batched adjacency matrix of shape (B, N, N) with 1 for a bond, 0 otherwise.
        max_hops (int): Maximum number of bond hops to consider for local grouping.

    Returns:
        torch.Tensor: Boolean mask of shape (B, N, N) where True indicates that a pair of atoms
                      is connected within max_hops bonds.
    """
    B, N, _ = adj.shape
    # Start with self-connections (0 hops): each atom is connected to itself.
    local_mask = torch.eye(N, dtype=torch.bool, device=adj.device).unsqueeze(0).expand(B, N, N)

    # Direct bonds (1 hop)
    current = (adj > 0).bool()  # (B, N, N)
    local_mask = local_mask | current

    # Expand connectivity for additional hops.
    for _ in range(2, max_hops + 1):
        # Use batched matrix multiplication to propagate connections.
        # Casting to float for multiplication, then thresholding.
        current = torch.matmul(current.float(), adj.float()) > 0
        local_mask = local_mask | current

    return local_mask


def adj_from_node_mask(mask, self_connect=False):
    """Creates an edge mask from a given node mask assuming all nodes are fully connected excluding self-connections

    Args:
        mask (torch.BoolTensor): Node mask tensor, shape [batch_size, num_nodes], True for real node False otherwise
        self_connect (bool): Whether to include self connections in the adjacency

    Returns:
        torch.Tensor: Adjacency tensor, shape [batch_size, num_nodes, num_nodes], 1 for real edge 0 otherwise
    """

    # Matrix mult gives us an outer product on the node mask, which is an edge mask
    adjacency = mask.unsqueeze(1) & mask.unsqueeze(2)  # [batch_size, num_nodes, num_nodes]

    # Remove diagonal connections
    if not self_connect:
        num_nodes = mask.size(1)
        node_idxs = torch.arange(num_nodes, device=adjacency.device)
        adjacency[:, node_idxs, node_idxs] = False

    return adjacency


def bonds_from_adj(adj: _T, lower_tri=True) -> tuple[_T, _T]:
    """Flatten an adjacency matrix into a 1D edge representation

    Args:
        adj (torch.Tensor): Adjacency matrix. shape [num_nodes, num_nodes].
        lower_tri (bool): Whether to only consider bonds which sit in the lower triangular of adj_matrix.

    Returns:
        bond_indices (torch.Tensor): Bond indices [num_edges, 2]
        bond_type (torch.Tensor): Bond types [num_edges]
    """
    assert adj.dim() == 2, "Adjacency matrix must be 2D"
    if lower_tri:
        adj = torch.tril(adj, diagonal=-1)
    u, v = torch.where(adj)
    bond_indices = torch.stack([u, v], dim=-1)
    bond_types = adj[u, v]
    return bond_indices, bond_types


def bonds_from_batched_adj(adj: _T, lower_tri=True) -> list[tuple[_T, _T]]:
    """Flatten a batched adjacency matrix into a 1D edge representation

    Args:
        adj (torch.Tensor): Batched djacency matrix, shape [batch_size, num_nodes, num_nodes].
        lower_tri (bool): Whether to only consider bonds which sit in the lower triangular of adj_matrix.

    Returns:
        List of bonds, each item is a tuple of (bond_indices, bond_types):
    """
    assert adj.dim() == 3, "Adjacency matrix must be 2D or 3D"
    return [bonds_from_adj(_adj, lower_tri=lower_tri) for _adj in adj]


def adj_from_edges(edge_indices: _T, edge_types: _T, n_nodes: int, symmetric: bool = False) -> _T:
    """Create adjacency matrix from a list of edge indices and types

    If an edge pair appears multiple times with different edge types, the adj element for that edge is undefined.

    Args:
        edge_indices (torch.Tensor): Edge list tensor, shape [n_edges, 2]. Pairs of (from_idx, to_idx).
        edge_types (torch.Tensor): Edge types, shape either [n_edges] or [n_edges, edge_types].
        n_nodes (int): Number of nodes in the adjacency matrix. This must be >= to the max node index in edges.
        symmetric (bool): Whether edges are considered symmetric. If True the adjacency matrix will also be symmetric,
                otherwise only the exact node indices within edges will be used to create the adjacency.

    Returns:
        torch.Tensor: Adjacency matrix tensor, shape [n_nodes, n_nodes] or
                [n_nodes, n_nodes, edge_types] if distributions over edge types are provided.
    """

    assert len(edge_indices.shape) == 2
    assert edge_indices.shape[0] == edge_types.shape[0]
    assert edge_indices.size(1) == 2
    adj = torch.zeros((n_nodes, n_nodes), device=edge_types.device, dtype=edge_types.dtype)
    u, v = torch.split(edge_indices, 1, dim=-1)
    u, v = u.squeeze(-1), v.squeeze(-1)
    adj[u, v] = edge_types
    if symmetric:
        adj[v, u] = edge_types
    return adj


def get_adj_mask(
    coords: _T,
    node_mask: _T | None,
    k: int | None = None,
    self_connect: bool = False,
):
    """Constuct adjacency mask
    # TODO: add desciption
    #"""

    batch_size, num_nodes, _ = coords.shape
    dev = coords.device

    # If node mask is None all nodes are real
    if node_mask is None:
        node_mask = torch.ones((batch_size, num_nodes), device=dev, dtype=torch.bool)

    adj_mask = adj_from_node_mask(node_mask, self_connect=True)

    if k is not None:
        raise NotImplementedError("k-nearest neighbours not implemented for adjacency mask")
        # TODO: for self-connect=True, we need to set k+1 instead of k.

    # Remove diagonal connections
    if not self_connect:
        node_idxs = torch.arange(num_nodes, device=dev)
        adj_mask[:, node_idxs, node_idxs] = False

    return adj_mask


def calc_cross_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Squared‐Euclidean distances between two point clouds.
    Args:
        x: [B, N1, D]
        y: [B, N2, D]
    Returns:
        dists: [B, N1, N2]
    """
    # (x_i - y_j)^2 summed over D
    diffs = x.unsqueeze(2) - y.unsqueeze(1)
    return (diffs * diffs).sum(-1)


def edges_from_two_sets(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    k: int | None = None,
    node_mask1: torch.Tensor | None = None,
    node_mask2: torch.Tensor | None = None,
    edge_format: str = "adjacency",
):
    """
    Build bipartite edges from set1→set2.

    Args:
      coords1: [B, N1, 3]
      coords2: [B, N2, 3]
      k:       how many neighbors in set2 for each node in set1 (None ⇒ all)
      node_mask1: [B, N1] 1=real node, 0=pad
      node_mask2: [B, N2]
      edge_format: "adjacency" or "list"
    Returns:
      If adjacency:
        adj: [B, N1, N2]  {0,1} mask
      If list:
        ( (idx1, idx2), edge_mask )
          idx1,idx2: [B, num_edges]  row/col indices
          edge_mask: [B, num_edges]  1=real edge, 0=pad
    """
    if edge_format not in ("adjacency", "list"):
        raise ValueError(f"Bad format {edge_format!r}")

    B, N1, _ = coords1.shape
    _, N2, _ = coords2.shape

    # default masks = all real
    if node_mask1 is None:
        node_mask1 = coords1.new_ones(B, N1, dtype=torch.int64)
    if node_mask2 is None:
        node_mask2 = coords2.new_ones(B, N2, dtype=torch.int64)

    # initial bipartite mask [B,N1,N2]
    valid = node_mask1[:, :, None] * node_mask2[:, None, :]

    # if k‑nearest
    if k is not None:
        k = min(k, N2)

        dists = calc_cross_distances(coords1, coords2)  # [B,N1,N2]
        dists = dists.masked_fill(valid == 0, float("inf"))

        # find the k closest in set2 for each node1
        _, best = dists.topk(k, dim=2, largest=False)  # [B,N1,k]

        # build new adjacency
        adj = torch.zeros_like(valid)
        batch_idx = torch.arange(B, device=coords1.device)[:, None, None].expand(-1, N1, k)
        row_idx = torch.arange(N1, device=coords1.device)[None, :, None].expand(B, -1, k)

        adj[batch_idx, row_idx, best] = 1
        adj = adj * valid  # ban any edges to pad‐nodes
    else:
        adj = valid

    if edge_format == "adjacency":
        return adj

    # --- now build the “list” form
    if k is not None:
        # fixed N1*k edges per batch
        E = N1 * k
        # flatten the same row_idx and best arrays we used above:
        idx1 = row_idx.reshape(B, E)
        idx2 = best.reshape(B, E)
        mask = adj[batch_idx, row_idx, best].reshape(B, E)
        return (idx1, idx2), mask

    else:
        # dynamic number per batch: fall back to gathering nonzeros per‐batch
        edge_idxs = []
        edge_masks = []
        for b in range(B):
            i, j = torch.nonzero(adj[b], as_tuple=True)
            edge_idxs.append((i, j))
            edge_masks.append(torch.ones_like(i, dtype=torch.int64))
        return edge_idxs, edge_masks


def gather_edge_features(pairwise_feats, adj_matrix):
    """Gather edge features for each node from pairwise features using the adjacency matrix

    All 'from nodes' (dimension 1 on the adj matrix) must have the same number of edges to 'to nodes'. Practically
    this means that the number of non-zero elements in dimension 2 of the adjacency matrix must always be the same.

    Args:
        pairwise_feats (torch.Tensor): Pairwise features tensor, shape [batch_size, num_nodes, num_nodes, num_feats]
        adj_matrix (torch.Tensor): Batched adjacency matrix, shape [batch_size, num_nodes, num_nodes]. It can contain
                any non-zero integer for connected nodes but must be 0 for unconnected nodes.

    Returns:
        torch.Tensor: Dense feature matrix, shape [batch_size, num_nodes, edges_per_node, num_feats]
    """

    # In case some of the connections don't use 1, create a 1s adjacency matrix
    adj_ones = torch.zeros_like(adj_matrix).int()
    adj_ones[adj_matrix != 0] = 1

    num_neighbours = adj_ones.sum(dim=2)
    feats_per_node = num_neighbours[0, 0].item()

    assert (num_neighbours == feats_per_node).all(), "All nodes must have the same number of connections"

    if len(pairwise_feats.size()) == 3:
        batch_size, num_nodes, _ = pairwise_feats.size()
        pairwise_feats = pairwise_feats.unsqueeze(3)

    elif len(pairwise_feats.size()) == 4:
        batch_size, num_nodes, _, _ = pairwise_feats.size()

    # nonzero() orders indices lexicographically with the last index changing the fastest, so we can reshape the
    # indices into a dense form with nodes along the outer axis and features along the inner
    gather_idxs = adj_ones.nonzero()[:, 2].reshape((batch_size, num_nodes, feats_per_node))
    batch_idxs = torch.arange(batch_size).view(-1, 1, 1)
    node_idxs = torch.arange(num_nodes).view(1, -1, 1)
    dense_feats = pairwise_feats[batch_idxs, node_idxs, gather_idxs, :]
    if dense_feats.size(-1) == 1:
        return dense_feats.squeeze(-1)

    return dense_feats


# *** Geometric Util Functions ***


# TODO rename? Maybe also merge with inter_distances
# TODO test unbatched and coord sets inputs
def calc_distances(coords, edges=None, sqrd=False, eps=1e-6):
    """Computes distances between connected nodes

    Takes an optional edges argument. If edges is None this will calculate distances between all nodes and return the
    distances in a batched square matrix [batch_size, num_nodes, num_nodes]. If edges is provided the distances are
    returned for each edge in a batched 1D format [batch_size, num_edges].

    Args:
        coords (torch.Tensor): Coordinate tensor, shape [batch_size, num_nodes, 3]
        edges (tuple): Two-tuple of connected node indices, each tensor has shape [batch_size, num_edges]
        sqrd (bool): Whether to return the squared distances
        eps (float): Epsilon to add before taking the square root for numical stability in the gradients

    Returns:
        torch.Tensor: Distances tensor, the shape depends on whether edges is provided (see above).
    """

    # TODO add checks

    # Create fake batch dim if unbatched
    unbatched = False
    if len(coords.size()) == 2:
        coords = coords.unsqueeze(0)
        unbatched = True

    if edges is None:
        coord_diffs = coords.unsqueeze(-2) - coords.unsqueeze(-3)
        sqrd_dists = torch.sum(coord_diffs * coord_diffs, dim=-1)

    else:
        edge_is, edge_js = edges
        batch_index = torch.arange(coords.size(0)).unsqueeze(1)
        coord_diffs = coords[batch_index, edge_js, :] - coords[batch_index, edge_is, :]
        sqrd_dists = torch.sum(coord_diffs * coord_diffs, dim=2)

    sqrd_dists = sqrd_dists.squeeze(0) if unbatched else sqrd_dists

    if sqrd:
        return sqrd_dists

    return torch.sqrt(sqrd_dists + eps)


def inter_distances(coords1, coords2, sqrd=False, eps=1e-6):
    # TODO add checks and doc

    # Create fake batch dim if unbatched
    unbatched = False
    if len(coords1.size()) == 2:
        coords1 = coords1.unsqueeze(0)
        coords2 = coords2.unsqueeze(0)
        unbatched = True

    coord_diffs = coords1.unsqueeze(2) - coords2.unsqueeze(1)
    sqrd_dists = torch.sum(coord_diffs * coord_diffs, dim=3)
    sqrd_dists = sqrd_dists.squeeze(0) if unbatched else sqrd_dists

    if sqrd:
        return sqrd_dists

    return torch.sqrt(sqrd_dists + eps)


def calc_com(coords, node_mask=None):
    """Calculates the centre of mass of a pointcloud

    Args:
        coords (torch.Tensor): Coordinate tensor, shape [*, num_nodes, 3]
        node_mask (torch.Tensor): Mask for points, shape [*, num_nodes], 1 for real node, 0 otherwise

    Returns:
        torch.Tensor: CoM of pointclouds with imaginary nodes excluded, shape [*, 1, 3]
    """

    node_mask = torch.ones_like(coords[..., 0]) if node_mask is None else node_mask

    assert node_mask.shape == coords[..., 0].shape

    num_nodes = node_mask.sum(dim=-1)
    real_coords = coords * node_mask.unsqueeze(-1)
    com = real_coords.sum(dim=-2) / num_nodes.unsqueeze(-1)
    return com.unsqueeze(-2)


def zero_com(coords, node_mask=None):
    """Sets the centre of mass for a batch of pointclouds to zero for each pointcloud

    Args:
        coords (torch.Tensor): Coordinate tensor, shape [*, num_nodes, 3]
        node_mask (torch.Tensor): Mask for points, shape [*, num_nodes], 1 for real node, 0 otherwise

    Returns:
        torch.Tensor: CoM-free coordinates, where imaginary nodes are excluded from CoM calculation
    """

    com = calc_com(coords, node_mask=node_mask)
    shifted = coords - com
    return shifted


def standardise_coords(coords, node_mask=None):
    """Convert coords into a standard normal distribution

    This will first remove the centre of mass from all pointclouds in the batch, then calculate the (biased) variance
    of the shifted coords and use this to produce a standard normal distribution.

    Args:
        coords (torch.Tensor):  Coordinate tensor, shape [batch_size, num_nodes, 3]
        node_mask (torch.Tensor): Mask for points, shape [batch_size, num_nodes], 1 for real node, 0 otherwise

    Returns:
        Tuple[torch.Tensor, float]: The standardised coords and the variance of the original coords
    """

    if node_mask is None:
        node_mask = torch.ones_like(coords)[:, :, 0]

    coord_idxs = node_mask.nonzero()
    real_coords = coords[coord_idxs[:, 0], coord_idxs[:, 1], :]

    variance = torch.var(real_coords, correction=0)
    std_dev = torch.sqrt(variance)

    result = (coords / std_dev) * node_mask.unsqueeze(2)
    return result, std_dev.item()


def rotate(coords: torch.Tensor, rotation: Rotation | TupleRot):
    """Rotate coordinates for a single molecule

    Args:
        coords (torch.Tensor): Unbatched coordinate tensor, shape [num_atoms, 3]
        rotation (Union[Rotation, Tuple[float, float, float]]): Can be either a scipy Rotation object or a tuple of
                rotation values in radians, (x, y, z). These are treated as extrinsic rotations. See the scipy docs
                (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) for info.

    Returns:
        torch.Tensor: Rotated coordinates
    """

    if not isinstance(rotation, Rotation):
        rotation = Rotation.from_euler("xyz", rotation)

    dtype, device = coords.dtype, coords.device
    coords_npy = coords.cpu().numpy()

    rotated = rotation.apply(coords_npy)
    rotated = torch.tensor(rotated, device=device, dtype=dtype)
    return rotated


def cartesian_to_spherical(coords):
    sqrd_dists = (coords * coords).sum(dim=-1)
    radii = torch.sqrt(sqrd_dists)
    inclination = torch.acos(coords[..., 2] / radii).unsqueeze(2)
    azimuth = torch.atan2(coords[..., 1], coords[..., 0]).unsqueeze(2)
    spherical = torch.cat((radii.unsqueeze(2), inclination, azimuth), dim=-1)
    return spherical


# *** Residue Utils ***


def convert_to_residue_format(
    atoms: torch.Tensor,
    charges: torch.Tensor,
    residues: torch.Tensor,
    residue_ids: torch.Tensor,
    adjacency: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    max_atoms_per_residue=14,
):
    """
    Convert atom-level data to residue-level format.

    Args:
        atoms: [B, N_atoms] - atom types
        charges: [B, N_atoms] - atom charges
        residues: [B, N_atoms] - residue types for each atom
        residue_ids: [B, N_atoms] - residue IDs for each atom
        adjacency: [B, N_atoms, N_atoms] - bond matrix
        coords: [B, N_atoms, 3] - atom coordinates
        mask: [B, N_atoms] - atom mask
        max_atoms_per_residue: int - maximum atoms per residue (default 14)

    Returns:
        Dictionary with residue-level formatted tensors:
        - atoms: [B, N_res, max_atoms_per_residue]
        - charges: [B, N_res, max_atoms_per_residue]
        - residues: [B, N_res, max_atoms_per_residue]
        - adjacency: [B, N_res, max_atoms_per_residue, max_atoms_per_residue]
        - coords: [B, N_res, max_atoms_per_residue, 3]
        - mask: [B, N_res, max_atoms_per_residue]
    """

    batch_size = coords.shape[0]
    device = coords.device
    dtype = coords.dtype

    # Process each batch
    batch_results = []

    for b in range(batch_size):
        # Get unique residue IDs for this batch
        unique_residue_ids = torch.unique(residue_ids[b][mask[b].bool()])
        n_residues = len(unique_residue_ids)

        # Initialize tensors for this batch
        batch_atoms = torch.zeros(n_residues, max_atoms_per_residue, dtype=torch.long, device=device)
        batch_charges = torch.zeros(n_residues, max_atoms_per_residue, dtype=torch.long, device=device)
        batch_residues = torch.zeros(n_residues, max_atoms_per_residue, dtype=torch.long, device=device)
        batch_coords = torch.zeros(n_residues, max_atoms_per_residue, 3, dtype=dtype, device=device)
        batch_adjs = torch.zeros(
            n_residues, max_atoms_per_residue, max_atoms_per_residue, dtype=torch.long, device=device
        )
        batch_mask = torch.zeros(n_residues, max_atoms_per_residue, dtype=torch.bool, device=device)

        # Fill in data for each residue
        for res_idx, res_id in enumerate(unique_residue_ids):
            # Find atoms belonging to this residue
            atom_indices = torch.where((residue_ids[b] == res_id) & mask[b].bool())[0]
            n_atoms_in_res = min(len(atom_indices), max_atoms_per_residue)

            if n_atoms_in_res > 0:
                # Take first max_atoms_per_residue atoms if residue has more
                selected_indices = atom_indices[:n_atoms_in_res]

                # Fill atom properties
                batch_atoms[res_idx, :n_atoms_in_res] = atoms[b, selected_indices]
                batch_charges[res_idx, :n_atoms_in_res] = charges[b, selected_indices]
                batch_residues[res_idx, :n_atoms_in_res] = residues[b, selected_indices]

                # Fill coordinates
                batch_coords[res_idx, :n_atoms_in_res] = coords[b, selected_indices]

                # Fill mask
                batch_mask[res_idx, :n_atoms_in_res] = True

                # Fill bond matrix (intra-residue bonds only)
                for i in range(n_atoms_in_res):
                    for j in range(n_atoms_in_res):
                        batch_adjs[res_idx, i, j] = adjacency[b, selected_indices[i], selected_indices[j]]

        batch_results.append(
            {
                "atoms": batch_atoms,
                "charges": batch_charges,
                "residues": batch_residues,
                "coords": batch_coords,
                "adjacency": batch_adjs,
                "mask": batch_mask,
            }
        )

    # Create batch indicator vector
    batch_ids = []
    for batch_idx, batch in enumerate(batch_results):
        n_residues_in_batch = batch["coords"].shape[0]
        batch_ids.append(torch.full((n_residues_in_batch,), batch_idx, dtype=torch.long, device=device))

    result = {}
    for key in batch_results[0].keys():
        # Concatenate all tensors directly (B*n_res, ...)
        result[key] = torch.cat([batch[key] for batch in batch_results], dim=0)

    # Add batch indicator vector
    result["batch_ids"] = torch.cat(batch_ids, dim=0)

    return result


def aggregate_atoms_by_mean(features: _T, atom_mask: _T, eps: float = 1e-8) -> _T:
    """
    Aggregate atom features to residue features using masked mean pooling.

    Args:
        features: Atom features of shape [B*N_res, 14, d_inv] or [B*N_res, 14, 3, d_equi]
        atom_mask: Mask for real atoms, shape [B*N_res, 14], 1 for real atoms, 0 for fake
        eps: Small value to avoid division by zero

    Returns:
        Residue features of shape [B*N_res, d_inv] or [B*N_res, 3, d_equi]
    """
    # Expand mask to match feature dimensions
    while atom_mask.dim() < features.dim():
        atom_mask = atom_mask.unsqueeze(-1)

    # Apply mask and compute mean
    masked_features = features * atom_mask
    summed_features = masked_features.sum(dim=1)  # Sum along atom dimension (dim=1)

    # Count real atoms per residue
    atom_counts = atom_mask.sum(dim=1)  # [B*N_res, ...]
    atom_counts = torch.clamp(atom_counts, min=eps)  # Avoid division by zero

    # Mean pooling
    mean_features = summed_features / atom_counts

    return mean_features


def pad_by_batch_id(data: _T, batch_ids: _T) -> tuple[_T, _T]:
    """
    Pads variable-length per-batch data into a dense tensor with padding.

    Args:
        data (torch.Tensor): Tensor of shape [num_items, ...] where items are grouped by `batch_ids`.
        batch_ids (torch.Tensor): Tensor of shape [num_items], with values in [0, B-1] indicating batch assignment.
        pad_dim (int): Not used currently, reserved for future extension if you want to pad along other dimensions.

    Returns:
        padded (torch.Tensor): Tensor of shape [B, N, ...], where N is max number of items in any batch.
        mask (torch.BoolTensor): Tensor of shape [B, N], 1 for real items, 0 for padding.
    """
    # Number of batches
    B = int(batch_ids.max().item()) + 1

    # Count number of items per batch
    counts = torch.bincount(batch_ids)

    # Max number of items in any batch (used for padding)
    N = counts.max().item()

    # Determine output shape: [B, N, ...]
    shape = list(data.shape[1:])  # trailing dimensions (e.g. [3, 64] or [384])
    padded_shape = [B, N] + shape

    # Allocate output tensor and mask
    padded = data.new_zeros(padded_shape)  # same dtype/device as input
    mask = torch.zeros(B, N, dtype=torch.bool, device=data.device)

    # Fill padded tensor and mask for each batch
    for b in range(B):
        idx = (batch_ids == b).nonzero(as_tuple=True)[0]  # indices belonging to batch `b`
        n = idx.size(0)
        padded[b, :n] = data[idx]  # copy real data
        mask[b, :n] = 1  # mark as real

    return padded, mask
