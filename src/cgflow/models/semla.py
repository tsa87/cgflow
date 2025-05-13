import numpy as np
import torch

import cgflow.util.functional as smolF


def adj_to_attn_mask(adj_matrix, pos_inf=False):
    """Assumes adj_matrix is only 0s and 1s"""

    inf = float("inf") if pos_inf else float("-inf")
    attn_mask = torch.zeros_like(adj_matrix.float())
    attn_mask[adj_matrix == 0] = inf

    # Ensure nodes with no connections (fake nodes) don't have all -inf in the attn mask
    # Otherwise we would have problems when softmaxing
    n_nodes = adj_matrix.sum(dim=-1)
    attn_mask[n_nodes == 0] = 0.0

    return attn_mask


# *************************************************************************************************
# *********************************** Helper Classes **********************************************
# *************************************************************************************************


class CoordNorm(torch.nn.Module):
    """Coordinate normalisation layer for coordinate sets with inductive bias towards molecules

    This layer allows 4 different types of coordinate normalisation (defined in the norm argument):
        1. 'none' - The coordinates are zero-centred and multiplied by learnable weights
        2. 'gvp' - Coords are zero-centred, scaled by learnable weights and each is scaled by sqrt(n_sets) / ||x_i||_2
        3. 'length' - Coords are zero-centred, multiplied by learnable weights and scaled by 1 / avg vector length

    Note that 'length' provides the same coordinate normalisation that is commonly used in current models but adapted
    to multiple coordinate sets, thereby allowing easier comparison to existing approaches.
    """

    def __init__(self, n_coord_sets, norm="length", zero_com=True, eps=1e-6):
        super().__init__()

        norm = "none" if norm is None else norm
        if norm not in ["none", "gvp", "length"]:
            raise ValueError(f"Unknown normalisation type '{norm}'")

        self.n_coord_sets = n_coord_sets
        self.norm = norm
        self.zero_com = zero_com
        self.eps = eps

        self.set_weights = torch.nn.Parameter(
            torch.ones((1, n_coord_sets, 1, 1)))

    def forward(self, coord_sets, node_mask):
        """Apply coordinate normlisation layer

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise

        Returns:
            torch.Tensor: Normalised coords, shape [batch_size, n_sets, n_nodes, 3]
        """

        # Zero the CoM in case it isn't already
        if self.zero_com:
            coord_sets = smolF.zero_com(coord_sets, node_mask)

        coord_sets = coord_sets * node_mask.unsqueeze(-1)

        n_atoms = node_mask.sum(dim=-1, keepdim=True)
        lengths = torch.linalg.vector_norm(coord_sets, dim=-1)

        if self.norm == "length":
            scaled_lengths = lengths.sum(dim=2, keepdim=True) / n_atoms
            coord_div = scaled_lengths.unsqueeze(-1) + self.eps

        elif self.norm == "gvp":
            coord_div = (lengths.unsqueeze(-1) + self.eps) / np.sqrt(
                self.n_coord_sets)

        else:
            coord_div = torch.ones_like(coord_sets)

        coord_sets = (coord_sets * self.set_weights) / coord_div
        coord_sets = coord_sets * node_mask.unsqueeze(-1)
        return coord_sets

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


class EdgeMessages(torch.nn.Module):

    def __init__(self,
                 d_model,
                 d_message,
                 d_out,
                 n_coord_sets,
                 d_ff=None,
                 d_edge=None,
                 eps=1e-6):
        super().__init__()

        edge_feats = 0 if d_edge is None else d_edge
        d_ff = d_out if d_ff is None else d_ff

        extra_feats = n_coord_sets + edge_feats
        in_feats = (d_message * 2) + extra_feats

        self.n_coord_sets = n_coord_sets
        self.d_edge = d_edge
        self.eps = eps

        self.coord_norm = CoordNorm(n_coord_sets, norm="none")
        self.node_norm = torch.nn.LayerNorm(d_model)
        self.edge_norm = torch.nn.LayerNorm(
            d_edge) if d_edge is not None else None

        self.node_proj = torch.nn.Linear(d_model, d_message)
        self.message_mlp = torch.nn.Sequential(torch.nn.Linear(in_feats, d_ff),
                                               torch.nn.SiLU(inplace=False),
                                               torch.nn.Linear(d_ff, d_out))

    def forward(self, coords, node_feats, node_mask, edge_feats=None):
        """Compute edge messages

        Args:
            coords (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node features, shape [batch_size, n_nodes, d_model]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise
            edge_feats (torch.Tensor): Incoming edge features, shape [batch_size, n_nodes, n_nodes, d_edge]

        Returns:
            torch.Tensor: Edge messages tensor, shape [batch_size, n_nodes, n_nodes, d_out]
        """

        batch_size, n_nodes, _ = tuple(node_feats.shape)

        if edge_feats is not None and self.d_edge is None:
            raise ValueError(
                "edge_feats was provided but the model was initialised with d_edge as None."
            )

        if edge_feats is None and self.d_edge is not None:
            raise ValueError(
                "The model was initialised with d_edge but no edge feats were provided to forward fn."
            )

        node_feats = self.node_norm(node_feats)

        coords = self.coord_norm(coords, node_mask).flatten(0, 1)
        coord_dotprods = torch.bmm(coords, coords.transpose(1, 2))
        coord_feats = coord_dotprods.unflatten(
            0, (-1, self.n_coord_sets)).movedim(1, -1)

        # Project to smaller dimension and create pairwise node features
        node_feats = self.node_proj(node_feats)
        node_feats_start = node_feats.unsqueeze(2).expand(
            batch_size, n_nodes, n_nodes, -1)
        node_feats_end = node_feats.unsqueeze(1).expand(
            batch_size, n_nodes, n_nodes, -1)
        node_pairs = torch.cat((node_feats_start, node_feats_end), dim=-1)

        in_edge_feats = torch.cat((node_pairs, coord_feats), dim=3)
        if edge_feats is not None:
            edge_feats = self.edge_norm(edge_feats)
            in_edge_feats = torch.cat((in_edge_feats, edge_feats), dim=-1)

        return self.message_mlp(in_edge_feats)


class NodeAttention(torch.nn.Module):

    def __init__(self, d_model, n_attn_heads, d_attn=None):
        super().__init__()

        d_attn = d_model if d_attn is None else d_attn
        d_head = d_model // n_attn_heads

        if d_attn % n_attn_heads != 0:
            raise ValueError(
                "n_attn_heads must divide d_model (or d_attn if provided) exactly."
            )

        self.d_model = d_model
        self.d_attn = d_attn
        self.n_attn_heads = n_attn_heads
        self.d_head = d_head

        self.feat_norm = torch.nn.LayerNorm(d_model)
        self.in_proj = torch.nn.Linear(d_model, d_attn)
        self.out_proj = torch.nn.Linear(d_attn, d_model)

    def forward(self, node_feats, messages, adj_matrix):
        """Accumulate edge messages to each node using attention-based message passing

        Args:
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            messages (torch.Tensor): Messages tensor, shape [batch_size, n_nodes, n_nodes, d_message]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_nodes, n_nodes]

        Returns:
            torch.Tensor: Accumulated node features, shape [batch_size, n_nodes, d_model]
        """

        attn_mask = adj_to_attn_mask(adj_matrix)
        messages = messages + attn_mask.unsqueeze(3)
        attentions = torch.softmax(messages, dim=2)

        node_feats = self.feat_norm(node_feats)
        proj_feats = self.in_proj(node_feats)
        head_feats = proj_feats.unflatten(-1, (self.n_attn_heads, self.d_head))

        # Put n_heads into the batch dim for both the features and the attentions
        # head_feats shape [B * n_heads, n_nodes, d_head]
        # attentions shape [B * n_heads, n_nodes, n_nodes]
        head_feats = head_feats.movedim(-2, 1).flatten(0, 1)
        attentions = attentions.movedim(-1, 1).flatten(0, 1)

        attn_out = torch.bmm(attentions.to(node_feats.dtype),
                             head_feats)

        # Apply variance preserving updates as proposed in GNN-VPA (https://arxiv.org/abs/2403.04747)
        weights = torch.sqrt((attentions**2).sum(dim=-1))
        attn_out = attn_out * weights.unsqueeze(-1)

        attn_out = attn_out.unflatten(0, (-1, self.n_attn_heads))
        attn_out = attn_out.movedim(1, -2).flatten(2, 3)
        return self.out_proj(attn_out)


class CoordAttention(torch.nn.Module):

    def __init__(self,
                 n_coord_sets,
                 proj_sets=None,
                 coord_norm="length",
                 eps=1e-6):
        super().__init__()

        proj_sets = n_coord_sets if proj_sets is None else proj_sets

        self.eps = eps

        self.coord_norm = CoordNorm(n_coord_sets, norm=coord_norm)
        self.coord_proj = torch.nn.Linear(n_coord_sets, proj_sets, bias=False)
        self.attn_proj = torch.nn.Linear(proj_sets, n_coord_sets, bias=False)

    def forward(self, coord_sets, messages, adj_matrix, node_mask):
        """Compute an attention update for coordinate sets

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            messages (torch.Tensor): Messages tensor, shape [batch_size, n_nodes, n_nodes, proj_sets]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_nodes, n_nodes]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise

        Returns:
            torch.Tensor: Updated coordinate sets, shape [batch_size, n_sets, n_nodes, 3]
        """

        coord_sets = self.coord_norm(coord_sets, node_mask)
        proj_coord_sets = self.coord_proj(coord_sets.transpose(1, -1))

        # proj_coord_sets shape [B, 3, N, P]
        # norm_dists shape [B, 1, N, N, P]
        vec_dists = proj_coord_sets.unsqueeze(3) - proj_coord_sets.unsqueeze(2)
        lengths = torch.linalg.vector_norm(vec_dists, dim=1, keepdim=True)
        norm_dists = vec_dists / (lengths + self.eps)

        attn_mask = adj_to_attn_mask(adj_matrix)
        messages = messages + attn_mask.unsqueeze(3)
        attentions = torch.softmax(messages, dim=2)

        # Dim 1 is currently 1 on dists so we need to unsqueeze attentions
        updates = norm_dists * attentions.unsqueeze(1)
        updates = updates.sum(dim=3)

        # Apply variance preserving updates as proposed in GNN-VPA (https://arxiv.org/abs/2403.04747)
        weights = torch.sqrt((attentions**2).sum(dim=2))
        updates = updates * weights.unsqueeze(1)

        # updates shape [B, 3, N, P] -> [B, S, N, 3]
        updates = self.attn_proj(updates).transpose(1, -1)
        return updates


class LengthsMLP(torch.nn.Module):

    def __init__(self, d_model, n_coord_sets, d_ff=None):
        super().__init__()

        d_ff = d_model * 4 if d_ff is None else d_ff

        self.node_ff = torch.nn.Sequential(
            torch.nn.Linear(d_model + n_coord_sets, d_ff),
            torch.nn.SiLU(inplace=False), torch.nn.Linear(d_ff, d_model))

    def forward(self, coord_sets, node_feats):
        """Pass data through the layer

        Assumes coords and node_feats have already been normalised

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]

        Returns:
            torch.Tensor: Updated node features, shape [batch_size, n_nodes, d_model]
        """

        lengths = torch.linalg.vector_norm(coord_sets, dim=-1).movedim(1, -1)
        in_feats = torch.cat((node_feats, lengths), dim=2)
        return self.node_ff(in_feats)


class EquivariantMLP(torch.nn.Module):

    def __init__(self, d_model, n_coord_sets, proj_sets=None):
        super().__init__()

        proj_sets = n_coord_sets if proj_sets is None else proj_sets

        self.node_proj = torch.nn.Sequential(
            torch.nn.Linear(d_model, proj_sets), torch.nn.SiLU(inplace=False),
            torch.nn.Linear(proj_sets, proj_sets))
        self.coord_proj = torch.nn.Linear(n_coord_sets, proj_sets, bias=False)
        self.attn_proj = torch.nn.Linear(proj_sets, n_coord_sets, bias=False)

    def forward(self, coord_sets, node_feats):
        """Pass data through the layer

        Assumes coords and node_feats have already been normalised

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]

        Returns:
            torch.Tensor: Updated coord_sets, shape [batch_size, n_sets, n_nodes, 3]
        """

        # inv_feats shape [B, 1, N, P]
        # proj_sets shape [B, 3, N, P]
        inv_feats = self.node_proj(node_feats).unsqueeze(1)
        proj_sets = self.coord_proj(coord_sets.transpose(1, -1))

        # Outer product with invariant features is equivariant, then sum over original coord sets
        attentions = inv_feats.unsqueeze(-1) * proj_sets.unsqueeze(-2)
        attentions = attentions.sum(-1)

        coords_out = self.attn_proj(attentions).transpose(1, -1)
        return coords_out


class NodeFeedForward(torch.nn.Module):

    def __init__(self,
                 d_model,
                 n_coord_sets,
                 d_ff=None,
                 proj_sets=None,
                 coord_norm="length",
                 fixed_equi=False):
        super().__init__()

        self.fixed_equi = fixed_equi

        self.node_norm = torch.nn.LayerNorm(d_model)
        self.invariant_mlp = LengthsMLP(d_model, n_coord_sets, d_ff=d_ff)

        if not fixed_equi:
            self.coord_norm = CoordNorm(n_coord_sets, norm=coord_norm)
            self.equivariant_mlp = EquivariantMLP(d_model,
                                                  n_coord_sets,
                                                  proj_sets=proj_sets)

    def forward(self, coord_sets, node_feats, node_mask):
        """Pass data through the layer

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise

        Returns:
            torch.Tensor, torch.Tensor: Updates to coords and node features
        """

        node_feats = self.node_norm(node_feats)

        if not self.fixed_equi:
            coord_sets = self.coord_norm(coord_sets, node_mask)

        out_node_feats = self.invariant_mlp(coord_sets, node_feats)

        if not self.fixed_equi:
            coord_sets = self.equivariant_mlp(coord_sets, node_feats)

        return coord_sets, out_node_feats


class BondRefine(torch.nn.Module):

    def __init__(self, d_model, d_message, d_edge, d_ff=None, norm_feats=True):
        super().__init__()

        d_ff = d_message if d_ff is None else d_ff
        in_feats = (2 * d_message) + d_edge + 2

        self.norm_feats = norm_feats

        if norm_feats:
            self.coord_norm = CoordNorm(1, norm="none")
            self.node_norm = torch.nn.LayerNorm(d_model)
            self.edge_norm = torch.nn.LayerNorm(d_edge)

        self.node_proj = torch.nn.Linear(d_model, d_message)
        self.message_mlp = torch.nn.Sequential(torch.nn.Linear(in_feats, d_ff),
                                               torch.nn.SiLU(inplace=False),
                                               torch.nn.Linear(d_ff, d_edge))

    def forward(self, coords, node_feats, node_mask, edge_feats):
        """Refine the bond predictions with a message passing layer that only updates bonds

        Args:
            coords (torch.Tensor): Coordinate tensor without coord sets, shape [batch_size, n_nodes, 3]
            node_feats (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_nodes], 1 for real, 0 otherwise
            edge_feats (torch.Tensor): Current edge features, shape [batch_size, n_nodes, n_nodes, d_edge]

        Returns:
            torch.Tensor: Bond predictions tensor, shape [batch_size, n_nodes, n_nodes, n_bond_types]
        """

        assert len(coords.shape) == 3

        batch_size, n_nodes, _ = tuple(node_feats.shape)

        # Calculate distances and dot products
        if self.norm_feats:
            coords = self.coord_norm(coords.unsqueeze(1),
                                     node_mask.unsqueeze(1)).squeeze(1)
            node_feats = self.node_norm(node_feats)
            edge_feats = self.edge_norm(edge_feats)

        coord_diffs = coords.unsqueeze(2) - coords.unsqueeze(1)
        dists = (coord_diffs * coord_diffs).sum(dim=-1).unsqueeze(-1)
        coord_dotprods = torch.bmm(coords, coords.transpose(1,
                                                            2)).unsqueeze(-1)

        # Project to smaller dimension and create pairwise node features
        node_feats = self.node_proj(node_feats)
        node_feats_i = node_feats.unsqueeze(2).expand(batch_size, n_nodes,
                                                      n_nodes, -1)
        node_feats_j = node_feats.unsqueeze(1).expand(batch_size, n_nodes,
                                                      n_nodes, -1)
        node_pairs = torch.cat((node_feats_i, node_feats_j), dim=-1)

        in_feats = torch.cat((node_pairs, dists, coord_dotprods, edge_feats),
                             dim=3)
        return self.message_mlp(in_feats)


# *************************************************************************************************
# ********************************** Equivariant Layers *******************************************
# *************************************************************************************************


class EquiMessagePassingLayer(torch.nn.Module):

    def __init__(
        self,
        d_model,
        d_message,
        n_coord_sets,
        n_attn_heads=None,
        d_message_hidden=None,
        d_edge_in=None,
        d_edge_out=None,
        coord_norm="length",
        eps=1e-6,
    ):
        super().__init__()

        n_attn_heads = d_message if n_attn_heads is None else n_attn_heads
        if d_model != ((d_model // n_attn_heads) * n_attn_heads):
            raise ValueError(
                f"n_attn_heads must exactly divide d_model, got {n_attn_heads} and {d_model}"
            )

        self.d_model = d_model
        self.d_message = d_message
        self.n_coord_sets = n_coord_sets
        self.n_attn_heads = n_attn_heads
        self.d_message_hidden = d_message_hidden
        self.d_edge_in = d_edge_in
        self.d_edge_out = d_edge_out
        self.d_coord_message = n_coord_sets
        self.eps = eps

        d_ff = d_model * 4
        d_attn = d_model
        d_message_out = n_attn_heads + self.d_coord_message
        d_message_out = d_message_out + d_edge_out if d_edge_out is not None else d_message_out

        if d_edge_in is not None:
            self.edge_feat_norm = torch.nn.LayerNorm(d_edge_in)

        self.node_ff = NodeFeedForward(
            d_model,
            n_coord_sets,
            d_ff=d_ff,
            proj_sets=d_message,
            coord_norm=coord_norm,
        )
        self.message_ff = EdgeMessages(d_model,
                                       d_message,
                                       d_message_out,
                                       n_coord_sets,
                                       d_ff=d_message_hidden,
                                       d_edge=d_edge_in,
                                       eps=eps)
        self.coord_attn = CoordAttention(n_coord_sets,
                                         self.d_coord_message,
                                         coord_norm=coord_norm,
                                         eps=eps)
        self.node_attn = NodeAttention(d_model, n_attn_heads, d_attn=d_attn)

    @property
    def hparams(self):
        return {
            "d_model": self.d_model,
            "d_message": self.d_message,
            "n_coord_sets": self.n_coord_sets,
            "n_attn_heads": self.n_attn_heads,
            "d_message_hidden": self.d_message_hidden,
        }

    def forward(self,
                coords,
                node_feats,
                adj_matrix,
                node_mask,
                edge_feats=None):
        """Pass data through the layer

        Args:
            coords (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            node_feats (torch.Tensor): Node features, shape [batch_size, n_nodes, d_model]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_nodes, n_nodes]
            node_mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise
            edge_feats (torch.Tensor): Incoming edge features, shape [batch_size, n_nodes, n_nodes, d_edge_in]

        Returns:
            Either a two-tuple of the new node coordinates and the new node features, or a three-tuple of the new
            node coords, new node features and new edge features.
        """

        if edge_feats is not None and self.d_edge_in is None:
            raise ValueError(
                "edge_feats was provided but the model was initialised with d_edge_in as None."
            )

        if edge_feats is None and self.d_edge_in is not None:
            raise ValueError(
                "The model was initialised with d_edge_in but no edge feats were provided to forward."
            )

        coord_updates, node_updates = self.node_ff(coords, node_feats,
                                                   node_mask)
        coords = coords + coord_updates
        node_feats = node_feats + node_updates

        messages = self.message_ff(coords,
                                   node_feats,
                                   node_mask,
                                   edge_feats=edge_feats)
        node_messages = messages[:, :, :, :self.n_attn_heads]
        coord_messages = messages[:, :, :,
                                  self.n_attn_heads:(self.n_attn_heads +
                                                     self.d_coord_message)]

        node_feats = node_feats + self.node_attn(node_feats, node_messages,
                                                 adj_matrix)
        coords = coords + self.coord_attn(coords, coord_messages, adj_matrix,
                                          node_mask)

        if self.d_edge_out is not None:
            edge_out = messages[:, :, :,
                                (self.n_attn_heads + self.d_coord_message):]
            edge_out = edge_feats + edge_out if edge_feats is not None else edge_out
            return coords, node_feats, edge_out

        return coords, node_feats
