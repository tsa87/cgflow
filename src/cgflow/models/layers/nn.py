import torch
import torch.nn as nn


class LengthsMLP(nn.Module):
    def __init__(self, d_model: int, n_coord_sets: int, d_ff: int | None = None):
        super().__init__()
        d_ff = d_model * 4 if d_ff is None else d_ff
        self.node_ff = nn.Sequential(
            nn.Linear(d_model + n_coord_sets, d_ff),
            nn.SiLU(inplace=False),
            nn.Linear(d_ff, d_model),
        )

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


class CoordNorm(nn.Module):
    def __init__(self, d_equi: int, zero_com: bool = False, eps: float = 1e-3):
        super().__init__()
        self.d_equi: int = d_equi
        self.zero_com: bool = zero_com
        self.eps: float = eps
        self.set_weights = nn.Parameter(torch.ones((1, 1, 1, d_equi)))

    def forward(self, coords: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """Apply coordinate normlisation layer

        Args:
            coords (torch.Tensor): Coordinate tensor, shape [B, N, 3, d_equi]
            node_mask (torch.BoolTensor): Mask for nodes, shape [B, N], True for real

        Returns:
            torch.Tensor: Normalised coords, shape [B, N, 3, d_equi]
        """
        # mask coords
        B, N = node_mask.shape
        coords = coords * node_mask.view(B, N, 1, 1)  # [B, N, 3, d_equi]
        n_nodes = node_mask.sum(dim=-1).clip(min=1)  # [B,]

        if self.zero_com:
            center = coords.sum(dim=-3) / n_nodes.view(B, 1, 1)  # [B, 3, d_equi]
            coords = coords - center.unsqueeze(-3)  # [B, N, 3, d_equi]

        lengths = torch.linalg.vector_norm(coords, dim=-2, keepdim=True)  # [B, N, 1, d_equi]
        scaled_lengths = lengths.sum(dim=1, keepdim=True) / n_nodes.view(B, 1, 1, 1)  # [B, 1, 1, d_equi]
        coords = (coords * self.set_weights) / (scaled_lengths + self.eps)
        return coords

    def reset_parameters(self):
        nn.init.ones_(self.weight)


class EquivariantMLP(nn.Module):
    def __init__(self, d_equi: int, d_inv: int):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(d_equi + d_inv, d_equi),
            nn.SiLU(inplace=False),
            nn.Linear(d_equi, d_equi),
            nn.Sigmoid(),
        )
        self.coord_proj = nn.Linear(d_equi, d_equi, bias=False)
        self.attn_proj = nn.Linear(d_equi, d_equi, bias=False)

    def forward(self, equis: torch.Tensor, invs: torch.Tensor) -> torch.Tensor:
        """Pass data through the layer
        Assumes coords and node_feats have already been normalised

        Args:
            equis (torch.Tensor): Equivariant features, shape [B, N, 3, d_equi]
            invs (torch.Tensor): Invariant features, shape [B, N, d_inv]

        Returns:
            torch.Tensor: Updated equivariant features, shape [B, N, 3, d_equi]
        """
        lengths = torch.linalg.vector_norm(equis, dim=2)
        invs = torch.cat((invs, lengths), dim=-1)
        inv_feats = self.node_proj(invs)  # [B, N, d_equi]
        proj_sets = self.coord_proj(equis)  # [B, N, 3, d_equi]
        gated_equis = inv_feats.unsqueeze(2) * proj_sets  # [B, N, 3, d_equi]
        equis_out = self.attn_proj(gated_equis)
        return equis_out


class PairwiseMessages(nn.Module):
    """Compute pairwise features for a set of query and a set of key nodes"""

    def __init__(
        self,
        d_equi: int | tuple[int, int],  # q_equi, k_equi
        d_inv: int | tuple[int, int],  # q_inv, k_inv
        d_edge: int | None,
        d_out: int,
        d_message: int,
        d_ff: int,
        include_dists: bool = True,
    ):
        super().__init__()
        if isinstance(d_inv, int):
            d_q_inv = d_k_inv = d_inv
        else:
            d_q_inv, d_k_inv = d_inv

        if isinstance(d_equi, int):
            d_q_equi = d_k_equi = d_equi
        else:
            d_q_equi, d_k_equi = d_equi

        in_feats = d_message * 3
        if include_dists:
            in_feats += d_message
        if d_edge:
            in_feats += d_edge

        self.d_message: int = d_message
        self.d_edge: int | None = d_edge
        self.include_dists: bool = include_dists

        self.q_inv_proj = nn.Linear(d_q_inv, d_message)
        self.k_inv_proj = nn.Linear(d_k_inv, d_message)
        self.q_equi_proj = nn.Linear(d_q_equi, d_message)
        self.k_equi_proj = nn.Linear(d_k_equi, d_message)
        self.message_mlp = nn.Sequential(
            nn.Linear(in_feats, d_ff),
            nn.SiLU(inplace=False),
            nn.Linear(d_ff, d_out),
        )

    def forward(
        self,
        q_equi: torch.Tensor,
        q_inv: torch.Tensor,
        k_equi: torch.Tensor | None = None,
        k_inv: torch.Tensor | None = None,
        edge_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce messages between query and key

        Args:
            q_equi (torch.Tensor): Equivariant query features, shape [B, N_q, 3, d_equi]
            q_inv (torch.Tensor): Invariant query features, shape [B, N_q, d_q_inv]
            k_equi (torch.Tensor): Equivariant key features, shape [B, N_kv, 3, d_equi]
            k_inv (torch.Tensor): Invariant key features, shape [B, N_kv, 3, d_kv_inv]
            edge_feats (torch.Tensor): Edge features, shape [B, N_q, N_kv, d_edge]

        Returns:
            torch.Tensor: Message matrix, shape [B, N_q, N_k, d_out]
        """
        if self.d_edge is None:
            assert edge_feats is None, "edge_feats was provided but the model was initialised with d_edge as None."
        else:
            assert self.d_edge is not None, (
                "The model was initialised with d_edge but no edge feats were provided to forward fn.",
            )

        if k_equi is None:
            k_equi = q_equi
        if k_inv is None:
            k_inv = q_inv

        q_inv = self.q_inv_proj(q_inv).unsqueeze(2).expand(-1, -1, k_inv.size(1), -1)
        k_inv = self.k_inv_proj(k_inv).unsqueeze(1).expand(-1, q_inv.size(1), -1, -1)

        q_equi = self.q_equi_proj(q_equi)  # [B, N_q, 3, d_message]
        k_equi = self.k_equi_proj(k_equi)  # [B, N_q, 3, d_message]
        q_equi_batched = q_equi.movedim(-1, 1).flatten(0, 1)  # [B * d_m, N_q, 3]
        k_equi_batched = k_equi.movedim(-1, 1).flatten(0, 1)  # [B * d_m, N_k, 3]
        dotprods = torch.bmm(q_equi_batched, k_equi_batched.transpose(1, 2))
        dotprods = dotprods.unflatten(0, (-1, self.d_message)).movedim(1, -1)  # [B, N_q, N_k, d_m]

        pairwise_feat_list = [q_inv, k_inv, dotprods]
        if self.include_dists:
            vec_dists = q_equi.unsqueeze(2) - k_equi.unsqueeze(1)
            dists = torch.linalg.vector_norm(vec_dists, dim=3)
            pairwise_feat_list.append(dists)
        if edge_feats is not None:
            pairwise_feat_list.append(edge_feats)

        pairwise_feats = torch.cat(pairwise_feat_list, dim=-1)  # [B, N_q, N_k, in_feats]
        pairwise_messages = self.message_mlp(pairwise_feats)  # [B, N_q, N_k, d_message]
        return pairwise_messages
