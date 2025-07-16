import torch
from torch import nn

from .nn import CoordNorm, EquivariantMLP, LengthsMLP, PairwiseMessages


def adj_to_attn_mask(adj: torch.Tensor, pos_inf=False):
    """Assumes adjacency mask is only 0s and 1s"""
    adj = adj.bool()  # ensaure adj is boolean

    inf = float("inf") if pos_inf else float("-inf")
    attn_mask = torch.zeros_like(adj, dtype=torch.float32)
    attn_mask[~adj] = inf

    # Ensure nodes with no connections (fake nodes) don't have all -inf in the attn mask
    # Otherwise we would have problems when softmaxing
    attn_mask[~adj.any(dim=-1)] = 0.0

    return attn_mask


class SemlaLayer(nn.Module):
    """Core layer of the Semla architecture.

    The layer contains a self-attention component and a feedforward component, by default. To turn on the conditional
    -attention component in addition to the others, set d_inv_cond to the number of invariant features in the
    conditional input. Note that currently d_equi must be the same for both attention inputs.
    """

    def __init__(
        self,
        d_equi: int,
        d_inv: int,
        d_edge: int,
        d_message: int,
        n_attn_heads: int,
        d_message_ff: int,
        d_inv_cond: int | None = None,
        d_equi_cond: int | None = None,
        d_edge_cond: int | None = None,
        use_condition: bool = False,
        fixed_equi: bool = False,
        zero_com: bool = False,
        eps: float = 1e-3,
    ):
        super().__init__()

        self.use_condition: int | None = use_condition
        self.fixed_equi: bool = fixed_equi

        # *** Self attention components ***
        self.self_attn_inv_norm = nn.LayerNorm(d_inv)

        if not fixed_equi:
            self.self_attn_equi_norm = CoordNorm(d_equi, zero_com=zero_com, eps=eps)

        self.self_attention = SemlaSelfAttention(
            d_equi,
            d_inv,
            d_edge,
            d_message,
            n_attn_heads,
            d_message_ff,
            fixed_equi=fixed_equi,
            eps=eps,
        )

        # *** Pocket-ligand cross attention components ***
        if self.use_condition:
            assert d_inv_cond is not None and d_equi_cond is not None and d_edge_cond is not None, (
                "Conditional attention requires d_inv_cond, d_equi_cond, and d_edge_cond to be specified."
            )
            self.cond_attn_self_inv_norm = nn.LayerNorm(d_inv)
            self.cond_attn_cond_inv_norm = nn.LayerNorm(d_inv_cond)
            self.cond_attn_equi_norm = CoordNorm(d_equi, zero_com=zero_com, eps=eps)
            self.cond_attention = SemlaCondAttention(
                d_equi,
                d_inv,
                d_message,
                n_attn_heads,
                d_message_ff,
                d_equi_cond=d_equi_cond,
                d_inv_cond=d_inv_cond,
                d_edge_cond=d_edge_cond,
                eps=eps,
            )

        # *** Feedforward components ***
        self.ff_inv_norm1 = nn.LayerNorm(d_inv)
        self.inv_ff = LengthsMLP(d_inv, d_equi)

        if not fixed_equi:
            self.ff_equi_norm = CoordNorm(d_equi, zero_com=zero_com, eps=eps)
            self.ff_inv_norm2 = nn.LayerNorm(d_inv)
            self.equi_ff = EquivariantMLP(d_equi, d_inv)

    def forward(
        self,
        equis: torch.Tensor,
        invs: torch.Tensor,
        edges: torch.Tensor,
        node_mask: torch.Tensor,
        adj_mask: torch.Tensor,
        cond_equis: torch.Tensor | None = None,
        cond_invs: torch.Tensor | None = None,
        cond_edges: torch.Tensor | None = None,
        cond_adj_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute output of Semla layer

        Args:
            equis (torch.Tensor): Equivariant features, shape [B, N, 3, d_equi]
            invs (torch.Tensor): Invariant features, shape [B, N, d_inv]
            edges (torch.Tensor): Edge features, shape [B, N, N, d_self_edge_in]
            node_mask (torch.Tensor): Mask for nodes, shape [B, N], 1 for real, 0 otherwise
            adj_mask (torch.Tensor): Adjacency matrix, shape [B, N, N], 1 is connected, 0 otherwise
            cond_equis (torch.Tensor): Cond equivariant features, shape [B, N_c, 3, d_equi]
            cond_invs (torch.Tensor): Cond invariant features, shape [B, N_c, d_inv_cond]
            cond_edges (torch.Tensor): Cond invariant features, shape [B, N, N_c, d_inv_cond]
            cond_adj_mask (torch.Tensor): Adj matrix to cond data, shape [B, N, N_c], 1 is connected, 0 otherwise

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
            Updated equivariant features: torch.Tensor
            updated invariant features: torch.Tensor

            Note that self pairwise features will be None if d_self_edge_out is None, and self
            -conditional pairwise features will be None if d_cond_edge_out is None.
            Tensor shapes: [B, N, 3, d_equi], [B, N, d_inv], [B, N, N, d_self_edge_out], [B, N, N_c, d_cond_edge_out]
        """

        # this style is better for torch.compile()
        if self.use_condition:
            assert cond_equis is not None, (
                "The layer was initialised with conditional attention but cond_equis is missing."
            )
            assert cond_invs is not None, (
                "The layer was initialised with conditional attention but cond_invs is missing."
            )
            assert cond_edges is not None, (
                "The layer was initialised with conditional attention but cond_edges is missing."
            )
            assert cond_adj_mask is not None, (
                "The layer was initialised with conditional attention but cond_adj_mask is missing."
            )

        # *** Self attention component ***
        invs_norm = self.self_attn_inv_norm(invs)
        equis_norm = self.self_attn_equi_norm(equis, node_mask) if not self.fixed_equi else equis
        equi_updates, inv_updates = self.self_attention.forward(equis_norm, invs_norm, edges, adj_mask)

        invs = invs + inv_updates
        equis = equis + equi_updates if not self.fixed_equi else equis

        # *** Conditional attention component ***
        if self.use_condition:
            equis, invs = self._cond_attention(equis, invs, cond_equis, cond_invs, cond_edges, node_mask, cond_adj_mask)

        # *** Feedforward component ***
        invs_norm = self.ff_inv_norm1(invs)
        equis_norm = self.ff_equi_norm(equis, node_mask) if not self.fixed_equi else equis
        inv_update = self.inv_ff(equis_norm.movedim(-1, 1), invs_norm)
        invs = invs + inv_update

        if not self.fixed_equi:
            invs_norm = self.ff_inv_norm2(invs)
            equi_update = self.equi_ff(equis_norm, invs_norm)
            equis = equis + equi_update

        return equis, invs

    def _cond_attention(
        self,
        equis: torch.Tensor,
        invs: torch.Tensor,
        cond_equis: torch.Tensor | None,
        cond_invs: torch.Tensor | None,
        cond_edges: torch.Tensor | None,
        node_mask: torch.Tensor,
        cond_adj_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self_invs_norm = self.cond_attn_self_inv_norm(invs)
        cond_invs_norm = self.cond_attn_cond_inv_norm(cond_invs)
        equis_norm = self.cond_attn_equi_norm(equis, node_mask)
        equi_updates, inv_updates = self.cond_attention(
            equis_norm, self_invs_norm, cond_equis, cond_invs_norm, cond_edges, cond_adj_mask
        )
        equis = equis + equi_updates
        invs = invs + inv_updates
        return equis, invs


class SemlaSelfAttention(nn.Module):
    def __init__(
        self,
        d_equi: int,
        d_inv: int,
        d_edge: int,
        d_message: int,
        n_heads: int,
        d_ff: int,
        fixed_equi: bool = False,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.d_equi: int = d_equi
        self.n_heads: int = n_heads
        self.d_edge: int = d_edge
        self.fixed_equi: int = fixed_equi

        d_out = n_heads
        if not fixed_equi:
            d_out = n_heads + d_equi
        self.messages = PairwiseMessages(d_equi, d_inv, d_edge, d_out, d_message, d_ff)
        self.inv_attn = _InvAttention(d_inv, n_attn_heads=n_heads)
        if not fixed_equi:
            self.equi_attn = _EquiAttention(d_equi, eps=eps)

    def forward(
        self,
        equis: torch.Tensor,
        invs: torch.Tensor,
        edges: torch.Tensor,
        adj_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Compute output of self attention layer

        Args:
            equis (torch.Tensor): Equivariant features, shape [B, N, 3, d_equi]
            invs (torch.Tensor): Invariant features, shape [B, N, d_inv]
            edges (torch.Tensor): Edge feature matrix, shape [B, N, N, d_edge]
            adj_mask (torch.Tensor): Adjacency matrix, shape [B, N, N], 1 is connected, 0 otherwise

        Returns:
            updated equi features: torch.Tensor
            updated inv features: torch.Tensor
        """
        messages = self.messages(equis, invs, equis, invs, edges)
        if not self.fixed_equi:
            inv_messages, equi_messages = torch.split(messages, [self.n_heads, self.d_equi], dim=-1)
            inv_updates = self.inv_attn(invs, inv_messages, adj_mask)
            equi_updates = self.equi_attn(equis, equi_messages, adj_mask)
        else:
            inv_updates = self.inv_attn(invs, messages, adj_mask)
            equi_updates = None

        return equi_updates, inv_updates


class SemlaCondAttention(nn.Module):
    def __init__(
        self,
        d_equi: int,
        d_inv: int,
        d_message: int,
        n_heads: int,
        d_ff: int,
        d_inv_cond: int,
        d_equi_cond: int,
        d_edge_cond: int,
        eps: float = 1e-3,
    ):
        super().__init__()
        # Use d_inv for the conditional inviariant features by default
        self.d_equi: int = d_equi
        self.n_heads: int = n_heads
        d_out = d_equi + n_heads
        self.messages: PairwiseMessages = PairwiseMessages(
            (d_equi, d_equi_cond), (d_inv, d_inv_cond), d_edge_cond, d_out, d_message, d_ff
        )
        self.equi_attn: _EquiAttention = _EquiAttention(d_equi, eps=eps)
        self.inv_attn: _InvAttention = _InvAttention(d_inv, n_attn_heads=n_heads, d_inv_cond=d_inv_cond)

    def forward(
        self,
        equis: torch.Tensor,
        invs: torch.Tensor,
        cond_equis: torch.Tensor,
        cond_invs: torch.Tensor,
        cond_edges: torch.Tensor,
        adj_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute output of conditional attention layer

        Args:
            equis (torch.Tensor): Equivariant features, shape [B, N, 3, d_equi]
            invs (torch.Tensor): Invariant features, shape [B, N, d_inv]
            cond_equis (torch.Tensor): Conditional equivariant features, shape [B, N_c, 3, d_equi]
            cond_invs (torch.Tensor): Conditional invariant features, shape [B, N_c, d_inv_cond]
            adj_mask (torch.Tensor): Adjacency matrix, shape [B, N, N_c], 1 is connected, 0 otherwise

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Updates to equi feats, inv feats, respectively
        """
        messages = self.messages(equis, invs, cond_equis, cond_invs, edge_feats=cond_edges)
        equi_messages, inv_messages = torch.split(messages, [self.d_equi, self.n_heads], dim=-1)

        equi_updates = self.equi_attn(cond_equis, equi_messages, adj_mask)
        inv_updates = self.inv_attn(cond_invs, inv_messages, adj_mask)
        return equi_updates, inv_updates


class _EquiAttention(nn.Module):
    def __init__(self, d_equi: int, eps: float = 1e-3):
        super().__init__()
        self.d_equi: int = d_equi
        self.eps: float = eps
        self.coord_proj = nn.Linear(d_equi, d_equi, bias=False)
        self.attn_proj = nn.Linear(d_equi, d_equi, bias=False)

    def forward(self, v_equi: torch.Tensor, messages: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        """Compute an attention update for equivariant features

        Args:
            v_equi (torch.Tensor): Coordinate tensor, shape [B, N_kv, 3, d_equi]
            messages (torch.Tensor): Messages tensor, shape [B, N_q, N_kv, d_equi]
            adj_mask (torch.Tensor): Adjacency matrix, shape [B, N_q, N_kv]

        Returns:
            torch.Tensor: Updates for equi features, shape [B, N_q, 3, d_equi]
        """

        proj_equi = self.coord_proj(v_equi)

        attn_mask = adj_to_attn_mask(adj_mask)
        messages = messages + attn_mask.unsqueeze(3)
        attentions = torch.softmax(messages, dim=2)

        # Attentions shape now [B * d_equi, N_q, N_kv]
        # proj_equi shape now [B * d_equi, N_kv, 3]
        attentions = attentions.movedim(-1, 1).flatten(0, 1)
        proj_equi = proj_equi.movedim(-1, 1).flatten(0, 1)
        attn_out = torch.bmm(attentions, proj_equi)

        # Apply variance preserving updates as proposed in GNN-VPA (https://arxiv.org/abs/2403.04747)
        weights = torch.sqrt((attentions**2).sum(dim=-1))
        attn_out = attn_out * weights.unsqueeze(-1)

        attn_out = attn_out.unflatten(0, (-1, self.d_equi)).movedim(1, -1)
        return self.attn_proj(attn_out)


class _InvAttention(nn.Module):
    def __init__(
        self,
        d_inv: int,
        n_attn_heads: int,
        d_inv_cond: int | None = None,
    ):
        super().__init__()
        d_inv_in: int = d_inv_cond if d_inv_cond is not None else d_inv
        d_head: int = d_inv_in // n_attn_heads
        if d_inv_in % n_attn_heads != 0:
            raise ValueError("n_attn_heads must divide d_inv or d_inv_cond (if provided) exactly.")

        self.d_inv: int = d_inv
        self.n_attn_heads: int = n_attn_heads
        self.d_head: int = d_head

        self.in_proj = nn.Linear(d_inv_in, d_inv_in)
        self.out_proj = nn.Linear(d_inv_in, d_inv)

    def forward(self, v_inv: torch.Tensor, messages: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        """Accumulate edge messages to each node using attention-based message passing

        Args:
            v_inv (torch.Tensor): Node feature tensor, shape [B, N_kv, d_inv or d_inv_cond if provided]
            messages (torch.Tensor): Messages tensor, shape [B, N_q, N_kv, d_message]
            adj_mask (torch.Tensor): Adjacency matrix, shape [B, N_q, N_kv]

        Returns:
            torch.Tensor: Updates to invariant features, shape [B, N_q, d_inv]
        """

        attn_mask = adj_to_attn_mask(adj_mask)
        messages = messages + attn_mask.unsqueeze(-1)
        attentions = torch.softmax(messages, dim=2)

        proj_feats = self.in_proj(v_inv)
        head_feats = proj_feats.unflatten(-1, (self.n_attn_heads, self.d_head))

        # Put n_heads into the batch dim for both the features and the attentions
        # head_feats shape [B * n_heads, N_kv, d_head]
        # attentions shape [B * n_heads, N_q, N_kv]
        head_feats = head_feats.movedim(-2, 1).flatten(0, 1)
        attentions = attentions.movedim(-1, 1).flatten(0, 1)
        attn_out = torch.bmm(attentions, head_feats)

        # Apply variance preserving updates as proposed in GNN-VPA (https://arxiv.org/abs/2403.04747)
        weights = torch.sqrt((attentions**2).sum(dim=-1))
        attn_out = attn_out * weights.unsqueeze(-1)

        attn_out = attn_out.unflatten(0, (-1, self.n_attn_heads))
        attn_out = attn_out.movedim(1, -2).flatten(2, 3)
        return self.out_proj(attn_out)
