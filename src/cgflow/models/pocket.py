import copy

import torch

import cgflow.util.functional as smolF
from cgflow.models.semla import BondRefine, LengthsMLP, adj_to_attn_mask

_T = torch.Tensor

# *****************************************************************************
# ******************************* Helper Modules ******************************
# *****************************************************************************


class _CoordNorm(torch.nn.Module):

    def __init__(self, d_equi, zero_com=True, eps=1e-6):
        super().__init__()

        self.d_equi = d_equi
        self.zero_com = zero_com
        self.eps = eps

        self.set_weights = torch.nn.Parameter(torch.ones((1, 1, 1, d_equi)))

    def forward(self, coord_sets, node_mask):
        """Apply coordinate normlisation layer

        Args:
            coord_sets (torch.Tensor): Coordinate tensor, shape [B, N, 3, d_equi]
            node_mask (torch.Tensor): Mask for nodes, shape [B, N], 1 for real, 0 otherwise

        Returns:
            torch.Tensor: Normalised coords, shape [B, N, 3, d_equi]
        """

        if self.zero_com:
            coord_sets = smolF.zero_com(coord_sets, node_mask)

        n_atoms = node_mask.sum(dim=-1).view(-1, 1, 1, 1)
        lengths = torch.linalg.vector_norm(coord_sets, dim=2, keepdim=True)
        scaled_lengths = lengths.sum(dim=1, keepdim=True) / n_atoms
        coord_sets = (coord_sets * self.set_weights) / (scaled_lengths +
                                                        self.eps)
        coord_sets = coord_sets * node_mask.unsqueeze(-1).unsqueeze(-1)

        return coord_sets

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


class _EquivariantMLP(torch.nn.Module):

    def __init__(self, d_equi, d_inv, original=False):
        super().__init__()

        self.original = original

        if original:
            self.node_proj = torch.nn.Sequential(
                torch.nn.Linear(d_inv, d_equi), torch.nn.SiLU(inplace=False),
                torch.nn.Linear(d_equi, d_equi))

        else:
            self.node_proj = torch.nn.Sequential(
                torch.nn.Linear(d_equi + d_inv, d_equi),
                torch.nn.SiLU(inplace=False),
                torch.nn.Linear(d_equi, d_equi),
                torch.nn.Sigmoid(),
            )

        self.coord_proj = torch.nn.Linear(d_equi, d_equi, bias=False)
        self.attn_proj = torch.nn.Linear(d_equi, d_equi, bias=False)

    def forward(self, equis, invs):
        """Pass data through the layer

        Assumes coords and node_feats have already been normalised

        Args:
            equis (torch.Tensor): Equivariant features, shape [B, N, 3, d_equi]
            invs (torch.Tensor): Invariant features, shape [B, N, d_inv]

        Returns:
            torch.Tensor: Updated equivariant features, shape [B, N, 3, d_equi]
        """

        if not self.original:
            lengths = torch.linalg.vector_norm(equis, dim=2)
            invs = torch.cat((invs, lengths), dim=-1)

        inv_feats = self.node_proj(invs).unsqueeze(2)
        proj_sets = self.coord_proj(equis)

        # inv_feats shape [B, N, 1, d_equi]
        # proj_sets shape [B, N, 3, d_equi]

        if self.original:
            attentions = inv_feats.unsqueeze(-1) * proj_sets.unsqueeze(-2)
            gated_equis = attentions.sum(-1)
        else:
            gated_equis = proj_sets * inv_feats

        equis_out = self.attn_proj(gated_equis)
        return equis_out


class _PairwiseMessages(torch.nn.Module):
    """Compute pairwise features for a set of query and a set of key nodes"""

    def __init__(self,
                 d_equi,
                 d_q_inv,
                 d_kv_inv,
                 d_message,
                 d_out,
                 d_ff,
                 d_edge=None,
                 include_dists=False):
        super().__init__()

        in_feats = (d_message * 2) + d_equi
        in_feats = in_feats + d_edge if d_edge is not None else in_feats
        in_feats = in_feats + d_equi if include_dists else in_feats

        self.d_equi = d_equi
        self.d_edge = d_edge
        self.include_dists = include_dists

        self.q_message_proj = torch.nn.Linear(d_q_inv, d_message)
        self.k_message_proj = torch.nn.Linear(d_kv_inv, d_message)

        self.message_mlp = torch.nn.Sequential(torch.nn.Linear(in_feats, d_ff),
                                               torch.nn.SiLU(inplace=False),
                                               torch.nn.Linear(d_ff, d_out))

    def forward(self, q_equi, q_inv, k_equi, k_inv, edge_feats=None):
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

        if edge_feats is not None and self.d_edge is None:
            raise ValueError(
                "edge_feats was provided but the model was initialised with d_edge as None."
            )

        if edge_feats is None and self.d_edge is not None:
            raise ValueError(
                "The model was initialised with d_edge but no edge feats were provided to forward fn."
            )

        q_equi_batched = q_equi.movedim(-1, 1).flatten(0, 1)
        k_equi_batched = k_equi.movedim(-1, 1).flatten(0, 1)

        dotprods = torch.bmm(q_equi_batched, k_equi_batched.transpose(1, 2))
        dotprods = dotprods.unflatten(0, (-1, self.d_equi)).movedim(1, -1)

        q_messages = self.q_message_proj(q_inv).unsqueeze(2).expand(
            -1, -1, k_inv.size(1), -1)
        k_messages = self.k_message_proj(k_inv).unsqueeze(1).expand(
            -1, q_inv.size(1), -1, -1)

        pairwise_feats = torch.cat((q_messages, k_messages, dotprods), dim=-1)

        if self.include_dists:
            vec_dists = q_equi.unsqueeze(2) - k_equi.unsqueeze(1)
            dists = torch.linalg.vector_norm(vec_dists, dim=3)
            pairwise_feats = torch.cat((pairwise_feats, dists), dim=-1)

        if edge_feats is not None:
            pairwise_feats = torch.cat((pairwise_feats, edge_feats), dim=-1)

        pairwise_messages = self.message_mlp(pairwise_feats)
        return pairwise_messages


class _EquiAttention(torch.nn.Module):

    def __init__(self, d_equi, eps=1e-6):
        super().__init__()

        self.d_equi = d_equi
        self.eps = eps

        self.coord_proj = torch.nn.Linear(d_equi, d_equi, bias=False)
        self.attn_proj = torch.nn.Linear(d_equi, d_equi, bias=False)

    def forward(self, v_equi, messages, adj_matrix):
        """Compute an attention update for equivariant features

        Args:
            v_equi (torch.Tensor): Coordinate tensor, shape [B, N_kv, 3, d_equi]
            messages (torch.Tensor): Messages tensor, shape [B, N_q, N_kv, d_equi]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [B, N_q, N_kv]

        Returns:
            torch.Tensor: Updates for equi features, shape [B, N_q, 3, d_equi]
        """

        proj_equi = self.coord_proj(v_equi)

        attn_mask = adj_to_attn_mask(adj_matrix)
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


class _InvAttention(torch.nn.Module):

    def __init__(self, d_inv, n_attn_heads, d_inv_cond=None):
        super().__init__()

        d_inv_in = d_inv_cond if d_inv_cond is not None else d_inv

        d_head = d_inv_in // n_attn_heads

        if d_inv_in % n_attn_heads != 0:
            raise ValueError(
                "n_attn_heads must divide d_inv or d_inv_cond (if provided) exactly."
            )

        self.d_inv = d_inv
        self.n_attn_heads = n_attn_heads
        self.d_head = d_head

        self.in_proj = torch.nn.Linear(d_inv_in, d_inv_in)
        self.out_proj = torch.nn.Linear(d_inv_in, d_inv)

    def forward(self, v_inv, messages, adj_matrix):
        """Accumulate edge messages to each node using attention-based message passing

        Args:
            v_inv (torch.Tensor): Node feature tensor, shape [B, N_kv, d_inv or d_inv_cond if provided]
            messages (torch.Tensor): Messages tensor, shape [B, N_q, N_kv, d_message]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [B, N_q, N_kv]

        Returns:
            torch.Tensor: Updates to invariant features, shape [B, N_q, d_inv]
        """

        attn_mask = adj_to_attn_mask(adj_matrix)
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


class SemlaSelfAttention(torch.nn.Module):

    def __init__(self,
                 d_equi,
                 d_inv,
                 d_message,
                 n_heads,
                 d_ff,
                 d_edge_in=None,
                 d_edge_out=None,
                 fixed_equi=False,
                 eps=1e-6):
        super().__init__()

        d_out = n_heads if fixed_equi else d_equi + n_heads
        d_out = d_out + d_edge_out if d_edge_out is not None else d_out

        messages = _PairwiseMessages(d_equi,
                                     d_inv,
                                     d_inv,
                                     d_message,
                                     d_out,
                                     d_ff,
                                     d_edge=d_edge_in)

        inv_attn = _InvAttention(d_inv, n_attn_heads=n_heads)

        self.d_equi = d_equi
        self.n_heads = n_heads
        self.d_edge_in = d_edge_in
        self.d_edge_out = d_edge_out
        self.fixed_equi = fixed_equi

        self.messages = messages
        self.inv_attn = inv_attn

        if not fixed_equi:
            self.equi_attn = _EquiAttention(d_equi, eps=eps)

    def forward(self, equis, invs, edges, adj_matrix):
        """Compute output of self attention layer

        Args:
            equis (torch.Tensor): Equivariant features, shape [B, N, 3, d_equi]
            invs (torch.Tensor): Invariant features, shape [B, N, d_inv]
            edges (torch.Tensor): Edge feature matrix, shape [B, N, N, d_edge]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [B, N, N], 1 is connected, 0 otherwise

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Updates to equi features, inv feats, edge features
            Note that equi features are None if fixed_equi is specified, and edge features are None if d_edge_out
            is None. This ordering is used to maintain consistency with the ordering in other modules and to help to
            ensure that errors will be thrown if the wrong output is taken.
        """

        messages = self.messages(equis, invs, equis, invs, edge_feats=edges)

        inv_messages = messages[..., :self.n_heads]
        inv_updates = self.inv_attn(invs, inv_messages, adj_matrix)

        equi_updates = None
        if not self.fixed_equi:
            equi_messages = messages[...,
                                     self.n_heads:self.n_heads + self.d_equi]
            equi_updates = self.equi_attn(equis, equi_messages, adj_matrix)

        edge_feats = None
        if self.d_edge_out is not None:
            edge_feats = messages[..., self.n_heads + self.d_equi:]

        return equi_updates, inv_updates, edge_feats


class SemlaCondAttention(torch.nn.Module):

    def __init__(self,
                 d_equi,
                 d_inv,
                 d_message,
                 n_heads,
                 d_ff,
                 d_inv_cond=None,
                 d_edge_in=None,
                 d_edge_out=None,
                 eps=1e-6):
        super().__init__()

        # Set the number of pairwise output features depending on whether edge features are generated or not
        d_out = d_equi + n_heads
        d_out = d_out if d_edge_out is None else d_out + d_edge_out

        # Use d_inv for the conditional inviariant features by default
        d_inv_cond = d_inv if d_inv_cond is None else d_inv_cond

        messages = _PairwiseMessages(d_equi,
                                     d_inv,
                                     d_inv_cond,
                                     d_message,
                                     d_out,
                                     d_ff,
                                     d_edge=d_edge_in)

        equi_attn = _EquiAttention(d_equi, eps=eps)
        inv_attn = _InvAttention(d_inv,
                                 n_attn_heads=n_heads,
                                 d_inv_cond=d_inv_cond)

        self.d_equi = d_equi
        self.n_heads = n_heads
        self.d_edge_in = d_edge_in
        self.d_edge_out = d_edge_out

        self.messages = messages
        self.equi_attn = equi_attn
        self.inv_attn = inv_attn

    def forward(self, equis, invs, cond_equis, cond_invs, edges, adj_matrix):
        """Compute output of conditional attention layer

        Args:
            equis (torch.Tensor): Equivariant features, shape [B, N, 3, d_equi]
            invs (torch.Tensor): Invariant features, shape [B, N, d_inv]
            cond_equis (torch.Tensor): Conditional equivariant features, shape [B, N_c, 3, d_equi]
            cond_invs (torch.Tensor): Conditional invariant features, shape [B, N_c, d_inv_cond]
            edges (torch.Tensor): Edge feature matrix, shape [B, N, N_c, d_edge]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [B, N, N_c], 1 is connected, 0 otherwise

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Updates to equi feats, inv feats, and edge feats,
            respectively. Note that the edge features will be None is d_edge_out is None.
        """

        messages = self.messages(equis,
                                 invs,
                                 cond_equis,
                                 cond_invs,
                                 edge_feats=edges)
        equi_messages = messages[..., :self.d_equi]
        inv_messages = messages[..., self.d_equi:self.d_equi + self.n_heads]

        edge_feats = None
        if self.d_edge_out is not None:
            edge_feats = messages[..., self.d_equi + self.n_heads:]

        equi_updates = self.equi_attn(cond_equis, equi_messages, adj_matrix)
        inv_updates = self.inv_attn(cond_invs, inv_messages, adj_matrix)

        return equi_updates, inv_updates, edge_feats


# *****************************************************************************
# ********************************* Semla Layer *******************************
# *****************************************************************************


class SemlaLayer(torch.nn.Module):
    """Core layer of the Semla architecture.

    The layer contains a self-attention component and a feedforward component, by default. To turn on the conditional
    -attention component in addition to the others, set d_inv_cond to the number of invariant features in the
    conditional input. Note that currently d_equi must be the same for both attention inputs.
    """

    def __init__(
        self,
        d_equi,
        d_inv,
        d_message,
        n_attn_heads,
        d_message_ff,
        d_inv_cond=None,
        d_self_edge_in=None,
        d_self_edge_out=None,
        d_cond_edge_in=None,
        d_cond_edge_out=None,
        fixed_equi=False,
        zero_com=False,
        eps=1e-6,
    ):
        super().__init__()

        if d_inv_cond is not None and fixed_equi:
            raise ValueError(
                "Equivariant features cannot be fixed when using conditional attention."
            )

        self.d_inv_cond = d_inv_cond
        self.d_self_edge_out = d_self_edge_out
        self.d_cond_edge_out = d_cond_edge_out
        self.fixed_equi = fixed_equi

        # *** Self attention components ***
        self.self_attn_inv_norm = torch.nn.LayerNorm(d_inv)

        if not fixed_equi:
            self.self_attn_equi_norm = _CoordNorm(d_equi,
                                                  zero_com=zero_com,
                                                  eps=eps)

        self.self_attention = SemlaSelfAttention(
            d_equi,
            d_inv,
            d_message,
            n_attn_heads,
            d_message_ff,
            d_edge_in=d_self_edge_in,
            d_edge_out=d_self_edge_out,
            fixed_equi=fixed_equi,
            eps=eps,
        )

        # *** Cross attention components ***
        if d_inv_cond is not None:
            self.cond_attn_self_inv_norm = torch.nn.LayerNorm(d_inv)
            self.cond_attn_cond_inv_norm = torch.nn.LayerNorm(d_inv_cond)
            self.cond_attn_equi_norm = _CoordNorm(d_equi,
                                                  zero_com=zero_com,
                                                  eps=eps)

            self.cond_attention = SemlaCondAttention(
                d_equi,
                d_inv,
                d_message,
                n_attn_heads,
                d_message_ff,
                d_inv_cond=d_inv_cond,
                d_edge_in=d_cond_edge_in,
                d_edge_out=d_cond_edge_out,
                eps=eps,
            )

        # *** Feedforward components ***
        self.ff_inv_norm = torch.nn.LayerNorm(d_inv)
        self.inv_ff = LengthsMLP(d_inv, d_equi)

        if not fixed_equi:
            self.ff_equi_norm = _CoordNorm(d_equi, zero_com=zero_com, eps=eps)
            self.equi_ff = _EquivariantMLP(d_equi, d_inv)

    def forward(
        self,
        equis,
        invs,
        edges,
        adj_matrix,
        node_mask,
        cond_equis=None,
        cond_invs=None,
        cond_edges=None,
        cond_adj_matrix=None,
    ):
        """Compute output of Semla layer

        Args:
            equis (torch.Tensor): Equivariant features, shape [B, N, 3, d_equi]
            invs (torch.Tensor): Invariant features, shape [B, N, d_inv]
            edges (torch.Tensor): Edge features, shape [B, N, N, d_self_edge_in]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [B, N, N], 1 is connected, 0 otherwise
            node_mask (torch.Tensor): Mask for nodes, shape [B, N], 1 for real, 0 otherwise
            cond_equis (torch.Tensor): Cond equivariant features, shape [B, N, 3, d_equi]
            cond_invs (torch.Tensor): Cond invariant features, shape [B, N, d_inv_cond]
            cond_edges (torch.Tensor): Edge features between self and cond, shape [B, N, N_c, d_cond_edge_in]
            cond_adj_matrix (torch.Tensor): Adj matrix to cond data, shape [B, N, N_c], 1 is connected, 0 otherwise

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Updated equivariant features, updated invariant features, self pairwise features, self-conditional
            pairwise features. Note that self pairwise features will be None if d_self_edge_out is None, and self
            -conditional pairwise features will be None if d_cond_edge_out is None.
            Tensor shapes: [B, N, 3, d_equi], [B, N, d_inv], [B, N, N, d_self_edge_out], [B, N, N_c, d_cond_edge_out]
        """

        # this style is better for torch.compile()
        if self.d_inv_cond:
            assert (
                cond_equis is not None
            ), "The layer was initialised with conditional attention but cond_equis is missing."
            assert (
                cond_invs is not None
            ), "The layer was initialised with conditional attention but cond_invs is missing."
            assert (
                cond_adj_matrix is not None
            ), "The layer was initialised with conditional attention but cond_adj_matrix is missing."

        # *** Self attention component ***
        invs_norm = self.self_attn_inv_norm(invs)
        equis_norm = equis if self.fixed_equi else self.self_attn_equi_norm(
            equis, node_mask)
        equi_updates, inv_updates, self_edge_feats = self.self_attention(
            equis_norm, invs_norm, edges, adj_matrix)

        invs = invs + inv_updates
        if not self.fixed_equi:
            equis = equis + equi_updates

        # *** Conditional attention component ***
        if self.d_inv_cond is not None:
            equis, invs, cond_edge_feats = self._compute_cond_attention(
                equis, invs, cond_equis, cond_invs, cond_edges, node_mask,
                cond_adj_matrix)
        else:
            cond_edge_feats = None

        # *** Feedforward component ***
        invs_norm = self.ff_inv_norm(invs)
        equis_norm = equis if self.fixed_equi else self.ff_equi_norm(
            equis, node_mask)

        inv_update = self.inv_ff(equis_norm.movedim(-1, 1), invs_norm)
        invs = invs + inv_update

        if not self.fixed_equi:
            equi_update = self.equi_ff(equis_norm, invs_norm)
            equis = equis + equi_update

        return equis, invs, self_edge_feats, cond_edge_feats

    def _compute_cond_attention(self, equis, invs, cond_equis, cond_invs,
                                cond_edges, node_mask, cond_adj_matrix):
        self_invs_norm = self.cond_attn_self_inv_norm(invs)
        cond_invs_norm = self.cond_attn_cond_inv_norm(cond_invs)
        equis_norm = self.cond_attn_equi_norm(equis, node_mask)

        equi_updates, inv_updates, cond_edge_feats = self.cond_attention(
            equis_norm, self_invs_norm, cond_equis, cond_invs_norm, cond_edges,
            cond_adj_matrix)

        equis = equis + equi_updates
        invs = invs + inv_updates

        return equis, invs, cond_edge_feats


# *****************************************************************************
# ************************ Encoder and Decoder Stacks *************************
# *****************************************************************************


class _InvariantEmbedding(torch.nn.Module):

    def __init__(
        self,
        d_inv,
        n_atom_types,
        n_bond_types,
        emb_size,
        n_charge_types=None,
        n_extra_feats=None,
        n_res_types=None,
        self_cond=False,
        max_size=None,
    ):
        super().__init__()

        n_embeddings = 2 if max_size is not None else 1
        n_embeddings = n_embeddings + 1 if n_charge_types is not None else n_embeddings
        n_embeddings = n_embeddings + 1 if n_res_types is not None else n_embeddings

        atom_in_feats = emb_size * n_embeddings
        atom_in_feats = atom_in_feats + n_atom_types if self_cond else atom_in_feats
        atom_in_feats = atom_in_feats + n_extra_feats if n_extra_feats is not None else atom_in_feats

        self.n_charge_types = n_charge_types
        self.n_extra_feats = n_extra_feats
        self.n_res_types = n_res_types
        self.self_cond = self_cond
        self.max_size = max_size

        self.atom_type_emb = torch.nn.Embedding(n_atom_types, emb_size)

        if n_charge_types is not None:
            self.atom_charge_emb = torch.nn.Embedding(n_charge_types, emb_size)

        if n_res_types is not None:
            self.res_type_emb = torch.nn.Embedding(n_res_types, emb_size)

        if max_size is not None:
            self.size_emb = torch.nn.Embedding(max_size, emb_size)

        self.atom_emb = torch.nn.Sequential(
            torch.nn.Linear(atom_in_feats, d_inv), torch.nn.SiLU(),
            torch.nn.Linear(d_inv, d_inv))

        self.bond_emb = torch.nn.Embedding(n_bond_types, emb_size)

        if self_cond:
            self.bond_proj = torch.nn.Linear(emb_size + n_bond_types, emb_size)

    def forward(
        self,
        atom_types,
        bond_types,
        atom_mask,
        atom_charges=None,
        extra_feats=None,
        res_types=None,
        cond_types=None,
        cond_bonds=None,
    ):
        if (cond_types is not None
                or cond_bonds is not None) and not self.self_cond:
            raise ValueError(
                "Conditional inputs were provided but the model was initialised with self_cond as False."
            )

        if (cond_types is None or cond_bonds is None) and self.self_cond:
            raise ValueError(
                "Conditional inputs must be provided if using self conditioning."
            )

        if self.n_charge_types is not None and atom_charges is None:
            raise ValueError(
                "The invariant embedding was initialised for charge embeddings but none were provided."
            )

        if self.n_extra_feats is not None and extra_feats is None:
            raise ValueError(
                "The invariant embedding was initialised with extra feats but none were provided."
            )

        invs = self.atom_type_emb(atom_types)

        if self.n_charge_types is not None:
            charge_feats = self.atom_charge_emb(atom_charges)
            invs = torch.cat((invs, charge_feats), dim=-1)

        if self.n_extra_feats is not None:
            invs = torch.cat((invs, extra_feats), dim=-1)

        if self.n_res_types is not None:
            residue_type_feats = self.res_type_emb(res_types)
            invs = torch.cat((invs, residue_type_feats), dim=-1)

        if self.max_size is not None:
            n_atoms = atom_mask.sum(dim=-1, keepdim=True)
            size_emb = self.size_emb(n_atoms).expand(-1, atom_mask.size(1), -1)
            invs = torch.cat((invs, size_emb), dim=-1)

        if self.self_cond:
            invs = torch.cat((invs, cond_types), dim=-1)

        invs = self.atom_emb(invs)

        edges = self.bond_emb(bond_types)
        if self.self_cond:
            edges = torch.cat((edges, cond_bonds), dim=-1)
            edges = self.bond_proj(edges)

        return invs, edges


class PocketEncoder(torch.nn.Module):

    def __init__(
        self,
        d_equi,
        d_inv,
        d_message,
        n_layers,
        n_attn_heads,
        d_message_ff,
        d_edge,
        n_atom_names,
        n_bond_types,
        n_res_types,
        n_charge_types=7,
        emb_size=64,
        fixed_equi=False,
        eps=1e-6,
    ):
        super().__init__()

        if fixed_equi and d_equi != 1:
            raise ValueError(
                f"If fixed_equi is True d_equi must be 1, got {d_equi}")

        self.d_equi = d_equi
        self.d_inv = d_inv
        self.d_message = d_message
        self.n_layers = n_layers
        self.n_attn_heads = n_attn_heads
        self.d_message_ff = d_message_ff
        self.d_edge = d_edge
        self.emb_size = emb_size
        self.fixed_equi = fixed_equi
        self.eps = eps

        # Embedding and encoding modules
        self.inv_emb = _InvariantEmbedding(d_inv,
                                           n_atom_names,
                                           n_bond_types,
                                           emb_size,
                                           n_charge_types=n_charge_types,
                                           n_res_types=n_res_types)
        self.bond_emb = _PairwiseMessages(d_equi, d_inv, d_inv, d_message,
                                          d_edge, d_message_ff, emb_size)

        if fixed_equi is not None:
            self.coord_emb = torch.nn.Linear(1, d_equi, bias=False)

        # Create a stack of encoder layers
        layer = SemlaLayer(
            d_equi,
            d_inv,
            d_message,
            n_attn_heads,
            d_message_ff,
            d_self_edge_in=d_edge,
            fixed_equi=fixed_equi,
            zero_com=False,
            eps=eps,
        )

        layers = self._get_clones(layer, n_layers)
        self.layers = torch.nn.ModuleList(layers)

    @property
    def hparams(self):
        return {
            "d_equi": self.d_equi,
            "d_inv": self.d_inv,
            "d_message": self.d_message,
            "n_layers": self.n_layers,
            "n_attn_heads": self.n_attn_heads,
            "d_message_ff": self.d_message_ff,
            "d_edge": self.d_edge,
            "emb_size": self.emb_size,
            "fixed_equi": self.fixed_equi,
            "eps": self.eps,
        }

    def forward(self,
                coords,
                atom_names,
                atom_charges,
                res_types,
                bond_types,
                atom_mask=None):
        """Encode the protein pocket into a learnable representation

        Args:
            coords (torch.Tensor): Coordinate tensor, shape [B, N, 3]
            atom_names (torch.Tensor): Atom name indices, shape [B, N]
            atom_charges (torch.Tensor): Atom charge indices, shape [B, N]
            residue_types (torch.Tensor): Residue type indices for each atom, shape [B, N]
            bond_types (torch.Tensor): Bond type indicies for each pair, shape [B, N, N]
            atom_mask (torch.Tensor): Mask for atoms, shape [B, N], 1 for real atom, 0 otherwise

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Equivariant and invariant features, [B, N, 3, d_equi] and [B, N, d_inv]
        """

        atom_mask = torch.ones_like(
            coords[..., 0]) if atom_mask is None else atom_mask
        # adj_matrix = smolF.adj_from_node_mask(atom_mask, self_connect=True)
        adj_matrix = smolF.edges_from_nodes(coords,
                                            k=None,
                                            node_mask=atom_mask,
                                            edge_format="adjacency",
                                            self_connect=True)

        coords = coords.unsqueeze(-1)
        equis = coords if self.fixed_equi else self.coord_emb(coords)

        invs, edges = self.inv_emb(atom_names,
                                   bond_types,
                                   atom_mask,
                                   atom_charges=atom_charges,
                                   res_types=res_types)
        edges = self.bond_emb(equis, invs, equis, invs, edges)
        edges = edges * adj_matrix.unsqueeze(-1)

        for layer in self.layers:
            equis, invs, _, _ = layer(equis, invs, edges, adj_matrix,
                                      atom_mask)

        return equis, invs

    def _get_clones(self, module, n):
        return [copy.deepcopy(module) for _ in range(n)]


class LigandDecoder(torch.nn.Module):
    """Class for generating ligands

    By default no pocket conditioning is used, to allow pocket conditioning set d_pocket_inv to the size of the pocket
    invariant feature vectors. d_equi must be the same for both pocket and ligand.
    """

    def __init__(
        self,
        d_equi,
        d_inv,
        d_message,
        n_layers,
        n_attn_heads,
        d_message_ff,
        d_edge,
        n_atom_types,
        n_bond_types,
        n_charge_types=7,
        emb_size=64,
        d_pocket_inv=None,
        n_interaction_types=None,
        n_extra_atom_feats=None,
        self_cond=False,
        eps=1e-6,
    ):
        super().__init__()

        self.d_equi = d_equi
        self.d_inv = d_inv
        self.d_message = d_message
        self.n_layers = n_layers
        self.n_attn_heads = n_attn_heads
        self.d_message_ff = d_message_ff
        self.d_edge = d_edge
        self.emb_size = emb_size
        self.d_pocket_inv = d_pocket_inv
        self.self_cond = self_cond
        self.eps = eps
        self.interactions = n_interaction_types is not None

        if d_pocket_inv is None and n_interaction_types is not None:
            raise ValueError(
                "Pocket conditioning is required for interaction encoding and prediction."
            )

        coord_proj_feats = 2 if self_cond else 1
        d_cond_edge_in = d_edge if n_interaction_types is not None else None
        d_cond_edge_out = d_edge if n_interaction_types is not None else None

        # *** Embedding and encoding modules ***

        self.inv_emb = _InvariantEmbedding(
            d_inv,
            n_atom_types,
            n_bond_types,
            emb_size,
            n_extra_feats=n_extra_atom_feats,
            self_cond=self_cond,
            max_size=512,
        )
        self.bond_emb = _PairwiseMessages(d_equi, d_inv, d_inv, d_message,
                                          d_edge, d_message_ff, emb_size)
        self.coord_emb = torch.nn.Linear(coord_proj_feats, d_equi, bias=False)

        # *** Layer stack ***

        enc_layer = SemlaLayer(
            d_equi,
            d_inv,
            d_message,
            n_attn_heads,
            d_message_ff,
            d_inv_cond=d_pocket_inv,
            d_self_edge_in=d_edge,
            d_cond_edge_in=d_cond_edge_in,
            zero_com=False,
            eps=eps,
        )
        layers = self._get_clones(enc_layer, n_layers - 1)

        # Create one final layer which also produces edge feature outputs
        dec_layer = SemlaLayer(
            d_equi,
            d_inv,
            d_message,
            n_attn_heads,
            d_message_ff,
            d_inv_cond=d_pocket_inv,
            d_self_edge_in=d_edge,
            d_self_edge_out=d_edge,
            d_cond_edge_in=d_cond_edge_in,
            d_cond_edge_out=d_cond_edge_out,
            zero_com=False,
            eps=eps,
        )
        layers.append(dec_layer)

        self.layers = torch.nn.ModuleList(layers)

        # *** Final norms and projections ***

        self.final_coord_norm = _CoordNorm(d_equi, zero_com=False, eps=eps)
        self.final_inv_norm = torch.nn.LayerNorm(d_inv)
        self.final_bond_norm = torch.nn.LayerNorm(d_edge)

        self.coord_out_proj = torch.nn.Linear(d_equi, 1, bias=False)
        self.atom_type_proj = torch.nn.Linear(d_inv, n_atom_types)
        self.atom_charge_proj = torch.nn.Linear(d_inv, n_charge_types)

        self.bond_refine = BondRefine(d_inv,
                                      d_message,
                                      d_edge,
                                      d_ff=d_inv,
                                      norm_feats=False)
        self.bond_proj = torch.nn.Linear(d_edge, n_bond_types)

        # *** Modules for interactions ***

        if n_interaction_types is not None:
            self.interaction_emb = torch.nn.Embedding(n_interaction_types,
                                                      d_edge)
            self.interaction_refine = _PairwiseMessages(1,
                                                        d_inv,
                                                        d_pocket_inv,
                                                        d_message,
                                                        d_edge,
                                                        d_message_ff,
                                                        d_edge=d_edge,
                                                        include_dists=True)
            self.interaction_proj = torch.nn.Linear(d_edge,
                                                    n_interaction_types)

    @property
    def hparams(self):
        return {
            "d_equi": self.d_equi,
            "d_inv": self.d_inv,
            "d_message": self.d_message,
            "n_layers": self.n_layers,
            "n_attn_heads": self.n_attn_heads,
            "d_message_ff": self.d_message_ff,
            "d_edge": self.d_edge,
            "emb_size": self.emb_size,
            "self_cond": self.self_cond,
            "eps": self.eps,
            "interactions": self.interactions,
        }

    def forward(
        self,
        coords,
        atom_types,
        bond_types,
        atom_mask=None,
        extra_feats=None,
        cond_coords=None,
        cond_atomics=None,
        cond_bonds=None,
        pocket_coords=None,
        pocket_equis=None,
        pocket_invs=None,
        pocket_atom_mask=None,
        interactions=None,
    ):
        """Generate ligand atom types, coords, charges and bonds

        Args:
            coords (torch.Tensor): Coordinate tensor, shape [B, N, 3]
            atom_types (torch.Tensor): Atom name indices, shape [B, N]
            bond_types (torch.Tensor): Bond type indicies for each pair, shape [B, N, N]
            atom_mask (torch.Tensor): Mask for atoms, shape [B, N], 1 for real atom, 0 otherwise
            extra_feats (torch.Tensor): Additional atom features, shape [B, N, n_extra_atom_feats]
            cond_coords (torch.Tensor): Self conditioning coords, shape [B, N, 3]
            cond_atomics (torch.Tensor): Self conditioning atom types, shape [B, N, n_atom_types]
            cond_bonds (torch.Tensor): Self conditioning bond types, shape [B, N, N, n_bond_types]
            pocket_coords (torch.Tensor): Original pocket coords, shape [B, N_p, 3]
            pocket_equis (torch.Tensor): Equivariant encoded pocket features, shape [B, N_p, d_equi]
            pocket_invs (torch.Tensor): Invariant encoded pocket features, shape [B, N_p, d_pocket_inv]
            pocket_atom_mask (torch.Tensor): Mask for pocket atom, shape [B, N_p], 1 for real, 0 otherwise
            interactions (torch.Tensor): Interaction types between pocket and ligand, shape [B, N, N_p]

        Returns:
            (predicted coordinates, atom type logits, bond logits, atom charge logits)
            All torch.Tensor, shapes:
                Coordinates: [B, N, 3],
                Type logits: [B, N, n_atom_types],
                Bond logits: [B, N, N, n_bond_types],
                Charge logits: [B, N, n_charge_types]
        """

        if self.self_cond:
            assert (
                cond_atomics is not None and cond_bonds is not None
            ), "Conditional inputs must be provided if using self conditioning."
        else:
            assert (
                cond_atomics is None and cond_bonds is None
            ), "Conditional inputs were provided but the model was initialised with self_cond as False."

        if self.d_pocket_inv:
            assert (
                pocket_invs is not None and pocket_equis is not None
            ), "Pocket cond inputs must be provided if using pocket conditioning."
        else:
            assert (
                pocket_invs is None and pocket_equis is None
            ), "Pocket cond inputs were provided but the model was not initialised for pocket cond."

        if self.interactions:
            assert (
                interactions is not None
            ), "The model was intialised with interactions but none were provided in forward."
        else:
            assert (
                interactions is None
            ), "Interactions were provided but the model was not initialised for interactions."

        atom_mask = torch.ones_like(
            coords[..., 0]) if atom_mask is None else atom_mask
        adj_matrix = smolF.adj_from_node_mask(atom_mask, self_connect=True)

        # Work out adj matrix between pocket and ligand, if required
        if self.d_pocket_inv is not None:
            cond_adj_matrix = atom_mask.float().unsqueeze(
                2) * pocket_atom_mask.float().unsqueeze(1)
            cond_adj_matrix = smolF.edges_from_two_sets(
                coords,
                pocket_coords,
                k=None,
                node_mask1=atom_mask,
                node_mask2=pocket_atom_mask,
                edge_format="adjacency")
            cond_adj_matrix = cond_adj_matrix.long()
        else:
            cond_adj_matrix = None

        # Embed interaction types, if required
        if self.interactions:
            interaction_feats = self.interaction_emb(interactions)
        else:
            interaction_feats = None

        # Project coords to d_equi
        coords = coords.unsqueeze(-1)
        if self.self_cond:
            coords = torch.cat((coords, cond_coords.unsqueeze(-1)), dim=-1)

        equis = self.coord_emb(coords)

        # Embed invariant features
        invs, edges = self.inv_emb(atom_types,
                                   bond_types,
                                   atom_mask,
                                   cond_types=cond_atomics,
                                   cond_bonds=cond_bonds,
                                   extra_feats=extra_feats)
        edges = self.bond_emb(equis, invs, equis, invs, edges)
        edges = edges * adj_matrix.unsqueeze(-1)

        # Iterate over Semla layers
        for layer in self.layers:
            equis, invs, edge_out, interaction_out = layer(
                equis,
                invs,
                edges,
                adj_matrix,
                atom_mask,
                cond_equis=pocket_equis,
                cond_invs=pocket_invs,
                cond_edges=interaction_feats,
                cond_adj_matrix=cond_adj_matrix,
            )

        # Project coords back to one equivariant feature
        equis = self.final_coord_norm(equis, atom_mask)
        out_coords = self.coord_out_proj(equis).squeeze(-1)

        # Project invariant features to atom and charge logits
        invs_norm = self.final_inv_norm(invs)
        atom_type_logits = self.atom_type_proj(invs_norm)
        charge_logits = self.atom_charge_proj(invs_norm)

        # Pass bonds through refinement layer and project to logits
        edge_norm = self.final_bond_norm(edge_out)
        edge_out = self.bond_refine(out_coords, invs_norm, atom_mask,
                                    edge_norm)
        bond_logits = self.bond_proj(edge_out + edge_out.transpose(1, 2))

        # Pass interactions through refinement layer and project to logits, if required
        if self.interactions:
            interaction_out = self.interaction_refine(
                out_coords.unsqueeze(-1), invs, pocket_invs,
                pocket_coords.unsqueeze(-1), interaction_out)
            interaction_logits = self.interaction_proj(interaction_out)

            return out_coords, atom_type_logits, bond_logits, charge_logits, interaction_logits

        return out_coords, atom_type_logits, bond_logits, charge_logits

    def forward_api(
        self,
        coords: _T,
        atom_types: _T,
        bond_types: _T,
        atom_mask: _T,
        adj_matrix: _T,
        pocket_equis: _T | None,
        pocket_invs: _T | None,
        complex_adj_matrix: _T,
        cond_coords: _T | None,
        cond_atomics: _T | None,
        cond_bonds: _T | None,
        extra_feats: _T,
        interactions: _T | None = None,
    ) -> tuple[_T, _T, _T]:
        if self.self_cond:
            assert (
                cond_atomics is not None and cond_bonds is not None
            ), "Conditional inputs must be provided if using self conditioning."
        else:
            assert (
                cond_atomics is None and cond_bonds is None
            ), "Conditional inputs were provided but the model was initialised with self_cond as False."

        if self.d_pocket_inv:
            assert (
                pocket_invs is not None and pocket_equis is not None
            ), "Pocket cond inputs must be provided if using pocket conditioning."
        else:
            assert (
                pocket_invs is None and pocket_equis is None
            ), "Pocket cond inputs were provided but the model was not initialised for pocket cond."

        if self.interactions:
            assert (
                interactions is not None
            ), "The model was intialised with interactions but none were provided in forward."
        else:
            assert (
                interactions is None
            ), "Interactions were provided but the model was not initialised for interactions."

        # Project coords to d_equi
        if self.self_cond:
            assert cond_coords is not None
            coords = torch.cat(
                (coords.unsqueeze(-1), cond_coords.unsqueeze(-1)), dim=-1)
        else:
            coords = coords.unsqueeze(-1)
        equis = self.coord_emb(coords)

        # Embed interaction types, if required
        if self.interactions:
            interaction_feats = self.interaction_emb(interactions)
        else:
            interaction_feats = None

        # Embed invariant features
        invs, edges = self.inv_emb(atom_types,
                                   bond_types,
                                   atom_mask,
                                   cond_types=cond_atomics,
                                   cond_bonds=cond_bonds,
                                   extra_feats=extra_feats)
        edges = self.bond_emb(equis, invs, equis, invs, edges)
        edges = edges * adj_matrix.unsqueeze(-1)

        # Iterate over Semla layers
        equis, invs = self.run_semla_layers(
            equis,
            invs,
            edges,
            adj_matrix,
            atom_mask,
            pocket_equis,
            pocket_invs,
            interaction_feats,
            complex_adj_matrix,
        )

        # Project coords back to one equivariant feature
        equis = self.final_coord_norm(equis, atom_mask)
        out_coords = self.coord_out_proj(equis).squeeze(-1)
        return out_coords, equis, invs

    def run_semla_layers(
        self,
        equis: _T,
        invs: _T,
        edges: _T,
        adj_matrix: _T,
        atom_mask: _T,
        pocket_equis: _T | None,
        pocket_invs: _T | None,
        interaction_feats: _T | None,
        complex_adj_matrix: _T,
    ):
        """for compile - Iterate over Semla layers"""
        for layer in self.layers:
            equis, invs, _, _ = layer(
                equis,
                invs,
                edges,
                adj_matrix,
                atom_mask,
                cond_equis=pocket_equis,
                cond_invs=pocket_invs,
                cond_edges=interaction_feats,
                cond_adj_matrix=complex_adj_matrix,
            )
        return equis, invs

    def _get_clones(self, module, n):
        return [copy.deepcopy(module) for _ in range(n)]


# *****************************************************************************
# ****************************** Overall Models *******************************
# *****************************************************************************


class LigandGenerator(torch.nn.Module):
    """Main entry point class for generating ligands.

    This class allows both unconditional and pocket-conditioned models to be created. The pocket-conditioned model
    can be created by passing in a PocketEncoder object with the pocket_enc argument, this will automatically setup
    the ligand decoder to use condition attention in addition to self attention. If pocket_enc is None the ligand
    decoder is setup as an unconditional generator.
    """

    def __init__(
        self,
        d_equi,
        d_inv,
        d_message,
        n_layers,
        n_attn_heads,
        d_message_ff,
        d_edge,
        n_atom_types,
        n_bond_types,
        n_charge_types=7,
        emb_size=64,
        n_interaction_types=None,
        n_extra_atom_feats=None,
        self_cond=False,
        pocket_enc=None,
        eps=1e-6,
    ):
        super().__init__()

        duplicate_pocket_equi = False
        if pocket_enc is not None:
            duplicate_pocket_equi = pocket_enc.d_equi == 1
            if not duplicate_pocket_equi and pocket_enc.d_equi != d_equi:
                raise ValueError(
                    "d_equi must be either the same for the pocket and ligand or 1 for the pocket."
                )

        d_pocket_inv = pocket_enc.d_inv if pocket_enc is not None else None

        self.d_equi = d_equi
        self.duplicate_pocket_equi = duplicate_pocket_equi

        self.n_atom_types = n_atom_types
        self.n_bond_types = n_bond_types
        self.n_charge_types = n_charge_types

        ligand_dec = LigandDecoder(
            d_equi,
            d_inv,
            d_message,
            n_layers,
            n_attn_heads,
            d_message_ff,
            d_edge,
            n_atom_types,
            n_bond_types,
            n_charge_types=n_charge_types,
            emb_size=emb_size,
            d_pocket_inv=d_pocket_inv,
            n_interaction_types=n_interaction_types,
            n_extra_atom_feats=n_extra_atom_feats,
            self_cond=self_cond,
            eps=eps,
        )

        self.pocket_enc = pocket_enc
        self.ligand_dec = ligand_dec

    @property
    def hparams(self):
        hparams = self.ligand_dec.hparams
        if self.pocket_enc is not None:
            pocket_hparams = {
                f"pocket-{name}": val
                for name, val in self.pocket_enc.hparams.items()
            }
            hparams = {**hparams, **pocket_hparams}

        return hparams

    def forward(
        self,
        coords,
        atom_types,
        bond_types,
        atom_mask=None,
        extra_feats=None,
        cond_coords=None,
        cond_atomics=None,
        cond_bonds=None,
        pocket_coords=None,
        pocket_atom_names=None,
        pocket_atom_charges=None,
        pocket_res_types=None,
        pocket_bond_types=None,
        pocket_atom_mask=None,
        interactions=None,
    ):

        pocket_equis = None
        pocket_invs = None

        if self.pocket_enc is not None:
            if None in [
                    pocket_coords, pocket_atom_names, pocket_atom_charges,
                    pocket_res_types, pocket_bond_types
            ]:
                raise ValueError(
                    "All pocket inputs must be provided if the model is created with pocket cond."
                )

            pocket_equis, pocket_invs = self.encode(
                pocket_coords,
                pocket_atom_names,
                pocket_atom_charges,
                pocket_res_types,
                pocket_bond_types,
                pocket_atom_mask=pocket_atom_mask,
            )

            if self.duplicate_pocket_equi:
                pocket_equis = pocket_equis.expand(-1, -1, -1, self.d_equi)

        decoder_out = self.decode(
            coords,
            atom_types,
            bond_types,
            atom_mask=atom_mask,
            extra_feats=extra_feats,
            cond_coords=cond_coords,
            cond_atomics=cond_atomics,
            cond_bonds=cond_bonds,
            pocket_coords=pocket_coords,
            pocket_equis=pocket_equis,
            pocket_invs=pocket_invs,
            pocket_atom_mask=pocket_atom_mask,
            interactions=interactions,
        )

        return decoder_out

    def encode(
        self,
        pocket_coords,
        pocket_atom_names,
        pocket_atom_charges,
        pocket_res_types,
        pocket_bond_types,
        pocket_atom_mask=None,
    ):
        if self.pocket_enc is None:
            raise ValueError(
                "Cannot call encode on a model initialised without a pocket encoder."
            )

        pocket_equis, pocket_invs = self.pocket_enc(
            pocket_coords,
            pocket_atom_names,
            pocket_atom_charges,
            pocket_res_types,
            pocket_bond_types,
            atom_mask=pocket_atom_mask,
        )

        return pocket_equis, pocket_invs

    def decode(
        self,
        coords,
        atom_types,
        bond_types,
        atom_mask=None,
        extra_feats=None,
        cond_coords=None,
        cond_atomics=None,
        cond_bonds=None,
        pocket_coords=None,
        pocket_equis=None,
        pocket_invs=None,
        pocket_atom_mask=None,
        interactions=None,
    ):
        if self.pocket_enc is not None and pocket_invs is None:
            raise ValueError(
                "The model was initialised with pocket conditioning but pocket_invs was not provided."
            )
        decoder_out = self.ligand_dec(
            coords,
            atom_types,
            bond_types,
            atom_mask=atom_mask,
            extra_feats=extra_feats,
            cond_coords=cond_coords,
            cond_atomics=cond_atomics,
            cond_bonds=cond_bonds,
            pocket_coords=pocket_coords,
            pocket_equis=pocket_equis,
            pocket_invs=pocket_invs,
            pocket_atom_mask=pocket_atom_mask,
            interactions=interactions,
        )
        return decoder_out
