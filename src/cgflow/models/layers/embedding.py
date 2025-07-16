import torch
import torch.nn as nn


class InvariantEmbedding(nn.Module):
    def __init__(
        self,
        d_inv: int,
        d_edge: int,
        n_atom_types: int,
        n_bond_types: int,
        n_charge_types: int,
        n_extra_feats: int = 0,
    ):
        super().__init__()

        self.n_extra_feats: int | None = n_extra_feats

        # atom feat
        self.atom_type_emb = nn.Embedding(n_atom_types, d_inv)
        self.atom_charge_emb = nn.Embedding(n_charge_types, d_inv)

        if n_extra_feats > 0:
            self.atom_proj = nn.Linear(d_inv + n_extra_feats, d_inv)
        else:
            self.atom_proj = nn.Identity()

        # bond feat
        self.bond_emb = nn.Embedding(n_bond_types, d_edge)

    def forward(
        self,
        atom_types: torch.Tensor,
        atom_charges: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
        extra_feats: torch.Tensor | None = None,
    ):
        if self.n_extra_feats is not None:
            assert extra_feats is not None, (
                "The invariant embedding was initialised with extra feats but none were provided."
            )
        else:
            assert extra_feats is None, (
                "The invariant embedding was initialised without extra feats but it is provided."
            )

        # === Node features === #
        inv = self.atom_type_emb(atom_types) + self.atom_charge_emb(atom_charges)
        if extra_feats is not None:
            inv = torch.cat([inv, extra_feats], dim=-1)
        x_inv = self.atom_proj(inv)

        # === Edge features === #
        x_edge = self.bond_emb(adjacency)

        return x_inv, x_edge
