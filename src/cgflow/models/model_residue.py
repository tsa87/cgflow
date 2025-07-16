import copy

import torch
from torch import nn

import cgflow.util.misc.functional as smolF
from cgflow.util.data.vocab import NUM_ATOMS, NUM_BOND_TYPES, NUM_CHARGES, NUM_RESIDUES
from cgflow.util.dataclasses import PocketBatch
from cgflow.util.registry import MODEL

from .layers.embedding import InvariantEmbedding
from .layers.nn import PairwiseMessages
from .layers.semla import SemlaLayer
from .model import PocketEmbedding, PocketEncoder, PocketEncoderConfig

_T = torch.Tensor


@MODEL.register(config=PocketEncoderConfig)
class ResidueEncoder(PocketEncoder):
    def __init__(
        self,
        config: PocketEncoderConfig,
        n_atom_types: int = NUM_ATOMS,
        n_bond_types: int = NUM_BOND_TYPES,
        n_res_types: int = NUM_RESIDUES,
        n_charge_types: int = NUM_CHARGES,
        n_residue_types: int = NUM_RESIDUES,
    ):
        super().__init__(config)

        if config.fixed_equi and config.d_equi != 1:
            raise ValueError(f"If fixed_equi is True d_equi must be 1, got {config.d_equi}")
        self.config: PocketEncoderConfig = config

        self.d_inv: int = config.d_inv
        self.d_equi: int = config.d_equi
        self.d_edge: int = config.d_edge

        self.d_message: int = config.d_message
        self.n_layers: int = config.n_layers
        self.n_attn_heads: int = config.n_attn_heads
        self.d_message_ff: int = config.d_message_ff
        self.fixed_equi: bool = config.fixed_equi
        self.eps: float = config.eps

        # Embedding and encoding modules
        res_emb_size = 128
        self.res_type_emb = nn.Embedding(n_res_types, res_emb_size)
        self.inv_emb = InvariantEmbedding(
            config.d_inv,
            config.d_edge,
            n_atom_types,
            n_bond_types,
            n_charge_types,
            n_extra_feats=res_emb_size,
        )
        self.bond_emb = PairwiseMessages(
            d_equi=config.d_equi,
            d_inv=config.d_inv,
            d_edge=config.d_edge,
            d_out=config.d_edge,
            d_message=config.d_message,
            d_ff=config.d_message_ff,
        )

        if not config.fixed_equi:
            self.coord_emb = nn.Linear(1, config.d_equi, bias=False)
        else:
            self.coord_emb = nn.Identity()

        # Create a stack of encoder layers
        layer = SemlaLayer(
            config.d_equi,
            config.d_inv,
            config.d_edge,
            config.d_message,
            config.n_attn_heads,
            config.d_message_ff,
            fixed_equi=config.fixed_equi,
            zero_com=False,
            eps=config.eps,
        )

        layers = self._get_clones(layer, config.n_layers)
        self.layers: list[SemlaLayer] = nn.ModuleList(layers)

    def forward(self, pocket: PocketBatch) -> PocketEmbedding:
        """Encode the protein pocket into a learnable representation
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Equivariant and invariant features and residue mask and c_alpha coordinates, [B, N_res, 3, d_equi] and [B, N_res, d_inv] and [B, N_res] and [B, N_res, 3]
        """

        # atom14 representation: [B, N] -> [B, L, 14]
        res_level_data = smolF.convert_to_residue_format(
            pocket.atoms,
            pocket.charges,
            pocket.residues,
            pocket.residue_ids,
            pocket.adjacency,
            pocket.coords,
            pocket.mask,
        )
        atom_types = res_level_data["atoms"]
        atom_charges = res_level_data["charges"]
        res_types = res_level_data["residues"]
        adjacency = res_level_data["adjacency"]
        coords = res_level_data["coords"]
        atom_mask = res_level_data["mask"]
        batch_ids = res_level_data["batch_ids"]

        c_alpha_coords = coords[:, 1, :]  # [B*N_res, 3]

        adj_mask = atom_mask.unsqueeze(-1) & atom_mask.unsqueeze(-2)

        x_equi = self.coord_emb(coords.unsqueeze(-1))

        res_emb = self.res_type_emb(res_types)
        x_inv, x_edge = self.inv_emb.forward(atom_types, atom_charges, adjacency, atom_mask, res_emb)

        x_edge = self.bond_emb.forward(x_equi, x_inv, x_equi, x_inv, x_edge)
        x_edge = x_edge * adj_mask.unsqueeze(-1)

        for layer in self.layers:
            x_equi, x_inv = layer(x_equi, x_inv, x_edge, atom_mask, adj_mask)

        # aggregate across residues by mean pooling across residues_group_ids
        equis_agg = smolF.aggregate_atoms_by_mean(x_equi, atom_mask)  # [total_N_res, 3, d_equi]
        invs_agg = smolF.aggregate_atoms_by_mean(x_inv, atom_mask)  # [total_N_res, d_inv]
        equis_agg, residue_mask = smolF.pad_by_batch_id(equis_agg, batch_ids)  # [B, N_res, 3, d_equi] and [B, N_res]
        invs_agg, _ = smolF.pad_by_batch_id(invs_agg, batch_ids)  # [B, N_res, d_inv] and [B, N_res]
        c_alpha_coords, _ = smolF.pad_by_batch_id(c_alpha_coords, batch_ids)  # [B, N_res, 3] and [B, N_res]
        return PocketEmbedding(c_alpha_coords, equis_agg, invs_agg, residue_mask)

    def _get_clones(self, module, n):
        return [copy.deepcopy(module) for _ in range(n)]
