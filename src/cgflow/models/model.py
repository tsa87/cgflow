import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch
from omegaconf import MISSING
from torch import nn

import cgflow.util.misc.functional as smolF
from cgflow.util.data.vocab import NUM_ATOMS, NUM_BOND_TYPES, NUM_CHARGES, NUM_RESIDUES
from cgflow.util.dataclasses import ConditionBatch, LigandBatch, PocketBatch
from cgflow.util.registry import MODEL

from .layers.embedding import InvariantEmbedding
from .layers.nn import CoordNorm, PairwiseMessages
from .layers.semla import SemlaLayer


@dataclass
class ModelConfig:
    _registry_: ClassVar[str] = "model"
    _type_: str


@dataclass
class LigandDecoderConfig(ModelConfig):
    _type_: str = "LigandDecoder"
    d_equi: int = 96
    d_inv: int = 384
    d_edge: int = 128
    n_layers: int = 12
    n_attn_heads: int = 64  # n heads = 6
    d_message: int = 64
    d_message_ff: int = 128
    d_pocket_inv: int = MISSING
    d_pocket_equi: int = MISSING
    time_cond_dim: int = 1  # 3 for CGFlow (time;rel-time;gen-time)
    self_cond: bool = True
    max_num_atoms: int | None = 60  # max atom numbers. larger sizes will be clamped to this value
    eps: float = 1e-3


@dataclass
class PocketEncoderConfig(ModelConfig):
    _type_: str = "PocketEncoder"
    d_equi: int = 96
    d_inv: int = 384
    d_edge: int = 128
    n_layers: int = 4
    n_attn_heads: int = 64  # n heads = 6
    d_message: int = 64
    d_message_ff: int = 128
    fixed_equi: bool = False  # If true, pocket equivariant features are not updated
    eps: float = 1e-3


@dataclass
class CGFlowEmbedding:
    coords: torch.Tensor  # [B, N, 3]
    equi: torch.Tensor  # [B, N, 3, d_equi]
    inv: torch.Tensor  # [B, N, d_inv]
    mask: torch.Tensor  # [B, N]


@dataclass
class PocketEmbedding(CGFlowEmbedding): ...


@dataclass
class LigandEmbedding(CGFlowEmbedding): ...


# *** Model Interface ***
@MODEL.register(name="BasePocketEncoder", config=PocketEncoderConfig)
class PocketEncoder(nn.Module, ABC):
    def __init__(self, config: PocketEncoderConfig):
        super().__init__()
        self.config: PocketEncoderConfig = config

    @abstractmethod
    def forward(self, pocket: PocketBatch) -> PocketEmbedding:
        """returns equivariant and invariant features of the pocket"""


# *** Model Interface ***
@MODEL.register(name="BaseLigandDecoder", config=LigandDecoderConfig)
class LigandDecoder(nn.Module, ABC):
    def __init__(self, config: LigandDecoderConfig):
        super().__init__()
        self.config: LigandDecoderConfig = config

    @abstractmethod
    def forward(
        self,
        ligand: LigandBatch,
        condition: ConditionBatch,
        pocket_embedding: PocketEmbedding,
        interaction_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate atom coordiates"""


class PosePrediction(nn.Module):
    """Main entry point class for pose prediction."""

    def __init__(self, pocket_enc: PocketEncoder, ligand_dec: LigandDecoder):
        super().__init__()
        self.pocket_enc: PocketEncoder = pocket_enc
        self.ligand_dec: LigandDecoder = ligand_dec
        # cehck dimension
        assert pocket_enc.d_inv == ligand_dec.d_pocket_inv

    def forward(
        self,
        ligand: LigandBatch | None = None,
        condition: ConditionBatch | None = None,
        pocket: PocketBatch | None = None,
        pocket_embedding: PocketEmbedding | None = None,
        interaction_mask: torch.Tensor | None = None,
        mode: str = "decode",  # [encode, decode]
    ) -> torch.Tensor | PocketEmbedding:
        assert mode in ["encode", "decode"], "Mode must be either 'encode' or 'decode'."

        if mode == "encode":
            assert pocket is not None, "Pocket must be provided for encoding."
            return self.pocket_enc.forward(pocket)
        else:
            assert ligand is not None and condition is not None, "Ligand and condition must be provided for decoding."
            # if pocket_embedding is not provided, encode the pocket
            if pocket_embedding is None:
                assert pocket is not None, "Pocket must be provided for encoding. (pocket_embedding is not providen)"
                pocket_embedding = self.pocket_enc(pocket)
            return self.ligand_dec(ligand, condition, pocket_embedding, interaction_mask)

    def encode(self, pocket: PocketBatch) -> PocketEmbedding:
        return self.pocket_enc.forward(pocket)

    def decode(
        self,
        ligand: LigandBatch,
        condition: ConditionBatch,
        pocket: PocketBatch | None = None,
        pocket_embedding: PocketEmbedding | None = None,
        interaction_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # if pocket_embedding is not provided, encode the pocket
        if pocket_embedding is None:
            assert pocket is not None, "Pocket must be provided for encoding. (pocket_embedding is not providen)"
            pocket_embedding = self.pocket_enc.forward(pocket)
        return self.ligand_dec.forward(ligand, condition, pocket_embedding, interaction_mask)

    def decode_with_embedding(
        self,
        ligand: LigandBatch,
        condition: ConditionBatch,
        pocket: PocketBatch | None = None,
        pocket_embedding: PocketEmbedding | None = None,
        interaction_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # if pocket_embedding is not provided, encode the pocket
        if pocket_embedding is None:
            assert pocket is not None, "Pocket must be provided for encoding. (pocket_embedding is not providen)"
            pocket_embedding = self.pocket_enc.forward(pocket)
        return self.ligand_dec.run(ligand, condition, pocket_embedding, interaction_mask)


@MODEL.register(config=PocketEncoderConfig)
class PocketEncoderV3(PocketEncoder):
    def __init__(
        self,
        config: PocketEncoderConfig,
        n_atom_types: int = NUM_ATOMS,
        n_bond_types: int = NUM_BOND_TYPES,
        n_res_types: int = NUM_RESIDUES,
        n_charge_types: int = NUM_CHARGES,
    ):
        super().__init__(config)

        if config.fixed_equi and config.d_equi != 1:
            raise ValueError(f"If fixed_equi is True d_equi must be 1, got {config.d_equi}")
        self.config: PocketEncoderConfig = config
        self.fixed_equi: bool = config.fixed_equi
        self.eps: float = config.eps

        self.d_inv: int = config.d_inv
        self.d_equi: int = config.d_equi

        # Embedding and encoding modules
        res_emb_size = config.d_inv
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
        # Project coords to d_equi
        x_equi = self.coord_emb(pocket.coords.unsqueeze(-1))  # [B, L, 3] -> [B, L, 3, Dequi]

        # get inv features
        res_emb = self.res_type_emb(pocket.residues)
        x_inv, x_edge = self.inv_emb(pocket.atoms, pocket.charges, pocket.adjacency, pocket.mask, res_emb)

        # get edge embedding with pairwise message passing
        adj_mask = smolF.get_adj_mask(pocket.coords, node_mask=pocket.mask, k=None, self_connect=True)  # [B, L, L]
        x_edge = self.bond_emb(x_equi, x_inv, x_equi, x_inv, x_edge)
        x_edge = x_edge * adj_mask.unsqueeze(-1)  # [B, L, L, Dedge]

        for layer in self.layers:
            x_equi, x_inv = layer.forward(x_equi, x_inv, x_edge, pocket.mask, adj_mask)

        return PocketEmbedding(pocket.coords, x_equi, x_inv, pocket.mask)

    def _get_clones(self, module, n):
        return [copy.deepcopy(module) for _ in range(n)]


@MODEL.register(config=LigandDecoderConfig)
class LigandDecoderV3(LigandDecoder):
    """Class for generating ligands

    By default no pocket conditioning is used, to allow pocket conditioning set d_pocket_inv to the size of the pocket
    invariant feature vectors. d_equi must be the same for both pocket and ligand.
    """

    def __init__(
        self,
        config: LigandDecoderConfig,
        n_atom_types: int = NUM_ATOMS,
        n_bond_types: int = NUM_BOND_TYPES,
        n_charge_types: int = NUM_CHARGES,
    ):
        super().__init__(config)
        self.config: LigandDecoderConfig = config
        self.time_cond: bool = config.time_cond_dim > 0
        self.self_cond: bool = config.self_cond
        self.max_num_atoms: int | None = config.max_num_atoms
        self.eps: float = config.eps

        self.d_inv: int = config.d_inv
        self.d_equi: int = config.d_equi
        self.d_pocket_inv: int = config.d_pocket_inv
        self.d_pocket_equi: int = config.d_equi

        # *** Embedding and encoding modules ***
        extra_dim = 128
        self.attachment_emb = nn.Parameter(torch.randn(extra_dim))
        if config.max_num_atoms is not None:
            self.size_emb = nn.Embedding(config.max_num_atoms, extra_dim)
        if self.time_cond:
            self.time_emb = nn.Linear(config.time_cond_dim, extra_dim)
        self.inv_emb = InvariantEmbedding(
            config.d_inv,
            config.d_edge,
            n_atom_types,
            n_bond_types,
            n_charge_types,
            n_extra_feats=extra_dim,
        )
        self.bond_emb = PairwiseMessages(
            d_equi=config.d_equi,
            d_inv=config.d_inv,
            d_edge=config.d_edge,
            d_out=config.d_edge,
            d_message=config.d_message,
            d_ff=config.d_message_ff,
        )
        coord_proj_feats = 2 if config.self_cond else 1
        # absolute coordinate embedding
        self.coord_emb = nn.Linear(coord_proj_feats, config.d_equi, bias=False)
        # distance embedding
        self.distance_emb = nn.Linear(1, config.d_edge, bias=False)

        # *** Layer stack ***
        enc_layer = SemlaLayer(
            config.d_equi,
            config.d_inv,
            config.d_edge,
            config.d_message,
            config.n_attn_heads,
            config.d_message_ff,
            d_inv_cond=config.d_pocket_inv,
            d_equi_cond=config.d_pocket_equi,
            d_edge_cond=config.d_edge,
            use_condition=True,
            zero_com=False,
            eps=config.eps,
        )
        layers = self._get_clones(enc_layer, config.n_layers)
        self.layers: list[SemlaLayer] = nn.ModuleList(layers)

        # *** Final norms and projections ***
        self.final_coord_norm = CoordNorm(config.d_equi, zero_com=False, eps=config.eps)
        self.coord_out_proj = nn.Linear(config.d_equi, 1, bias=False)

    def forward(
        self,
        ligand: LigandBatch,
        condition: ConditionBatch,
        pocket_embedding: PocketEmbedding,
        interaction_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate ligand atom types, coords, charges and bonds

        Args:
            ligand (LigandBatch): Batch of ligand data
            condition (Conditionbatch): Batch of Conditions (time condition; self condition)
            pocket_equis (torch.Tensor): Equivariant encoded pocket features, shape [B, N_p, d_pocket_equi]
            pocket_invs (torch.Tensor): Invariant encoded pocket features, shape [B, N_p, d_pocket_inv]
            pocket_mask (torch.Tensor): Mask of pocket features, shape [B, N_p]

        Returns:
            predicted coordinates: torch.Tensor ([B, N, 3])
        """
        return self.run(ligand, condition, pocket_embedding, interaction_mask)[0]

    def run(
        self,
        ligand: LigandBatch,
        condition: ConditionBatch,
        pocket_embedding: PocketEmbedding,
        interaction_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate ligand atom types, coords, charges and bonds

        Args:
            ligand (LigandBatch): Batch of ligand data
            condition (Conditionbatch): Batch of Conditions (time condition; self condition)
            pocket_equis (torch.Tensor): Equivariant encoded pocket features, shape [B, N_p, d_pocket_equi]
            pocket_invs (torch.Tensor): Invariant encoded pocket features, shape [B, N_p, d_pocket_inv]
            pocket_mask (torch.Tensor): Mask of pocket features, shape [B, N_p]

        Returns:
            predicted coordinates: torch.Tensor ([B, N, 3])
            equivariant features: torch.Tensor ([B, N, 3, d_equi])
            invariant features: torch.Tensor ([B, N, d_inv])

        """

        # Project coords to d_equi
        coords = ligand.coords.unsqueeze(-1)
        if self.self_cond:
            coords = torch.cat((coords, condition.self_cond.unsqueeze(-1)), dim=-1)
        x_equi = self.coord_emb.forward(coords)  # [B, L, 3, Dequi]

        # Embed invariant features
        extra_emb = ligand.attachments.unsqueeze(-1) * self.attachment_emb.view(1, 1, -1)  # [B, L, d_extra]
        if self.max_num_atoms is not None:
            n_atoms = ligand.mask.sum(dim=-1).clamp(max=self.max_num_atoms - 1)  # [B,]
            size_emb = self.size_emb(n_atoms)  # [B, d_extra]
            extra_emb = extra_emb + size_emb.unsqueeze(1)  # [B, L, d_extra]
        if self.time_cond:
            time_emb = self.time_emb(condition.time_cond)  # [B, L, d_extra]
            extra_emb = extra_emb + time_emb  # [B, L, d_extra]
        x_inv, x_edge = self.inv_emb(ligand.atoms, ligand.charges, ligand.adjacency, ligand.mask, extra_emb)

        # get edge embedding with pairwise message passing
        adj_mask = smolF.get_adj_mask(ligand.coords, ligand.mask, self_connect=True)
        x_edge = self.bond_emb(x_equi, x_inv, x_equi, x_inv, x_edge)
        x_edge = x_edge * adj_mask.unsqueeze(-1)

        # Message passing btw protein-ligand
        cond_equi, cond_inv = (pocket_embedding.equi, pocket_embedding.inv)

        # match the shape: [B, N-poc, 3, 1] -> [B, N-poc, 3, d_equi]
        if cond_equi.shape[-1] == 1:
            cond_equi = cond_equi.expand(-1, -1, -1, self.ligand_dec.d_equi)

        # Project protein-ligand edge to d_equi
        pocket_ligand_dist = ligand.coords.unsqueeze(-2) - pocket_embedding.coords.unsqueeze(-3)  # [B, L, P, 3]
        pocket_ligand_dist = pocket_ligand_dist.norm(dim=-1, keepdim=True)  # [B, L, P, 1]
        cond_edge = self.distance_emb(pocket_ligand_dist)  # [B, L, P, d_edge]

        # use interaction_mask as the attention bias [B, L, P]
        if interaction_mask is None:
            # if interaction_mask is not provided, create full interaction mask
            interaction_mask = ligand.mask.unsqueeze(-1) & pocket_embedding.mask.unsqueeze(-2)

        # Iterate over Semla layers
        for layer in self.layers:
            x_equi, x_inv = layer.forward(
                x_equi,
                x_inv,
                x_edge,
                ligand.mask,
                adj_mask,
                cond_equis=cond_equi,
                cond_invs=cond_inv,
                cond_edges=cond_edge,
                cond_adj_mask=interaction_mask,
            )

        # Project coords back to one equivariant feature
        x_equi_norm = self.final_coord_norm(x_equi, ligand.mask)
        out_coords = self.coord_out_proj(x_equi_norm).squeeze(-1)
        return out_coords, x_equi, x_inv

    def _get_clones(self, module, n):
        return [copy.deepcopy(module) for _ in range(n)]
