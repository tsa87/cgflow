from typing import Any

import torch
from torch import Tensor, nn

import cgflow.util.functional as smolF
from cgflow.data.interpolate import ARGeometricInterpolant
from cgflow.models.fm import Integrator
from cgflow.models.pocket import LigandGenerator, PocketEncoder

from .cfg import CGFlowConfig
from .utils import N_ATOM_TYPES, N_BOND_TYPES, N_CHARGE_TYPES, N_EXTRA_ATOM_FEATS, N_RESIDUE_TYPES


class CGFlowInference(nn.Module):
    integrator: Integrator
    pocket_encoder: PocketEncoder | None
    gen: LigandGenerator

    def __init__(
        self,
        config: CGFlowConfig,
        interpolant: ARGeometricInterpolant,
        model_state_dict: Any,
        device: str | torch.device = "cpu",
        compile: bool = False,
    ):
        super().__init__()
        self.cfg: CGFlowConfig = config
        self.device: torch.device = torch.device(device)

        self.interpolant = interpolant

        self.integrator = Integrator(
            self.cfg.num_inference_steps,
            coord_noise_std=0,
            type_strategy="no-change",
            bond_strategy="no-change",
            cat_noise_level=0,
            type_mask_index=None,
            bond_mask_index=None,
        )

        pocket_encoder = PocketEncoder(
            d_equi=1 if self.cfg.fixed_equi else self.cfg.n_coord_sets,
            d_inv=self.cfg.pocket_d_inv,
            d_message=self.cfg.d_message,
            n_layers=self.cfg.pocket_n_layers,
            n_attn_heads=self.cfg.n_attn_heads,
            d_message_ff=self.cfg.d_message_hidden,
            d_edge=self.cfg.d_edge,
            n_atom_names=N_ATOM_TYPES,
            n_bond_types=N_BOND_TYPES,
            n_res_types=N_RESIDUE_TYPES,
            fixed_equi=self.cfg.fixed_equi,
        )

        self.gen = LigandGenerator(
            self.cfg.n_coord_sets,
            self.cfg.d_model,
            self.cfg.d_message,
            self.cfg.n_layers,
            self.cfg.n_attn_heads,
            self.cfg.d_message_hidden,
            self.cfg.d_edge,
            N_ATOM_TYPES,
            N_BOND_TYPES,
            N_CHARGE_TYPES,
            n_extra_atom_feats=N_EXTRA_ATOM_FEATS,
            self_cond=self.cfg.self_condition,
            pocket_enc=pocket_encoder,
        )

        self.load_state_dict(model_state_dict, strict=False)
        self.gen.eval()
        self.gen = self.gen.to(self.device)

        self.pocket_encoder = self.gen.pocket_enc
        self.gen.pocket_enc = None

        self.decoder = self.gen.ligand_dec
        if compile:
            # compile semla layers - balancing the compiling time compared to full compiling of decoding
            self.gen.ligand_dec.run_semla_layers = torch.compile(
                self.gen.ligand_dec.run_semla_layers,
                dynamic=True,
                fullgraph=False,
                mode="reduce-overhead",
            )

    @torch.no_grad()
    def encode_pocket(self, pocket_data: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.pocket_encoder is not None:
            for key in ["coords", "atomics", "charges", "residues", "bonds", "mask"]:
                assert (
                    pocket_data[key] is not None
                ), f"All pocket inputs must be provided if the model is created with pocket cond. ({key} is missed)"
            pocket_equis, pocket_invs = self.pocket_encoder.forward(
                pocket_data["coords"],
                pocket_data["atomics"],
                pocket_data["charges"],
                pocket_data["residues"],
                pocket_data["bonds"],
                pocket_data["mask"],
            )
            if self.gen.duplicate_pocket_equi:
                pocket_equis = pocket_equis.expand(-1, -1, -1, self.gen.d_equi)
            pocket_data["equis"] = pocket_equis
            pocket_data["invs"] = pocket_invs

        return pocket_data

    @torch.no_grad()
    def run(
        self,
        curr: dict[str, Tensor],
        pocket: dict[str, Tensor],
        gen_steps: Tensor,
        curr_step: int,
        is_last: bool,
        return_traj: bool,
    ) -> tuple[list[tuple[Tensor, Tensor]], tuple[Tensor, Tensor]]:
        """model inference for binding pose prediction

        Parameters
        ----------
        curr : dict[str, Tensor]
            current states of molecules
        pocket : dict[str, Tensor]
            pocket information
        gen_steps : Tensor
            what generation step each atom was added in
        curr_step : int
            current generation step
        is_last : bool
            whether the generation is finished
        return_traj : bool
            if True, return all trajectory

        Returns
        -------
        list[tuple[Tensor, Tensor]]
            - trajectory of xt      [num_trajs, num_atoms, 3]
            - trajectory of \\hatx1 [num_trajs, num_atoms, 3]
        tuple[Tensor, Tensor]
            - Z_equi: [num_batches, num_atoms, 3, Ndim]
            - Z_inv: [num_batches, num_atoms, Ndim]
        """
        # Compute the start and end times for each interpolation interval
        curr_time = curr_step * self.cfg.t_per_ar_action
        times = torch.full((curr["coords"].size(0),), curr_time, device=self.device)
        gen_times = torch.clamp(gen_steps * self.cfg.t_per_ar_action, max=self.cfg.max_action_t)

        # If we are at the last step, we need to make sure that the end time is 1.0
        if is_last:
            end_time = 1.0
        else:
            end_time = (curr_step + 1) * self.cfg.t_per_ar_action

        # flow matching input
        curr = curr.copy()
        cond: dict[str, Tensor] | None = None
        trajectory: list[tuple[Tensor, Tensor]] = []

        # flow matching inference
        step_size = 1.0 / self.cfg.num_inference_steps
        num_steps = max(1, round((end_time - curr_time) / step_size))
        hidden_state: tuple[Tensor, Tensor] | None = None
        for t in range(num_steps):
            curr, predicted, cond, hidden_state = self._step(curr, pocket, times, gen_times, step_size, cond)
            times = times + step_size
            if (t == num_steps - 1) or return_traj:
                trajectory.append((curr["coords"].cpu(), predicted["coords"].cpu()))
        assert hidden_state is not None
        return trajectory, hidden_state

    def _step(
        self,
        curr: dict[str, Tensor],
        pocket: dict[str, Tensor],
        times: Tensor,
        gen_times: Tensor,
        step_size: float,
        cond: dict[str, Tensor] | None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor] | None, tuple[Tensor, Tensor]]:
        # Compute relative times for each atom
        # if rel_times == 1, it means the atom should be at ground truth already
        expanded_times = times.unsqueeze(1).expand(-1, gen_times.size(1))
        rel_times = self.interpolant._compute_rel_time(expanded_times, gen_times)
        # Also compute the end times for each atom
        end_times = torch.clamp(gen_times + self.interpolant.max_interp_time, max=1.0)
        assert rel_times.shape == end_times.shape == curr["coords"].shape[:2]

        # calculate self-conditioning
        if self.cfg.self_condition and cond is None:
            cond = self.self_conditioning(curr, pocket, times, rel_times)

        # Predict Vector Field (for CGFlow, we predict the coordinates)
        coords, hidden_feats = self.forward(curr, pocket, times, rel_times, cond)
        predicted = {"coords": coords}

        # We take a step with the predicted coordinates
        # prior is set to None - as it shouldn't matter if we are using the no-change strategy
        prior = None
        updated = self.integrator.step(curr, predicted, prior, times, step_size, end_times)

        # We now update the current state, and self-conditioning
        curr["coords"] = updated["coords"]
        if cond is not None:
            cond["coords"] = predicted["coords"]
        return curr, predicted, cond, hidden_feats

    def self_conditioning(
        self,
        curr: dict[str, Tensor],
        pocket: dict[str, Tensor],
        times: Tensor,
        rel_times: Tensor,
    ) -> dict[str, Tensor]:
        """Calculate self-conditioning"""
        self_cond_atomics = smolF.one_hot_encode_tensor(curr["atomics"], N_ATOM_TYPES)
        self_cond_bonds = smolF.one_hot_encode_tensor(curr["bonds"], N_BOND_TYPES)
        self_cond = {
            "coords": torch.zeros_like(curr["coords"]),
            "atomics": torch.zeros_like(self_cond_atomics),
            "bonds": torch.zeros_like(self_cond_bonds),
        }
        coords, _ = self.forward(curr, pocket, times, rel_times, self_cond)
        self_cond["coords"] = coords
        self_cond["atomics"] = self_cond_atomics
        self_cond["bonds"] = self_cond_bonds
        return self_cond

    def forward(
        self,
        curr: dict[str, Tensor],
        pocket: dict[str, Tensor],
        times: Tensor,
        rel_times: Tensor,
        cond: dict[str, Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Predict molecular coordinates and atom types

        Args:
            curr (dict[str, Tensor]): Batched pointcloud data
            pocket (dict[str, Tensor]): Batched pointcloud data of pocket
            times (torch.Tensor): Interpolation times between 0 and 1, shape [batch_size]
            rel_times (torch.Tensor): Interpolation times between 0 and 1, shape [batch_size, N]
            cond (dict[str, Tensor]): Predictions from previous step, if we are using self conditioning

        Returns:
            (predicted coordinates, atom type logits (unnormalised probabilities))
            Both torch.Tensor, shapes [batch_size, num_atoms, 3] and [batch_size, num atoms, vocab_size]
        """
        # TODO: move it to data processing
        if "adj_matrix" not in curr:
            # curr["adj_matrix"] = smolF.adj_from_node_mask(curr["mask"], self_connect=True)
            curr["adj_matrix"] = smolF.edges_from_nodes(curr["coords"], k=None, node_mask=curr["mask"], edge_format="adjacency", self_connect=True)
        if "complex_adj_matrix" not in curr:
            complex_adj_matrix = curr["mask"].unsqueeze(2) & pocket["mask"].unsqueeze(1)
            curr["complex_adj_matrix"] = complex_adj_matrix.long()

        # Prepare time embedding
        Natom = curr["coords"].shape[1]
        lig_times = times.view(-1, 1, 1).repeat(1, Natom, 1)
        lig_rel_times = rel_times.view(-1, Natom, 1)
        extra_feats = torch.cat([lig_times, lig_rel_times], dim=-1)

        coords, equi, inv = self.decoder.forward_api(
            coords=curr["coords"],
            atom_types=curr["atomics"],
            bond_types=curr["bonds"],
            atom_mask=curr["mask"],
            adj_matrix=curr["adj_matrix"],
            pocket_equis=pocket["equis"],
            pocket_invs=pocket["invs"],
            complex_adj_matrix=curr["complex_adj_matrix"],
            cond_coords=cond["coords"] if cond is not None else None,
            cond_atomics=cond["atomics"] if cond is not None else None,
            cond_bonds=cond["bonds"] if cond is not None else None,
            extra_feats=extra_feats,
        )
        hidden_emb = (equi, inv)
        return coords.clone(), hidden_emb
