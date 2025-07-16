from pathlib import Path
from typing import Self

import torch
from omegaconf import OmegaConf
from torch import Tensor, nn

from cgflow.models.model import PocketEmbedding, PosePrediction
from cgflow.models.utils import Integrator
from cgflow.util.dataclasses import ConditionBatch, LigandBatch, PocketBatch

from .buildutil import build_model

# TODO: check dependencies on lightning


class CGFlowInference(nn.Module):
    def __init__(self, config, model: PosePrediction, device: str | torch.device = "cpu"):
        super().__init__()
        self.cfg = config
        self.device: torch.device = torch.device(device)
        self.model: PosePrediction = model

        # model dimension
        self.d_equi: int = self.model.ligand_dec.config.d_equi
        self.d_inv: int = self.model.ligand_dec.config.d_inv
        self.d_equi_pocket: int = self.model.pocket_enc.config.d_equi
        self.d_inv_pocket: int = self.model.pocket_enc.config.d_inv

        # interpolant config
        self.prior_std: int = config.prior_dist.noise_std
        self.max_ar_steps: int = config.interpolant.max_num_cuts + 1
        self.t_per_ar_action: float = config.interpolant.t_per_ar_action
        self.max_interp_time: float = config.interpolant.max_interp_time

        # integrater
        self.integrator: Integrator = Integrator(noise_std=0)

    @classmethod
    def from_pretrained(cls, pretrained_path: str | Path, device: str | torch.device = "cpu") -> Self:
        # NOTE: load flow-matching module
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
        config = OmegaConf.create(ckpt["hyper_parameters"])
        model: PosePrediction = build_model(config, device)

        # load state dict
        if any(k.startswith("ema_model") for k in ckpt["state_dict"]):
            state_dict = {
                k.replace("ema_model.module.", ""): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("ema_model.")
            }
            state_dict.pop("ema_model.n_averaged")
        else:
            state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        model.load_state_dict(state_dict, strict=True)

        if torch.device(device).type == "cuda":
            model = model.to(torch.bfloat16)

        return cls(config, model, device)

    @torch.no_grad()
    def encode_pocket(self, pocket: PocketBatch) -> PocketEmbedding:
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            embedding = self.model.encode(pocket)
        return embedding

    def prior_like(self, ref_coords: torch.Tensor) -> torch.Tensor:
        # set prior for new fragment atoms
        return torch.randn_like(ref_coords) * self.prior_std

    @torch.no_grad()
    def run(
        self,
        curr: LigandBatch,
        pocket_embedding: PocketEmbedding,
        gen_steps: torch.Tensor,
        curr_step: int,
        num_inference_steps: int = 50,
    ) -> tuple[list[tuple[LigandBatch, LigandBatch]], tuple[Tensor, Tensor]]:
        """model inference for binding pose prediction

        Parameters
        ----------
        curr : LigandBatch
            current states of molecules
        pocket_embedding : PocketBatch
            pocket embeddings
        gen_steps : Tensor
            what generation step each atom was added in
        curr_step : int
            current generation step
        num_inference_steps : int
            number of total inference steps to take

        Returns
        -------
        list[tuple[LigandBatch, LigandBatch]]
            - trajectory of xt
            - trajectory of x1-hat
        tuple[Tensor, Tensor]
            - x_equi: [num_batches, num_atoms, 3, Ndim]
            - x_inv: [num_batches, num_atoms, Ndim]
        """
        # Compute generated times for each atom
        gen_times = gen_steps * self.t_per_ar_action  # [batch_size, num_atoms]
        end_times = self._compute_end_time(gen_times)  # end time for each atom

        # Compute the start and end times
        curr_t = curr_step * self.t_per_ar_action
        if curr_step == self.max_ar_steps - 1:
            end_t = 1.0
        else:
            end_t = curr_t + self.t_per_ar_action

        step_size = 1.0 / num_inference_steps
        num_steps = max(1, round((end_t - curr_t) / step_size))

        # initialize self conditioning
        predicted = curr.copy_with(coords=torch.zeros_like(curr.coords))

        t = curr_t
        hidden_state: tuple[Tensor, Tensor] | None = None
        trajectory: list[tuple[LigandBatch, LigandBatch]] = []
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            for _ in range(num_steps):
                self_condition = predicted
                predicted, hidden_state = self.forward(curr, self_condition, pocket_embedding, gen_times, t)
                curr = self.integrator.step(curr, predicted, t, step_size, end_times)
                trajectory.append((curr.to("cpu"), predicted.to("cpu")))
                t += step_size
        assert hidden_state is not None
        return trajectory, hidden_state

    @torch.no_grad()
    def forward(
        self,
        curr: LigandBatch,
        self_condition: LigandBatch,
        pocket_embedding: PocketEmbedding,
        gen_times: Tensor,
        time: float,
    ) -> tuple[LigandBatch, tuple[Tensor, Tensor]]:
        # calc time condition
        times = torch.full_like(gen_times, time)
        rel_times = self._compute_rel_time(time, gen_times)
        time_cond = torch.stack([times, rel_times, gen_times], dim=-1)  # [batch_size, num_atoms, 3]

        # get condition - time cond and self cond
        condition = ConditionBatch(time_cond, self_condition.coords, curr.mask)

        # predict coords
        pred_coords, x_equi, x_inv = self.model.decode_with_embedding(
            curr, condition, pocket_embedding=pocket_embedding
        )
        return curr.copy_with(coords=pred_coords), (x_equi, x_inv)

    def _compute_rel_time(self, t: float | torch.Tensor, gen_times: torch.Tensor) -> torch.Tensor:
        """
        Compute the relative time of each atom in the interpolated molecule
        t = 1 means the atom is fully interpolated
        t < 0 mean the atom has not been generated yet
        """
        total_time = 1 - gen_times
        if self.max_interp_time:
            total_time = torch.clamp(total_time, max=self.max_interp_time)

        rel_time = (t - gen_times) / total_time
        return torch.clamp(rel_time, max=1)

    def _compute_end_time(self, gen_times: torch.Tensor) -> torch.Tensor:
        """
        Compute the relative time of each atom in the interpolated molecule
        t = 1 means the atom coords xt is same to x1-hat
        t < 1 means the atom coords xt should be interpolated more
        """
        return torch.clamp(gen_times + self.max_interp_time, max=1.0)
