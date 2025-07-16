from dataclasses import dataclass
from typing import Any, ClassVar

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from rdkit import Chem
from torch.optim.lr_scheduler import LinearLR
from torchmetrics import MetricCollection

import cgflow.util.metrics as Metrics
from cgflow.data.interpolate import ARGeometricInterpolant
from cgflow.models.model import PocketEmbedding, PosePrediction
from cgflow.models.utils import Integrator, mol_from_tensor, mols_from_batch
from cgflow.util.dataclasses import ConditionBatch, LigandBatch, LigandTensor, PocketBatch
from cgflow.util.misc.functional import pad_tensors
from cgflow.util.registry import CFM, merge_base_config

_T = torch.Tensor


@dataclass
class CFMConfig:
    _registry_: ClassVar[str] = "cfm"
    _type_: str


@dataclass
class MolecularCFMConfig(CFMConfig):
    _type_: str = "MolecularCFM"

    # cfm config
    self_condition: bool = True

    # model training
    use_ema: bool = True
    dist_loss_weight: float = 0.0  # TODO: add dist loss
    loss_fn: str = "hubor"
    lr: float = 0.0003
    lr_schedule: str = "constant"
    warmup_steps: int | None = 10000

    # sampling (for validation)
    inference_noise_std: float = 0.0
    sampling_steps: int = 50
    sampling_strategy: str = "linear"
    # metric (for validation)
    use_energy_metric: bool = True
    use_complex_metric: bool = False

    debug: bool = False  # if True, return trajectory


@CFM.register()  # here we do not use default config
class MolecularCFM(L.LightningModule):
    config_class: type[CFMConfig] = MolecularCFMConfig

    def __init__(self, config, model: PosePrediction, **kwargs):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(config))
        self.global_config = config
        self.config: MolecularCFMConfig = merge_base_config(config.cfm, config_class=self.config_class)

        self.model: PosePrediction = model

        # hparam
        self.self_condition: bool = self.config.self_condition

        self.use_ema: bool = self.config.use_ema
        if self.use_ema:
            avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
            self.ema_model: PosePrediction = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=avg_fn)

        self.lr: float = self.config.lr
        self.lr_schedule = self.config.lr_schedule
        self.warmup_steps = self.config.warmup_steps

        self.loss_fn = self.config.loss_fn
        self.dist_loss_weight: float = self.config.dist_loss_weight

        # sampling for valdation
        self.integrator: Integrator = Integrator(self.config.inference_noise_std)
        self.sampling_steps: int = self.config.sampling_steps
        self.sampling_strategy: str = self.config.sampling_strategy

        # set metrics
        # Conformer metrics for molecules that are not changing
        metrics = {
            "align-rmsd": Metrics.MolecularPairRMSD(),
            "rmsd": Metrics.MolecularPairRMSD(align=False),
            "centroid-rmsd": Metrics.CentroidRMSD(),
        }
        self.conf_metrics = MetricCollection(metrics, compute_groups=False)

        # Energy metrics
        if self.config.use_energy_metric:
            metrics = {
                "energy-validity": Metrics.EnergyValidity(),
                "opt-energy-validity": Metrics.EnergyValidity(optimise=True),
                "energy": Metrics.AverageEnergy(),
                "energy-per-atom": Metrics.AverageEnergy(per_atom=True),
                "strain": Metrics.AverageStrainEnergy(),
                "strain-per-atom": Metrics.AverageStrainEnergy(per_atom=True),
                "opt-rmsd": Metrics.AverageOptRmsd(),
            }
            self.energy_metrics = MetricCollection(metrics, compute_groups=False)
        else:
            self.energy_metrics = None

        if self.config.use_complex_metric:
            metrics = {
                "clash": Metrics.Clash(),
                "interactions": Metrics.Interactions(),
            }
            self.complex_metrics = MetricCollection(metrics, compute_groups=False)
        else:
            self.complex_metrics = None

        self._init_params()

    def configure_optimizers(self):
        # optimizer
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, amsgrad=True, foreach=True, weight_decay=0.0)

        # lr scheduler
        if self.lr_schedule == "constant":
            warmup_steps = 0 if self.warmup_steps is None else self.warmup_steps
            scheduler = LinearLR(opt, start_factor=1e-2, total_iters=warmup_steps)
        else:
            raise ValueError(f"LR schedule {self.lr_schedule} is not supported.")

        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def forward(
        self,
        ligand: LigandBatch | None = None,
        pocket: PocketBatch | None = None,
        condition: ConditionBatch | None = None,
        pocket_embedding: PocketEmbedding | None = None,
        training: bool = False,
        mode: str = "decode",
    ) -> PocketEmbedding | LigandBatch:
        """Predict molecular coordinates
        TODO: add docs
        """
        # Whether to use the EMA version of the model or not
        if not training and self.use_ema:
            model = self.ema_model
        else:
            model = self.model

        if mode == "encode":
            # return PocketEmbedding
            assert ligand is None, "Ligand must not be provided for encoding mode."
            assert pocket is not None, "Pocket must be provided for encoding mode."
            assert condition is None, "Condition must not be provided for encoding mode."
            assert pocket_embedding is None, "Pocket embedding is not used in encoding mode."
            return model(pocket=pocket, mode="encode")
        else:
            # return predicted LigandBatch
            assert ligand is not None, "Ligand must be provided for decoding."
            assert condition is not None, "Condition must be provided for decoding."
            if pocket_embedding is None:
                assert pocket is not None, "Pocket must be provided for decoding without pocket_embedding."
            # predict coordinates
            pred_coords = model(ligand=ligand, condition=condition, pocket_embedding=pocket_embedding, mode="decode")
            return ligand.copy_with(coords=pred_coords)

    def training_step(self, batch, b_idx: int) -> _T:
        # get pocket embedding
        pocket = PocketBatch(**batch["pocket_mol"])
        pocket_embedding = self(pocket=pocket, training=True, mode="encode")

        # true label
        true_ligand = LigandBatch(**batch["ligand_mol"])
        # copy with interpolated coordinates
        interp_ligand = true_ligand.copy_with(coords=batch["xt"])

        # get condition
        # time condition
        times = batch["time"]  # [B,]
        # NOTE: we use 3 to match AR one
        time_cond = times.view(-1, 1, 1).expand(-1, interp_ligand.length, 3)  # [B, L, 3]
        # self condition
        self_cond = torch.zeros_like(interp_ligand.coords)  # [B,L,3]
        condition = ConditionBatch(time_cond, self_cond, interp_ligand.mask)

        # If training with self conditioning, half the time generate a conditional batch by setting cond to zeros
        if self.self_condition:
            if torch.rand(1).item() > 0.5:
                with torch.no_grad():
                    self_conditioning = self(
                        ligand=interp_ligand,
                        condition=condition,
                        pocket_embedding=pocket_embedding,
                        training=True,
                        mode="decode",
                    )
                condition.self_cond = self_conditioning.coords

        # predict coordinates (x1-hat)
        pred_ligand = self(
            ligand=interp_ligand, condition=condition, pocket_embedding=pocket_embedding, training=True, mode="decode"
        )

        loss, losses = self._loss(true_ligand, pred_ligand)
        if not loss.isfinite():
            print("---------------------------")
            print(f"[Training] Skipping batch {b_idx}")
            print("true ligand pos:", true_ligand.coords.isfinite().any().item())
            print("pred ligand pos:", pred_ligand.coords.isfinite().any().item())
            print("interp ligand pos:", interp_ligand.coords.isfinite().any().item())
            print("coord loss:", losses["coord-loss"])
            print("dist loss:", losses.get("dist-loss", None))
            print("---------------------------")
            if losses["coord-loss"].isfinite():
                loss = losses["coord-loss"]
            else:
                loss = (true_ligand.coords * 0.0).mean()
            return loss

        for k, v in losses.items():
            self.log(f"train/{k}", v, on_step=True, logger=True)
        self.log("train/loss", loss, prog_bar=True, on_step=True, logger=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.use_ema:
            self.ema_model.update_parameters(self.model)

    def validation_step(self, batch, b_idx):
        label = LigandBatch(**batch["ligand_mol"])
        recon = self.reconstruct(batch)
        label_rdmols = mols_from_batch(label)
        recon_rdmols = mols_from_batch(recon)
        self.conf_metrics.update(recon_rdmols, label_rdmols)
        if self.energy_metrics:
            self.energy_metrics.update(recon_rdmols)
        if self.complex_metrics:
            self.complex_metrics.update(recon_rdmols, batch["pocket_raw"])

    def on_validation_epoch_end(self):
        conf_metrics_results = self.conf_metrics.compute()
        energy_metrics_results = self.energy_metrics.compute() if self.energy_metrics else {}
        complex_metrics_results = self.complex_metrics.compute() if self.complex_metrics else {}
        metrics = {
            **energy_metrics_results,
            **complex_metrics_results,
            **conf_metrics_results,
        }
        for k, v in metrics.items():
            progbar = True if k == "rmsd" else False
            self.log(f"val/{k}", v, on_epoch=True, logger=True, prog_bar=progbar, sync_dist=True)

        self.conf_metrics.reset()
        if self.energy_metrics:
            self.energy_metrics.reset()
        if self.complex_metrics:
            self.complex_metrics.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def predict_step(self, batch, batch_idx) -> list[Chem.Mol]:
        recon = self.reconstruct(batch)
        return mols_from_batch(recon)

    def _loss(self, label: LigandBatch, pred: LigandBatch):
        assert label.coords.shape == pred.coords.shape, (
            "Predicted and true positions must have the same shape before masking."
        )
        pred_pos = pred.coords[pred.mask]
        true_pos = label.coords[label.mask]
        assert pred_pos.shape == true_pos.shape, "Predicted and true positions must have the same shape after masking."

        # TODO: mean for batch-level? then, how can we do without for iteration?
        # TODO: current code occurs NaN... why?

        # HACK: I don't know why NaN error happes...
        # to prevent NaN error, remove infinite values
        mask = pred_pos.isfinite().all(-1)
        if mask.all():
            pass
        elif mask.any():
            pred_pos, true_pos = pred_pos[mask], true_pos[mask]
        else:
            print("all pred is infinite.")
            return (true_pos * 0.0).mean(), {}

        coord_loss = self.__calc_loss(pred_pos, true_pos)
        losses = {"coord-loss": coord_loss}
        loss = losses["coord-loss"]  # default loss is coord loss

        if self.dist_loss_weight > 0:
            distance_loss = self.calc_distance_loss(pred.coords, label.coords, label.adjacency, n_hops=3)
            losses["dist-loss"] = distance_loss
            loss = loss + self.dist_loss_weight * distance_loss
        return loss, losses

    def calc_distance_loss(self, input: _T, target: _T, adjacency: _T, n_hops: int = 3) -> _T:
        """calculate distance loss between predicted and true coordinates

        Parameters
        ----------
        input : torch.Tensor
            input coordinates (batch_size, num_nodes, 3)
        target : torch.Tensor
            target coordinates (batch_size, num_nodes, 3)
        adjacency : torch.Tensor
            adjacency matrix (batch_size, num_nodes, num_nodes)
        n_hops : int
            number of hops to consider for distance calculation, default is 3 (1: bond, 2: angle, 3: dihedral)

        Returns
        -------
        torch.Tensor
            distance loss
        """
        assert n_hops >= 1, "n_hops must be at least 1"
        with torch.no_grad():
            adj_float = adjacency.float()
            if n_hops > 1:
                adj_n_power = torch.matrix_power(adj_float, n_hops)
                adj_n_hop = adj_n_power
            else:
                adj_n_hop = adjacency

            edge_indices = torch.nonzero(adj_n_hop, as_tuple=False)
            if edge_indices.numel() == 0:
                return torch.tensor(0.0, device=input.device)

        batch_idcs, u, v = torch.split(edge_indices, 1, dim=-1)
        pred_dists = (input[batch_idcs, u] - input[batch_idcs, v]).norm(dim=-1)
        with torch.no_grad():
            true_dists = (target[batch_idcs, u] - target[batch_idcs, v]).norm(dim=-1)

        mask = pred_dists.isfinite()
        if mask.all():
            pass
        elif mask.any():
            pred_dists, true_dists = pred_dists[mask], true_dists[mask]
        else:
            print("some pred is infinite.")
            return torch.tensor(0.0, device=input.device)
        return self.__calc_loss(pred_dists, true_dists)

    def __calc_loss(self, input: _T, target: _T, loss_fn: str | None = None, reduction: str = "mean", **kwargs) -> _T:
        loss_fn = self.loss_fn if loss_fn is None else loss_fn
        if loss_fn == "huber":
            return F.huber_loss(input, target, reduction=reduction, **kwargs)
        elif loss_fn == "mse":
            return F.mse_loss(input, target, reduction=reduction, **kwargs)
        elif loss_fn == "mae":
            return F.l1_loss(input, target, reduction=reduction, **kwargs)
        else:
            raise NotImplementedError(f"Loss function '{loss_fn}' is not implemented.")

    @torch.no_grad()
    def reconstruct(self, batch) -> LigandBatch:
        trajectory = self.run_reconstruct(batch, self.sampling_steps, self.sampling_strategy)

        # for debug - to be removed
        if self.config.debug:
            w3 = Chem.SDWriter("./true.sdf")
            w1 = Chem.SDWriter("./gen_traj_state.sdf")
            w2 = Chem.SDWriter("./gen_traj_pred.sdf")
            for curr, pred in trajectory:
                curr_mol = mol_from_tensor(curr[0])
                pred_mol = mol_from_tensor(pred[0])
                w1.write(curr_mol)
                w2.write(pred_mol)
            w3.write(mol_from_tensor(LigandBatch(**batch["ligand_mol"])[0]))
            w1.close()
            w2.close()
            w3.close()

        return trajectory[-1][0]  # return the last state of the trajectory

    @torch.no_grad()
    def run_reconstruct(self, batch, steps: int, strategy: str = "linear") -> list[tuple[LigandBatch, LigandBatch]]:
        if strategy == "linear":
            time_points = np.linspace(0, 1, steps + 1).tolist()
        elif strategy == "log":
            time_points = (1 - np.geomspace(0.01, 1.0, steps + 1)).tolist()
            time_points.reverse()
        else:
            raise ValueError(f"Unknown ODE integration strategy '{strategy}'")

        batch = self.to_device(batch, self.device)

        # pre-calculate pocket embedding
        pocket = PocketBatch(**batch["pocket_mol"])
        pocket_embedding = self(pocket=pocket, training=False, mode="encode")

        # true ligand
        label_ligand = LigandBatch(**batch["ligand_mol"])
        # copy ligand with prior coordinates
        curr = label_ligand.copy_with(coords=batch["x0"])
        B, L = curr.atoms.shape

        step_size = 1 / steps
        predicted = curr.copy_with(coords=torch.zeros_like(curr.coords))  # Initialize predicted with zeros

        trajectory: list[tuple[LigandBatch, LigandBatch]] = []  # (curr, pred)
        for i in range(steps):
            time = i * step_size

            # Set condition  (note: here we use '3' as dim to match CGFlow's one)
            time_cond = torch.full((B, L, 3), time, device=curr.coords.device, dtype=curr.coords.dtype)
            self_cond = predicted.coords
            condition = ConditionBatch(time_cond, self_cond, curr.mask)

            # Predict coordinates (x1-hat)
            predicted = self(
                ligand=curr, condition=condition, pocket_embedding=pocket_embedding, training=False, mode="decode"
            )
            # We take a step with the predicted coordinates
            curr = self.integrator.step(curr, predicted, time, step_size)

            # Add trajectory
            trajectory.append((curr, predicted))

        return trajectory

    def to_device(self, batch, device: str | torch.device) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: self.to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, tuple | list):
            return [self.to_device(v, device) for v in batch]
        else:
            return batch

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)


@CFM.register()
class ARMolecularCFM(MolecularCFM):
    config_class: type[CFMConfig] = MolecularCFMConfig

    def __init__(self, config, model: PosePrediction, interpolant: ARGeometricInterpolant, **kwargs):
        super().__init__(config, model, **kwargs)
        self.interpolant = interpolant

    def training_step(self, batch, b_idx: int):
        # get pocket embedding
        pocket = PocketBatch(**batch["pocket_mol"])
        pocket_embedding = self(pocket=pocket, training=True, mode="encode")

        # true label - here we use the masked ligand instead of full ligand
        label_ligand = LigandBatch(**batch["masked_ligand_mol"])
        # copy with interpolated coordinates
        interp_ligand = label_ligand.copy_with(coords=batch["xt"])

        # get condition
        # time condition
        rel_times = batch["rel_times"]  # [B,L]
        gen_times = batch["gen_times"]  # [B,L]
        time = batch["time"].unsqueeze(1).expand_as(rel_times)  # [B,] -> [B,L]
        time_cond = torch.stack([time, rel_times, gen_times], dim=-1)  # [B,L,3]
        # self condition
        self_cond = torch.zeros_like(interp_ligand.coords)  # [B,L,3]
        condition = ConditionBatch(time_cond, self_cond, interp_ligand.mask)

        # If training with self conditioning, half the time generate a conditional batch by setting cond to zeros
        if self.self_condition:
            if torch.rand(1).item() > 0.5:
                with torch.no_grad():
                    pred = self(
                        ligand=interp_ligand,
                        condition=condition,
                        pocket_embedding=pocket_embedding,
                        training=True,
                        mode="decode",
                    )
                condition.self_cond = pred.coords

        # predict coordinates (x1-hat)
        pred_ligand = self(
            ligand=interp_ligand,
            condition=condition,
            pocket_embedding=pocket_embedding,
            training=True,
            mode="decode",
        )
        loss, losses = self._loss(label_ligand, pred_ligand)

        for k, v in losses.items():
            self.log(f"train/{k}", v, on_step=True, logger=True)
        self.log("train/loss", loss, prog_bar=True, on_step=True, logger=True)
        return loss

    @torch.no_grad()
    def run_reconstruct(self, batch, steps: int, strategy: str = "linear") -> list[tuple[LigandBatch, LigandBatch]]:
        # here we implement function based on linear interpolation
        assert self.sampling_strategy == "linear"

        batch = self.to_device(batch, self.device)

        # pre-calculate pocket embedding
        pocket = PocketBatch(**batch["pocket_mol"])
        pocket_embedding = self(pocket=pocket, training=False, mode="encode")

        # true ligand
        # here we use masked ligand instead of full ligand to get attachment information
        # we note that we set constant time = 0.99 for validation/test.
        label_ligand = LigandBatch(**batch["masked_ligand_mol"])
        # to add fragment sequentially
        gen_times = batch["gen_times"]
        # copy with prior coordinates
        prior_ligand = label_ligand.copy_with(coords=batch["x0"])

        # Compute the time points, and initalize the times
        # TODO: add various time scheduling (currently we only use linear)
        time = 0
        step_sizes: list[float] = [1.0 / steps] * steps  # equal step sizes

        # state
        curr_full: LigandBatch = prior_ligand  # store states for generated atoms; prior for not generated ones
        curr: LigandBatch = prior_ligand  # initialized at the first step
        predicted: LigandBatch = prior_ligand  # initialized at the first step
        prev_gen_atom_cnt: int = 0

        trajectory: list[tuple[LigandBatch, LigandBatch]] = []  # (curr, pred)
        for step_size in step_sizes:
            # Compute relative times for each atom
            rel_times = self.interpolant._compute_rel_time(time, gen_times)  # [B, L]
            expanded_times = torch.full_like(rel_times, time, device=gen_times.device, dtype=gen_times.dtype)  # [B, L]
            # Also compute the end times for each atom
            end_times = torch.clamp(gen_times + self.interpolant.max_interp_time, max=1.0)

            # Compute the mask for generated atoms based on relative times
            is_gens = rel_times >= 0  # [B, L]
            is_gens[~label_ligand.mask] = False

            # chech whether the new fragment is being added
            gen_atom_cnt = int(is_gens.sum().item())
            if gen_atom_cnt != prev_gen_atom_cnt:
                # update structures when the new fragment is added
                # we also zero-initialize the self-conditioning (predicted)
                curr = self.mask_ligand_batch(curr_full, is_gens)  # [B, L] -> [B, Lmasked]
                predicted = curr.copy_with(coords=torch.zeros_like(curr.coords))  # [B, Lmasked]
                prev_gen_atom_cnt = gen_atom_cnt

            # Mask time # [B, L] -> [B, Lmasked]
            _times = pad_tensors([v[is_gen] for v, is_gen in zip(expanded_times, is_gens, strict=True)])
            _rel_times = pad_tensors([v[is_gen] for v, is_gen in zip(rel_times, is_gens, strict=True)])
            _gen_times = pad_tensors([v[is_gen] for v, is_gen in zip(gen_times, is_gens, strict=True)])
            _end_times = pad_tensors([v[is_gen] for v, is_gen in zip(end_times, is_gens, strict=True)])
            assert _times.shape == _rel_times.shape == _gen_times.shape == (curr.batch_size, curr.length), (
                "shape is not mathed to atoms"
            )

            # Set condition
            time_cond = torch.stack([_times, _rel_times, _gen_times], dim=-1)  # [B, Lmasked, 3]
            self_cond = predicted.coords  # [B, Lmasked, 3]
            condition = ConditionBatch(time_cond, self_cond, curr.mask)  # [B, Lmasked]

            # Predict coordinates (x1-hat)
            predicted = self(ligand=curr, condition=condition, pocket_embedding=pocket_embedding, training=False)

            # We take a step with the predicted coordinates
            curr = self.integrator.step(curr, predicted, time, step_size, end_t=_end_times)

            # update coordinate storage
            curr_full.coords[is_gens] = curr.coords[curr.mask.bool()]  # [B, L, 3]

            # add trajectory
            trajectory.append((curr, predicted))

            # Update the times
            time += step_size

        # After the generation is finished, the ligand should have the same length as the label ligand
        assert len(curr) == len(label_ligand)

        return trajectory

    def mask_ligand_batch(self, batch: LigandBatch, masks: _T):
        """mask out atoms that have not been generated yet using the masks provided"""
        tensors: list[LigandTensor] = []
        for i in range(batch.batch_size):
            m = masks[i]
            tensors.append(
                LigandTensor(
                    batch.atoms[i][m],
                    batch.charges[i][m],
                    batch.attachments[i][m],
                    batch.adjacency[i][m][:, m],
                    batch.coords[i][m, :],
                )
            )
        return LigandBatch.from_tensors(tensors)
