import torch
import numpy as np


from semlaflow.data.interpolate import ARGeometricInterpolant
from semlaflow.models.fm import BaseMolecularCFM
from semlaflow.data.datamodules import GeometricInterpolantDM
from semlaflow.models.pocket import PocketEncoder
from semlaflow.util.molrepr import GeometricMol
from semlaflow.util.functional import pad_tensors

from semlaflow.models.complex_fm import _compute_complex_model_args

import semlaflow.util.complex_metrics as ComplexMetrics
from torchmetrics import MetricCollection

_T = torch.Tensor


def mask_mol(mol: GeometricMol, mask: _T) -> GeometricMol:
    """Mask out atoms that have not been generated yet in the molecule"""
    mask = mask.bool().cpu()

    coords = mol.coords[mask]
    atomics = mol.atomics[mask]
    interp_adj = mol.adjacency[mask][:, mask]

    interp_mol_seq_length = coords.shape[0]
    # This makes bond_types a tensor of shape (n_bonds, bond_type_dim)
    bond_indices = torch.ones((interp_mol_seq_length, interp_mol_seq_length)).nonzero()
    bond_types = interp_adj[bond_indices[:, 0], bond_indices[:, 1]]

    # GeometricMol should have a start time attribute for each of its atom
    masked_mol = GeometricMol(coords, atomics, bond_indices=bond_indices, bond_types=bond_types)
    return masked_mol


class ARMolecularCFM(BaseMolecularCFM):
    def __init__(self, ar_interpolant: ARGeometricInterpolant, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.distill
        assert self.type_loss_weight == 0.0
        assert self.bond_loss_weight == 0.0
        assert self.charge_loss_weight == 0.0

        self.interpolant = ar_interpolant
        self.collator = GeometricInterpolantDM(None, None, None, 0)

    def _extract_from_batch(self, batch):
        prior, data, interpolated, ar_masked_data, times, rel_times, gen_times = batch

        # Make sure that the relative times are the same shape as the interpolated coords (without the xyz dimension)
        assert (
            rel_times.shape == interpolated["coords"].shape[:2]
        ), f"{rel_times.shape} != {interpolated['coords'].shape}"

        return {
            "prior": prior,
            "data": data,
            "interpolated": interpolated,
            "ar_masked_data": ar_masked_data,
            "times": times,
            "rel_times": rel_times,
            "gen_times": gen_times,
        }

    def _compute_forward_args(
        self,
        batch_dict,
        training=True,
        cond_batch=None,
    ):
        return {
            "batch": batch_dict["interpolated"],
            "t": batch_dict["times"],
            "t_rel": batch_dict["rel_times"],
            "training": training,
            "cond_batch": cond_batch,
        }

    def training_step(self, batch, b_idx):
        batch_dict = self._extract_from_batch(batch)
        interpolated = batch_dict["interpolated"]
        ar_masked_data = batch_dict["ar_masked_data"]
        holo_mols = batch_dict.get("holo_mols", None)

        cond_batch = None

        # If training with self conditioning, half the time generate a conditional batch by setting cond to zeros
        if self.self_condition:
            cond_batch = {
                "coords": torch.zeros_like(interpolated["coords"]),
                "atomics": ar_masked_data["atomics"],
                "bonds": ar_masked_data["bonds"],
            }

            if torch.rand(1).item() > 0.5:
                with torch.no_grad():
                    model_input = self._compute_forward_args(batch_dict, training=False, cond_batch=cond_batch)
                    cond_coords, cond_types, cond_bonds, _ = self(**model_input)
                    cond_batch = {
                        "coords": cond_coords,
                        "atomics": ar_masked_data["atomics"],
                        "bonds": ar_masked_data["bonds"],
                    }

        model_input = self._compute_forward_args(batch_dict, training=True, cond_batch=cond_batch)
        coords, types, bonds, charges = self(**model_input)

        # The categorical data remains the same
        predicted = {
            "coords": coords,
            "atomics": ar_masked_data["atomics"],
            "bonds": ar_masked_data["bonds"],
            "charges": ar_masked_data["charges"],
        }

        # We are computing the loss of the predicted with respect to
        # the data with the atoms that have not been generated yet masked out
        losses = self._loss(ar_masked_data, interpolated, predicted, holo_mols)
        loss = sum(list(losses.values()))

        for name, loss_val in losses.items():
            self.log(f"train-{name}", loss_val, on_step=True, logger=True)

        self.log("train-loss", loss, prog_bar=True, on_step=True, logger=True)

        return loss

    def _compute_gen_batch(self, batch):
        prior, data, interpolated, ar_masked_data, times, rel_times, gen_times = batch
        gen_batch = self._generate(prior, gen_times, self.integrator.steps, self.sampling_strategy)
        return gen_batch

    def _compute_data_mols(self, batch):
        prior, data, interpolated, ar_masked_data, times, rel_times, gen_times = batch
        return self._generate_mols(data, rescale=True)

    def _compute_interp_mols(self, batch):
        raise NotImplementedError("This method is not used in AR models")

    def _compute_model_args(self, batch, t, t_rel, cond_batch=None):
        """Predict molecular coordinates and atom types

        Args:
            batch (dict[str, Tensor]): Batched pointcloud data
            t (torch.Tensor): Interpolation times between 0 and 1, shape [batch_size]
            t_rel (torch.Tensor): Relative times between 0 and 1, shape [batch_size, num_atoms]
            training (bool): Whether to run forward in training mode
            cond_batch (dict[str, Tensor]): Predictions from previous step, if we are using self conditioning

        Returns:
            (predicted coordinates, atom type logits (unnormalised probabilities))
            Both torch.Tensor, shapes [batch_size, num_atoms, 3] and [batch_size, num atoms, vocab_size]
        """
        model_args = super()._compute_model_args(batch, t, cond_batch)

        # Append relative times to the input features
        rel_times = t_rel.unsqueeze(2)
        model_args["inv_feats"] = torch.cat((model_args["inv_feats"], rel_times), dim=2)
        return model_args

    def forward(self, batch, t, t_rel, training=False, cond_batch=None, *args, **kwargs):
        return super().forward(
            batch=batch,
            t=t,
            t_rel=t_rel,
            training=training,
            cond_batch=cond_batch,
            *args,
            **kwargs,
        )

    def _step(self, curr, times, gen_times, step_size, *args, **kwargs):
        # Compute the is_gen mask for atoms based on gen times
        expanded_times = times.unsqueeze(1).expand(-1, gen_times.size(1))

        # Compute relative times for each atom
        # if rel_times < 0, it means the atom has not been generated yet
        # if rel_times == 1, it means the atom should be at ground truth already
        rel_times = self.interpolant._compute_rel_time(expanded_times, gen_times)

        # Also compute the end times for each atom
        end_times = torch.clamp(gen_times + self.interpolant.max_interp_time, max=1.0)

        # Compute the is_gen mask for atoms based on relative times
        is_gens = rel_times >= 0

        # We convert mask out atoms that have not been generated yet
        curr_masked = self.mask_mol_batch(curr, is_gens)

        # Make padding atoms False - this is important for updating the coords and relative times
        mol_size = curr["mask"].sum(dim=1)
        for i in range(len(is_gens)):
            is_gens[i, mol_size[i] :] = False

        # We adjust relative times to be padded to the size as curr_masked coords
        masked_rel_times = pad_tensors([rel_time[is_gen] for rel_time, is_gen in zip(rel_times, is_gens, strict=False)])
        masked_end_times = pad_tensors([end_time[is_gen] for end_time, is_gen in zip(end_times, is_gens, strict=False)])
        assert masked_rel_times.shape == curr_masked["coords"].shape[:2]
        assert masked_end_times.shape == curr_masked["coords"].shape[:2]

        if self.self_condition:
            cond = {
                "coords": torch.zeros_like(curr_masked["coords"]),
                "atomics": curr_masked["atomics"],
                "bonds": curr_masked["bonds"],
            }

            coords, _, _, _ = self(
                batch=curr_masked,
                t=times,
                t_rel=masked_rel_times,
                training=False,
                cond_batch=cond,
                *args,
                **kwargs,
            )

            cond = {
                "coords": coords,
                "atomics": curr_masked["atomics"],
                "bonds": curr_masked["bonds"],
            }

        else:
            cond = None

        coords, _, _, _ = self(
            batch=curr_masked,
            t=times,
            t_rel=masked_rel_times,
            training=False,
            cond_batch=cond,
            *args,
            **kwargs,
        )

        predicted = {
            "coords": coords,
            "atomics": curr["atomics"],  # atomics remain the same as the prior
            "bonds": curr["bonds"],  # these won't be used during the innerloop
            "charges": curr["charges"],
            "mask": curr["mask"],
        }

        # We take a step with the predicted coordinates
        # prior is set to None - as it shouldn't matter if we are using the no-change strategy
        curr_masked = self.integrator.step(
            curr_masked,
            predicted,
            None,
            times,
            step_size,
            end_t=masked_end_times,
        )

        # We now update the current batch with updated coords
        curr["coords"][is_gens] = curr_masked["coords"][curr_masked["mask"].bool()]

        # Update the times
        times = times + step_size
        return curr, predicted, times

    def _step_interval(
        self, curr, gen_times, steps, curr_time, end_time, *args, **kwargs
    ) -> tuple[dict[str, _T], dict[str, _T], _T]:
        assert end_time > curr_time
        time_points = np.linspace(curr_time, end_time, steps + 1).tolist()
        times = torch.full((curr["coords"].size(0),), curr_time, device=self.device)
        step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:], strict=False)]

        curr = {k: v.clone() for k, v in curr.items()}

        with torch.no_grad():
            for step_size in step_sizes:
                curr, predicted, times = self._step(curr, times, gen_times, step_size, *args, **kwargs)
        return curr, predicted, times

    def _step_interval_inference(
        self,
        curr: dict[str, _T],
        gen_times: _T,
        steps: int,
        curr_time: float,
        end_time: float,
        return_traj: bool,
        *args,
        **kwargs,
    ) -> list[tuple[_T, _T]]:
        # TODO: add desc
        assert end_time > curr_time
        curr = {k: v.clone() for k, v in curr.items()}

        step_size = (end_time - curr_time) / steps
        trajectory: list[tuple[_T, _T]] = []
        times = torch.full((curr["coords"].size(0),), curr_time, device=self.device)
        with torch.no_grad():
            for t in range(steps):
                curr, predicted, times = self._step(curr, times, gen_times, step_size, *args, **kwargs)
                if (t == steps - 1) or return_traj:
                    trajectory.append((curr["coords"].cpu().clone(), predicted["coords"].cpu().clone()))
        return trajectory

    def _generate(self, prior, gen_times, steps, strategy="linear", *args, **kwargs):
        assert strategy == "linear"

        # Compute the time points, and initalize the times
        time_points = np.linspace(0, 1, steps + 1).tolist()
        times = torch.zeros(prior["coords"].size(0), device=self.device)
        step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:], strict=False)]

        # intialize the current batch with the prior
        curr = {k: v.clone() for k, v in prior.items()}

        with torch.no_grad():
            for step_size in step_sizes:
                curr, predicted, times = self._step(curr, times, gen_times, step_size, *args, **kwargs)

        predicted["coords"] = predicted["coords"] * self.coord_scale

        # Ensure that the final values are the same as the prior
        assert (predicted["atomics"] == prior["atomics"]).all()
        assert (predicted["bonds"] == prior["bonds"]).all()

        return predicted

    def mask_mol_batch(self, batch, masks):
        """
        For a dictionary batch of molecules, mask out atoms that have not been generated yet
        using the masks provided

        Args:
            batch (dict[str, Tensor]): Batched GeometricMol data
            masks (list[Tensor]): List of masks for each molecule in the batch
        """

        # Build the molecules back
        curr_mols = self.builder.smol_from_tensors(
            coords=batch["coords"],
            atom_dists=batch["atomics"],
            mask=batch["mask"],
            bond_dists=batch["bonds"],
            charge_dists=batch["charges"],
        )

        # Mask out atoms that have not been generated yet
        # only take masks that are the same length as the molecule
        masked_mols = [mask_mol(mol, mask[: mol.seq_length]) for mol, mask in zip(curr_mols, masks, strict=False)]

        # Collate the masked mols into a batch
        masked_batch = self.collator._collate_objs(masked_mols)

        masked_batch = {k: v.to(self.device) for k, v in masked_batch.items()}
        return masked_batch

    def predict_step(self, batch, batch_idx):
        # NOTE: Adapt for different batch formats, use gen_times for generate
        prior, _, _, _, _, _, gen_times = batch
        gen_batch = self._generate(prior, gen_times, self.integrator.steps, self.sampling_strategy)
        gen_mols = self._generate_mols(gen_batch)
        return gen_mols


class ARComplexMolecularCFM(ARMolecularCFM):
    def __init__(
        self,
        ar_interpolant: ARGeometricInterpolant,
        pocket_encoder: PocketEncoder,
        use_gvp: bool,
        use_complex_metrics: bool,
        *args,
        **kwargs,
    ):
        super().__init__(ar_interpolant, *args, **kwargs)

        if use_gvp:
            self.pocket_encoder = pocket_encoder
        else:
            self.pocket_encoder = None

        if use_complex_metrics:
            complex_metrics = {
                "clash": ComplexMetrics.Clash(),
                "interactions": ComplexMetrics.Interactions(),
            }
            self.complex_metrics = MetricCollection(complex_metrics, compute_groups=False)
        else:
            self.complex_metrics = None

    def _extract_from_batch(self, batch):
        (
            prior,
            data,
            interpolated,
            ar_masked_data,
            holo_mols,
            holo_pocks,
            times,
            rel_times,
            gen_times,
        ) = batch

        # Make sure that the relative times are the same shape as the interpolated coords (without the xyz dimension)
        assert (
            rel_times.shape == interpolated["coords"].shape[:2]
        ), f"{rel_times.shape} != {interpolated['coords'].shape}"

        return {
            "prior": prior,
            "data": data,
            "interpolated": interpolated,
            "ar_masked_data": ar_masked_data,
            "holo_mols": holo_mols,
            "holo_pocks": holo_pocks,
            "times": times,
            "rel_times": rel_times,
            "gen_times": gen_times,
        }

    def _compute_forward_args(self, batch_dict, training=True, cond_batch=None):
        model_args = super()._compute_forward_args(batch_dict, training, cond_batch)
        model_args["holo"] = batch_dict["holo_mols"]
        model_args["holo_pocks"] = batch_dict["holo_pocks"]
        return model_args

    def _compute_gen_batch(self, batch):
        (
            prior,
            data,
            interpolated,
            ar_masked_data,
            holo_mols,
            holo_pocks,
            times,
            rel_times,
            gen_times,
        ) = batch
        gen_batch = self._generate(
            prior,
            gen_times,
            self.integrator.steps,
            self.sampling_strategy,
            holo=holo_mols,
            holo_pocks=holo_pocks,
        )
        return gen_batch

    def _compute_data_mols(self, batch):
        (
            prior,
            data,
            interpolated,
            ar_masked_data,
            holo_mols,
            holo_pocks,
            times,
            rel_times,
            gen_times,
        ) = batch
        return self._generate_mols(data, rescale=True)

    def _compute_interp_mols(self, batch):
        raise NotImplementedError("This method is not used in AR models")

    def _compute_model_args(self, batch, t, t_rel, cond_batch=None, holo=None, holo_pocks=None):
        # NOTE: Concatenate relative time into model input
        model_args = _compute_complex_model_args(
            batch,
            t,
            cond_batch=cond_batch,
            holo=holo,
            holo_pocks=holo_pocks,
            pocket_encoder=self.pocket_encoder,
        )

        rel_times = t_rel.unsqueeze(2)
        model_args["lig_inv_feats"] = torch.cat((rel_times, model_args["lig_inv_feats"]), dim=2)

        if holo is not None:
            dummy_pocket_rel_times = torch.zeros_like(model_args["pro_inv_feats"][:, :, 0]).unsqueeze(2)
            model_args["pro_inv_feats"] = torch.cat((dummy_pocket_rel_times, model_args["pro_inv_feats"]), dim=2)
        return model_args

    def forward(
        self,
        batch,
        t,
        t_rel,
        training=False,
        cond_batch=None,
        holo=None,
        holo_pocks=None,
    ):
        return super().forward(
            batch=batch,
            t=t,
            t_rel=t_rel,
            training=training,
            cond_batch=cond_batch,
            holo=holo,
            holo_pocks=holo_pocks,
        )

    def _generate(self, prior, gen_times, steps, strategy="linear", holo=None, holo_pocks=None):
        return super()._generate(
            prior=prior,
            gen_times=gen_times,
            steps=steps,
            strategy=strategy,
            holo=holo,
            holo_pocks=holo_pocks,
        )

    def _compute_complex_metrics(self, batch):
        # Compute the complex metrics additionally
        (
            prior,
            data,
            interpolated,
            ar_masked_data,
            holo_mols,
            holo_pocks,
            times,
            rel_times,
            gen_times,
        ) = batch

        # Check if it's not a psuedo complex batch
        if holo_pocks is not None and self.complex_metrics is not None:
            gen_batch = self._compute_gen_batch(batch)
            gen_mols = self._generate_mols(gen_batch)
            self.complex_metrics.update(gen_mols, holo_pocks)

    def validation_step(self, batch, b_idx):
        super().validation_step(batch, b_idx)
        self._compute_complex_metrics(batch)

    def _compute_and_log_complex_metrics(self):
        if self.complex_metrics is not None:
            complex_metrics_results = self.complex_metrics.compute()
            self.complex_metrics.reset()
            for metric, value in complex_metrics_results.items():
                progbar = True if metric == "validity" else False
                self.log(f"val-{metric}", value, on_epoch=True, logger=True, prog_bar=progbar)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        self._compute_and_log_complex_metrics()

    def predict_step(self, batch, batch_idx):
        # NOTE: Adapt for different batch formats, use gen_times for generate
        batch_dict = self._extract_from_batch(batch)
        gen_batch = self._generate(
            prior=batch_dict["prior"],
            gen_times=batch_dict["gen_times"],
            steps=self.integrator.steps,
            strategy=self.sampling_strategy,
            holo=batch_dict["holo_mols"],
            holo_pocks=batch_dict["holo_pocks"],
        )
        gen_mols = self._generate_mols(gen_batch)
        return gen_mols
