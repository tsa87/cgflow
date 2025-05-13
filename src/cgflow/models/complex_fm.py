from functools import partial

import numpy as np
import torch

from cgflow.data.datamodules import GeometricInterpolantDM
from cgflow.data.interpolate import ARGeometricInterpolant
from cgflow.models.fm import MolecularCFM
from cgflow.util.functional import pad_tensors
from cgflow.util.molrepr import GeometricMol

_T = torch.Tensor
_BatchT = dict[str, _T]


def mask_mol(mol: GeometricMol, mask: _T) -> GeometricMol:
    """Mask out atoms that have not been generated yet in the molecule"""
    mask = mask.bool().cpu()

    coords = mol.coords[mask]
    atomics = mol.atomics[mask]
    interp_adj = mol.adjacency[mask][:, mask]

    interp_mol_seq_length = coords.shape[0]
    # This makes bond_types a tensor of shape (n_bonds, bond_type_dim)
    bond_indices = torch.ones(
        (interp_mol_seq_length, interp_mol_seq_length)).nonzero()
    bond_types = interp_adj[bond_indices[:, 0], bond_indices[:, 1]]

    # GeometricMol should have a start time attribute for each of its atom
    masked_mol = GeometricMol(coords,
                              atomics,
                              bond_indices=bond_indices,
                              bond_types=bond_types)
    return masked_mol


class ARMolecularCFM(MolecularCFM):

    def __init__(self, ar_interpolant: ARGeometricInterpolant, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.interpolant = ar_interpolant
        self.collator = GeometricInterpolantDM(None, None, None, 0)

    def _parse_batch(self, batch):
        if len(batch) == 7:
            prior, data, interpolated, masked_data, times, rel_times, gen_times = batch
            pocket = None
            pocket_raw = None
        elif len(batch) == 9:
            prior, data, interpolated, masked_data, pocket, pocket_raw, times, rel_times, gen_times = batch
        else:
            raise ValueError(
                f"Batch must be of length 4 or 6, not {len(batch)}")

        assert (rel_times.shape == interpolated["coords"].shape[:2]
                ), f"{rel_times.shape} != {interpolated['coords'].shape}"
        return prior, data, interpolated, masked_data, pocket, pocket_raw, times, rel_times, gen_times

    def forward(self,
                batch,
                t,
                t_rel,
                training=False,
                cond_batch=None,
                pocket_batch=None):
        """Predict molecular coordinates and atom types

        Args:
            batch (dict[str, Tensor]): Batched pointcloud data
            t (torch.Tensor): Interpolation times between 0 and 1, shape [batch_size]
            t_rel (torch.Tensor): Interpolation times between 0 and 1, shape [batch_size, N]
            training (bool): Whether to run forward in training mode
            cond_batch (dict[str, Tensor]): Predictions from previous step, if we are using self conditioning
            pocket_batch (dict[str, Tensor]): Batch pointcloud data for pocket

        Returns:
            (predicted coordinates, atom type logits (unnormalised probabilities))
            Both torch.Tensor, shapes [batch_size, num_atoms, 3] and [batch_size, num atoms, vocab_size]
        """

        lig_coords = batch["coords"]
        lig_atom_types = batch["atomics"]
        lig_bonds = batch["bonds"]
        lig_mask = batch["mask"]
        # Prepare invariant atom features
        lig_times = t.view(-1, 1, 1).expand(-1, lig_coords.size(1), -1)
        lig_rel_times = t_rel.unsqueeze(-1)
        extra_feats = torch.cat([lig_times, lig_rel_times], dim=-1)

        if pocket_batch is not None:
            pro_coords = pocket_batch["coords"]
            pro_atom_types = pocket_batch["atomics"]
            pro_charges = pocket_batch["charges"]
            pro_residues = pocket_batch["residues"]
            pro_bonds = pocket_batch["bonds"]
            pro_mask = pocket_batch["mask"]
        else:
            pro_coords = None
            pro_atom_types = None
            pro_charges = None
            pro_residues = None
            pro_bonds = None
            pro_mask = None

        # Whether to use the EMA version of the model or not
        if not training and self.ema_gen is not None:
            model = self.ema_gen
        else:
            model = self.gen

        wrapped_model = partial(
            model,
            coords=lig_coords,
            atom_types=lig_atom_types,
            bond_types=lig_bonds,
            atom_mask=lig_mask,
            extra_feats=extra_feats,
            pocket_coords=pro_coords,
            pocket_atom_names=pro_atom_types,
            pocket_atom_charges=pro_charges,
            pocket_res_types=pro_residues,
            pocket_bond_types=pro_bonds,
            pocket_atom_mask=pro_mask,
        )

        if cond_batch is not None:
            out = wrapped_model(cond_coords=cond_batch["coords"],
                                cond_atomics=cond_batch["atomics"],
                                cond_bonds=cond_batch["bonds"])
        else:
            out = wrapped_model()

        return out

    # TODO this part
    def training_step(self, batch, b_idx):
        try:
            prior, data, interpolated, masked_data, pocket, pocket_raw, times, rel_times, gen_times = self._parse_batch(
                batch)
            masked_data = self._batch_to_onehot(masked_data)
            cond_batch = None

            # If training with self conditioning, half the time generate a conditional batch by setting cond to zeros
            if self.self_condition:
                cond_batch = {
                    "coords":
                    torch.zeros_like(interpolated["coords"],
                                     device=self.device),
                    "atomics":
                    torch.zeros_like(masked_data["atomics"],
                                     device=self.device),
                    "bonds":
                    torch.zeros_like(masked_data["bonds"], device=self.device),
                }

                if torch.rand(1).item() > 0.5:
                    with torch.no_grad():
                        cond_coords, cond_types, cond_bonds, _ = self(
                            interpolated,
                            times,
                            rel_times,
                            training=True,
                            cond_batch=cond_batch,
                            pocket_batch=pocket)

                        cond_batch = {
                            "coords": cond_coords,
                            "atomics": masked_data["atomics"],
                            "bonds": masked_data["bonds"],
                        }

            coords, types, bonds, charges = self(interpolated,
                                                 times,
                                                 rel_times,
                                                 training=True,
                                                 cond_batch=cond_batch,
                                                 pocket_batch=pocket)
            predicted = {
                "coords": coords,
                "atomics": masked_data["atomics"],
                "bonds": masked_data["bonds"],
                "charges": masked_data["charges"],
            }

            losses = self._loss(data, interpolated, predicted)
            loss = sum(list(losses.values()))

            for name, loss_val in losses.items():
                self.log(f"train-{name}", loss_val, on_step=True, logger=True)

            self.log("train-loss",
                     loss,
                     prog_bar=True,
                     on_step=True,
                     logger=True)

            return loss
        except Exception as e:
            print(f"[Training] Skipping batch {b_idx}: {e}")
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return loss

    def validation_step(self, batch, b_idx):
        # try:
        prior, data, interpolated, masked_data, pocket, pocket_raw, times, rel_times, gen_times = self._parse_batch(
            batch)

        gen_batch = self._generate(
            prior,
            gen_times,
            self.integrator.steps,
            self.sampling_strategy,
            pocket_batch=pocket,
        )
        stabilities = self._generate_stabilities(gen_batch)
        gen_mols = self._generate_mols(gen_batch)

        self.stability_metrics.update(stabilities)
        self.gen_metrics.update(gen_mols)
        if pocket_raw is not None and self.complex_metrics:
            self.complex_metrics.update(gen_mols, pocket_raw)

        if self.conformer_metrics:
            data = self._batch_to_onehot(data)
            data_mols = self._generate_mols(data, rescale=True)
            self.conf_metrics.update(gen_mols, data_mols)
        # except Exception as e:
        #     print(f"[Validation] Skipping batch {b_idx}: {e}")
        #     loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        #     return loss

    def _generate(self,
                  prior,
                  gen_times,
                  steps,
                  strategy="linear",
                  pocket_batch=None):
        assert strategy == "linear"

        # Compute the time points, and initalize the times
        time_points = np.linspace(0, 1, steps + 1).tolist()
        times = torch.zeros(prior["coords"].size(0), device=self.device)
        step_sizes = [
            t1 - t0
            for t0, t1 in zip(time_points[:-1], time_points[1:], strict=False)
        ]

        # intialize the current batch with the prior
        curr = {k: v.clone() for k, v in prior.items()}

        with torch.no_grad():
            for step_size in step_sizes:
                # Compute the is_gen mask for atoms based on gen times
                expanded_times = times.unsqueeze(1).expand(
                    -1, gen_times.size(1))

                # Compute relative times for each atom
                # if rel_times < 0, it means the atom has not been generated yet
                # if rel_times == 1, it means the atom should be at ground truth already
                rel_times = self.interpolant._compute_rel_time(
                    expanded_times, gen_times)

                # Also compute the end times for each atom
                end_times = torch.clamp(gen_times +
                                        self.interpolant.max_interp_time,
                                        max=1.0)

                # Compute the is_gen mask for atoms based on relative times
                is_gens = rel_times >= 0

                # We convert mask out atoms that have not been generated yet
                curr_masked = self.mask_mol_batch(curr, is_gens)

                # Make padding atoms False - this is important for updating the coords and relative times
                mol_size = curr["mask"].sum(dim=1)
                for i in range(len(is_gens)):
                    is_gens[i, mol_size[i]:] = False

                # We adjust relative times to be padded to the size as curr_masked coords
                masked_rel_times = pad_tensors([
                    rel_time[is_gen] for rel_time, is_gen in zip(
                        rel_times, is_gens, strict=False)
                ])
                masked_end_times = pad_tensors([
                    end_time[is_gen] for end_time, is_gen in zip(
                        end_times, is_gens, strict=False)
                ])
                assert masked_rel_times.shape == curr_masked[
                    "coords"].shape[:2]
                assert masked_end_times.shape == curr_masked[
                    "coords"].shape[:2]

                if self.self_condition:
                    cond_batch = {
                        "coords":
                        torch.zeros_like(curr_masked["coords"],
                                         device=self.device),
                        "atomics":
                        torch.zeros((*curr_masked["atomics"].shape,
                                     self.gen.n_atom_types),
                                    device=self.device),
                        "bonds":
                        torch.zeros((*curr_masked["bonds"].shape,
                                     self.gen.n_bond_types),
                                    device=self.device),
                    }

                    coords, _, _, _ = self(
                        curr_masked,
                        times,
                        masked_rel_times,
                        training=False,
                        cond_batch=cond_batch,
                        pocket_batch=pocket_batch,
                    )

                    # Cond batch must be one-hot encoded
                    cond_batch = {
                        "coords": coords,
                        "atomics": curr_masked["atomics"],
                        "bonds": curr_masked["bonds"],
                    }
                    cond_batch = self._batch_to_onehot(cond_batch)

                else:
                    cond_batch = None

                coords, _, _, _ = self(
                    curr_masked,
                    times,
                    masked_rel_times,
                    training=False,
                    cond_batch=cond_batch,
                    pocket_batch=pocket_batch,
                )

                # predicted must also be one hot encoded
                predicted = {
                    "coords": coords,
                    "atomics":
                    curr["atomics"],  # atomics remain the same as the prior
                    "bonds":
                    curr["bonds"],  # these won't be used during the innerloop
                    "charges": curr["charges"],
                    "mask": curr["mask"],
                }
                predicted = self._batch_to_onehot(predicted)

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
                curr["coords"][is_gens] = curr_masked["coords"][
                    curr_masked["mask"].bool()].to(curr["coords"].dtype)

                # Update the times
                times = times + step_size

        predicted["coords"] = predicted["coords"] * self.coord_scale

        # Ensure that the final values are the same as the prior
        prior_ohot = self._batch_to_onehot(prior)
        assert (predicted["atomics"] == prior_ohot["atomics"]).all()
        assert (predicted["bonds"] == prior_ohot["bonds"]).all()

        return predicted

    def mask_mol_batch(self, batch, masks):
        """
        For a dictionary batch of molecules, mask out atoms that have not been generated yet
        using the masks provided

        Args:
            batch (dict[str, Tensor]): Batched GeometricMol data
            masks (list[Tensor]): List of masks for each molecule in the batch
        """
        batch = self._batch_to_onehot(batch)

        # Build the molecules back
        curr_mols = self.builder.smol_from_tensors(
            coords=batch["coords"],
            atom_dists=batch["atomics"],
            mask=batch["mask"],
            bond_dists=batch["bonds"],
            charge_dists=batch["charges"],
            extract_from_dist=True,
        )

        # Mask out atoms that have not been generated yet
        # only take masks that are the same length as the molecule
        masked_mols = [
            mask_mol(mol, mask[:mol.seq_length])
            for mol, mask in zip(curr_mols, masks, strict=False)
        ]

        # Collate the masked mols into a batch
        masked_batch = self.collator._collate_objs(masked_mols)

        masked_batch = {k: v.to(self.device) for k, v in masked_batch.items()}

        return masked_batch

    def predict_step(self, batch, batch_idx):
        prior, data, interpolated, masked_data, pocket, pocket_raw, times, rel_times, gen_times = self._parse_batch(
            batch)
        gen_batch = self._generate(
            prior,
            gen_times,
            self.integrator.steps,
            self.sampling_strategy,
            pocket_batch=pocket,
        )
        gen_mols = self._generate_mols(gen_batch)
        return gen_mols
