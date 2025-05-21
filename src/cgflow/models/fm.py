from concurrent.futures import ThreadPoolExecutor
from functools import partial

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, OneCycleLR
from torchmetrics import MetricCollection

import argparse
import cgflow.util.complex_metrics as ComplexMetrics
import cgflow.util.functional as smolF
import cgflow.util.loss as smolL
import cgflow.util.metrics as Metrics
import cgflow.util.rdkit as smolRD
from cgflow.models.pocket import LigandGenerator
from cgflow.util.molrepr import GeometricMol
from cgflow.util.tokeniser import Vocabulary

_T = torch.Tensor
_BatchT = dict[str, _T]


class Integrator:

    def __init__(
        self,
        steps,
        coord_noise_std=0.0,
        type_strategy="mask",
        bond_strategy="mask",
        cat_noise_level=0,
        type_mask_index=None,
        bond_mask_index=None,
        eps=1e-5,
    ):

        self._check_cat_sampling_strategy(type_strategy, type_mask_index,
                                          "type")
        self._check_cat_sampling_strategy(bond_strategy, bond_mask_index,
                                          "bond")

        self.steps = steps
        self.coord_noise_std = coord_noise_std
        self.type_strategy = type_strategy
        self.bond_strategy = bond_strategy
        self.cat_noise_level = cat_noise_level
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index
        self.eps = eps

    @property
    def hparams(self):
        return {
            "integration-steps": self.steps,
            "integration-coord-noise-std": self.coord_noise_std,
            "integration-type-strategy": self.type_strategy,
            "integration-bond-strategy": self.bond_strategy,
            "integration-cat-noise-level": self.cat_noise_level,
        }

    def step(
        self,
        curr: _BatchT,
        predicted: _BatchT,
        prior: _BatchT,
        t: _T,
        step_size: float,
        end_t: _T | None = None,
    ) -> _BatchT:
        # end_t is optional tensor of shape [batch_size, number_of_atoms]
        # t is a tensor of shape [batch_size]
        assert t.dim() == 1, "Time tensor must be 1D"
        batch_size, n_lig_atoms, _ = curr["coords"].shape

        device = curr["coords"].device

        # *** Coord update step ***
        # if end_t is None, we assume that the end time is 1
        if end_t is None:
            end_t = torch.ones((batch_size, n_lig_atoms), device=t.device)
        # Compute the time left for each atom's interpolation
        time_left = end_t.unsqueeze(-1) - t.view(-1, 1, 1)

        # if time left < step_size, we should just set the coords to the predicted coords
        # by clamping the time left to the step size
        time_left = torch.clamp(time_left, min=step_size)

        coord_velocity = (predicted["coords"] - curr["coords"]) / time_left
        coord_velocity += torch.randn_like(
            coord_velocity) * self.coord_noise_std
        coords = curr["coords"] + (step_size * coord_velocity)

        # *** Atom type update step ***
        if self.type_strategy == "linear":
            vocab_size = predicted["atomics"].size(-1)
            one_hots = torch.eye(vocab_size,
                                 device=device).unsqueeze(0).unsqueeze(0)
            type_velocity = one_hots - prior["atomics"].unsqueeze(-1)
            type_velocity = (type_velocity *
                             predicted["atomics"].unsqueeze(-2)).sum(-1)
            atomics = curr["atomics"] + (step_size * type_velocity)

        # Masking strategy from Discrete Flow Models paper (https://arxiv.org/abs/2402.04997)
        elif self.type_strategy == "mask":
            atomics = self._mask_sampling_step(curr["atomics"],
                                               predicted["atomics"], t,
                                               self.type_mask_index, step_size)

        # Uniform sampling strategy from Discrete Flow Models paper
        elif self.type_strategy == "uniform-sample":
            atomics = self._uniform_sample_step(curr["atomics"],
                                                predicted["atomics"], t,
                                                step_size)
        elif self.type_strategy == "no-change":
            atomics = curr["atomics"]

        # *** Bond update step ***
        if self.type_strategy == "linear":
            n_bonds = predicted["bonds"].size(-1)
            one_hots = torch.eye(n_bonds,
                                 device=device).view(1, 1, 1, n_bonds, n_bonds)
            bond_velocity = one_hots - prior["bonds"].unsqueeze(-1)
            bond_velocity = (bond_velocity *
                             predicted["bonds"].unsqueeze(-2)).sum(-1)
            bonds = curr["bonds"] + (step_size * bond_velocity)

        elif self.bond_strategy == "mask":
            bonds = self._mask_sampling_step(curr["bonds"], predicted["bonds"],
                                             t, self.bond_mask_index,
                                             step_size)

        elif self.bond_strategy == "uniform-sample":
            bonds = self._uniform_sample_step(curr["bonds"],
                                              predicted["bonds"], t, step_size)
        elif self.bond_strategy == "no-change":
            bonds = curr["bonds"]

        updated = {
            "coords": coords,
            "atomics": atomics,
            "bonds": bonds,
            "mask": curr["mask"]
        }
        return updated

    # TODO test with mask sampling
    def _mask_sampling_step(self, curr, pred_dist, t, mask_index, step_size):
        n_categories = pred_dist.size(-1)

        pred = torch.distributions.Categorical(pred_dist).sample()

        ones = [1] * (len(pred.shape) - 1)
        times = t.view(-1, *ones)

        # Choose elements to unmask
        limit = step_size * (1 + (self.cat_noise_level * times)) / (1 - times)
        unmask = torch.rand_like(pred.float()) < limit
        unmask = unmask * (curr == mask_index)

        # Choose elements to mask
        mask = torch.rand_like(pred.float()) < step_size * self.cat_noise_level
        mask = mask * (curr != self.type_mask_index)
        mask[t + step_size >= 1.0] = 0.0

        # Applying unmasking and re-masking
        curr[unmask] = pred[unmask]
        curr[mask] = mask_index

        return curr

    def _uniform_sample_step(self, curr, pred_dist, t, step_size):
        n_categories = pred_dist.size(-1)

        curr = curr.unsqueeze(-1)
        pred_probs_curr = torch.gather(pred_dist, -1, curr)

        # Setup batched time tensor and noise tensor
        ones = [1] * (len(pred_dist.shape) - 1)
        times = t.view(-1, *ones).clamp(min=self.eps, max=1.0 - self.eps)
        noise = torch.zeros_like(times)
        noise[times + step_size < 1.0] = self.cat_noise_level

        # Off-diagonal step probs
        mult = (1 + ((2 * noise) * (n_categories - 1) * times)) / (1 - times)
        first_term = step_size * mult * pred_dist
        second_term = step_size * noise * pred_probs_curr
        step_probs = (first_term + second_term).clamp(max=1.0)

        # On-diagonal step probs
        step_probs.scatter_(-1, curr, 0.0)
        diags = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        step_probs.scatter_(-1, curr, diags)

        # Sample and convert back to one-hot so that all strategies represent data the same way
        samples = torch.distributions.Categorical(step_probs).sample()
        return samples

    def _check_cat_sampling_strategy(self, strategy, mask_index, name):
        if strategy not in [
                "linear",
                "dirichlet",
                "mask",
                "uniform-sample",
                "no-change",
        ]:
            raise ValueError(
                f"{name} sampling strategy '{strategy}' is not supported.")

        if strategy == "mask" and mask_index is None:
            raise ValueError(
                f"{name}_mask_index must be provided if using the mask sampling strategy."
            )


class MolBuilder:

    def __init__(self, vocab, n_workers=16):
        self.vocab = vocab
        self.n_workers = n_workers
        self._executor = None

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def _startup(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(self.n_workers)

    def mols_from_smiles(self, smiles, explicit_hs=False):
        self._startup()
        futures = [
            self._executor.submit(smolRD.mol_from_smiles, smi, explicit_hs)
            for smi in smiles
        ]
        mols = [future.result() for future in futures]
        self.shutdown()
        return mols

    def mols_from_tensors(self,
                          coords,
                          atom_dists,
                          mask,
                          bond_dists=None,
                          charge_dists=None,
                          sanitise=True):
        extracted = self._extract_mols(coords,
                                       atom_dists,
                                       mask,
                                       bond_dists=bond_dists,
                                       charge_dists=charge_dists)

        self._startup()
        build_fn = partial(self._mol_from_tensors, sanitise=sanitise)
        futures = [
            self._executor.submit(build_fn, *items) for items in extracted
        ]
        mols = [future.result() for future in futures]
        self.shutdown()

        return mols

    # TODO move into from_tensors method of GeometricMolBatch
    def smol_from_tensors(
        self,
        coords,
        atom_dists,
        mask,
        bond_dists,
        charge_dists,
        extract_from_dist=False,
    ):
        """
        Generate GeometricMol objects from tensors
        If extract_from_dist then extract atomics and bonds from dist tensors
        """
        extracted = self._extract_mols(coords,
                                       atom_dists,
                                       mask,
                                       bond_dists=bond_dists,
                                       charge_dists=charge_dists)
        self._startup()
        build_fn = partial(self._smol_from_tensors,
                           extract_from_dist=extract_from_dist)
        futures = [
            self._executor.submit(build_fn, *items) for items in extracted
        ]
        smol_mols = [future.result() for future in futures]
        self.shutdown()

        return smol_mols

    def _mol_from_tensors(self,
                          coords,
                          atom_dists,
                          bond_dists=None,
                          charge_dists=None,
                          sanitise=True):
        tokens = self._mol_extract_atomics(atom_dists)
        bonds = self._mol_extract_bonds(
            bond_dists) if bond_dists is not None else None
        charges = self._mol_extract_charges(
            charge_dists) if charge_dists is not None else None
        return smolRD.mol_from_atoms(coords.to(torch.float32).numpy(),
                                     tokens,
                                     bonds=bonds,
                                     charges=charges,
                                     sanitise=sanitise)

    def _smol_from_tensors(self,
                           coords,
                           atom_dists,
                           bond_dists,
                           charge_dists,
                           extract_from_dist=False):
        n_atoms = coords.size(0)
        charges = torch.tensor(self._mol_extract_charges(charge_dists))

        if extract_from_dist:
            atomics = torch.argmax(atom_dists, dim=1)
            bonds = torch.tensor(self._mol_extract_bonds(bond_dists))
            bond_indices = bonds[:, :2].long()
            bond_types = bonds[:, 2]
            mol = GeometricMol(coords, atomics, bond_indices, bond_types,
                               charges)
        else:
            bond_indices = torch.ones((n_atoms, n_atoms)).nonzero()
            bond_types = bond_dists[bond_indices[:, 0], bond_indices[:, 1], :]
            mol = GeometricMol(coords, atom_dists, bond_indices, bond_types,
                               charges)
        return mol

    def mol_stabilities(self, coords, atom_dists, mask, bond_dists,
                        charge_dists):
        extracted = self._extract_mols(coords,
                                       atom_dists,
                                       mask,
                                       bond_dists=bond_dists,
                                       charge_dists=charge_dists)
        mol_atom_stabilities = [
            self.atom_stabilities(*items) for items in extracted
        ]
        return mol_atom_stabilities

    def atom_stabilities(self, coords, atom_dists, bond_dists, charge_dists):
        n_atoms = coords.shape[0]

        atomics = self._mol_extract_atomics(atom_dists)
        bonds = self._mol_extract_bonds(bond_dists)
        charges = self._mol_extract_charges(charge_dists).tolist()

        # Recreate the adj to ensure it is symmetric
        bond_indices = torch.tensor(bonds[:, :2])
        bond_types = torch.tensor(bonds[:, 2])
        adj = smolF.adj_from_edges(bond_indices,
                                   bond_types,
                                   n_atoms,
                                   symmetric=True)

        adj[adj == 4] = 1.5
        valencies = adj.sum(dim=-1).long()

        stabilities = []
        for i in range(n_atoms):
            atom_type = atomics[i]
            charge = charges[i]
            valence = valencies[i].item()

            if atom_type not in Metrics.ALLOWED_VALENCIES:
                stabilities.append(False)
                continue

            allowed = Metrics.ALLOWED_VALENCIES[atom_type]
            atom_stable = Metrics._is_valid_valence(valence, allowed, charge)
            stabilities.append(atom_stable)

        return stabilities

    # Separate each molecule from the batch
    def _extract_mols(self,
                      coords,
                      atom_dists,
                      mask,
                      bond_dists=None,
                      charge_dists=None):
        coords_list = []
        atom_dists_list = []
        bond_dists_list = []
        charge_dists_list = []

        n_atoms = mask.sum(dim=1)
        for idx in range(coords.size(0)):
            mol_atoms = n_atoms[idx]
            mol_coords = coords[idx, :mol_atoms, :].cpu()
            mol_token_dists = atom_dists[idx, :mol_atoms, :].cpu()

            coords_list.append(mol_coords)
            atom_dists_list.append(mol_token_dists)

            if bond_dists is not None:
                mol_bond_dists = bond_dists[
                    idx, :mol_atoms, :mol_atoms, :].cpu()
                bond_dists_list.append(mol_bond_dists)
            else:
                bond_dists_list.append(None)

            if charge_dists is not None:
                mol_charge_dists = charge_dists[idx, :mol_atoms, :].cpu()
                charge_dists_list.append(mol_charge_dists)
            else:
                charge_dists_list.append(None)

        zipped = zip(coords_list,
                     atom_dists_list,
                     bond_dists_list,
                     charge_dists_list,
                     strict=False)
        return zipped

    # Take index with highest probability and convert to token
    def _mol_extract_atomics(self, atom_dists):
        vocab_indices = torch.argmax(atom_dists, dim=1).tolist()
        tokens = self.vocab.tokens_from_indices(vocab_indices)
        return tokens

    # Convert to atomic number bond list format
    def _mol_extract_bonds(self, bond_dists):
        bond_types = torch.argmax(bond_dists, dim=-1)
        bonds = smolF.bonds_from_adj(bond_types)
        return bonds.long().numpy()

    # Convert index from model to actual atom charge
    def _mol_extract_charges(self, charge_dists):
        charge_types = torch.argmax(charge_dists, dim=-1).tolist()
        charges = [smolRD.IDX_CHARGE_MAP[idx] for idx in charge_types]
        return np.array(charges)


# *********************************************************************************************************************
# ******************************************** Lightning Flow Matching Models *****************************************
# *********************************************************************************************************************


class MolecularCFM(L.LightningModule):

    def __init__(
        self,
        gen: LigandGenerator,
        vocab: Vocabulary,
        lr: float,
        integrator: Integrator,
        coord_scale: float = 1.0,
        type_strategy: str = "ce",
        bond_strategy: str = "ce",
        dist_loss_weight: float = 0.0,
        type_loss_weight: float = 1.0,
        bond_loss_weight: float = 1.0,
        charge_loss_weight: float = 1.0,
        pairwise_metrics: bool = True,
        complex_metrics: bool = False,
        use_ema: bool = True,
        compile_model: bool = True,
        self_condition: bool = False,
        distill: bool = False,
        lr_schedule: str = "constant",
        sampling_strategy: str = "linear",
        warm_up_steps: int | None = None,
        total_steps: int | None = None,
        train_smiles: list[str] | None = None,
        type_mask_index: int | None = None,
        bond_mask_index: int | None = None,
        args: argparse.Namespace | None = None,
        **kwargs,
    ):
        super().__init__()

        if type_strategy not in ["mse", "ce", "mask", "no-change"]:
            raise ValueError(
                f"Unsupported type training strategy '{type_strategy}'")

        if bond_strategy not in ["ce", "mask", "no-change"]:
            raise ValueError(
                f"Unsupported bond training strategy '{bond_strategy}'")

        if lr_schedule not in ["constant", "one-cycle"]:
            raise ValueError(f"LR scheduler {lr_schedule} not supported.")

        if lr_schedule == "one-cycle" and total_steps is None:
            raise ValueError(
                "total_steps must be provided when using the one-cycle LR scheduler."
            )

        if distill:
            raise ValueError("Distillation is not supported yet.")

        if lr_schedule == "one-cycle" and warm_up_steps is not None:
            print(
                "Note: warm_up_steps is currently ignored if schedule is one-cycle"
            )

        self.gen: LigandGenerator = gen
        self.vocab = vocab
        self.lr = lr
        self.coord_scale = coord_scale
        self.type_strategy = type_strategy
        self.bond_strategy = bond_strategy
        self.dist_loss_weight = dist_loss_weight
        self.type_loss_weight = type_loss_weight
        self.bond_loss_weight = bond_loss_weight
        self.charge_loss_weight = charge_loss_weight
        self.pairwise_metrics = pairwise_metrics
        self.complex_metrics = complex_metrics
        self.compile_model = compile_model
        self.self_condition = self_condition
        self.lr_schedule = lr_schedule
        self.sampling_strategy = sampling_strategy
        self.warm_up_steps = warm_up_steps
        self.total_steps = total_steps
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index

        builder = MolBuilder(vocab)

        if use_ema:
            avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
            ema_gen = torch.optim.swa_utils.AveragedModel(gen,
                                                          multi_avg_fn=avg_fn)

        if compile_model:
            self.gen = self._compile_model(gen)

        self.integrator = integrator
        self.builder = builder
        self.ema_gen = ema_gen if use_ema else None

        # Anything else passed into kwargs will also be saved
        # hparams = {
        #     "lr": lr,
        #     "coord_scale": coord_scale,
        #     "dist_loss_weight": dist_loss_weight,
        #     "type_loss_weight": type_loss_weight,
        #     "bond_loss_weight": bond_loss_weight,
        #     "type_strategy": type_strategy,
        #     "bond_strategy": bond_strategy,
        #     "self_condition": self_condition,
        #     "distill": distill,
        #     "lr_schedule": lr_schedule,
        #     "sampling_strategy": sampling_strategy,
        #     "use_ema": use_ema,
        #     "compile_model": compile_model,
        #     "warm_up_steps": warm_up_steps,
        #     **gen.hparams,
        #     **integrator.hparams,
        #     **kwargs,
        # }
        self.save_hyperparameters(vars(args))

        stability_metrics = {
            "atom-stability": Metrics.AtomStability(),
            "molecule-stability": Metrics.MoleculeStability(),
        }
        gen_metrics = {
            "validity": Metrics.Validity(),
            "fc-validity": Metrics.Validity(connected=True),
            "uniqueness": Metrics.Uniqueness(),
            "energy-validity": Metrics.EnergyValidity(),
            "opt-energy-validity": Metrics.EnergyValidity(optimise=True),
            "energy": Metrics.AverageEnergy(),
            "energy-per-atom": Metrics.AverageEnergy(per_atom=True),
            "strain": Metrics.AverageStrainEnergy(),
            "strain-per-atom": Metrics.AverageStrainEnergy(per_atom=True),
            "opt-rmsd": Metrics.AverageOptRmsd(),
        }

        if train_smiles is not None and self.type_strategy != "no-change":
            print("Creating RDKit mols from training SMILES...")
            train_mols = self.builder.mols_from_smiles(train_smiles,
                                                       explicit_hs=True)
            train_mols = [mol for mol in train_mols if mol is not None]

            print("Initialising novelty metric...")
            gen_metrics["novelty"] = Metrics.Novelty(train_mols)
            print("Novelty metric complete.")

        self.stability_metrics = MetricCollection(stability_metrics,
                                                  compute_groups=False)
        self.gen_metrics = MetricCollection(gen_metrics, compute_groups=False)

        if complex_metrics:
            complex_metrics = {
                "clash": ComplexMetrics.Clash(),
                "interactions": ComplexMetrics.Interactions(),
            }
            self.complex_metrics = MetricCollection(complex_metrics,
                                                    compute_groups=False)

        if pairwise_metrics:
            pair_metrics = {
                "mol-accuracy": Metrics.MolecularAccuracy(),
                "pair-rmsd": Metrics.MolecularPairRMSD(),
            }
            self.pair_metrics = MetricCollection(pair_metrics,
                                                 compute_groups=False)

        # Conformer metrics for molecules that are not changing
        self.conformer_metrics = True if type_strategy == "no-change" and bond_strategy == "no-change" else False
        if self.conformer_metrics:
            conf_metrics = {
                "conformer-rmsd": Metrics.MolecularPairRMSD(),
                "conformer-no-align-rmsd":
                Metrics.MolecularPairRMSD(align=False),
                "conformer-centroid-rmsd": Metrics.CentroidRMSD(),
            }
            self.conf_metrics = MetricCollection(conf_metrics,
                                                 compute_groups=False)

        self._init_params()

    def _parse_batch(self, batch):
        if len(batch) == 4:
            prior, data, interpolated, times = batch
            pocket = None
            pocket_raw = None
        elif len(batch) == 6:
            prior, data, interpolated, pocket, pocket_raw, times = batch
        else:
            raise ValueError(
                f"Batch must be of length 4 or 6, not {len(batch)}")
        return prior, data, interpolated, pocket, pocket_raw, times

    def forward(self,
                batch,
                t,
                training=False,
                cond_batch=None,
                pocket_batch=None):
        """Predict molecular coordinates and atom types

        Args:
            batch (dict[str, Tensor]): Batched pointcloud data
            t (torch.Tensor): Interpolation times between 0 and 1, shape [batch_size]
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
            extra_feats=lig_times,
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

    def training_step(self, batch, b_idx):
        try:
            prior, data, interpolated, pocket, pocket_raw, times = self._parse_batch(
                batch)

            cond_batch = None

            # If training with self conditioning, half the time generate a conditional batch by setting cond to zeros
            if self.self_condition:
                cond_batch = {
                    "coords":
                    torch.zeros_like(interpolated["coords"],
                                     device=self.device),
                    "atomics":
                    torch.zeros((*interpolated["atomics"].shape,
                                 self.gen.n_atom_types),
                                device=self.device),
                    "bonds":
                    torch.zeros(
                        (*interpolated["bonds"].shape, self.gen.n_bond_types),
                        device=self.device),
                }

                if torch.rand(1).item() > 0.5:
                    with torch.no_grad():
                        cond_coords, cond_types, cond_bonds, _ = self(
                            interpolated,
                            times,
                            training=True,
                            cond_batch=cond_batch,
                            pocket_batch=pocket)
                        cond_batch = {
                            "coords": cond_coords,
                            "atomics": F.softmax(cond_types, dim=-1),
                            "bonds": F.softmax(cond_bonds, dim=-1),
                        }

            coords, types, bonds, charges = self(interpolated,
                                                 times,
                                                 training=True,
                                                 cond_batch=cond_batch,
                                                 pocket_batch=pocket)
            predicted = {
                "coords": coords,
                "atomics": types,
                "bonds": bonds,
                "charges": charges
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
            print(f"[Train] Skipping batch {b_idx}: {e}")
            return None

    def on_train_batch_end(self, outputs, batch, b_idx):
        if self.ema_gen is not None:
            self.ema_gen.update_parameters(self.gen)

    def validation_step(self, batch, b_idx):
        try:
            prior, data, interpolated, pocket, pocket_raw, times = self._parse_batch(
                batch)

            gen_batch = self._generate(prior,
                                       self.integrator.steps,
                                       self.sampling_strategy,
                                       pocket_batch=pocket)
            stabilities = self._generate_stabilities(gen_batch)
            gen_mols = self._generate_mols(gen_batch)

            self.stability_metrics.update(stabilities)
            self.gen_metrics.update(gen_mols)
            if pocket_raw is not None and self.complex_metrics:
                self.complex_metrics.update(gen_mols, pocket_raw)

            # Also measure the model's ability to recreate the original molecule when a bit of prior noise has been added
            if self.pairwise_metrics:
                gen_interp_steps = max(
                    1, int((1 - times[0].item()) * self.integrator.steps))
                gen_interp_batch = self._generate(interpolated,
                                                  gen_interp_steps,
                                                  pocket_batch=pocket)
                gen_interp_mols = self._generate_mols(gen_interp_batch)
                data = self._batch_to_onehot(data)
                data_mols = self._generate_mols(data, rescale=True)
                self.pair_metrics.update(gen_interp_mols, data_mols)

            if self.conformer_metrics:
                data = self._batch_to_onehot(data)
                data_mols = self._generate_mols(data, rescale=True)
                self.conf_metrics.update(gen_mols, data_mols)
        except Exception as e:
            print(f"[Validation] Skipping batch {b_idx}: {e}")
            return None

    def on_validation_epoch_end(self):
        stability_metrics_results = self.stability_metrics.compute()
        gen_metrics_results = self.gen_metrics.compute()
        complex_metrics_results = self.complex_metrics.compute(
        ) if self.complex_metrics else {}
        pair_metrics_results = self.pair_metrics.compute(
        ) if self.pairwise_metrics else {}
        conf_metrics_results = self.conf_metrics.compute(
        ) if self.conformer_metrics else {}

        metrics = {
            **stability_metrics_results,
            **gen_metrics_results,
            **pair_metrics_results,
            **complex_metrics_results,
            **conf_metrics_results,
        }

        for metric, value in metrics.items():
            progbar = True if metric == "validity" else False
            self.log(
                f"val-{metric}",
                float(value),
                on_epoch=True,
                logger=True,
                prog_bar=progbar,
                sync_dist=True,
            )

        self.stability_metrics.reset()
        self.gen_metrics.reset()

        if self.pairwise_metrics:
            self.pair_metrics.reset()
        if self.conformer_metrics:
            self.conf_metrics.reset()
        if self.complex_metrics:
            self.complex_metrics.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def predict_step(self, batch, batch_idx):
        prior, data, interpolated, pocket, pocket_raw, times = self._parse_batch(
            batch)
        gen_batch = self._generate(prior,
                                   self.integrator.steps,
                                   self.sampling_strategy,
                                   pocket_batch=pocket)
        gen_mols = self._generate_mols(gen_batch)
        return gen_mols

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.gen.parameters(),
                               lr=self.lr,
                               amsgrad=True,
                               foreach=True,
                               weight_decay=0.0)

        if self.lr_schedule == "constant":
            warm_up_steps = 0 if self.warm_up_steps is None else self.warm_up_steps
            scheduler = LinearLR(opt,
                                 start_factor=1e-2,
                                 total_iters=warm_up_steps)

        # TODO could use warm_up_steps to shift peak of one cycle
        elif self.lr_schedule == "one-cycle":
            scheduler = OneCycleLR(opt,
                                   max_lr=self.lr,
                                   total_steps=self.total_steps,
                                   pct_start=0.3)

        else:
            raise ValueError(
                f"LR schedule {self.lr_schedule} is not supported.")

        config = {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
        return config

    def _compile_model(self, model):
        return torch.compile(model,
                             dynamic=False,
                             fullgraph=True,
                             mode="reduce-overhead")

    def _loss(self, data, interpolated, predicted):
        pred_coords = predicted["coords"]
        coords = data["coords"]
        mask = data["mask"].unsqueeze(2).float()  # B, N, 1

        coord_loss = F.mse_loss(pred_coords, coords, reduction="none")
        coord_loss = (coord_loss * mask).mean(dim=(1, 2))

        if self.dist_loss_weight > 0:
            hop_mask = smolF.compute_local_mask_batched(data["bonds"],
                                                        max_hops=2)
            dist_loss = smolL.pairwise_dist_loss(pred_coords,
                                                 coords,
                                                 mask=hop_mask)  # B, N, N
            dist_loss = dist_loss * mask
        else:
            dist_loss = torch.tensor(0.0, device=coords.device)

        type_loss = (self._type_loss(data, interpolated, predicted)
                     if self.type_loss_weight > 0 else torch.tensor(
                         0.0, device=coords.device))
        bond_loss = (self._bond_loss(data, interpolated, predicted)
                     if self.bond_loss_weight > 0 else torch.tensor(
                         0.0, device=coords.device))
        charge_loss = (self._charge_loss(data, predicted)
                       if self.charge_loss_weight > 0 else torch.tensor(
                           0.0, device=coords.device))

        coord_loss = coord_loss.mean()
        dist_loss = dist_loss.mean() * self.dist_loss_weight
        type_loss = type_loss.mean() * self.type_loss_weight
        bond_loss = bond_loss.mean() * self.bond_loss_weight
        charge_loss = charge_loss.mean() * self.charge_loss_weight

        losses = {
            "coord-loss": coord_loss,
            "dist-loss": dist_loss,
            "type-loss": type_loss,
            "bond-loss": bond_loss,
            "charge-loss": charge_loss,
        }
        return losses

    def _type_loss(self, data, interpolated, predicted, eps=1e-3):
        pred_logits = predicted["atomics"]
        atomics = data["atomics"]
        mask = data["mask"].unsqueeze(2)
        batch_size, num_atoms, _ = pred_logits.size()

        atomics = atomics.flatten(0, 1)
        type_loss = F.cross_entropy(pred_logits.flatten(0, 1),
                                    atomics,
                                    reduction="none")
        type_loss = type_loss.unflatten(0,
                                        (batch_size, num_atoms)).unsqueeze(2)

        n_atoms = mask.sum(dim=(1, 2)) + eps

        # If we are training with masking, only compute the loss on masked types
        if self.type_strategy == "mask":
            masked_types = interpolated["atomics"] == self.type_mask_index
            n_atoms = masked_types.sum(dim=-1) + eps
            type_loss = type_loss * masked_types.float().unsqueeze(-1)

        type_loss = (type_loss * mask).sum(dim=(1, 2)) / n_atoms
        return type_loss

    def _bond_loss(self, data, interpolated, predicted, eps=1e-3):
        pred_logits = predicted["bonds"]
        mask = data["mask"]
        bonds = data["bonds"]
        batch_size, num_atoms, _, _ = pred_logits.size()

        bond_loss = F.cross_entropy(pred_logits.flatten(0, 2),
                                    bonds.flatten(0, 2),
                                    reduction="none")
        bond_loss = bond_loss.unflatten(0, (batch_size, num_atoms, num_atoms))

        adj_matrix = smolF.adj_from_node_mask(mask, self_connect=True)
        n_bonds = adj_matrix.sum(dim=(1, 2)) + eps

        # Only compute loss on masked bonds if we are training with masking strategy
        if self.bond_strategy == "mask":
            masked_bonds = interpolated["bonds"] == self.bond_mask_index
            n_bonds = masked_bonds.sum(dim=(1, 2)) + eps
            bond_loss = bond_loss * masked_bonds.float()

        bond_loss = (bond_loss * adj_matrix).sum(dim=(1, 2)) / n_bonds
        return bond_loss

    def _charge_loss(self, data, predicted, eps=1e-3):
        pred_logits = predicted["charges"]
        charges = data["charges"]
        mask = data["mask"]
        batch_size, num_atoms, _ = pred_logits.size()

        charges = charges.flatten(0, 1)
        charge_loss = F.cross_entropy(pred_logits.flatten(0, 1),
                                      charges,
                                      reduction="none")
        charge_loss = charge_loss.unflatten(0, (batch_size, num_atoms))

        n_atoms = mask.sum(dim=1) + eps
        charge_loss = (charge_loss * mask).sum(dim=1) / n_atoms
        return charge_loss

    def _batch_to_onehot(self, batch):
        """Convert a batch of molecules to one-hot encoding"""
        one_hot_batch = {}
        for k, v in batch.items():
            if k == "atomics":
                one_hot_batch[k] = smolF.one_hot_encode_tensor(
                    v, self.gen.n_atom_types).float()
            elif k == "bonds":
                one_hot_batch[k] = smolF.one_hot_encode_tensor(
                    v, self.gen.n_bond_types).float()
            elif k == "charges":
                one_hot_batch[k] = smolF.one_hot_encode_tensor(
                    v, self.gen.n_charge_types).float()
            else:
                one_hot_batch[k] = v
        return one_hot_batch

    def _generate(self, prior, steps, strategy="linear", pocket_batch=None):
        if strategy == "linear":
            time_points = np.linspace(0, 1, steps + 1).tolist()

        elif strategy == "log":
            time_points = (1 - np.geomspace(0.01, 1.0, steps + 1)).tolist()
            time_points.reverse()

        else:
            raise ValueError(f"Unknown ODE integration strategy '{strategy}'")

        times = torch.zeros(prior["coords"].size(0), device=self.device)
        step_sizes = [
            t1 - t0
            for t0, t1 in zip(time_points[:-1], time_points[1:], strict=False)
        ]
        curr = {k: v.clone() for k, v in prior.items()}

        cond_batch = {
            "coords":
            torch.zeros_like(prior["coords"], device=self.device),
            "atomics":
            torch.zeros((*prior["atomics"].shape, self.gen.n_atom_types),
                        device=self.device),
            "bonds":
            torch.zeros((*prior["bonds"].shape, self.gen.n_bond_types),
                        device=self.device),
        }

        with torch.no_grad():
            for step_size in step_sizes:
                cond = cond_batch if self.self_condition else None
                coords, type_logits, bond_logits, charge_logits = self(
                    curr,
                    times,
                    training=False,
                    cond_batch=cond,
                    pocket_batch=pocket_batch)

                type_probs = F.softmax(type_logits, dim=-1)
                bond_probs = F.softmax(bond_logits, dim=-1)
                charge_probs = F.softmax(charge_logits, dim=-1)

                cond_batch = {
                    "coords": coords,
                    "atomics": type_probs,
                    "bonds": bond_probs
                }
                predicted = {
                    "coords": coords,
                    "atomics": type_probs,
                    "bonds": bond_probs,
                    "charges": charge_probs,
                    "mask": curr["mask"],
                }

                curr = self.integrator.step(curr, predicted, prior, times,
                                            step_size)

                # For the no-change strategy,
                # ensure that the curr values are the same as the prior
                if self.bond_strategy == "no-change":
                    assert (prior["bonds"] == curr["bonds"]).all()
                if self.type_strategy == "no-change":
                    assert (prior["atomics"] == curr["atomics"]).all()

                times = times + step_size

        predicted["coords"] = predicted["coords"] * self.coord_scale

        # If using the no-change strategy ensure that the final values are the same as the prior
        if self.bond_strategy == "no-change" and self.type_strategy == "no-change":
            prior_ohot = self._batch_to_onehot(prior)
            predicted["atomics"] = prior_ohot["atomics"]
            predicted["bonds"] = prior_ohot["bonds"]
            predicted["charges"] = prior_ohot["charges"]

        return predicted

    def _generate_mols(self, generated, sanitise=True, rescale=False):
        coords = generated["coords"]
        if rescale:  # Rescale the coordinates back to the original scale
            coords = coords * self.coord_scale

        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]
        masks = generated["mask"]

        mols = self.builder.mols_from_tensors(coords,
                                              atom_dists,
                                              masks,
                                              bond_dists=bond_dists,
                                              charge_dists=charge_dists,
                                              sanitise=sanitise)
        return mols

    def _generate_smols(self, generated, extract_from_dist=True):
        coords = generated["coords"]
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]
        masks = generated["mask"]
        smols = self.builder.smol_from_tensors(
            coords,
            atom_dists,
            masks,
            bond_dists,
            charge_dists,
            extract_from_dist=extract_from_dist,
        )
        return smols

    def _generate_stabilities(self, generated):
        coords = generated["coords"]
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]
        masks = generated["mask"]
        stabilities = self.builder.mol_stabilities(coords, atom_dists, masks,
                                                   bond_dists, charge_dists)
        return stabilities

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
