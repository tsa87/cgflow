from functools import cached_property
from pathlib import Path
from typing import Self
from collections.abc import Generator

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem
from torch import Tensor

import biotite.structure.io.pdb as pdb

from semlaflow.data.datamodules import GeometricInterpolantDM
from semlaflow.data.interpolate import ARGeometricInterpolant, HarmonicSDE
from semlaflow.models.ar_fm import ARComplexMolecularCFM
from semlaflow.models.fm import Integrator
from semlaflow.models.pocket import PocketEncoder
from semlaflow.util.molrepr import GeometricMol, GeometricMolBatch
from semlaflow.util.pocket import ProteinPocket
import semlaflow.util.rdkit as smolRD
import semlaflow.util.functional as smolF
from semlaflow.util.tokeniser import Vocabulary

from cgflow.utils import extract_pocket
from .cfg import SemlaFlowConfig, get_cfg_crossdocked

GEOM_COORDS_STD_DEV = 2.407038688659668
PLINDER_COORDS_STD_DEV = GEOM_COORDS_STD_DEV  # 2.2693647416252976
PLINDER_LIG_CENTER_STD_DEV = 1.866057902527167
GEOM_DRUGS_BUCKET_LIMITS = [24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 96, 192]
PLINDER_BUCKET_LIMITS = [128, 170, 256, 512, 1024]


BATCH_COST = 2048
BUCKET_SIZES = [32, 64]
MAX_NUM_BATCH = 32


def build_vocab():
    # Need to make sure PAD has index 0
    special_tokens = ["<PAD>", "<MASK>"]
    core_atoms = ["H", "C", "N", "O", "F", "P", "S", "Cl"]
    other_atoms = ["Br", "B", "Al", "Si", "As", "I", "Hg", "Bi"]
    tokens = special_tokens + core_atoms + other_atoms
    return Vocabulary(tokens)


class SemlaFlowAPI:
    def __init__(
        self,
        ckpt_path: str | Path,
        pocket_path: str | Path,
        config: SemlaFlowConfig | None = None,
        device: str | torch.device = "cpu",
        fp16: bool = True,
    ):
        self.pocket_path = Path(pocket_path)
        self.cfg: SemlaFlowConfig = config if config is not None else get_cfg_crossdocked()
        self.device: torch.device = torch.device(device)
        self.fp16: bool = fp16

        # load pocket
        pdb_file = pdb.PDBFile.read(str(pocket_path))
        pocket_atoms = pdb.get_structure(pdb_file, include_bonds=True)[0]
        self.pocket: ProteinPocket = ProteinPocket.from_pocket_atoms(pocket_atoms, infer_res_bonds=True)

        # create semlaflow
        self.vocab: Vocabulary = build_vocab()
        self.dm = SemlaFlow_DM(self.cfg, self.vocab, self.pocket, self.device)
        self.model: ARComplexMolecularCFM = self.construct_model(ckpt_path)
        self.model.gen = self.model._compile_model(self.model.gen)
        self.model.eval()

        # pre-calculate pocket encoding
        self.holo_data = self.encode_pocket()
        assert self.model.pocket_encoder is None

    @classmethod
    def from_protein(
        cls,
        ckpt_path: str | Path,
        protein_path: str | Path,
        ref_ligand_path: str | Path,
        config: SemlaFlowConfig | None = None,
        device: str | torch.device = "cpu",
        force_pocket_extract: bool = False,
        fp16: bool = True,
        **kwargs,
    ) -> Self:
        config = get_cfg_crossdocked(**kwargs) if config is None else config
        extract_algorithm = config.dataset
        pocket_path = cls.extract_pocket(protein_path, ref_ligand_path, extract_algorithm, force_pocket_extract)
        obj = cls(ckpt_path, pocket_path, config, device, fp16=fp16)
        return obj

    @staticmethod
    def extract_pocket(
        protein_path: str | Path,
        ref_ligand_path: str | Path,
        extract_algorithm: str,
        force_pocket_extract: bool = False,
        fp16: bool = True,
    ) -> Path:
        protein_path = Path(protein_path)
        ref_ligand_path = Path(ref_ligand_path)
        if extract_algorithm == "plinder":
            extract_func = extract_pocket.extract_pocket_plinder
        elif extract_algorithm == "crossdocked":
            extract_func = extract_pocket.extract_pocket_crossdocked_v1
        else:
            raise ValueError("extract_algorithm")
        return extract_func(protein_path, ref_ligand_path, force_pocket_extract=force_pocket_extract)

    @torch.no_grad()
    def step(
        self,
        mols: list[Chem.Mol],
        curr_step: int,
        is_last: bool = False,
        inplace: bool = False,
        return_traj: bool = False,
    ) -> tuple[list[Chem.Mol], list[np.ndarray], list[np.ndarray]]:
        """Predict Binding Pose in an autoregressive manner

        Parameters
        ----------
        mols : list[Chem.Mol]
            molecules, the newly added atoms' coordinates are (0, 0, 0)
        curr_step : int
            current generation step
        is_last : bool
            whether the generation is finished
        inplace : bool
            if True, input molecule informations are updated
        return_traj : bool
            whether return all flow matching trajectory
            if False, only the last output is return (num_trajs = 1)

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            - trajectory of xt      [num_trajs, num_atoms, 3]
            - trajectory of \\hatx1 [num_trajs, num_atoms, 3]
        """
        if not inplace:
            mols = [Chem.Mol(mol) for mol in mols]

        # set gen order when it is unlabeled
        for mol in mols:
            self.set_gen_order(mol, curr_step)

        # mask dummy atoms
        masked_mols, idcs_list = zip(*[remove_dummy(mol) for mol in mols], strict=True)

        # run semlaflow
        if self.fp16:
            with torch.autocast("cuda" if self.device == torch.device("cuda") else "cpu", dtype=torch.float16):
                __traj_xt_list, __traj_x1_list = self._step(masked_mols, curr_step, is_last, return_traj)
        else:
            __traj_xt_list, __traj_x1_list = self._step(masked_mols, curr_step, is_last, return_traj)

        # Add dummy atoms & pose stateate
        upd_mols: list[Chem.Mol] = []
        traj_xt_list: list[np.ndarray] = []
        traj_x1_list: list[np.ndarray] = []
        for i, mol in enumerate(mols):
            num_atoms = mol.GetNumAtoms()
            atom_indices = idcs_list[i]
            traj_xt = __traj_xt_list[i]
            traj_x1 = __traj_x1_list[i]

            # add dummy atom
            traj_xt = expand_trajs(traj_xt, atom_indices, num_atoms)
            traj_x1 = expand_trajs(traj_x1, atom_indices, num_atoms)

            # stateate pose
            mol.GetConformer().SetPositions(traj_xt[-1])

            # add to list
            upd_mols.append(mol)
            traj_xt_list.append(traj_xt)
            traj_x1_list.append(traj_x1)
        return upd_mols, traj_xt_list, traj_x1_list

    @torch.no_grad()
    def _step(
        self,
        mols: list[Chem.Mol],
        curr_step: int,
        is_last: bool = False,
        return_traj: bool = False,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Predict Binding Pose in an autoregressive manner

        Parameters
        ----------
        mols : list[Chem.Mol]
            molecule w/o dummy atoms
        curr_step : int
            current generation step
        is_last : bool
            whether the generation is finished
        return_traj : bool
            whether return all flow matching trajectory

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            - trajectory of xt      [num_trajs, num_atoms, 3]
            - trajectory of \\hatx1 [num_trajs, num_atoms, 3]
        """
        datas = [self.get_ligand_data(mol) for mol in mols]
        data_list = [data[0] for data in datas]
        gen_order_list = [data[1] for data in datas]

        # sort with data size to minimize padding
        lengths = [len(orders) for orders in gen_order_list]
        sorted_indices = sorted(range(len(datas)), key=lambda i: lengths[i])
        inverse_indices = sorted(range(len(sorted_indices)), key=lambda i: sorted_indices[i])

        # create loader
        sorted_data_list = [data_list[i] for i in sorted_indices]
        sorted_gen_order_list = [gen_order_list[i] for i in sorted_indices]
        loader = self.dm.iterator(sorted_data_list, sorted_gen_order_list)

        sorted_xt_traj_list: list[np.ndarray] = []
        sorted_x1_traj_list: list[np.ndarray] = []
        for (curr, prior_coords), gen_steps in loader:
            # set coordinates of newly added atoms to prior
            newly_added = gen_steps == curr_step
            curr["coords"][newly_added, :] = prior_coords[newly_added, :]

            # Move all the data to device
            curr = {k: v.to(self.device) for k, v in curr.items()}
            gen_steps = gen_steps.to(self.device)

            # Match the batch size
            num_data = gen_steps.shape[0]
            holo = {k: v[:num_data] for k, v in self.holo_data.items()}

            # flow matching inference for binding pose prediction
            rescale = lambda pos: self.dm.rescale_coords(pos.cpu().numpy())  # noqa
            process = lambda v: (rescale(v[0]), rescale(v[1]))  # noqa

            trajectory = self.run_flow_matching(curr, holo, gen_steps, curr_step, is_last, return_traj)
            rescaled_trajectory = list(map(process, trajectory))
            traj_for_each_ligand = [[(xt[i], x1[i]) for xt, x1 in rescaled_trajectory] for i in range(num_data)]

            # add conformer
            masks = curr["mask"].cpu().bool().numpy()
            for i in range(num_data):
                mask = masks[i]
                traj = traj_for_each_ligand[i]
                xt_traj = np.stack([xt[mask] for xt, _ in traj], axis=0, dtype=np.float_)
                x1_traj = np.stack([x1[mask] for _, x1 in traj], axis=0, dtype=np.float_)
                sorted_xt_traj_list.append(xt_traj)
                sorted_x1_traj_list.append(x1_traj)

        # reordering
        xt_traj_list = [sorted_xt_traj_list[i] for i in inverse_indices]
        x1_traj_list = [sorted_x1_traj_list[i] for i in inverse_indices]
        return xt_traj_list, x1_traj_list

    def run_flow_matching(
        self,
        curr: dict[str, Tensor],
        holo: dict[str, Tensor],
        gen_steps: Tensor,
        curr_step: int,
        is_last: bool,
        return_traj: bool,
    ) -> list[tuple[Tensor, Tensor]]:
        """model inference for binding pose prediction

        Parameters
        ----------
        curr : dict[str, Tensor]
            current states of molecules
        holo : dict[str, Tensor]
            states of protein (repeated)
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
        """
        # Compute the start and end times for each interpolation interval
        curr_time = curr_step * self.cfg.t_per_ar_action
        gen_times = torch.clamp(gen_steps * self.cfg.t_per_ar_action, max=self.cfg.max_action_t)

        # If we are at the last step, we need to make sure that the end time is 1.0
        if is_last:
            end_time = 1.0
        else:
            end_time = (curr_step + 1) * self.cfg.t_per_ar_action

        # flow matching inference
        num_fm_steps = max(1, int((end_time - curr_time) / (1.0 / self.cfg.num_inference_steps)))
        trajectory = self.model._step_interval_inference(
            curr, gen_times, num_fm_steps, curr_time, end_time, holo=holo, return_traj=return_traj
        )
        return trajectory

    def set_gen_order(self, mol: Chem.Mol, step: int):
        gen_orders: list[int] = []
        coords = mol.GetConformer().GetPositions()
        for aidx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(aidx)
            pos = coords[aidx]
            if (pos == 0.0).all():
                order = step
                atom.SetIntProp("gen_order", order)
            else:
                if atom.HasProp("gen_order"):
                    order = atom.GetIntProp("gen_order")
                else:
                    order = step - 1  # e.g., C-[*] -> the information of C is removed during the rxn
                    atom.SetIntProp("gen_order", order)
            gen_orders.append(order)
        return gen_orders

    def get_ligand_data(self, mol: Chem.Mol):
        g = GeometricMol.from_rdkit(mol)
        gen_orders: list[int] = [atom.GetIntProp("gen_order") for atom in mol.GetAtoms()]
        return g, gen_orders

    def encode_pocket(self) -> dict[str, Tensor]:
        """Perform pocket encoding (gvp) during initialization
        See `_compute_complex_model_args()` in semlaflow.model.complex_fm
        Instead perform pocket encoding of the fixed pocket for each sample,
        this pre-calculate the pocket encoding when API is initialized.
        """

        holo_pock, holo_mol = self.dm.transform_holo(self.pocket)
        holo_data = self.dm._batch_to_dict(GeometricMolBatch.from_list([holo_mol]))
        holo_data = {k: v.to(self.device) for k, v in holo_data.items()}

        # pre-calculate pocket encoding
        pocket_encoder = self.model.pocket_encoder
        if pocket_encoder is not None:
            if isinstance(pocket_encoder, PocketEncoder):
                holo_atomics_emb = pocket_encoder.forward_single(holo_pock, holo_data)
            else:
                raise ValueError(type(pocket_encoder), "pocket encoder should be PocketEncoder")
            holo_data["atomics"] = holo_atomics_emb

        # prevent the redundant pocket encoding
        self.model.pocket_encoder = None

        # repeat data for batching
        # [1, ...] -> [Nbatch, ...]
        for key, val in holo_data.items():
            holo_data[key] = val.repeat((MAX_NUM_BATCH,) + (1,) * (val.dim() - 1))

        if self.device is not torch.device("cpu"):
            torch.cuda.empty_cache()

        return holo_data

    def construct_model(self, ckpt_path: str | Path | None = None):
        from semlaflow.buildutil.build_model import get_categorical_config, get_complex_semla_model

        cat_config = get_categorical_config(self.cfg, self.vocab)
        integrator = Integrator(
            self.cfg.num_inference_steps,
            type_strategy=cat_config.sampling_strategy,
            bond_strategy=cat_config.sampling_strategy,
            cat_noise_level=self.cfg.cat_sampling_noise_level,
            type_mask_index=cat_config.type_mask_index,
            bond_mask_index=cat_config.bond_mask_index,
        )

        egnn_gen = get_complex_semla_model(self.cfg, self.vocab, cat_config)
        pocket_enc = PocketEncoder()
        fm_model = ARComplexMolecularCFM(
            ar_interpolant=self.dm.interpolant,
            pocket_encoder=pocket_enc,
            use_gvp=(self.cfg.pocket_encoding == "gvp"),
            use_complex_metrics=False,
            gen=egnn_gen,
            vocab=self.vocab,
            lr=self.cfg.lr,
            integrator=integrator,
            coord_scale=PLINDER_COORDS_STD_DEV,
            dist_loss_weight=self.cfg.dist_loss_weight,
            type_loss_weight=self.cfg.type_loss_weight,
            bond_loss_weight=self.cfg.bond_loss_weight,
            charge_loss_weight=self.cfg.charge_loss_weight,
            pairwise_metrics=False,
            use_ema=self.cfg.use_ema,
            compile_model=False,
            self_condition=self.cfg.self_condition,
            distill=False,
            type_mask_index=integrator.type_mask_index,
            bond_mask_index=integrator.bond_mask_index,
        )

        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            fm_model.load_state_dict(checkpoint["state_dict"])
        return fm_model.to(self.device)


class SemlaFlow_DM(GeometricInterpolantDM):
    def __init__(
        self,
        cfg: SemlaFlowConfig,
        vocab: Vocabulary,
        pocket: ProteinPocket,
        device: torch.device,
    ):

        self.cfg: SemlaFlowConfig = cfg
        self.vocab: Vocabulary = vocab
        self.coord_center = torch.mean(pocket.coords, dim=0).numpy()
        self.coord_std: float = PLINDER_COORDS_STD_DEV
        self.device: torch.device = device

        self.interpolant = SemlaFlow_Interpolant(
            # default
            vocab=vocab,
            decomposition_strategy=cfg.decomposition_strategy,
            ordering_strategy=cfg.ordering_strategy,
            coord_noise_std=cfg.coord_noise_std_dev,
            t_per_ar_action=cfg.t_per_ar_action,
            max_action_t=cfg.max_action_t,
            max_interp_time=cfg.max_interp_time,
            max_num_cuts=cfg.max_num_cuts,
            # for test
            fixed_time=0.9,
        )

        super().__init__(
            train_dataset=None,
            val_dataset=None,
            test_dataset=None,
            batch_size=BATCH_COST,
            test_interpolant=self.interpolant,
            bucket_limits=GEOM_DRUGS_BUCKET_LIMITS if self.c_alpha_only else PLINDER_BUCKET_LIMITS,
            bucket_cost_scale=cfg.bucket_cost_scale,
            pad_to_bucket=False,
            num_workers=cfg.num_workers,
        )

    def iterator(
        self, mols: list[GeometricMol], gen_orders: list[list[int]]
    ) -> Generator[tuple[tuple[dict[str, Tensor], Tensor], Tensor]]:
        lengthes = [len(v) for v in gen_orders]
        sample_idcs = [[] for _ in BUCKET_SIZES]
        for idx, size in enumerate(lengthes):
            in_bucket = False
            for k, threshold in enumerate(BUCKET_SIZES):
                if size < threshold:
                    sample_idcs[k].append(idx)
                    in_bucket = True
                    break
            if not in_bucket:
                sample_idcs[-1].append(idx)

        def get_batch(batch_idxs):
            batch_mols = [self.transform_ligand(mols[i]) for i in batch_idxs]
            batch_gen_orders = [gen_orders[i] for i in batch_idxs]
            return (self.collate_data(batch_mols), self.collate_gen_orders(batch_gen_orders))

        for bucket, bucket_cost in zip(sample_idcs, BUCKET_SIZES, strict=True):
            curr_cost = 0
            batch = []
            for idx in bucket:
                if (curr_cost > BATCH_COST) or (len(batch) == MAX_NUM_BATCH):
                    yield get_batch(batch)
                    curr_cost = 0
                    batch = []
                batch.append(idx)
                curr_cost = len(batch) * min(lengthes[idx], bucket_cost)
            if len(batch) > 0:
                yield get_batch(batch)

    def collate_data(self, data_list: list[GeometricMol]) -> tuple[dict[str, Tensor], Tensor]:
        return self._collate(data_list, dataset="test")

    def collate_gen_orders(self, gen_order_list: list[list[int]]) -> torch.Tensor:
        gen_orders = [torch.tensor(v) for v in gen_order_list]
        return pad_sequence(gen_orders, batch_first=True, padding_value=0)

    def transform_holo(self, holo: ProteinPocket):
        """transform holo structure (zero_com, scaling)"""
        holo = holo.shift(-self.coord_center)
        scaled_holo = holo.scale(1.0 / self.coord_std)

        scaled_holo = scaled_holo.select_c_alpha_atoms() if self.c_alpha_only else scaled_holo.copy()
        scaled_holo_mol = scaled_holo.to_geometric_mol()

        # One-hot encode either the C_alpha atoms or all atoms in the holo
        if self.c_alpha_only:
            res_names = scaled_holo.atoms.res_name
            atomics = smolF.one_hot_encode_tensor(smolF.aa_to_index(res_names), len(smolRD.IDX_RESIDUE_MAP))
        else:
            atomics = smolF.one_hot_encode_atomics(scaled_holo_mol.atomics, self.vocab)
        bond_types = smolF.one_hot_encode_tensor(scaled_holo_mol.bond_types, self.n_bond_types)
        charges = smolF.charge_to_index(scaled_holo_mol.charges)
        scaled_holo_mol = scaled_holo_mol._copy_with(atomics=atomics, bond_types=bond_types, charges=charges)
        return holo, scaled_holo_mol

    def transform_ligand(self, ligand: GeometricMol) -> GeometricMol:
        """transform ligand structure (zero_com, scaling)"""
        ligand = ligand.shift(-self.coord_center)
        scaled_ligand = ligand.scale(1.0 / self.coord_std)
        return scaled_ligand._copy_with(
            atomics=smolF.one_hot_encode_atomics(scaled_ligand.atomics, self.vocab),
            bond_types=smolF.one_hot_encode_tensor(scaled_ligand.bond_types, self.n_bond_types),
            charges=smolF.charge_to_index(scaled_ligand.charges),
        )

    # rescaling
    def rescale_coords(self, coords: np.ndarray):
        return self.coord_std * coords + self.coord_center.reshape(1, 1, -1)

    @cached_property
    def c_alpha_only(self) -> bool:
        return self.cfg.pocket_encoding in ["c-alpha", "gvp"]

    @cached_property
    def n_bond_types(self) -> int:
        cat_strategy = self.cfg.categorical_strategy
        n_bond_types = len(smolRD.BOND_IDX_MAP.keys()) + 1
        n_bond_types = n_bond_types + 1 if cat_strategy == "mask" else n_bond_types
        return n_bond_types


class SemlaFlow_Interpolant(ARGeometricInterpolant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        conf_coord_strategy = self.conf_noise_sampler.coord_noise
        self.conf_noise_sampler = ConfNoiseSampler_FixedSeed(coord_noise=conf_coord_strategy, lamb=1.0)

    def interpolate(self, datas: list[GeometricMol]):
        priors = []
        to_mols = []
        for to_mol in datas:
            # We generate the from_mols by copying the to_mols and using sampled coordinates
            priors.append(self.conf_noise_sampler.sample(to_mol))
            to_mols.append(to_mol)
        return to_mols, priors


class ConfNoiseSampler_FixedSeed:
    """For reproducibility of pose predictoin"""

    def __init__(self, coord_noise: str = "gaussian", lamb=1.0):
        if coord_noise not in ["gaussian", "harmonic"]:
            raise ValueError(f"coord_noise must be 'gaussian' or 'harmonic', got {coord_noise}")
        self.coord_noise = coord_noise
        self.lamb = torch.tensor([lamb])

    def sample(self, geo_mol: GeometricMol) -> Tensor:
        seed = 42 + geo_mol.seq_length
        generator = torch.Generator()
        generator.manual_seed(seed)
        noise = torch.randn(geo_mol.coords.shape, generator=generator)
        if self.coord_noise == "harmonic":
            assert geo_mol.is_connected, "Molecule must be connected for harmonic noise."
            num_nodes = geo_mol.seq_length
            try:
                D, P = HarmonicSDE.diagonalize(num_nodes, geo_mol.bond_indices, lamb=self.lamb)
            except Exception as e:
                raise ValueError(
                    (num_nodes, geo_mol.bond_indices, self.lamb), "Could not diagonalize the harmonic SDE. "
                ) from e
            # Negative eigenvalues may arise due to numerical instability
            assert torch.all(D >= -1e-5), f"Negative eigenvalues found: {D}"
            D = torch.clamp(D, min=1e-6)
            prior = P @ (noise / torch.sqrt(D)[:, None])
        elif self.coord_noise == "gaussian":
            prior = noise
        else:
            raise ValueError(self.coord_noise)

        return prior


def remove_dummy(mol: Chem.Mol) -> tuple[Chem.RWMol, list[int]]:
    non_star_idcs = [i for i, atom in enumerate(mol.GetAtoms()) if atom.GetSymbol() != "*"]
    non_star_mol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            non_star_mol.RemoveAtom(atom.GetIdx())
    return non_star_mol, non_star_idcs


def expand_coords(coords: np.ndarray, atom_indices: list[int], num_atoms: int) -> np.ndarray:
    """Expands [V', 3] coordinates to [V, 3], filling unspecified indices with zero."""
    if coords.shape[0] == num_atoms:
        return coords.astype(np.float_)
    else:
        expanded_coords = np.zeros((num_atoms, 3), dtype=np.float_)
        expanded_coords[atom_indices] = coords
        return expanded_coords.copy()


def expand_trajs(coords: np.ndarray, atom_indices: list[int], num_atoms: int) -> np.ndarray:
    """Expands [T, V', 3] coordinates to [T, V, 3], filling unspecified indices with zero."""
    if coords.shape[1] == num_atoms:
        return coords
    else:
        num_trajs = coords.shape[0]
        expanded_coords = np.zeros((num_trajs, num_atoms, 3))
        expanded_coords[:, atom_indices] = coords
        return expanded_coords.copy()
