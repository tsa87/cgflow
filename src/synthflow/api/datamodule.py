from collections.abc import Generator

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

import cgflow.util.functional as smolF
from cgflow.data.datamodules import GeometricInterpolantDM
from cgflow.data.interpolate import ARGeometricInterpolant, HarmonicSDE
from cgflow.util.molrepr import GeometricMol
from cgflow.util.pocket import ProteinPocket
from cgflow.util.tokeniser import Vocabulary

from .cfg import CGFlowConfig
from .utils import PLINDER_COORDS_STD_DEV

BATCH_COST = 8000
BUCKET_SIZES = [16, 32, 48, 64]
MAX_NUM_BATCH = 64


class CGFlow_DM(GeometricInterpolantDM):
    def __init__(
        self,
        cfg: CGFlowConfig,
        vocab: Vocabulary,
        device: torch.device,
    ):

        self.cfg: CGFlowConfig = cfg
        self.vocab: Vocabulary = vocab
        self.coord_std: float = PLINDER_COORDS_STD_DEV
        self.device: torch.device = device

        self.interpolant = CGFlow_Interpolant(
            # default
            vocab=vocab,
            decomposition_strategy=cfg.decomposition_strategy,
            ordering_strategy=cfg.ordering_strategy,
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
            num_workers=0,
            bucket_limits=[32, 64],  # dummy
            bucket_cost_scale="linear",  # dummy
            pad_to_bucket=False,  # dummy
        )

    def set_center(self, center: np.ndarray):
        self._tmp_coord_center = center

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

    def transform_pocket(self, pocket: ProteinPocket) -> GeometricMol:
        """transform pocket structure (zero_com, scaling)"""
        scaled_pocket = pocket.shift(-self._tmp_coord_center).scale(1.0 / self.coord_std)
        scaled_pocket_mol = scaled_pocket.to_geometric_mol()

        # One-hot encode either the C_alpha atoms or all atoms in the pocket
        scaled_pocket_mol = scaled_pocket_mol._copy_with(
            atomics=smolF.atomics_to_index(scaled_pocket_mol.atomics, self.vocab),
            charges=smolF.charge_to_index(scaled_pocket_mol.charges),
        )
        return scaled_pocket_mol

    def transform_ligand(self, ligand: GeometricMol) -> GeometricMol:
        """transform ligand structure (zero_com, scaling)"""
        ligand = ligand.shift(-self._tmp_coord_center)
        scaled_ligand = ligand.scale(1.0 / self.coord_std)
        return scaled_ligand._copy_with(
            atomics=smolF.atomics_to_index(scaled_ligand.atomics, self.vocab),
            charges=smolF.charge_to_index(scaled_ligand.charges),
        )

    # rescaling
    def rescale_coords(self, coords: np.ndarray):
        return self.coord_std * coords + self._tmp_coord_center.reshape(1, 1, -1)


class CGFlow_Interpolant(ARGeometricInterpolant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        conf_coord_strategy = self.conf_noise_sampler.coord_noise
        self.conf_noise_sampler = ConfNoiseSampler_FixedSeed(coord_noise=conf_coord_strategy, lamb=1.0)

    def interpolate(self, datas: list[GeometricMol]) -> tuple[list[dict[str, Tensor]], list[Tensor]]:
        priors = []
        to_mols = []
        for to_mol in datas:
            # We generate the from_mols by copying the to_mols and using sampled coordinates
            priors.append(self.conf_noise_sampler.sample(to_mol))
            to_mols.append(to_mol)
        return to_mols, priors


class ConfNoiseSampler_FixedSeed:
    """For reproducibility of pose prediction
    """

    def __init__(self, coord_noise: str = "gaussian", lamb=1.0):
        if coord_noise not in ["gaussian", "harmonic"]:
            raise ValueError(f"coord_noise must be 'gaussian' or 'harmonic', got {coord_noise}")
        self.coord_noise = coord_noise
        self.lamb = torch.tensor([lamb])

    def sample(self, geo_mol: GeometricMol) -> Tensor:
        # use fixed seed for deterministic prediction
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
