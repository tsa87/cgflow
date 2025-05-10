import torch
import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment

import semlaflow.util.functional as smolF
from semlaflow.util.harmonic import HarmonicSDE
import semlaflow.util.rdkit as smolRD
import semlaflow.util.algorithms as smolA
from semlaflow.util.tokeniser import Vocabulary
from semlaflow.util.molrepr import SmolMol, SmolBatch, GeometricMol, GeometricMolBatch
from semlaflow.util.pocket import PocketComplex

SCALE_OT_FACTOR = 0.2

_InterpT = tuple[list[SmolMol], list[SmolMol], list[SmolMol], torch.Tensor]
_GeometricInterpT = tuple[list[GeometricMol], list[GeometricMol], list[GeometricMol], torch.Tensor]


class Interpolant(ABC):

    @property
    @abstractmethod
    def hparams(self):
        raise NotImplementedError

    @abstractmethod
    def interpolate(self, to_batch: list[SmolMol]) -> _InterpT:
        pass


class NoiseSampler(ABC):

    @property
    def hparams(self):
        raise NotImplementedError

    @abstractmethod
    def sample_molecule(self, num_atoms: int) -> SmolMol:
        pass

    @abstractmethod
    def sample_batch(self, num_atoms: list[int]) -> SmolBatch:
        pass


class CoordNoiseSampler:

    def __init__(self, coord_noise: str = "gaussian"):
        if coord_noise != "gaussian":
            raise NotImplementedError(f"Coord noise {coord_noise} is not supported.")
        self.coord_noise = coord_noise
        self.coord_dist = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

    def sample_coords(self, n_atoms: int) -> torch.Tensor:
        coords = self.coord_dist.sample((n_atoms, 3))
        return coords


class ConfNoiseSampler:

    def __init__(self, coord_noise: str = "gaussian", lamb=1.0):
        if coord_noise not in ["gaussian", "harmonic"]:
            raise ValueError(f"coord_noise must be 'gaussian' or 'harmonic', got {coord_noise}")
        self.coord_noise = coord_noise
        self.lamb = torch.tensor([lamb])

    def sample(self, geo_mol: GeometricMol):
        if self.coord_noise == "harmonic":
            assert geo_mol.is_connected, "Molecule must be connected for harmonic noise."
            num_nodes = geo_mol.seq_length
            try:
                D, P = HarmonicSDE.diagonalize(num_nodes, geo_mol.bond_indices, lamb=self.lamb)
            except Exception as e:
                print(num_nodes, geo_mol.bond_indices, self.lamb)
                raise ValueError("Could not diagonalize the harmonic SDE.") from e

            # Negative eigenvalues may arise due to numerical instability
            assert torch.all(D >= -1e-5), f"Negative eigenvalues found: {D}"
            D = torch.clamp(D, min=1e-6)

            noise = torch.randn_like(geo_mol.coords)
            prior = P @ (noise / torch.sqrt(D)[:, None])

        elif self.coord_noise == "gaussian":
            prior = torch.randn_like(geo_mol.coords)

        return prior


class GeometricNoiseSampler(NoiseSampler):

    def __init__(
        self,
        vocab_size: int,
        n_bond_types: int,
        coord_noise: str = "gaussian",
        type_noise: str = "uniform-sample",
        bond_noise: str = "uniform-sample",
        scale_ot: bool = False,
        zero_com: bool = True,
        type_mask_index: int | None = None,
        bond_mask_index: int | None = None,
    ):
        if coord_noise != "gaussian":
            raise NotImplementedError(f"Coord noise {coord_noise} is not supported.")

        self._check_cat_noise_type(type_noise, type_mask_index, "type")
        self._check_cat_noise_type(bond_noise, bond_mask_index, "bond")

        self.vocab_size = vocab_size
        self.n_bond_types = n_bond_types
        self.coord_noise = coord_noise
        self.type_noise = type_noise
        self.bond_noise = bond_noise
        self.scale_ot = scale_ot
        self.zero_com = zero_com
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index

        self.coord_noise_sampler = CoordNoiseSampler(coord_noise)
        self.atomic_dirichlet = torch.distributions.Dirichlet(torch.ones(vocab_size))
        self.bond_dirichlet = torch.distributions.Dirichlet(torch.ones(n_bond_types))

    @property
    def hparams(self):
        return {
            "coord-noise": self.coord_noise,
            "type-noise": self.type_noise,
            "bond-noise": self.bond_noise,
            "noise-scale-ot": self.scale_ot,
            "zero-com": self.zero_com,
        }

    def sample_molecule(self, n_atoms: int) -> GeometricMol:
        # Sample coords and scale, if required
        coords = self.coord_noise_sampler.sample_coords(n_atoms)
        if self.scale_ot:
            coords = coords * np.log(n_atoms + 1) * SCALE_OT_FACTOR

        # Sample atom types
        if self.type_noise == "dirichlet":
            atomics = self.atomic_dirichlet.sample((n_atoms,))

        elif self.type_noise == "uniform-dist":
            atomics = torch.ones((n_atoms, self.vocab_size)) / self.vocab_size

        elif self.type_noise == "mask":
            atomics = torch.zeros((n_atoms, self.vocab_size), dtype=torch.float32)
            atomics[:, self.type_mask_index] = 1.0

        elif self.type_noise == "uniform-sample":
            atomics = torch.randint(0, self.vocab_size, (n_atoms,))
            atomics = smolF.one_hot_encode_tensor(atomics, self.vocab_size)

        else:
            raise ValueError(self.type_noise)

        # Create bond indices and sample bond types
        bond_indices = torch.ones((n_atoms, n_atoms)).nonzero()
        n_bonds = bond_indices.size(0)

        if self.bond_noise == "dirichlet":
            bond_types = self.bond_dirichlet.sample((n_bonds,))

        elif self.bond_noise == "uniform-dist":
            bond_types = torch.ones((n_bonds, self.n_bond_types)) / self.n_bond_types

        elif self.bond_noise == "mask":
            bond_types = torch.tensor(self.bond_mask_index).repeat(n_bonds)
            bond_types = smolF.one_hot_encode_tensor(bond_types, self.n_bond_types)

        elif self.bond_noise == "uniform-sample":
            bond_types = torch.randint(0, self.n_bond_types, size=(n_bonds,))
            bond_types = smolF.one_hot_encode_tensor(bond_types, self.n_bond_types)

        else:
            raise ValueError(self.bond_noise)

        # Create smol mol object
        mol = GeometricMol(coords, atomics, bond_indices=bond_indices, bond_types=bond_types)
        if self.zero_com:
            mol = mol.zero_com()

        return mol

    def sample_batch(self, num_atoms: list[int]) -> GeometricMolBatch:
        mols = [self.sample_molecule(n) for n in num_atoms]
        batch = GeometricMolBatch.from_list(mols)
        return batch

    def _check_cat_noise_type(self, noise_type, mask_index, name):
        if noise_type not in [
            "dirichlet",
            "uniform-dist",
            "mask",
            "uniform-sample",
            "no-change",
        ]:
            raise ValueError(f"{name} noise {noise_type} is not supported.")

        if noise_type == "mask" and mask_index is None:
            raise ValueError(f"{name}_mask_index must be provided if {name}_noise is 'mask'.")


class GeometricInterpolant(Interpolant):

    def __init__(
        self,
        prior_sampler: GeometricNoiseSampler,
        coord_interpolation: str = "linear",
        type_interpolation: str = "unmask",
        bond_interpolation: str = "unmask",
        coord_noise_std: float = 0.0,
        type_dist_temp: float = 1.0,
        equivariant_ot: bool = False,
        batch_ot: bool = False,
        time_alpha: float = 1.0,
        time_beta: float = 1.0,
        fixed_time: float | None = None,
        conf_coord_strategy: str = "gaussian",
    ):

        if fixed_time is not None and (fixed_time < 0 or fixed_time > 1):
            raise ValueError("fixed_time must be between 0 and 1 if provided.")

        if coord_interpolation != "linear":
            raise ValueError(f"coord interpolation '{coord_interpolation}' not supported.")

        if type_interpolation not in ["dirichlet", "unmask", "no-change"]:
            raise ValueError(f"type interpolation '{type_interpolation}' not supported.")

        if bond_interpolation not in ["dirichlet", "unmask", "no-change"]:
            raise ValueError(f"bond interpolation '{bond_interpolation}' not supported.")

        if type_interpolation == "no-change" or bond_interpolation == "no-change":
            if equivariant_ot:
                raise ValueError("equivariant_ot cannot be used with no-change type or bond interpolation.")
            if type_interpolation != bond_interpolation:
                raise ValueError("type and bond interpolation must be the same if one is no-change.")

        self.prior_sampler = prior_sampler
        self.coord_interpolation = coord_interpolation
        self.type_interpolation = type_interpolation
        self.bond_interpolation = bond_interpolation
        self.coord_noise_std = coord_noise_std
        self.type_dist_temp = type_dist_temp
        self.equivariant_ot = equivariant_ot
        self.align_vector = equivariant_ot
        self.batch_ot = batch_ot
        self.time_alpha = time_alpha if fixed_time is None else None
        self.time_beta = time_beta if fixed_time is None else None
        self.fixed_time = fixed_time

        self.time_dist = torch.distributions.Beta(time_alpha, time_beta)

        if self.type_interpolation == "no-change":
            self.conf_noise_sampler = ConfNoiseSampler(coord_noise=conf_coord_strategy, lamb=1.0)

    @property
    def hparams(self):
        prior_hparams = {f"prior-{k}": v for k, v in self.prior_sampler.hparams.items()}
        hparams = {
            "coord-interpolation": self.coord_interpolation,
            "type-interpolation": self.type_interpolation,
            "bond-interpolation": self.bond_interpolation,
            "coord-noise-std": self.coord_noise_std,
            "type-dist-temp": self.type_dist_temp,
            "equivariant-ot": self.equivariant_ot,
            "batch-ot": self.batch_ot,
            "time-alpha": self.time_alpha,
            "time-beta": self.time_beta,
            **prior_hparams,
        }

        if self.fixed_time is not None:
            hparams["fixed-interpolation-time"] = self.fixed_time

        return hparams

    def interpolate(self, to_mols: list[GeometricMol]) -> _GeometricInterpT:
        batch_size = len(to_mols)
        num_atoms = max([mol.seq_length for mol in to_mols])

        if self.type_interpolation != "no-change":
            from_mols = [self.prior_sampler.sample_molecule(num_atoms) for _ in to_mols]

            # Choose best possible matches for the whole batch if using batch OT
            if self.batch_ot:
                from_mols = [mol.zero_com() for mol in from_mols]
                to_mols = [mol.zero_com() for mol in to_mols]
                from_mols = self._ot_map(from_mols, to_mols)

            # Within match_mols either just truncate noise to match size of data molecule
            # Or also permute and rotate the noise to best match data molecule
            else:
                from_mols = [
                    self._match_mols(from_mol, to_mol) for from_mol, to_mol in zip(from_mols, to_mols, strict=False)
                ]

        # We set the categorical values of the from_mols to be the same as the to_mols
        elif self.type_interpolation == "no-change":
            # We generate the from_mols by copying the to_mols and using random coordinates
            # Note we don't do equivariant
            from_mols = [to_mol._copy_with(coords=self.conf_noise_sampler.sample(to_mol)) for to_mol in to_mols]
            if self.align_vector:
                # Rotate the molecules to optimal angle
                from_mols = [
                    from_mol.rotate(Rotation.align_vectors(to_mol.coords, from_mol.coords)[0])
                    for to_mol, from_mol in zip(to_mols, from_mols, strict=False)
                ]

        if self.fixed_time is not None:
            times = torch.tensor([self.fixed_time] * batch_size)
        else:
            times = self.time_dist.sample((batch_size,))

        tuples = zip(from_mols, to_mols, times.tolist(), strict=False)
        interp_mols = [self._interpolate_mol(from_mol, to_mol, t) for from_mol, to_mol, t in tuples]
        return from_mols, to_mols, interp_mols, list(times)

    def _ot_map(self, from_mols: list[GeometricMol], to_mols: list[GeometricMol]) -> list[GeometricMol]:
        """Permute the from_mols batch so that it forms an approximate mini-batch OT map with to_mols"""

        mol_matrix = []
        cost_matrix = []

        # Create matrix with to mols on outer axis and from mols on inner axis
        for to_mol in to_mols:
            best_from_mols = [self._match_mols(from_mol, to_mol) for from_mol in from_mols]
            best_costs = [self._match_cost(mol, to_mol) for mol in best_from_mols]
            mol_matrix.append(list(best_from_mols))
            cost_matrix.append(list(best_costs))

        row_indices, col_indices = linear_sum_assignment(np.array(cost_matrix))
        best_from_mols = [mol_matrix[r][c] for r, c in zip(row_indices, col_indices, strict=False)]
        return best_from_mols

    def _match_mols(self, from_mol: GeometricMol, to_mol: GeometricMol) -> GeometricMol:
        """Permute the from_mol to best match the to_mol and return the permuted from_mol"""

        if to_mol.seq_length > from_mol.seq_length:
            raise RuntimeError("from_mol must have at least as many atoms as to_mol.")

        # Find best permutation first, then best rotation
        # As done in Equivariant Flow Matching (https://arxiv.org/abs/2306.15030)

        # Keep the same number of atoms as the data mol in the noise mol
        from_mol = from_mol.permute(list(range(to_mol.seq_length)))

        if not self.equivariant_ot:
            return from_mol

        cost_matrix = smolF.inter_distances(to_mol.coords.cpu(), from_mol.coords.cpu(), sqrd=True)
        _, from_mol_indices = linear_sum_assignment(cost_matrix.numpy())
        from_mol = from_mol.permute(from_mol_indices.tolist())

        padded_coords = smolF.pad_tensors([from_mol.coords.cpu(), to_mol.coords.cpu()])
        from_mol_coords = padded_coords[0].numpy()
        to_mol_coords = padded_coords[1].numpy()

        if self.align_vector:
            rotation, _ = Rotation.align_vectors(to_mol_coords, from_mol_coords)
            from_mol = from_mol.rotate(rotation)

        return from_mol

    def _match_cost(self, from_mol: GeometricMol, to_mol: GeometricMol) -> float:
        """Calculate MSE between mol coords as a match cost"""

        sqrd_dists = smolF.inter_distances(from_mol.coords.cpu(), to_mol.coords.cpu(), sqrd=True)
        mse = sqrd_dists.mean().item()
        return mse

    def _interpolate_mol(self, from_mol: GeometricMol, to_mol: GeometricMol, t: float) -> GeometricMol:
        """Interpolates mols which have already been sampled according to OT map, if required"""

        if from_mol.seq_length != to_mol.seq_length:
            raise RuntimeError("Both molecules to be interpolated must have the same number of atoms.")

        # Interpolate coords and add gaussian noise
        coords_mean = (from_mol.coords * (1 - t)) + (to_mol.coords * t)
        coords_noise = torch.randn_like(coords_mean) * self.coord_noise_std
        coords = coords_mean + coords_noise

        if self.type_interpolation == "dirichlet":
            to_atomics = torch.softmax(to_mol.atomics / self.type_dist_temp, dim=-1)
            atomics_mean = (from_mol.atomics * (1 - t)) + (to_atomics * t)
            atomics = torch.distributions.Dirichlet(atomics_mean).sample()

        elif self.type_interpolation == "unmask":
            atom_mask = torch.rand(from_mol.seq_length) > t
            to_atomics = torch.argmax(to_mol.atomics, dim=-1)
            from_atomics = torch.argmax(from_mol.atomics, dim=-1)
            to_atomics[atom_mask] = from_atomics[atom_mask]
            atomics = smolF.one_hot_encode_tensor(to_atomics, to_mol.atomics.size(-1))

        elif self.type_interpolation == "no-change":
            atomics = to_mol.atomics

        # Interpolate bonds
        if self.bond_interpolation == "dirichlet":
            to_adj = torch.softmax(to_mol.adjacency / self.type_dist_temp, dim=-1)
            adj_mean = (from_mol.adjacency * (1 - t)) + (to_adj * t)
            interp_adj = torch.distributions.Dirichlet(adj_mean).sample()

        elif self.bond_interpolation == "unmask":
            to_adj = torch.argmax(to_mol.adjacency, dim=-1)
            from_adj = torch.argmax(from_mol.adjacency, dim=-1)
            bond_mask = torch.rand_like(from_adj.float()) > t
            to_adj[bond_mask] = from_adj[bond_mask]
            interp_adj = smolF.one_hot_encode_tensor(to_adj, to_mol.adjacency.size(-1))

        elif self.bond_interpolation == "no-change":
            interp_adj = to_mol.adjacency

        bond_indices = torch.ones((from_mol.seq_length, from_mol.seq_length)).nonzero()
        bond_types = interp_adj[bond_indices[:, 0], bond_indices[:, 1]]

        interp_mol = GeometricMol(coords, atomics, bond_indices=bond_indices, bond_types=bond_types)
        return interp_mol


class GeometricComplexInterpolant(GeometricInterpolant):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # We cannot rotate the ligand noise in the pocket
        self.align_vector = False

        if self.batch_ot:
            raise ValueError("batch_ot cannot be used with GeometricComplexInterpolant.")

    def interpolate(self, to_complexs: list[PocketComplex]):
        # We interpolate the mol and keep the pocket fixed
        to_mols = [to_complex.ligand for to_complex in to_complexs]
        from_mols, to_mols, interp_mols, t = super().interpolate(to_mols)

        holo_mols = [to_complex.holo_mol for to_complex in to_complexs]
        holo_pocks = [to_complex.holo for to_complex in to_complexs]
        return from_mols, to_mols, interp_mols, holo_mols, holo_pocks, t


class PseudoGeometricComplexInterpolant(GeometricInterpolant):
    """This interpolant is used to interpolate ligand only dataset for
    a model capable of taking in both ligand and holo molecules.
    """

    def __init__(self, align_vector=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.align_vector = align_vector

        if self.batch_ot:
            raise ValueError("batch_ot cannot be used with GeometricComplexInterpolant.")

    def interpolate(self, to_mols: list[GeometricMol]):
        from_mols, to_mols, interp_mols, t = super().interpolate(to_mols)
        holo_mols = [None for _ in to_mols]
        holo_pocks = [None for _ in to_mols]
        return from_mols, to_mols, interp_mols, holo_mols, holo_pocks, t


class ARGeometricInterpolant(Interpolant):
    """
    This is a modified interpolant which discretizes time
    in addition, it also generates the order in which atoms are generated

    decomposition_strategy: str
        The strategy used to decompose the molecule into a generation order

    ordering_strategy: str
        To enforce whether the generation order should maintain a connected molecule graph

    t_per_ar_action: float
        The time for each autoregressive action

    max_action_t: float
        The maximum time for which an autoregressive action can be selected

    max_interp_time: Optional[float]
        The maximum time for which the atom can be interpolated for. If not provided, the
        atom will be interpolated from its generation time to the end of time.

    max_num_cuts: Optional[int]
        The maximum number of BRICS cuts to make. If not provided, the number of cuts is not limited.

    """

    def __init__(
        self,
        vocab: Vocabulary,
        decomposition_strategy: str = "atom",  # brics
        ordering_strategy: str = "connected",  # random
        coord_noise_std: float = 0.0,
        time_alpha: float = 1.0,
        time_beta: float = 1.0,
        t_per_ar_action: float = 0.01,
        max_action_t: float = 1.0,
        fixed_time: float | None = None,
        max_interp_time: float | None = None,
        max_num_cuts: int | None = None,
        min_group_size: int | None = None,
        conf_coord_strategy: str = "gaussian",
    ):
        if decomposition_strategy not in ["atom", "brics", "reaction", "rotatable"]:
            raise ValueError(f"decomposition strategy '{decomposition_strategy}' not supported.")

        if max_action_t < 0.0 or max_action_t > 1:
            raise ValueError("max_action_t must be between 0 and 1 if provided.")

        if fixed_time is not None and (fixed_time < 0.0 or fixed_time > 1):
            raise ValueError("fixed_time must be between 0 and 1 if provided.")

        if max_interp_time is not None and (max_interp_time < 0.0 or max_interp_time > 1):
            raise ValueError("max_interp_time must be between 0 and 1 if provided.")

        if max_num_cuts is not None and decomposition_strategy == "atom":
            raise ValueError("max_num_cuts cannot be provided for atom decomposition.")

        self.vocab = vocab
        self.decomposition_strategy = decomposition_strategy
        self.ordering_strategy = ordering_strategy
        self.coord_noise_std = coord_noise_std
        self.time_alpha = time_alpha if fixed_time is None else None
        self.time_beta = time_beta if fixed_time is None else None
        self.t_per_ar_action = t_per_ar_action

        self.max_action_t = max_action_t
        self.fixed_time = fixed_time
        self.max_interp_time = max_interp_time
        self.max_num_cuts = max_num_cuts
        self.min_group_size = min_group_size
        self.time_dist = torch.distributions.Beta(time_alpha, time_beta)

        self.conf_noise_sampler = ConfNoiseSampler(coord_noise=conf_coord_strategy, lamb=1.0)

    @property
    def hparams(self):
        return {
            "t_per_ar_action": self.t_per_ar_action,
            "coord_noise_std": self.coord_noise_std,
            "time_alpha": self.time_alpha,
            "time_beta": self.time_beta,
        }

    def _get_gen_order(self, to_mol: GeometricMol) -> torch.Tensor:
        """Get the generation order for the molecule"""
        rdkit_mol = to_mol.to_rdkit(self.vocab, sanitise=True)
        if rdkit_mol is None:
            return torch.zeros(to_mol.seq_length, dtype=torch.long)
        rdkit_smi = smolRD.smiles_from_mol(rdkit_mol)

        try:
            group, group_connectivity = smolRD.get_decompose_assignment(
                rdkit_mol, self.decomposition_strategy, self.max_num_cuts, self.min_group_size
            )
        except Exception:
            print(f"Could not decompose molecule {rdkit_smi}")
            return torch.zeros(to_mol.seq_length, dtype=torch.long)

        # Generate an order for the groups
        num_groups = max(group.values()) + 1

        if self.ordering_strategy == "random":
            group_order = torch.randperm(num_groups)
        elif self.ordering_strategy == "connected":
            group_order = smolA.sample_connected_trajectory_bfs(group_connectivity)

        # Compute a generation order index for each atom
        gen_order = torch.zeros(len(group), dtype=torch.long)
        for i, group_idx in group.items():
            gen_order[i] = group_order[group_idx]

        return gen_order

    def interpolate_single(self, to_mol: GeometricMol, times: torch.Tensor) -> _GeometricInterpT:
        from_mol = to_mol._copy_with(coords=self.conf_noise_sampler.sample(to_mol))

        # Generate the order in which atoms are generated
        if self.decomposition_strategy == "atom":
            gen_order = torch.randperm(to_mol.seq_length)
        else:
            assert self.decomposition_strategy in ["brics", "reaction", "rotatable"]
            gen_order = self._get_gen_order(to_mol)

        gen_time = torch.clamp(gen_order * self.t_per_ar_action, max=self.max_action_t)
        res = [self._interpolate_mol(from_mol, to_mol, t, gen_time) for t in times.tolist()]
        interp_mols, masked_to_mols, rel_times = zip(*res, strict=False)

        return (
            from_mol,
            to_mol,
            interp_mols,
            masked_to_mols,
            list(times),
            list(rel_times),
            gen_time,
        )

    def interpolate(self, to_mols: list[GeometricMol], times: torch.Tensor | None = None) -> _GeometricInterpT:
        batch_size = len(to_mols)

        # We generate the from_mols by copying the to_mols and using sampled coordinates
        from_mols = [to_mol._copy_with(coords=self.conf_noise_sampler.sample(to_mol)) for to_mol in to_mols]

        if times is None:
            if self.fixed_time is not None:
                times = torch.tensor([self.fixed_time] * batch_size)
            else:
                times = self.time_dist.sample((batch_size,))

        # Generate the order in which atoms are generated
        if self.decomposition_strategy == "atom":
            gen_orders = [torch.randperm(to_mol.seq_length) for to_mol in to_mols]
        # Use RXN or BRICS to break the molecule into fragments
        else:
            assert self.decomposition_strategy in ["brics", "reaction", "rotatable"]
            gen_orders = [self._get_gen_order(to_mol) for to_mol in to_mols]

        # The time for each atom to be generated
        # We clamp the generated times to the maximum action time
        gen_times = [torch.clamp(go * self.t_per_ar_action, max=self.max_action_t) for go in gen_orders]

        tuples = zip(from_mols, to_mols, times.tolist(), gen_times, strict=False)
        res = [self._interpolate_mol(from_mol, to_mol, t, gen_time) for from_mol, to_mol, t, gen_time in tuples]
        interp_mols, masked_to_mols, rel_times = zip(*res, strict=False)

        return (
            from_mols,
            to_mols,
            interp_mols,
            masked_to_mols,
            list(times),
            list(rel_times),
            gen_times,
        )

    def _compute_rel_time(self, t: torch.Tensor, gen_times: torch.Tensor) -> torch.Tensor:
        """
        Compute the relative time of each atom in the interpolated molecule
        t = 1 means the atom is fully interpolated
        t < 0 mean the atom has not been generated yet
        """
        total_time = 1 - gen_times
        if self.max_interp_time:
            total_time = torch.clamp(total_time, max=self.max_interp_time)

        rel_time = (t - gen_times) / total_time
        rel_time = torch.clamp(rel_time, max=1)
        return rel_time

    def _interpolate_mol(
        self,
        from_mol: GeometricMol,
        to_mol: GeometricMol,
        t: float,
        gen_time: torch.Tensor,
    ) -> tuple[GeometricMol, GeometricMol, torch.Tensor]:
        """
        returns:
            interp_mol: the interpolated molecule
            masked_to_mol: the molecule with atoms that have not been generated yet masked out
            rel_time: the relative time of each atom in the interpolated molecule [n_atoms]
            is_gen: a boolean tensor indicating whether the atom has been generated yet [n_atoms]
        """

        if from_mol.seq_length != to_mol.seq_length:
            raise RuntimeError("Both molecules to be interpolated must have the same number of atoms.")

        assert gen_time.shape[0] == to_mol.coords.shape[0]

        # Get the relative time of each atom
        # measured by the time since the atom was generated
        # as a fraction of the total time it will be interpolated for

        # The total time is the time till the end or the max interpolation time
        rel_time = self._compute_rel_time(t, gen_time)

        # indicates whether the atom has been generated yet
        is_gen = rel_time >= 0
        rel_time = rel_time.unsqueeze(1)

        # Interpolate coords and add gaussian noise
        coords_mean = (from_mol.coords * (1 - rel_time)) + (to_mol.coords * rel_time)
        coords_noise = torch.randn_like(coords_mean) * self.coord_noise_std
        coords = coords_mean + coords_noise

        # We do not interpolate atomics and adjacency matrix as they
        # have been generated by the autoregressive model
        atomics = to_mol.atomics
        interp_adj = to_mol.adjacency

        # We should mask out the atoms that have not been generated yet
        coords = coords[is_gen]
        atomics = atomics[is_gen]
        interp_adj = interp_adj[is_gen][:, is_gen]
        rel_time = rel_time[is_gen]

        interp_mol_seq_length = coords.shape[0]
        # This makes bond_types a tensor of shape (n_bonds, bond_type_dim)
        bond_indices = torch.ones((interp_mol_seq_length, interp_mol_seq_length)).nonzero()
        bond_types = interp_adj[bond_indices[:, 0], bond_indices[:, 1]]

        # GeometricMol should have a start time attribute for each of its atom
        interp_mol = GeometricMol(coords, atomics, bond_indices=bond_indices, bond_types=bond_types)

        # We now also remove atoms that are not generated yet from the original to_mol
        # atomics, bond indices, and bond_types are the same for both molecules
        masked_to_mol = GeometricMol(
            to_mol.coords[is_gen],
            atomics,
            bond_indices=bond_indices,
            bond_types=bond_types,
        )

        return interp_mol, masked_to_mol, rel_time.squeeze(-1)


class ARGeometricComplexInterpolant(ARGeometricInterpolant):
    def interpolate_single(self, to_complex: PocketComplex, times: torch.Tensor):
        return super().interpolate_single(to_complex.ligand, times=times)

    def interpolate(self, to_complexs: list[PocketComplex], times: torch.Tensor | None = None) -> tuple:
        # We interpolate the mol and keep the pocket fixed
        to_mols = [to_complex.ligand for to_complex in to_complexs]
        from_mols, to_mols, interp_mols, masked_to_mols, t, rel_t, gen_t = super().interpolate(to_mols, times=times)

        holo_mols = [to_complex.holo_mol for to_complex in to_complexs]
        holo_pocks = [to_complex.holo for to_complex in to_complexs]
        return (
            from_mols,
            to_mols,
            interp_mols,
            masked_to_mols,
            holo_mols,
            holo_pocks,
            t,
            rel_t,
            gen_t,
        )


class PseudoARGeometricComplexInterpolant(ARGeometricInterpolant):
    def interpolate_single(self, to_complex: PocketComplex, times: torch.Tensor):
        return super().interpolate_single(to_complex.ligand, times=times)

    def interpolate(self, to_mols: list[GeometricMol], times: torch.Tensor | None = None) -> tuple:
        from_mols, to_mols, interp_mols, masked_to_mols, t, rel_t, gen_t = super().interpolate(to_mols, times=times)
        holo_mols = [None for _ in to_mols]
        holo_pocks = [None for _ in to_mols]
        return (
            from_mols,
            to_mols,
            interp_mols,
            masked_to_mols,
            holo_mols,
            holo_pocks,
            t,
            rel_t,
            gen_t,
        )
