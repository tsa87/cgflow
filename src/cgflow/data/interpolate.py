from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Generic, TypeVar

import torch

import cgflow.util.data.rdkit as smolRD
import cgflow.util.misc.algorithms as smolA
import cgflow.util.misc.functional as smolF
from cgflow.data.prior_dist import PriorDistribution
from cgflow.data.time_dist import TimeDistribution
from cgflow.util.data.molrepr import GeometricMol, LigandMol, PocketMol
from cgflow.util.data.pocket import PocketComplex, ProteinPocket
from cgflow.util.registry import INTERPOLANT

_InterpT = TypeVar("_InterpT", bound="InterpT")


@dataclass
class InterpolantConfig:
    _registry_: ClassVar[str] = "interpolant"
    _type_: str


@dataclass
class GeometricInterpolantConfig(InterpolantConfig):
    noise_std: float = 0.2


@dataclass
class ARGeometricInterpolantConfig(GeometricInterpolantConfig):
    _type_: str = "ARGeometricInterpolant"
    # for decomposition_strategy
    decomposition_strategy: str = "reaction"
    ordering_strategy: str = "connected"
    max_num_cuts: int = 2  # cut up to N+1 fragments
    min_group_size: int = 5  # min fragment size.
    # for time scheduling
    t_per_ar_action: float = 0.33  # t_per_ar_action * max_num_cuts should be less than 1
    max_interp_time: float = 1.0  # if fragment is added at t, it is denoised until t + max_interp_time


@dataclass
class InterpT:
    x0: torch.Tensor
    xt: torch.Tensor
    x1: torch.Tensor
    time: float
    ligand_mol: LigandMol
    pocket_mol: PocketMol
    pocket_raw: ProteinPocket | None

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass
class GeometricInterpT(InterpT): ...


@dataclass
class ARGeometricInterpT(GeometricInterpT):
    gen_times: torch.Tensor
    rel_times: torch.Tensor
    masked_ligand_mol: LigandMol


@INTERPOLANT.register(config=InterpolantConfig)
class Interpolant(ABC, Generic[_InterpT]):
    def __init__(self, config: InterpolantConfig, prior_dist: PriorDistribution, time_dist: TimeDistribution):
        self.config: InterpolantConfig = config
        self.prior_dist: PriorDistribution = prior_dist
        self.time_dist: TimeDistribution = time_dist

    def __call__(self, complex: PocketComplex, time: float | None = None) -> _InterpT:
        """
        Interpolates the given complex at a specific time.
        If time is None, it samples a time from the distribution.
        """
        return self.interpolate(complex, time)

    @abstractmethod
    def interpolate(self, complex: PocketComplex, time: float | None = None) -> _InterpT:
        pass

    @abstractmethod
    def interpolate_traj(self, complex: PocketComplex, times: list[float]) -> list[_InterpT]:
        pass


@INTERPOLANT.register(config=GeometricInterpolantConfig)
class GeometricInterpolant(Interpolant[GeometricInterpT]):
    def __init__(self, config: GeometricInterpolantConfig, prior_dist: PriorDistribution, time_dist: TimeDistribution):
        super().__init__(config, prior_dist, time_dist)
        self.noise_std: float = config.noise_std

    def interpolate(self, complex: PocketComplex, time: float | None = None) -> GeometricInterpT:
        """Interpolates molecular coordinates"""
        time = self.time_dist.sample() if time is None else time
        ligand = complex.ligand
        x0, xt = self._interpolate_mol(ligand, time)
        return GeometricInterpT(x0, xt, ligand.coords, time, ligand, complex.holo_mol, complex.holo)

    def _interpolate_mol(self, mol: LigandMol, time: float) -> tuple[torch.Tensor, torch.Tensor]:
        coords = mol.coords
        prior = self.prior_dist.sample(coords)
        coords_mean = (prior * (1 - time)) + (coords * time)
        coords_noise = torch.randn_like(coords_mean) * self.noise_std
        interp = coords_mean + coords_noise
        return prior, interp

    def interpolate_traj(self, complex: PocketComplex, times: list[float]) -> list[GeometricInterpT]:
        # Generate the order in which atoms are generated
        mol = complex.ligand
        coords = mol.coords
        prior = self.prior_dist.sample(coords)

        traj: list[GeometricInterpT] = []

        for t in times:
            coords_mean = (prior * (1 - t)) + (coords * t)
            coords_noise = torch.randn_like(coords_mean) * self.noise_std
            interp = coords_mean + coords_noise
            traj.append(
                GeometricInterpT(
                    prior,
                    interp,
                    coords,
                    t,
                    mol,
                    complex.holo_mol,
                    None,  # complex.holo,
                )
            )
        return traj


@INTERPOLANT.register(config=ARGeometricInterpolantConfig)
class ARGeometricInterpolant(Interpolant[ARGeometricInterpT]):
    """
    This is a modified interpolant which discretizes time
    in addition, it also generates the order in which atoms are generated

    # decomposition
    decomposition_strategy: str
        The strategy used to decompose the molecule into a generation order
    ordering_strategy: str
        To enforce whether the generation order should maintain a connected molecule graph
    max_num_cuts: Optional[int]
        The maximum number of BRICS cuts to make. If not provided, the number of cuts is not limited.

    # time scheduling
    t_per_ar_action: float
        The time for each autoregressive action
    max_interp_time: float
        The maximum time for which the atom can be interpolated for. If not provided, the
        atom will be interpolated from its generation time to the end of time.
    """

    def __init__(
        self, config: ARGeometricInterpolantConfig, prior_dist: PriorDistribution, time_dist: TimeDistribution
    ):
        super().__init__(config, prior_dist, time_dist)
        self.noise_std: float = config.noise_std

        # Auto-regressive strategy
        if config.decomposition_strategy not in ["brics", "reaction", "rotatable"]:
            raise ValueError(f"decomposition strategy '{config.decomposition_strategy}' not supported.")
        if not (0.0 <= config.max_interp_time <= 1.0):
            raise ValueError("max_interp_time must be between 0 and 1 if provided.")
        assert config.max_num_cuts * config.t_per_ar_action <= 1, (
            f"max_num_cuts ({config.max_num_cuts}) * t_per_ar_action ({config.t_per_ar_action}) must be less than or equal to 1."
        )

        self.decomposition_strategy: str = config.decomposition_strategy
        self.ordering_strategy: str = config.ordering_strategy
        self.t_per_ar_action: float = config.t_per_ar_action
        self.max_interp_time: float = config.max_interp_time
        self.max_num_cuts: int = config.max_num_cuts
        self.min_group_size: int = config.min_group_size

    def interpolate(self, complex: PocketComplex, time: float | None = None) -> ARGeometricInterpT:
        time = self.time_dist.sample() if time is None else time
        ligand = complex.ligand
        x0, xt, gen_times, rel_times, masked_ligand = self._interpolate_mol(complex.ligand, time)
        return ARGeometricInterpT(
            x0, xt, ligand.coords, time, ligand, complex.holo_mol, complex.holo, gen_times, rel_times, masked_ligand
        )

    def _interpolate_mol(
        self, mol: LigandMol, time: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, LigandMol]:
        """
        returns:
            prior: the prior coordinates
            interp: the interpolated coordinates
            rel_time: the relative time of each atom in the interpolated molecule [n_atoms]
            gen_time: the generated time of each atom [n_atoms]
            masked_mol: the molecule with atoms that have not been generated yet masked out
        """
        # Generate the order in which atoms are generated
        assert self.decomposition_strategy in ["brics", "reaction", "rotatable"]
        # FIXME: remove this assertion
        assert self.decomposition_strategy == "reaction", (
            "For debugging purposes, only reaction decomposition is supported, Please remove this assertion."
        )
        mol = mol.copy()
        is_attachments, gen_orders = self._break_mol(mol)  # noqa
        mol.attachments = is_attachments

        # The time for each atom to be generated
        # We clamp the generated times to the maximum action time
        gen_times = gen_orders * self.t_per_ar_action

        # Get the relative time of each atom
        # measured by the time since the atom was generated
        # as a fraction of the total time it will be interpolated for
        rel_times = self._compute_rel_time(time, gen_times)
        # indicates whether the atom has been generated yet
        is_gen = rel_times >= 0

        # Mask out the atoms that have not been generated yet
        masked_mol = mol.get_masked_mol(is_gen)
        gen_times = gen_times[is_gen]
        rel_times = rel_times[is_gen]

        # Interpolate coords and add gaussian noise
        prior = self.prior_dist(masked_mol.coords)
        w = rel_times.unsqueeze(1)  # (1, 1)
        coords_mean = (prior * (1 - w)) + masked_mol.coords * w  # (V, 3)
        coords_noise = torch.randn_like(coords_mean) * self.noise_std
        interp = coords_mean + coords_noise
        return prior, interp, gen_times, rel_times, masked_mol

    def interpolate_traj(self, complex: PocketComplex, times: list[float]) -> list[ARGeometricInterpT]:
        # Generate the order in which atoms are generated
        assert self.decomposition_strategy in ["brics", "reaction", "rotatable"]
        mol = complex.ligand
        is_attachments, gen_orders = self._break_mol(mol)  # noqa
        gen_times = gen_orders * self.t_per_ar_action

        prior = self.prior_dist(mol.coords)
        traj: list[ARGeometricInterpT] = []
        for t in times:
            rel_times = self._compute_rel_time(t, gen_times)
            is_gen = rel_times >= 0

            # We should mask out the atoms that have not been generated yet
            _masked_atoms = mol.atoms[is_gen]
            _masked_charges = mol.charges[is_gen]
            _masked_coords = mol.coords[is_gen]
            _masked_is_attachments = is_attachments[is_gen]
            _masked_prior = prior[is_gen]
            _masked_gen_times = gen_times[is_gen]
            _masked_rel_times = rel_times[is_gen]

            # we also need to mask bond indices and types
            _masked_adj = mol.adjacency[is_gen][:, is_gen]
            _masked_bond_indices, _masked_bond_types = smolF.bonds_from_adj(_masked_adj)
            masked_mol = LigandMol(
                _masked_atoms,
                _masked_charges,
                _masked_bond_indices,
                _masked_bond_types,
                _masked_coords,
                _masked_is_attachments,
            )

            # Interpolate coords and add gaussian noise
            w = _masked_rel_times.unsqueeze(1)  # (V, 1)
            coords_mean = (_masked_prior * (1 - w)) + _masked_coords * w  # (V, 3)
            coords_noise = torch.randn_like(coords_mean) * self.noise_std
            interp_coords = coords_mean + coords_noise

            interp = ARGeometricInterpT(
                prior,
                interp_coords,
                _masked_coords,
                t,
                mol,
                complex.holo_mol,
                None,  # complex.holo,
                _masked_gen_times,
                _masked_rel_times,
                masked_mol,
            )
            traj.append(interp)
        return traj

    def _break_mol(self, mol: GeometricMol) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the generation order for the molecule"""
        rdkit_mol = mol.to_rdkit(sanitise=True)
        if rdkit_mol is None:
            print("Fail to convert GeometricMol to RDKit mol")
            return torch.zeros(mol.seq_length, dtype=torch.bool), torch.zeros(mol.seq_length, dtype=torch.long)

        try:
            attachment_indices, group, group_connectivity = smolRD.get_decompose_assignment(
                rdkit_mol, self.decomposition_strategy, self.max_num_cuts, self.min_group_size
            )
        except Exception:
            rdkit_smi = smolRD.smiles_from_mol(rdkit_mol)
            print(f"Could not decompose molecule {rdkit_smi}")
            return torch.zeros(mol.seq_length, dtype=torch.bool), torch.zeros(mol.seq_length, dtype=torch.int32)

        # Generate an order for the groups
        num_atoms = len(group)
        num_groups = max(group.values()) + 1

        if self.ordering_strategy == "random":
            group_order = torch.randperm(num_groups)
        elif self.ordering_strategy == "connected":
            group_order = smolA.sample_connected_trajectory_bfs(group_connectivity)
        else:
            raise ValueError(f"Unknown ordering strategy: {self.ordering_strategy}")

        # Compute a generation order index for each atom
        gen_order = torch.zeros(num_atoms, dtype=torch.int32)
        for i, group_idx in group.items():
            gen_order[i] = group_order[group_idx]

        # Get attachments
        is_attachments = torch.zeros(num_atoms, dtype=torch.bool)
        is_attachments[list(attachment_indices)] = True

        return is_attachments, gen_order

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
        rel_time = torch.clamp(rel_time, max=1)
        return rel_time
