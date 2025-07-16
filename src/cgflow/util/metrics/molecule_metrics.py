import torch
from rdkit import Chem
from torchmetrics import Metric

import cgflow.util.data.rdkit as smolRD


class GenerativeMetric(Metric):
    # TODO add metric attributes - see torchmetrics doc

    def __init__(self, **kwargs):
        # Pass extra kwargs (defined in Metric class) to parent
        super().__init__(**kwargs)

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()


class PairMetric(Metric):
    def __init__(self, **kwargs):
        # Pass extra kwargs (defined in Metric class) to parent
        super().__init__(**kwargs)

    def update(self, predicted: list[Chem.rdchem.Mol], actual: list[Chem.rdchem.Mol]) -> None:
        raise NotImplementedError()

    def compute(self) -> torch.Tensor:
        raise NotImplementedError()


class EnergyValidity(GenerativeMetric):
    n_valid: torch.Tensor
    total: torch.Tensor

    def __init__(self, optimise=False, **kwargs):
        super().__init__(**kwargs)

        self.optimise = optimise

        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        num_mols = len(mols)

        if self.optimise:
            mols = [smolRD.optimise_mol(mol) for mol in mols if mol is not None]

        energies = [smolRD.calc_energy(mol) for mol in mols if mol is not None]
        valid_energies = [energy for energy in energies if energy is not None]

        self.n_valid += len(valid_energies)
        self.total += num_mols

    def compute(self) -> torch.Tensor:
        return self.n_valid.float() / self.total


class AverageEnergy(GenerativeMetric):
    """Average energy for molecules for which energy can be calculated

    Note that the energy cannot be calculated for some molecules (specifically invalid ones) and the pose optimisation
    is not guaranteed to succeed. Molecules for which the energy cannot be calculated do not count towards the metric.

    This metric doesn't require that input molecules have been sanitised by RDKit, however, it is usually a good idea
    to do this anyway to ensure that all of the required molecular and atom properties are calculated and stored.
    """

    n_energy: torch.Tensor
    n_valid_energies: torch.Tensor

    def __init__(self, optimise: bool = False, per_atom: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.optimise: bool = optimise
        self.per_atom: bool = per_atom

        self.add_state("energy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid_energies", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        if self.optimise:
            mols = [smolRD.optimise_mol(mol) for mol in mols if mol is not None]

        energies = [smolRD.calc_energy(mol, per_atom=self.per_atom) for mol in mols if mol is not None]
        valid_energies = [energy for energy in energies if energy is not None]

        self.energy += sum(valid_energies)
        self.n_valid_energies += len(valid_energies)

    def compute(self) -> torch.Tensor:
        return self.energy / self.n_valid_energies


class AverageStrainEnergy(GenerativeMetric):
    """
    The strain energy is the energy difference between a molecule's pose and its optimised pose. Estimated using RDKit.
    Only calculated when all of the following are true:
    1. The molecule is valid and an energy can be calculated
    2. The pose optimisation succeeds
    3. The energy can be calculated for the optimised pose

    Note that molecules which do not meet these criteria will not count towards the metric and can therefore give
    unexpected results. Use the EnergyValidity metric with the optimise flag set to True to track the proportion of
    molecules for which this metric can be calculated.

    This metric doesn't require that input molecules have been sanitised by RDKit, however, it is usually a good idea
    to do this anyway to ensure that all of the required molecular and atom properties are calculated and stored.
    """

    total_energy_diff: torch.Tensor
    n_valid: torch.Tensor

    def __init__(self, per_atom: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.per_atom: bool = per_atom

        self.add_state("total_energy_diff", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        opt_mols = [(idx, smolRD.optimise_mol(mol)) for idx, mol in list(enumerate(mols)) if mol is not None]
        energies = [(idx, smolRD.calc_energy(mol, per_atom=self.per_atom)) for idx, mol in opt_mols if mol is not None]
        valids = [(idx, energy) for idx, energy in energies if energy is not None]
        if len(valids) == 0:
            return

        valid_indices = [idx for idx, _ in valids]
        valid_energies = [energy for _, energy in valids]
        original_energies = [smolRD.calc_energy(mols[idx], per_atom=self.per_atom) for idx in valid_indices]
        energy_diffs = [
            orig - opt for orig, opt in zip(original_energies, valid_energies, strict=False) if orig is not None
        ]

        self.total_energy_diff += sum(energy_diffs)
        self.n_valid += len(energy_diffs)

    def compute(self) -> torch.Tensor:
        return self.total_energy_diff / self.n_valid


class AverageOptRmsd(GenerativeMetric):
    """
    Average RMSD between a molecule and its optimised pose. Only calculated when all of the following are true:
    1. The molecule is valid
    2. The pose optimisation succeeds

    Note that molecules which do not meet these criteria will not count towards the metric and can therefore give
    unexpected results.
    """

    total_rmsd: torch.Tensor
    n_valid: torch.Tensor

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total_rmsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.rdchem.Mol]) -> None:
        valids = [(idx, smolRD.optimise_mol(mol)) for idx, mol in list(enumerate(mols)) if mol is not None]
        valids = [(idx, mol) for idx, mol in valids if mol is not None]

        if len(valids) == 0:
            return

        valid_indices = [idx for idx, _ in valids]
        opt_mols = [mol for _, mol in valids]
        original_mols = [mols[idx] for idx in valid_indices]
        rmsds = [
            smolRD.calc_rmsd(mol1, mol2, isomorphism=False) for mol1, mol2 in zip(original_mols, opt_mols, strict=False)
        ]

        self.total_rmsd += sum(rmsds)
        self.n_valid += len(rmsds)

    def compute(self) -> torch.Tensor:
        return self.total_rmsd / self.n_valid


class MolecularPairRMSD(PairMetric):
    total_rmsd: torch.Tensor
    n_valid: torch.Tensor

    def __init__(self, align=True, **kwargs):
        super().__init__(**kwargs)
        self.align = align

        self.add_state("total_rmsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predicted: list[Chem.rdchem.Mol], actual: list[Chem.rdchem.Mol]) -> None:
        valid_pairs = [
            (pred, act) for pred, act in zip(predicted, actual, strict=False) if pred is not None and act is not None
        ]
        rmsds = [smolRD.calc_rmsd(pred, act, isomorphism=True, align=self.align) for pred, act in valid_pairs]

        self.total_rmsd += sum(rmsds)
        self.n_valid += len(rmsds)

    def compute(self) -> torch.Tensor:
        return self.total_rmsd / self.n_valid


class CentroidRMSD(PairMetric):
    total_rmsd: torch.Tensor
    n_valid: torch.Tensor

    def __init__(self, align=True, **kwargs):
        super().__init__(**kwargs)
        self.align = align

        self.add_state("total_rmsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predicted: list[Chem.rdchem.Mol], actual: list[Chem.rdchem.Mol]) -> None:
        valid_pairs = [
            (pred, act) for pred, act in zip(predicted, actual, strict=False) if pred is not None and act is not None
        ]
        rmsds = [smolRD.centroid_distance(pred, act) for pred, act in valid_pairs]

        self.total_rmsd += sum(rmsds)
        self.n_valid += len(rmsds)

    def compute(self) -> torch.Tensor:
        return self.total_rmsd / self.n_valid
