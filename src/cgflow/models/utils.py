import torch
from rdkit import Chem

import cgflow.util.data.rdkit as smolRD
import cgflow.util.data.vocab as smolV
import cgflow.util.misc.functional as smolF
from cgflow.util.dataclasses import LigandBatch, LigandTensor

_T = torch.Tensor


def mols_from_batch(batch: LigandBatch, sanitise=True) -> list[Chem.Mol]:
    tensors = batch.to_tensors()
    return [mol_from_tensor(t, sanitise=sanitise) for t in tensors]


def mol_from_tensor(ligand: LigandTensor, sanitise: bool = True) -> Chem.Mol:
    ligand = ligand.to("cpu")
    bond_indices, bond_types = smolF.bonds_from_adj(ligand.adjacency)
    return smolRD.construct_mol(
        smolV.decode_atoms(ligand.atoms),
        smolV.decode_charges(ligand.charges),
        bond_indices.tolist(),
        smolV.decode_bond_types(bond_types),
        ligand.coords.double().numpy(),
        sanitise,
    )


class Integrator:
    def __init__(self, noise_std: float = 0.0, eps: float = 1e-5):
        self.noise_std: float = noise_std
        self.eps: float = eps

    def step(
        self, curr: LigandBatch, predicted: LigandBatch, t: _T | float, step_size: float, end_t: _T | None = None
    ) -> LigandBatch:
        # end_t is optional tensor of shape [batch_size, number_of_atoms]
        # t is a tensor of shape [batch_size]
        device = curr.device

        # if end_t is None, we assume that the end time is 1
        if end_t is None:
            end_t = torch.ones((1, 1), device=device)  # [batch_size, number_of_atoms]

        # Compute the time left for each atom's interpolation
        if isinstance(t, torch.Tensor):
            assert t.ndim <= 1, "Time tensor must be scalar or 1D"
            t = t.view(-1, 1)
        time_left = end_t - t  # [batch_size, number_of_atoms]

        # to avoid numerical issue (fp16), we directly calculate delta(xt) from xt and x1-hat
        step = step_size / time_left
        # if time_left < step_size, we should just set the coords to the predicted coords
        step = torch.clamp(step, max=1.0)
        delta_coords = (predicted.coords - curr.coords) * step.unsqueeze(-1)  # [batch_size, number_of_atoms, 3]

        if self.noise_std > 0:
            raise ValueError("Here we use ODE sampler")
            noise = torch.randn_like(delta_coords) * self.noise_std
            delta_coords = delta_coords + noise * step_size
        return curr.copy_with(coords=curr.coords + delta_coords)
