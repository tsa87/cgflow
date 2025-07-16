import torch
from rdkit import Chem
from torch import Tensor


# Helper functions
def remove_dummy(mol: Chem.Mol) -> tuple[Chem.RWMol, list[bool]]:
    """return a new mol without dummy atoms."""
    is_valid: list[bool] = [atom.GetSymbol() != "*" for atom in mol.GetAtoms()]
    mol_wo_dummy = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            mol_wo_dummy.RemoveAtom(atom.GetIdx())
    # return RWMol to avoid sanitizie issue
    return mol_wo_dummy, is_valid


def extend_tensor(x: Tensor, is_valid: list[bool], is_batched: bool = False) -> Tensor:
    """Extend the tensor with padding
    - is_batched=True  : [V', ...] to [V, ...]
    - is_batched=False : [B,V', ...] to [B,V, ...]
    """
    num_atoms = len(is_valid)
    if is_batched:
        if x.shape[1] == num_atoms:
            return x
        else:
            expanded_x = torch.zeros((x.shape[0], num_atoms, *x.shape[2:]), dtype=x.dtype)
            expanded_x[:, is_valid] = x
            return expanded_x
    else:
        if x.shape[0] == num_atoms:
            return x
        else:
            expanded_x = torch.zeros((num_atoms, *x.shape[1:]), dtype=x.dtype)
            expanded_x[is_valid] = x
            return expanded_x


def pad_tensors(tensors: list[Tensor], pad_dim: int = 0) -> Tensor:
    """Pad a list of tensors with zeros

    All dimensions other than pad_dim must have the same shape. A single tensor is returned with the batch dimension
    first, where the batch dimension is the length of the tensors list.

    Args:
        tensors (list[Tensor]): List of tensors
        pad_dim (int): Dimension on tensors to pad. All other dimensions must be the same size.

    Returns:
        Tensor: Batched, padded tensor, if pad_dim is 0 then shape [B, L, *] where L is length of longest tensor.
    """

    if pad_dim != 0:
        # TODO
        raise NotImplementedError()

    padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    return padded
