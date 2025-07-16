import torch
from rdkit import Chem

# fmt: off
ATOMS: tuple[str, ...] = (
    # common
    "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I",
    # others
    "X"
)
# fmt: on
ATOM_VOCAB: dict[str, int] = {v: i for i, v in enumerate(ATOMS)}
NUM_ATOMS: int = len(ATOMS)

BOND_TYPES: dict[int, Chem.BondType] = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}
BOND_VOCAB: dict[Chem.BondType, int] = {v: i for i, v in BOND_TYPES.items()}
NUM_BOND_TYPES: int = len(BOND_TYPES) + 1  # 0: for unconnected;

CHARGES: tuple[int, ...] = (0, 1, 2, 3, -1, -2, -3)
CHARGE_VOCAB: dict[int, int] = {v: i for i, v in enumerate(CHARGES)}
NUM_CHARGES: int = len(CHARGES)

# fmt: off
RESIDUES: tuple[str, ...] = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK",
)
# fmt: on
RESIDUE_VOCAB: dict[str, int] = {v: i for i, v in enumerate(RESIDUES)}
NUM_RESIDUES: int = len(RESIDUES)


def encode_atoms(
    atoms: list[str],
    dtype: torch.dtype = torch.int32,
    allow_unk: bool = False,
) -> torch.Tensor:
    if allow_unk:
        unk_idx = NUM_ATOMS - 1
        return torch.tensor([ATOM_VOCAB.get(v, unk_idx) for v in atoms], dtype=dtype)
    else:
        assert all(v in ATOM_VOCAB for v in atoms), "Invalid atom in input"
        return torch.tensor([ATOM_VOCAB[v] for v in atoms], dtype=dtype)


def decode_atoms(tensor: torch.Tensor) -> list[str]:
    assert tensor.dtype in (torch.int16, torch.int32, torch.int64), (
        f"Tensor must be of type int16, int32 or int64, but got {tensor.dtype}"
    )
    tokens: list[int] = tensor.tolist()
    return [ATOMS[i] for i in tokens]


def encode_bond_types(bond_types: list[Chem.BondType], dtype: torch.dtype = torch.int32) -> torch.Tensor:
    assert all(v in BOND_VOCAB for v in bond_types), "Invalid atom in input"
    return torch.tensor([BOND_VOCAB[v] for v in bond_types], dtype=dtype)


def decode_bond_types(tensor: torch.Tensor) -> list[Chem.BondType]:
    assert tensor.dtype in (torch.int16, torch.int32, torch.int64), (
        f"Tensor must be of type int16, int32 or int64, but got {tensor.dtype}"
    )
    tokens: list[int] = tensor.tolist()
    return [BOND_TYPES[i] for i in tokens]


def encode_charges(charges: list[int], dtype: torch.dtype = torch.int32) -> torch.Tensor:
    assert all(v in CHARGE_VOCAB for v in charges), "Invalid atom in input"
    return torch.tensor([CHARGE_VOCAB[v] for v in charges], dtype=dtype)


def decode_charges(tensor: torch.Tensor) -> list[int]:
    assert tensor.dtype in (torch.int16, torch.int32, torch.int64), (
        f"Tensor must be of type int16, int32 or int64, but got {tensor.dtype}"
    )
    tokens: list[int] = tensor.tolist()
    return [CHARGES[i] for i in tokens]


def encode_residues(
    residues: list[str],
    dtype: torch.dtype = torch.int32,
    allow_unk: bool = False,
) -> torch.Tensor:
    if allow_unk:
        # FIXME: I miss to set allow_unk = True, so now we just convert UNK to GLY
        # unk_idx = NUM_RESIDUES - 1
        unk_idx = 0
        return torch.tensor([RESIDUE_VOCAB.get(v, unk_idx) for v in residues], dtype=dtype)
    else:
        assert all(v in RESIDUE_VOCAB for v in residues), "Invalid atom in input"
        return torch.tensor([RESIDUE_VOCAB[v] for v in residues], dtype=dtype)


def decode_residues(tensor: torch.Tensor) -> list[str]:
    assert tensor.dtype in (torch.int16, torch.int32, torch.int64), "Tensor must be of type int16, int32 or int64"
    tokens: list[int] = tensor.tolist()
    return [RESIDUES[i] for i in tokens]
