from copy import deepcopy
from typing import Self

from torch import Tensor

import cgflow.util.rdkit as smolRD
from cgflow.util.tokeniser import Vocabulary

GEOM_COORDS_STD_DEV = 2.407038688659668
PLINDER_COORDS_STD_DEV = GEOM_COORDS_STD_DEV  # 2.2693647416252976

_SPECIAL_TOKENS = ["<PAD>", "<MASK>"]
_CORE_ATOMS = ["H", "C", "N", "O", "F", "P", "S", "Cl"]
_OTHER_ATOMS = ["Br", "B", "Al", "Si", "As", "I", "Hg", "Bi"]

_TOKENS = _SPECIAL_TOKENS + _CORE_ATOMS + _OTHER_ATOMS
N_ATOM_TYPES = len(_TOKENS)
N_EXTRA_ATOM_FEATS = 2  # times, relative times
N_RESIDUE_TYPES = len(smolRD.IDX_RESIDUE_MAP)
N_BOND_TYPES = len(smolRD.BOND_IDX_MAP) + 1
N_CHARGE_TYPES = len(smolRD.CHARGE_IDX_MAP)


def build_vocab():
    return Vocabulary(_TOKENS)


class ComplexBatch:
    def __init__(self, ligands: dict[str, Tensor], pockets: dict[str, Tensor]):
        self.ligands: dict[str, Tensor] = ligands  # batch
        self.pockets: dict[str, Tensor] = pockets  # batch

        self.complexs: dict[str, Tensor] = {}
        self.complexs["adj_matrix"] = self.ligand_atom_mask.unsqueeze(2) & self.pocket_atom_mask.unsqueeze(1)

    @property
    def ligand_coords(self) -> Tensor:
        return self.ligands["coords"]

    @property
    def ligand_atom_types(self) -> Tensor:
        return self.ligands["atomics"]

    @property
    def ligand_bond_types(self) -> Tensor:
        return self.ligands["bonds"]

    @property
    def ligand_atom_mask(self) -> Tensor:
        # TODO: implement when mask is None
        assert self.ligands["mask"] is not None
        return self.ligands["mask"]

    @property
    def pocket_coords(self) -> Tensor:
        return self.pockets["coords"]

    @property
    def pocket_equis(self) -> Tensor | None:
        return self.pockets.get("equis", None)

    @property
    def pocket_invs(self) -> Tensor | None:
        return self.pockets.get("invs", None)

    @property
    def pocket_atom_mask(self) -> Tensor:
        # TODO: implement when mask is None
        assert self.pockets["mask"] is not None
        return self.pockets["mask"]

    def clone(self, deep_copy: bool = False) -> Self:
        if deep_copy:
            return deepcopy(self)
        else:
            return self.__class__(
                self.ligands.copy(),
                self.pockets.copy(),
            )
