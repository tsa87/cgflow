import copy
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import networkx as nx
import numpy as np
import torch
from rdkit import Chem

import cgflow.util.data.rdkit as smolRD
import cgflow.util.data.vocab as smolV
import cgflow.util.misc.functional as smolF
from cgflow.util.dataclasses import BaseTensor, LigandTensor, PocketTensor

# Type aliases
_T = torch.Tensor
TCoord = tuple[float, float, float]

# Constants
PICKLE_PROTOCOL = 4

# **********************
# *** Util functions ***
# **********************


def _check_type(obj, obj_type, name="object"):
    if not isinstance(obj, obj_type):
        raise TypeError(
            f"{name} must be an instance of {obj_type} or one of its subclasses, got {type(obj)}"
        )


def _check_shape(tensor: _T, shape: tuple[int, ...], name="object"):
    assert tensor.shape == shape, f"Shape ({tensor.shape}) of {name} must be in {shape}"


def _check_shape_len(tensor, allowed, name="object"):
    num_dims = len(tensor.size())
    allowed = [allowed] if isinstance(allowed, int) else allowed
    if num_dims not in allowed:
        raise RuntimeError(
            f"Number of dimensions of {name} must be in {str(allowed)}, got {num_dims}"
        )


def _check_shapes_equal(t1, t2, dims=None):
    if dims is None:
        if t1.size() != t2.size():
            raise RuntimeError(
                f"objects must have the same shape, got {t1.shape} and {t2.shape}"
            )
        else:
            return

    if isinstance(dims, int):
        dims = [dims]

    t1_dims = [t1.size(dim) for dim in dims]
    t2_dims = [t2.size(dim) for dim in dims]
    if t1_dims != t2_dims:
        raise RuntimeError(
            f"Expected dimensions {str(dims)} to match, got {t1.size()} and {t2.size()}"
        )


def _check_dict_key(map, key, dict_name="dictionary"):
    if key not in map:
        raise RuntimeError(f"{dict_name} must contain key {key}")


# *************************
# *** MolRepr Interface ***
# *************************


class SmolMol(ABC):
    """Interface for molecule representations for the Smol library"""

    node_keys: list[str] = []
    edge_keys: list[str] = []
    attr_keys: list[str] = node_keys + edge_keys

    def __init__(self, str_id: str | None):
        self._str_id: str | None = str_id

    # *** Interface util functions for all molecule representations ***

    def __len__(self):
        return self.seq_length

    def __str__(self):
        if self._str_id is not None:
            return self._str_id
        return super().__str__()

    @property
    def str_id(self):
        return self.__str__()

    # for save/load
    def to_bytes(self) -> bytes:
        dict_repr: dict[str, Any] = {"id": self._str_id}
        for k in self.attr_keys:
            v = getattr(self, k)
            if isinstance(v, _T):
                v = v.numpy()
            dict_repr[k] = v
        byte_obj = pickle.dumps(dict_repr, protocol=PICKLE_PROTOCOL)
        return byte_obj

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        obj = pickle.loads(data)
        # TODO: remove this once we have a new version of the data
        if "atomics" in obj:
            obj["atoms"] = obj["atomics"]

        _check_type(obj, dict, "unpickled object")
        for k in ["id"] + cls.attr_keys:
            _check_dict_key(obj, k)

        attrs = {}
        for k in cls.attr_keys:
            v = obj[k]
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            attrs[k] = v

        return cls(str_id=obj["id"], **attrs)

    # *** Conversion functions for molecule objects ***
    @property
    @abstractmethod
    def seq_length(self) -> int:
        ...

    @classmethod
    @abstractmethod
    def from_rdkit(cls, mol: Chem.Mol, remove_hs: bool = True) -> Self:
        ...

    @abstractmethod
    def to_rdkit(self) -> Chem.Mol:
        ...

    # *** Copy functions ***
    # Note: only performs a shallow copy
    def copy(self) -> Self:
        return copy.copy(self)

    @abstractmethod
    def copy_with(self, *args) -> Self:
        ...


# *******************************
# *** MolRepr Implementations ***
# *******************************


class GeometricMol(SmolMol):
    node_keys: list[str] = ["atoms", "charges", "coords"]
    edge_keys: list[str] = ["bond_indices", "bond_types"]
    attr_keys: list[str] = node_keys + edge_keys

    def __init__(
        self,
        atoms: _T,
        charges: _T,
        bond_indices: _T,
        bond_types: _T,
        coords: _T,
        str_id: str | None = None,
    ):
        super().__init__(str_id)

        # Check that each tensor has correct number of dimensions
        Natom = atoms.shape[0]
        Nbond = bond_indices.shape[0]
        _check_shape(atoms, (Natom, ), "atoms")
        _check_shape(coords, (Natom, 3), "coords")
        _check_shape(charges, (Natom, ), "coords")
        _check_shape(bond_indices, (Nbond, 2), "bond indices")
        _check_shape(bond_types, (Nbond, ), "bond types")
        self.atoms: _T = atoms.int()
        self.charges: _T = charges.int()
        self.bond_indices: _T = bond_indices.int()
        self.bond_types: _T = bond_types.int()
        self.coords: _T = coords.float()

        assert (self.atoms >= 0).all(), "atom type error"
        # assert (self.charges >= 0).all(), "charge type error" Why did we have this?
        assert (self.bond_types >= 1).all(), "bond type error"
        assert self.coords.isfinite().all(), "coord error"

        self._adjacency: _T | None = None

        # cache object
        self._rdkit_mol: Chem.Mol | None = None
        self._sanitized_rdkit_mol: Chem.Mol | None = None

    # *** General Properties ***

    @property
    def seq_length(self) -> int:
        return self.atoms.shape[0]

    # Note: this will always return a symmetric NxN matrix
    @property
    def adjacency(self) -> _T:
        if self._adjacency is None:
            bond_indices = self.bond_indices
            bond_types = self.bond_types
            adjacency = smolF.adj_from_edges(bond_indices,
                                             bond_types,
                                             self.seq_length,
                                             symmetric=True)
            self._adjacency = adjacency
        return self._adjacency

    @property
    def com(self) -> _T:
        return self.coords.mean(dim=0)

    @property
    def ligand_mask(self) -> _T:
        """Returns a 1D mask, 1 for ligand, 2 for pocket."""
        return torch.ones(self.seq_length, dtype=torch.int)

    @property
    def is_connected(self) -> bool:
        G = nx.Graph()
        G.add_nodes_from(range(self.seq_length))  # Add all nodes
        G.add_edges_from(self.bond_indices.tolist())  # Add edges
        return nx.is_connected(G)

    # *** Interface Methods ***

    @classmethod
    def from_rdkit(cls, mol: Chem.Mol, remove_hs: bool = True) -> Self:
        assert mol.GetNumConformers(
        ) > 0, "Molecule must have at least one conformer."
        if remove_hs:
            mol = Chem.RemoveHs(mol)
        smiles = smolRD.smiles_from_mol(mol)

        coords = np.array(mol.GetConformer().GetPositions(), dtype=np.float32)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
        bond_indices = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                        for bond in mol.GetBonds()]
        bond_types = [bond.GetBondType() for bond in mol.GetBonds()]

        # convert to tensor
        coords_t = torch.from_numpy(coords)
        atoms_t = smolV.encode_atoms(atoms, torch.int32)
        charges_t = smolV.encode_charges(charges, torch.int32)
        bond_indices_t = torch.tensor(bond_indices,
                                      dtype=torch.int32).view(-1, 2)
        bond_types_t = smolV.encode_bond_types(bond_types, torch.int32)
        return cls(atoms_t,
                   charges_t,
                   bond_indices_t,
                   bond_types_t,
                   coords_t,
                   str_id=smiles)

    @classmethod
    def from_sdf(cls, sdf_file: str | Path) -> Self:
        with open(sdf_file) as f:
            sdf_str = f.read()
        mol = Chem.MolFromMolBlock(sdf_str)
        assert mol is not None, "Could not parse SDF string."
        return cls.from_rdkit(mol)

    def to_rdkit(self, sanitise: bool = False) -> Chem.Mol:
        atoms = smolV.decode_atoms(self.atoms)
        charges = smolV.decode_charges(self.charges)
        bond_types = smolV.decode_bond_types(self.bond_types)
        bond_indices = self.bond_indices.tolist()
        coords = self.coords.double().numpy()
        mol = smolRD.construct_mol(atoms, charges, bond_indices, bond_types,
                                   coords, sanitise)
        return mol

    def copy_with(self, coords: _T) -> Self:
        obj = self.copy()
        obj.coords = coords
        return obj

    def zero_com(self) -> Self:
        shifted = self.coords - self.com.unsqueeze(0)
        return self.copy_with(coords=shifted)

    def rotate(self, rotation: np.ndarray | TCoord) -> Self:
        rotated = smolF.rotate(self.coords, rotation)
        return self.copy_with(coords=rotated)

    def shift(self, shift: float | np.ndarray | TCoord) -> Self:
        shift_tensor = torch.tensor(shift).view(1, 3)
        shifted = self.coords + shift_tensor
        return self.copy_with(coords=shifted)

    def scale(self, scale: float) -> Self:
        scaled = self.coords * scale
        return self.copy_with(coords=scaled)

    def to_tensor(self) -> BaseTensor:
        """Convert the molecule to a tensor representation."""
        raise NotImplementedError()


class LigandMol(GeometricMol):
    node_keys: list[str] = ["atoms", "charges", "coords", "stems"]

    def __init__(
        self,
        atoms: _T,
        charges: _T,
        bond_indices: _T,
        bond_types: _T,
        coords: _T,
        attachments: _T | None = None,
        str_id: str | None = None,
    ):
        super().__init__(atoms, charges, bond_indices, bond_types, coords,
                         str_id)
        Natom = self.seq_length
        if attachments is None:
            attachments = torch.zeros_like(self.atoms, dtype=torch.bool)
        _check_shape(attachments, (Natom, ), "attachments")
        self.attachments: _T = attachments.bool()

    def to_tensor(self) -> LigandTensor:
        """Convert the molecule to a tensor representation."""
        return LigandTensor(self.atoms, self.charges, self.attachments,
                            self.adjacency, self.coords)

    def get_masked_mol(self, mask: _T) -> Self:
        atoms = self.atoms[mask]
        charges = self.charges[mask]
        coords = self.coords[mask]
        is_attachments = self.attachments[mask]
        adjacency = self.adjacency[mask][:, mask]
        bond_indices, bond_types = smolF.bonds_from_adj(adjacency)
        return self.__class__(atoms, charges, bond_indices, bond_types, coords,
                              is_attachments)


class PocketMol(GeometricMol):
    node_keys: list[str] = [
        "atom", "charges", "coords", "residues", "residue_ids"
    ]
    edge_keys: list[str] = ["bond_indices", "bond_types"]
    attr_keys: list[str] = node_keys + edge_keys

    def __init__(
        self,
        atoms: _T,
        charges: _T,
        bond_indices: _T,
        bond_types: _T,
        coords: _T,
        residues: _T,
        residue_ids: _T,
        str_id: str | None = None,
    ):
        super().__init__(atoms, charges, bond_indices, bond_types, coords,
                         str_id)
        Natom = self.seq_length
        _check_shape(residues, (Natom, ), "residues")
        _check_shape(residue_ids, (Natom, ), "residue_ids")
        self.residues: _T = residues.int()
        self.residue_ids: _T = residue_ids.int()

    def to_tensor(self) -> PocketTensor:
        """Convert the molecule to a tensor representation."""
        return PocketTensor(self.atoms, self.charges, self.residues,
                            self.residue_ids, self.adjacency, self.coords)
