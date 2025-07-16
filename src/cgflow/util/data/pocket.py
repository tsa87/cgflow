import copy
import pickle
import tempfile
from pathlib import Path
from typing import Any, Self

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np
import torch
from Bio.PDB import PDBIO, PDBParser, Select
from biotite.structure import AtomArray
from biotite.structure.bonds import BondList
from rdkit import Chem

import cgflow.util.data.vocab as smolV
from cgflow.util.data.molrepr import LigandMol, PocketMol

# Type aliases
_T = torch.Tensor
TCoord = tuple[float, float, float]

TDevice = torch.device | str

PICKLE_PROTOCOL = 4
BIOTITE_QUADRUPLE_BOND_IDX = int(struc.BondType.QUADRUPLE)
BIOTITE_AROMATIC_BOND_START_IDX = int(struc.BondType.AROMATIC_SINGLE)
assert BIOTITE_QUADRUPLE_BOND_IDX == 4
assert BIOTITE_AROMATIC_BOND_START_IDX == 5  # (aro-single, aro-double, aro-triple, aro)

# **********************
# *** Util functions ***
# **********************


def _check_type(obj, obj_types, name="object"):
    if not isinstance(obj, obj_types):
        raise TypeError(
            f"{name} must be an instance of {obj_types} or one of its subclasses, got {type(obj)}"
        )


def _check_dict_key(map, key, dict_name="dictionary"):
    if key not in map:
        raise RuntimeError(f"{dict_name} must contain key {key}")


# ************************
# *** Module functions ***
# ************************


class ResidueFilterSelect(Select):

    def __init__(self, verbose=False):
        super().__init__()
        self.seen_atoms = {}  # Track unique atoms
        self.verbose = verbose

    def accept_residue(self, residue):
        # Check if the residue contains all of "N", "CA", "C", and "O"
        atom_names = {atom.get_name() for atom in residue}
        required_atoms = {"N", "CA", "C", "O"}
        is_complete = required_atoms.issubset(atom_names)
        if not is_complete and self.verbose:
            print(
                f"Residue {residue.get_resname()} {residue.get_id()} is missing atoms {required_atoms - atom_names}"
            )

        return is_complete

    def accept_atom(self, atom):
        # Get the residue containing this atom
        residue = atom.get_parent()

        # Unique identifier for the residue: (chain ID, residue ID)
        residue_id = (residue.get_parent().get_id(), residue.get_id())

        # Initialize a set to track atom names for this residue if not already done
        if residue_id not in self.seen_atoms:
            self.seen_atoms[residue_id] = set()

        # Check if the atom name is already in the residue's atom set
        atom_name = atom.get_name()
        if atom_name in self.seen_atoms[residue_id]:
            if self.verbose:
                print(
                    f"Duplicate atom {atom_name} in residue {residue.get_resname()} {residue.get_id()}"
                )
            return False  # Duplicate atom in the same residue
        else:
            # Add the atom name to the residue's atom set and accept it
            self.seen_atoms[residue_id].add(atom_name)
            return True


# **********************************
# *** Pocket and Complex Classes ***
# **********************************

# TODO implement own version of AtomArray and BondArray for small molecules
# Use these for Smol molecule implementations

# TODO make atoms and bonds internal and don't expose them when implementing fleshed-out version


class ProteinPocket:

    def __init__(self,
                 atom_array: AtomArray,
                 bond_list: BondList,
                 str_id: str | None = None):
        self._check_atom_array(atom_array)
        self._check_bond_list(bond_list)
        if "charge" not in atom_array.get_annotation_categories():
            atom_array.add_annotation("charge", np.float32)

        self.atom_array: AtomArray = atom_array
        self.bond_list: BondList = bond_list
        self.str_id: str | None = str_id

    @property
    def seq_length(self) -> int:
        return self.n_residues

    @property
    def n_residues(self) -> int:
        residues = list(self.iterate_chain_residues())
        return len(set(residues))

    def iterate_chain_residues(self) -> list[tuple[str, int]]:
        return list(
            zip(self.atom_array.chain_id, self.atom_array.res_id, strict=True))

    @property
    def coords(self) -> _T:
        return torch.as_tensor(self.atom_array.coord, dtype=torch.float32)

    @property
    def atoms(self) -> _T:
        return smolV.encode_atoms(self.atom_array.element.tolist(),
                                  allow_unk=True)

    @property
    def charges(self) -> _T:
        return smolV.encode_charges(
            self.atom_array.charge.astype(np.int32).tolist())

    @property
    def residues(self) -> _T:
        return smolV.encode_residues(self.atom_array.res_name.tolist(),
                                     allow_unk=True)

    @property
    def bond_indices(self) -> _T:
        return torch.as_tensor(self.bond_list.as_array()[:, :2],
                               dtype=torch.int32)

    @property
    def residue_ids(self) -> _T:
        """Returns unique residue ID for each atom (combination of res_id and chain_id)."""
        # Create unique residue identifiers by combining res_id and chain_id
        chain_res_ids = self.iterate_chain_residues()
        # Create a mapping from unique (res_id, chain_id) pairs to consecutive integers
        unique_res_chain_ids = list(set(chain_res_ids))
        unique_res_chain_ids.sort()  # Sort for consistent ordering
        id_map = {
            res_chain: idx
            for idx, res_chain in enumerate(unique_res_chain_ids)
        }
        # Map each atom to its residue ID
        residue_ids = [id_map[res_chain] for res_chain in chain_res_ids]
        return torch.tensor(residue_ids, dtype=torch.int32)

    # NOTE this needs to call self.bonds so that the bond types are converted to the same types we use for ligand
    @property
    def bond_types(self) -> _T:
        bond_types = torch.as_tensor(self.bond_list.as_array()[:, 2],
                                     dtype=torch.int32)
        bond_types[bond_types == 0] = smolV.BOND_VOCAB[Chem.BondType.SINGLE]
        bond_types[bond_types >=
                   BIOTITE_AROMATIC_BOND_START_IDX] = smolV.BOND_VOCAB[
                       Chem.BondType.AROMATIC]
        return bond_types

    def __len__(self) -> int:
        return self.seq_length

    # *** Subset functions ***
    def select_atoms(self,
                     mask: np.ndarray,
                     str_id: str | None = None) -> Self:
        """Select atoms in the pocket using a binary np mask. True means keep the atom. Returns a copy."""

        # Numpy will throw an error if the size of the mask doesn't match the atoms so don't handle this explicitly
        atom_struc = self.atom_array.copy()
        atom_struc.bonds = self.bond_list.copy()
        atom_subset = atom_struc[mask]

        bond_subset = atom_subset.bonds
        atom_subset.bonds = None

        str_id = str_id if str_id is not None else self.str_id
        return self.__class__(atom_subset, bond_subset, str_id)

    def select_residues_by_distance(self, centroid: tuple[float, float, float]
                                    | np.ndarray, cutoff: float) -> Self:
        """
        Select residues that have at least one atom within the given cutoff distance from the centroid.

        Args:
            centroid: The reference point (x, y, z) to measure distances from
            cutoff: Maximum distance in Angstrom for a residue to be included

        Returns:
            A new ProteinPocket containing only the selected residues
        """
        centroid_arr = np.array(centroid)

        # Calculate distances from each atom to the centroid
        distances = np.sqrt(
            np.sum((self.atom_array.coord - centroid_arr)**2, axis=1))

        # Get residue IDs and chain IDs for all atoms
        res_chain_ids = [(c, str(r)) for c, r in self.iterate_chain_residues()]
        unique_res_chain_ids = list(set(res_chain_ids))

        # For each residue, check if any of its atoms are within the cutoff
        residues_to_keep = []
        for chain_id, res_id in unique_res_chain_ids:
            # Create a mask for atoms in this residue
            residue_mask = np.array([(c == chain_id) and (str(r) == res_id)
                                     for c, r in self.iterate_chain_residues()
                                     ])

            # Check if any atom in this residue is within cutoff
            if np.any(distances[residue_mask] <= cutoff):
                residues_to_keep.append((chain_id, res_id))

        # Create a mask for all atoms in the residues we want to keep
        selection_mask = np.array([(c, str(r)) in residues_to_keep
                                   for c, r in self.iterate_chain_residues()])

        # Return a new pocket with only the selected residues
        return self.select_atoms(selection_mask)

    def select_c_alpha_atoms(self) -> Self:
        """Select only the C-alpha atoms in the pocket. Returns a copy."""

        c_alpha_mask = self.atom_array.atom_name == "CA"
        return self.select_atoms(c_alpha_mask)

    # *** Geometric functions ***

    def rotate(self, rotation: TCoord | np.ndarray) -> Self:
        rotated = struc.rotate(self.atom_array, rotation)
        return self.copy_with(atom_array=rotated)

    def shift(self, shift: tuple[float, float, float] | np.ndarray) -> Self:
        shift_arr = np.array(shift).reshape(3)
        shifted = struc.translate(self.atom_array, shift_arr)
        return self.copy_with(atom_array=shifted)

    def scale(self, scale: float) -> Self:
        atoms = self.atom_array.copy()
        atoms.coord *= scale
        return self.copy_with(atom_array=atoms)

    # *** Conversion functions ***

    @classmethod
    def from_pocket_atoms(cls,
                          atom_array: AtomArray,
                          infer_res_bonds: bool = False) -> Self:
        # Will either infer bonds or bonds will be taken from the atoms (bonds on atoms could be None)
        if infer_res_bonds:
            bond_list = struc.connect_via_residue_names(atom_array,
                                                        inter_residue=True)
        else:
            bond_list = atom_array.bonds
        return cls(atom_array, bond_list)

    @classmethod
    def from_protein(
        cls,
        structure: AtomArray,
        chain_id: int,
        res_ids: list[int],
        infer_res_bonds: bool = False,
    ) -> Self:
        chain = structure[structure.chain_id == chain_id]
        atom_array = chain[np.isin(chain.res_id, res_ids)]
        return cls.from_pocket_atoms(atom_array,
                                     infer_res_bonds=infer_res_bonds)

    @staticmethod
    def sanitize_pdb_read(pdb_file: str | Path) -> Any:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("structure", pdb_file)
        io = PDBIO()
        io.set_structure(structure)
        with tempfile.NamedTemporaryFile() as f:
            io.save(f.name, ResidueFilterSelect())
            file = pdb.PDBFile.read(f.name)
        return file

    @classmethod
    def from_pdb(cls,
                 pdb_file: str | Path,
                 infer_res_bonds: bool = False,
                 sanitize: bool = True) -> Self:
        if sanitize:
            file = cls.sanitize_pdb_read(pdb_file)
        else:
            file = pdb.PDBFile.read(pdb_file)
        atoms = file.get_structure()

        # Multiple chains detected. Using first chain.
        if len(atoms[0]) > 1:
            atoms = atoms[0]
            assert len(atoms) > 1, "The first chain has no more than one atom."

        return cls.from_pocket_atoms(atoms, infer_res_bonds=infer_res_bonds)

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        obj = pickle.loads(data)
        _check_dict_key(obj, "atom_array")
        _check_dict_key(obj, "bond_list")
        _check_dict_key(obj, "str_id")

        atom_array = obj["atom_array"]
        bond_list = obj["bond_list"]
        str_id = obj["str_id"]
        if str_id is None:
            str_id = str(hash(data))
        return cls(atom_array, bond_list, str_id=str_id)

    def to_bytes(self) -> bytes:
        dict_repr = {
            "atom_array": self.atom_array,
            "bond_list": self.bond_list,
            "str_id": self.str_id
        }
        byte_obj = pickle.dumps(dict_repr, protocol=PICKLE_PROTOCOL)
        return byte_obj

    def to_geometric_mol(self) -> PocketMol:
        """Convert pocket to Smol GeometricMol format"""
        return PocketMol(
            self.atoms,
            self.charges,
            self.bond_indices,
            self.bond_types,
            self.coords,
            self.residues,
            self.residue_ids,
        )

    # *** IO functions ***

    def write_pdb(self,
                  filepath: str | Path,
                  include_bonds: bool = False) -> None:
        """Ensure all chains have a one-character chain ID before writing a valid PDB file."""

        # Ensure chain IDs are single characters
        unique_chains = list(set(
            self.atom_array.chain_id))  # Get unique chain IDs
        chain_map = {
            chain: chr(65 + i)
            for i, chain in enumerate(unique_chains[:26])
        }  # Map to 'A'-'Z'

        # Assign new single-character chain IDs
        self.atom_array.chain_id = [
            chain_map[chain] for chain in self.atom_array.chain_id
        ]

        if include_bonds:
            self.atom_array.bonds = self.bond_list
        else:
            self.atom_array.bonds = None

        # Create and write PDB file
        pdb_file = pdb.PDBFile()
        pdb.set_structure(pdb_file, self.atom_array)
        pdb_file.write(Path(filepath))

    # *** Other helper functions ***

    def copy(self) -> Self:
        """Creates a deep copy of this object"""
        atom_copy = self.atom_array.copy()
        bond_copy = self.bond_list.copy()
        return self.__class__(atom_copy, bond_copy, self.str_id)

    def copy_with(self,
                  atom_array: AtomArray | None = None,
                  bond_list: BondList | None = None) -> Self:
        atom_copy = atom_array if atom_array is not None else self.atom_array.copy(
        )
        bond_copy = bond_list if bond_list is not None else self.bond_list.copy(
        )
        return self.__class__(atom_copy, bond_copy, self.str_id)

    @staticmethod
    def _check_atom_array(atoms: AtomArray):
        annotations = atoms.get_annotation_categories()
        # coord doesn't exist in annotations but should always be in atom array
        # so no need to check for coords
        # Check required annotations are provided
        _check_dict_key(annotations, "res_name", "atom array")
        _check_dict_key(annotations, "element", "atom array")

    @staticmethod
    def _check_bond_list(bonds: BondList):
        bond_arr = bonds.as_array()
        bond_types = bond_arr[:, 2]
        if np.any(bond_types == BIOTITE_QUADRUPLE_BOND_IDX):
            raise RuntimeError("Quadruple bonds are not supported.")


class PocketComplex:

    def __init__(
        self,
        holo: ProteinPocket,
        ligand: LigandMol,
        holo_mol: PocketMol
        | None = None,  # GeometricMol representation of holo pocket
        interactions: np.ndarray | None = None,
        metadata: dict | None = None,
        device: TDevice | None = None,
    ):
        metadata = {} if metadata is None else metadata

        PocketComplex._check_interactions(interactions, holo, ligand)

        self.ligand = ligand
        self.interactions = interactions
        self.metadata = metadata

        self.holo: ProteinPocket = holo
        self._holo_mol: PocketMol | None = holo_mol

        self._device = device

    @property
    def holo_mol(self) -> PocketMol:
        assert self._holo_mol is not None, "Holo mol is not set. Use to_geometric_mol() to set it."
        return self._holo_mol

    def __len__(self) -> int:
        return self.seq_length

    @property
    def seq_length(self) -> int:
        return len(self.holo) + len(self.ligand)

    @property
    def ligand_length(self) -> int:
        return len(self.ligand)

    @property
    def pocket_length(self) -> int:
        return len(self.holo)

    @property
    def system_id(self) -> str:
        return self.metadata.get("system_id")

    @property
    def str_id(self) -> str:
        return self.ligand.str_id

    @property
    def ligand_mask(self) -> _T:
        """Returns a 1D mask, 1 for ligand, 2 for pocket."""

        mask = ([1] * len(self.ligand)) + ([2] * len(self.holo))
        return torch.tensor(mask, dtype=torch.int)

    @property
    def holo_com(self) -> _T:
        return torch.mean(self.holo.coords, dim=0, dtype=torch.float32)

    @property
    def ligand_com(self) -> _T:
        return torch.mean(self.ligand.coords, dim=0, dtype=torch.float32)

    def to_bytes(self) -> bytes:
        dict_repr = {
            "holo": self.holo.to_bytes(),
            "ligand": self.ligand.to_bytes(),
            "interactions": self.interactions,
            "metadata": self.metadata,
        }

        byte_obj = pickle.dumps(dict_repr, protocol=PICKLE_PROTOCOL)
        return byte_obj

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        obj = pickle.loads(data)
        if isinstance(obj, tuple):
            lig_bytes, holo_bytes = obj
            obj = {"ligand": lig_bytes, "holo": holo_bytes, "metadata": None}

        _check_dict_key(obj, "holo")
        _check_dict_key(obj, "ligand")

        holo = ProteinPocket.from_bytes(obj["holo"])
        ligand = LigandMol.from_bytes(obj["ligand"])
        interactions = obj.get("interactions")
        metadata = obj.get("metadata")
        return cls(holo, ligand, interactions=interactions, metadata=metadata)

    def copy_with(
        self,
        holo: ProteinPocket | None = None,
        ligand: LigandMol | None = None,
        holo_mol: PocketMol | None = None,
        interactions: np.ndarray | None = None,
    ) -> Self:
        holo_copy = self.holo.copy() if holo is None else holo
        ligand_copy = self.ligand.copy() if ligand is None else ligand
        if holo_mol is None:
            holo_mol_copy = self.holo_mol.copy(
            ) if self._holo_mol is not None else None
        else:
            holo_mol_copy = holo_mol

        if interactions is None:
            interactions_copy = np.copy(
                self.interactions) if self.interactions is not None else None
        else:
            interactions_copy = interactions

        metadata_copy = copy.deepcopy(self.metadata)

        complex_copy = self.__class__(
            holo_copy,
            ligand_copy,
            holo_mol=holo_mol_copy,
            interactions=interactions_copy,
            metadata=metadata_copy,
            device=self._device,
        )
        return complex_copy

    @staticmethod
    def _check_interactions(interactions, holo, ligand):
        if interactions is None:
            return

        int_shape = tuple(interactions.shape)

        if int_shape[0] != len(holo):
            err = f"Dim 0 of interactions must match the length of the holo pocket, got {int_shape[0]} and {len(holo)}"
            raise ValueError(err)

        if int_shape[1] != len(ligand):
            err = f"Dim 1 of interactions must match the length of the ligand, got {int_shape[0]} and {len(ligand)}"
            raise ValueError(err)

    # *** Geometric functions ***
    def zero_holo_com(self) -> Self:
        # Shift the complex so that the holo com is at the origin
        shift = self.holo_com.numpy() * -1
        return self.shift(shift)

    def zero_ligand_com(self) -> Self:
        # Shift the complex so that the holo com is at the origin
        shift = self.ligand_com.numpy() * -1
        return self.shift(shift)

    def rotate(self, rotation: TCoord | np.ndarray) -> Self:
        return self.copy_with(
            holo=self.holo.rotate(rotation),
            ligand=self.ligand.rotate(rotation),
            interactions=self.interactions,
        )

    def shift(self, shift: TCoord | np.ndarray) -> Self:
        return self.copy_with(
            holo=self.holo.shift(shift),
            ligand=self.ligand.shift(shift),
            interactions=self.interactions,
        )

    def scale(self, scale: float) -> Self:
        return self.copy_with(
            holo=self.holo.scale(scale),
            ligand=self.ligand.scale(scale),
            interactions=self.interactions,
        )
