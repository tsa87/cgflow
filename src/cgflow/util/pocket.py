from __future__ import annotations

import copy
import pickle
import tempfile
from collections.abc import Sequence
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
import torch
from Bio.PDB import PDBIO, PDBParser, Select
from biotite.structure import AtomArray, BondList
from rdkit import Chem

import cgflow.util.functional as smolF
import cgflow.util.rdkit as smolRD
from cgflow.util.molrepr import GeometricMol
from cgflow.util.rdkit import PeriodicTable

# Type aliases
_T = torch.Tensor
TCoord = tuple[float, float, float]

TDevice = torch.device | str

PICKLE_PROTOCOL = 4
BIOTITE_AROMATIC_BOND_START_IDX = 5

PT = PeriodicTable()

# **********************
# *** Util functions ***
# **********************


def get_dummy_pocket():
    dummy_atom = struc.Atom([0.0, 0.0, 0.0], element="C")
    dummy_atom_array = struc.array([dummy_atom])
    dummy_bond_list = struc.BondList(1)
    dummy_pocket = ProteinPocket(dummy_atom_array, dummy_bond_list)
    return dummy_pocket


def _check_type(obj, obj_types, name="object"):
    if not isinstance(obj, obj_types):
        raise TypeError(
            f"{name} must be an instance of {obj_types} or one of its subclasses, got {type(obj)}"
        )


def _check_dim_shape(tensor, dim, allowed, name="object"):
    shape = tensor.size(dim)
    allowed = [allowed] if isinstance(allowed, int) else allowed
    if shape not in allowed:
        raise RuntimeError(
            f"Shape of {name} for dim {dim} must be in {allowed}")


def _check_dict_key(map, key, dict_name="dictionary"):
    if key not in map:
        raise RuntimeError(f"{dict_name} must contain key {key}")


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


def _check_shape_len(tensor, allowed, name="object"):
    num_dims = len(tensor.size())
    allowed = [allowed] if isinstance(allowed, int) else allowed
    if num_dims not in allowed:
        raise RuntimeError(
            f"Number of dimensions of {name} must be in {str(allowed)}, got {num_dims}"
        )


def _check_shape_len_if_exists(tensor, allowed, name="object"):
    if tensor is not None:
        _check_shape_len(tensor, allowed, name)


# ************************
# *** Module functions ***
# ************************


def calc_system_metrics(system: PocketComplex) -> dict:
    metrics = {}

    # TODO calculate some holo and ligand metrics
    if system.apo is None:
        return metrics

    backbone_mask = struc.filter_peptide_backbone(system.holo.atoms)
    apo_backbone = system.apo.atoms[backbone_mask]
    holo_backbone = system.holo.atoms[backbone_mask]

    pocket_rmsd = struc.rmsd(system.apo.atoms, system.holo.atoms)
    backbone_rmsd = struc.rmsd(apo_backbone, holo_backbone)

    metrics["pocket_rmsd"] = pocket_rmsd
    metrics["backbone_rmsd"] = backbone_rmsd

    return metrics


def holo_residue_complete(holo: ProteinPocket):
    """Check if the holo has all the required atoms for a residue"""
    parser = PDBParser()

    with tempfile.NamedTemporaryFile() as f:
        holo.write_pdb(f.name)
        s = parser.get_structure("x", f.name)
        res_list = list(s.get_residues())

        for res in res_list:
            if not (("N" in res) and ("CA" in res) and ("C" in res) and
                    ("O" in res)):
                return False
        return True


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
                 atoms: AtomArray,
                 bonds: BondList,
                 str_id: str | None = None):
        self._check_atom_array(atoms)

        if "charge" not in atoms.get_annotation_categories():
            atoms.add_annotation("charge", np.float32)

        self.atoms = atoms
        self.bonds = bonds

        # if str_id is not None:
        self.str_id = str_id
        # else:
        #     self.str_id = "".join([three_to_one.get(res.resname, "X") for res in res_list])

    @property
    def seq_length(self) -> int:
        return len(self.atoms)

    @property
    def res_length(self) -> int:
        return len(
            set([
                str(res_id) + chain_id for res_id, chain_id in zip(
                    self.atoms.res_id, self.atoms.chain_id, strict=False)
            ]))

    @property
    def atom_symbols(self) -> list[str]:
        return self.atoms.element.tolist()

    @property
    def n_residues(self) -> int:
        res_ids = set(self.atoms.res_id)
        return len(res_ids)

    @property
    def coords(self) -> _T:
        return torch.tensor(self.atoms.coord)

    @property
    def atomics(self) -> _T:
        atoms = [
            "Se" if el == "SE" else el for el in self.atoms.element.tolist()
        ]
        return torch.tensor([PT.atomic_from_symbol(atom) for atom in atoms])

    @property
    def charges(self) -> _T:
        return torch.tensor(self.atoms.charge).long()

    @property
    def bonds_tensor(self) -> _T:
        bonds = torch.tensor(self.bonds.as_array().astype(np.int32))

        bond_types = bonds[:, 2]
        # TODO move this check to constructor
        # Don't support quadruple bonds
        if (bond_types == 4).sum().item() != 0:
            raise RuntimeError("Quadruple bonds are not supported.")

        # Biotite splits aromatic bonds into single and double but we follow RDKit and map everything to the same
        aromatic_bond_idx = smolRD.BOND_IDX_MAP[Chem.BondType.AROMATIC]
        bonds[:, 2][bonds[:, 2] >=
                    BIOTITE_AROMATIC_BOND_START_IDX] = aromatic_bond_idx
        return bonds

    @property
    def bond_indices(self) -> _T:
        return self.bonds_tensor[:, :2]

    # NOTE this needs to call self.bonds so that the bond types are converted to the same types we use for ligand
    @property
    def bond_types(self) -> _T:
        return self.bonds_tensor[:, 2]

    def __len__(self) -> int:
        return self.seq_length

    # *** Subset functions ***
    def select_atoms(self,
                     mask: np.ndarray,
                     str_id: str | None = None) -> ProteinPocket:
        """Select atoms in the pocket using a binary np mask. True means keep the atom. Returns a copy."""

        # Numpy will throw an error if the size of the mask doesn't match the atoms so don't handle this explicitly
        atom_struc = self.atoms.copy()
        atom_struc.bonds = self.bonds.copy()
        atom_subset = atom_struc[mask]

        bond_subset = atom_subset.bonds
        atom_subset.bonds = None

        str_id = str_id if str_id is not None else self.str_id
        pocket = ProteinPocket(atom_subset, bond_subset, str_id)
        return pocket

    def select_residues_by_distance(self, centroid: tuple[float, float, float]
                                    | np.ndarray,
                                    cutoff: float) -> ProteinPocket:
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
            np.sum((self.atoms.coord - centroid_arr)**2, axis=1))

        # Get residue IDs and chain IDs for all atoms
        res_chain_ids = [(str(res_id), chain_id) for res_id, chain_id in zip(
            self.atoms.res_id, self.atoms.chain_id, strict=False)]
        unique_res_chain_ids = list(set(res_chain_ids))

        # For each residue, check if any of its atoms are within the cutoff
        residues_to_keep = []
        for res_id, chain_id in unique_res_chain_ids:
            # Create a mask for atoms in this residue
            residue_mask = np.array([
                (str(r) == res_id) and (c == chain_id) for r, c in zip(
                    self.atoms.res_id, self.atoms.chain_id, strict=False)
            ])

            # Check if any atom in this residue is within cutoff
            if np.any(distances[residue_mask] <= cutoff):
                residues_to_keep.append((res_id, chain_id))

        # Create a mask for all atoms in the residues we want to keep
        selection_mask = np.array([
            (str(r), c) in residues_to_keep for r, c in zip(
                self.atoms.res_id, self.atoms.chain_id, strict=False)
        ])

        # Return a new pocket with only the selected residues
        return self.select_atoms(selection_mask)

    def select_c_alpha_atoms(self) -> ProteinPocket:
        """Select only the C-alpha atoms in the pocket. Returns a copy."""

        c_alpha_mask = self.atoms.atom_name == "CA"
        return self.select_atoms(c_alpha_mask)

    def add_hs(self, str_id: str | None = None) -> ProteinPocket:
        import hydride

        atom_struc = self.atoms.copy()
        atom_struc.bonds = self.bonds.copy()

        # Adding hydrogen requires removing all hydrogens first
        atom_no_hs = atom_struc[atom_struc.element != "H"]
        # Add hydrogens to the pocket
        atom_with_hs, _ = hydride.add_hydrogen(atom_no_hs)

        bonds_with_hs = atom_with_hs.bonds
        atom_with_hs.bonds = None

        str_id = str_id if str_id is not None else self.str_id
        pocket = ProteinPocket(atom_with_hs, bonds_with_hs, str_id=str_id)
        return pocket

    def remove_hs(self, str_id: str | None = None) -> ProteinPocket:
        """Returns a copy of the object with hydrogens removed"""

        atom_struc = self.atoms.copy()
        atom_struc.bonds = self.bonds.copy()
        atoms_no_hs = atom_struc[atom_struc.element != "H"]

        bonds_no_hs = atoms_no_hs.bonds
        atoms_no_hs.bonds = None

        str_id = str_id if str_id is not None else self.str_id
        pocket = ProteinPocket(atoms_no_hs, bonds_no_hs, str_id=str_id)
        return pocket

    # *** Geometric functions ***

    def rotate(self, rotation: TCoord) -> ProteinPocket:
        rotation_arr = np.array(rotation)
        rotated = struc.rotate(self.atoms, rotation_arr)
        return self.copy_with(atoms=rotated)

    def shift(self,
              shift: tuple[float, float, float] | np.ndarray) -> ProteinPocket:
        shift_arr = np.array(shift)
        shifted = struc.translate(self.atoms, shift_arr)
        return self.copy_with(atoms=shifted)

    def scale(self, scale: float) -> ProteinPocket:
        atoms = self.atoms.copy()
        atoms.coord *= scale
        return self.copy_with(atoms=atoms)

    # *** Conversion functions ***

    @staticmethod
    def from_pocket_atoms(atoms: AtomArray,
                          infer_res_bonds: bool = False) -> ProteinPocket:
        # Will either infer bonds or bonds will be taken from the atoms (bonds on atoms could be None)
        if infer_res_bonds:
            bonds = struc.connect_via_residue_names(atoms, inter_residue=True)
        else:
            bonds = atoms.bonds

        return ProteinPocket(atoms, bonds)

    @staticmethod
    def from_protein(
        structure: AtomArray,
        chain_id: int,
        res_ids: list[int],
        infer_res_bonds: bool = False,
    ) -> ProteinPocket:
        chain = structure[structure.chain_id == chain_id]
        pocket = chain[np.isin(chain.res_id, res_ids)]
        return ProteinPocket.from_pocket_atoms(pocket,
                                               infer_res_bonds=infer_res_bonds)

    @staticmethod
    def sanitize_pdb_read(pdb_file: str | Path) -> ProteinPocket:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("structure", pdb_file)
        io = PDBIO()
        io.set_structure(structure)
        with tempfile.NamedTemporaryFile() as f:
            io.save(f.name, ResidueFilterSelect())
            file = pdb.PDBFile.read(f.name)
        return file

    @staticmethod
    def from_pdb(pdb_file: str | Path,
                 infer_res_bonds: bool = False,
                 sanitize: bool = True) -> ProteinPocket:
        if sanitize:
            file = ProteinPocket.sanitize_pdb_read(pdb_file)
        else:
            file = pdb.PDBFile.read(pdb_file)

        atoms = file.get_structure()

        # Multiple chains detected. Using first chain.
        if len(atoms[0]) > 1:
            atoms = atoms[0]
            assert len(atoms) > 1, "The first chain has no more than one atom."

        return ProteinPocket.from_pocket_atoms(atoms,
                                               infer_res_bonds=infer_res_bonds)

    @staticmethod
    def from_bytes(data: bytes) -> ProteinPocket:
        obj = pickle.loads(data)

        _check_dict_key(obj, "atoms")
        _check_dict_key(obj, "bonds")
        _check_dict_key(obj, "str_id")

        atoms = obj["atoms"]
        bonds = obj["bonds"]
        str_id = obj["str_id"]

        if str_id is None:
            str_id = str(hash(data))

        pocket = ProteinPocket(atoms, bonds, str_id=str_id)
        return pocket

    def to_geometric_mol(self) -> GeometricMol:
        """Convert pocket to Smol GeometricMol format"""

        atoms = [
            "Se" if el == "SE" else el for el in self.atoms.element.tolist()
        ]
        atomics = torch.tensor([PT.atomic_from_symbol(atom) for atom in atoms])
        coords = torch.tensor(self.atoms.coord)
        charges = torch.tensor(self.atoms.charge)
        residues = smolF.aa_to_index(self.atoms.res_name.tolist())

        bonds = self.bonds_tensor.numpy()
        bond_indices = torch.tensor(bonds[:, :2].astype(np.int32))
        bond_types = torch.tensor(bonds[:, 2].astype(np.int32))

        mol = GeometricMol(coords,
                           atomics,
                           bond_indices,
                           bond_types,
                           charges=charges,
                           residues=residues)
        return mol

    def to_bytes(self) -> bytes:
        dict_repr = {
            "atoms": self.atoms,
            "bonds": self.bonds,
            "str_id": self.str_id
        }
        byte_obj = pickle.dumps(dict_repr, protocol=PICKLE_PROTOCOL)
        return byte_obj

    def to_prolif(self):
        import MDAnalysis as mda
        import prolif as plf

        # Create a copy of the holo with chain id always only char
        struc_copy = self.atoms.copy()
        struc_copy.chain_id = ["A"] * len(struc_copy)
        pocket_copy = ProteinPocket(struc_copy, self.bonds)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Write to pdb but don't include bonds since this doesn't seem to be working for pdb files
            write_path = Path(tmp_dir) / "pocket.pdb"
            pocket_copy.write_pdb(write_path, include_bonds=False)

            # Create prolif molecule by first reading using MDAnalysis
            holo_mda = mda.Universe(str(write_path.resolve()),
                                    guess_bonds=True)
            holo_mol = plf.Molecule.from_mda(holo_mda)

        return holo_mol

    # *** IO functions ***

    def write_cif(self,
                  filepath: str | Path,
                  include_bonds: bool = False) -> None:
        if include_bonds:
            self.atoms.bonds = self.bonds
        else:
            self.atoms.bonds = None

        cif_file = pdbx.CIFFile()
        pdbx.set_structure(cif_file, self.atoms, include_bonds=include_bonds)
        cif_file.write(Path(filepath))

        if include_bonds:
            self.atoms.bonds = None

    def write_pdb(self,
                  filepath: str | Path,
                  include_bonds: bool = False) -> None:
        """Ensure all chains have a one-character chain ID before writing a valid PDB file."""

        # Ensure chain IDs are single characters
        unique_chains = list(set(self.atoms.chain_id))  # Get unique chain IDs
        chain_map = {
            chain: chr(65 + i)
            for i, chain in enumerate(unique_chains[:26])
        }  # Map to 'A'-'Z'

        # Assign new single-character chain IDs
        self.atoms.chain_id = [
            chain_map[chain] for chain in self.atoms.chain_id
        ]

        if include_bonds:
            self.atoms.bonds = self.bonds
        else:
            self.atoms.bonds = None

        # Create and write PDB file
        pdb_file = pdb.PDBFile()
        pdb.set_structure(pdb_file, self.atoms)
        pdb_file.write(Path(filepath))

    # *** Other helper functions ***

    def copy(self) -> ProteinPocket:
        """Creates a deep copy of this object"""

        atom_copy = self.atoms.copy()
        bond_copy = self.bonds.copy()
        str_id_copy = self.str_id[:] if self.str_id is not None else None

        pocket_copy = ProteinPocket(atom_copy, bond_copy, str_id_copy)
        return pocket_copy

    def copy_with(self,
                  atoms: AtomArray | None = None,
                  bonds: BondList | None = None) -> ProteinPocket:

        atom_copy = atoms if atoms is not None else self.atoms.copy()
        bond_copy = bonds if bonds is not None else self.bonds.copy()
        str_id_copy = self.str_id[:] if self.str_id is not None else None

        pocket_copy = ProteinPocket(atom_copy, bond_copy, str_id_copy)
        return pocket_copy

    def _check_atom_array(self, atoms: AtomArray) -> None:
        annotations = atoms.get_annotation_categories()

        # coord doesn't exist in annotations but should always be in atom array
        # so no need to check for coords

        # Check required annotations are provided
        _check_dict_key(annotations, "res_name", "atom array")
        _check_dict_key(annotations, "element", "atom array")


class PocketComplex:

    def __init__(
        self,
        holo: ProteinPocket,
        ligand: GeometricMol,
        holo_mol: GeometricMol
        | None = None,  # GeometricMol representation of holo pocket
        apo: ProteinPocket | None = None,
        interactions: np.ndarray | None = None,
        metadata: dict | None = None,
        device: TDevice | None = None,
    ):
        metadata = {} if metadata is None else metadata

        PocketComplex._check_holo_apo_match(holo, apo)
        PocketComplex._check_interactions(interactions, holo, ligand)

        self.ligand = ligand
        self.apo = apo
        self.interactions = interactions
        self.metadata = metadata

        self.holo = holo
        self.holo_mol = holo_mol
        # If holo_mol is not provided, use holo pocket as holo_mol
        self.holo_repr = holo_mol if holo_mol is not None else holo

        self._device = device

    def __len__(self) -> int:
        return self.seq_length

    @property
    def seq_length(self) -> int:
        return len(self.holo.coords) + len(self.ligand.coords)

    @property
    def ligand_length(self) -> int:
        return len(self.ligand)

    @property
    def pocket_length(self) -> int:
        return len(self.holo_repr)

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
        return torch.mean(self.holo.coords, dim=0)

    @property
    def ligand_com(self) -> _T:
        return torch.mean(self.ligand.coords, dim=0)

    def to_bytes(self) -> bytes:
        dict_repr = {
            "holo": self.holo.to_bytes(),
            "ligand": self.ligand.to_bytes(),
            "interactions": self.interactions,
            "metadata": self.metadata,
        }

        if self.apo is not None:
            dict_repr["apo"] = self.apo.to_bytes()

        byte_obj = pickle.dumps(dict_repr, protocol=PICKLE_PROTOCOL)
        return byte_obj

    @staticmethod
    def from_bytes(data: bytes) -> PocketComplex:
        obj = pickle.loads(data)

        _check_dict_key(obj, "holo")
        _check_dict_key(obj, "ligand")
        _check_dict_key(obj, "metadata")

        holo = ProteinPocket.from_bytes(obj["holo"])
        ligand = GeometricMol.from_bytes(obj["ligand"])
        apo = ProteinPocket.from_bytes(
            obj["apo"]) if obj.get("apo") is not None else None
        interactions = obj.get("interactions")

        system = PocketComplex(holo,
                               ligand,
                               apo=apo,
                               interactions=interactions,
                               metadata=obj["metadata"])
        return system

    def remove_hs(self, include_ligand: bool = False) -> PocketComplex:
        """Remove hydrogen atoms, by default Hs are only removed from the protein."""

        if include_ligand:
            raise NotImplementedError(
                "Removing Hs from the ligand is not supported.")

        protein_atom_mask = self.holo.atoms.element != "H"
        holo_subset = self.holo.select_atoms(protein_atom_mask)

        apo_subset = self.apo.select_atoms(
            protein_atom_mask) if self.apo is not None else None
        interactions_subset = self.interactions[
            protein_atom_mask, :, :] if self.interactions is not None else None

        subset = PocketComplex(
            holo_subset,
            self.ligand,
            apo=apo_subset,
            interactions=interactions_subset,
            metadata=self.metadata,
        )
        return subset

    # TODO
    def remove_heavy_atom_hbonds(self) -> PocketComplex:
        """Remove any h-bonds where both interacting atoms are heavy"""
        raise NotImplementedError()

    def store_metrics_(self):
        self.metadata["metrics"] = calc_system_metrics(self)

    def copy_with(
        self,
        holo: ProteinPocket | None = None,
        ligand: GeometricMol | None = None,
        holo_mol: GeometricMol | None = None,
        apo: ProteinPocket | None = None,
        interactions: np.ndarray | None = None,
    ):
        holo_copy = self.holo.copy() if holo is None else holo
        ligand_copy = self.ligand.copy() if ligand is None else ligand
        if holo_mol is None:
            holo_mol_copy = self.holo_mol.copy(
            ) if self.holo_mol is not None else None
        else:
            holo_mol_copy = holo_mol

        if apo is None:
            apo_copy = self.apo.copy() if self.apo is not None else None
        else:
            apo_copy = apo

        if interactions is None:
            interactions_copy = np.copy(
                self.interactions) if self.interactions is not None else None
        else:
            interactions_copy = interactions

        metadata_copy = copy.deepcopy(self.metadata)

        complex_copy = PocketComplex(
            holo_copy,
            ligand_copy,
            holo_mol=holo_mol_copy,
            apo=apo_copy,
            interactions=interactions_copy,
            metadata=metadata_copy,
            device=self._device,
        )
        return complex_copy

    @staticmethod
    def _check_holo_apo_match(holo: ProteinPocket, apo: ProteinPocket):
        if apo is None:
            return

        # Check sizes match
        if len(holo) != len(apo):
            raise ValueError(
                f"Apo and holo must have the same number of atoms, got {len(apo)} and {len(holo)}"
            )

        # Check atom names match
        if not (holo.atoms.atom_name == apo.atoms.atom_name).all():
            raise ValueError("All apo and holo atom names must match.")

        # Check bonds match (biotite checks deep equality and return a single bool)
        if holo.bonds != apo.bonds:
            raise ValueError("All apo and holo bonds must match.")

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
    def zero_holo_com(self) -> PocketComplex:
        # Shift the complex so that the holo com is at the origin
        shift = self.holo_com.numpy() * -1
        return self.shift(shift)

    def rotate(self, rotation: TCoord) -> PocketComplex:
        rotation_arr = np.array(rotation)
        return self.copy_with(
            holo=self.holo.rotate(rotation_arr),
            ligand=self.ligand.rotate(rotation_arr),
            apo=self.apo.rotate(rotation_arr)
            if self.apo is not None else None,
            interactions=self.interactions,
        )

    def shift(self, shift: tuple[float, float, float]) -> PocketComplex:
        return self.copy_with(
            holo=self.holo.shift(shift),
            ligand=self.ligand.shift(shift),
            apo=self.apo.shift(shift) if self.apo is not None else None,
            interactions=self.interactions,
        )

    def scale(self, scale: float) -> PocketComplex:
        return self.copy_with(
            holo=self.holo.scale(scale),
            ligand=self.ligand.scale(scale),
            apo=self.apo.scale(scale) if self.apo is not None else None,
            interactions=self.interactions,
        )


class PocketComplexBatch(Sequence):

    def __init__(self, systems: list[PocketComplex]):
        for system in systems:
            _check_type(system, PocketComplex, "system")

        self._systems = systems

    # *** Useful properties ***
    @property
    def seq_length(self) -> list[int]:
        return [system.seq_length for system in self._systems]

    @property
    def batch_size(self) -> int:
        return len(self._systems)

    @property
    def coords(self) -> _T:
        coords = [system.coords for system in self._systems]
        return smolF.pad_tensors(coords)

    @property
    def atomics(self) -> _T:
        atomics = [system.atomics for system in self._systems]
        return smolF.pad_tensors(atomics)

    @property
    def charges(self) -> _T:
        charges = [system.charges for system in self._systems]
        return smolF.pad_tensors(charges)

    @property
    def bond_indices(self) -> _T:
        return self.bonds[:, :, :2]

    @property
    def bond_types(self) -> _T:
        return self.bonds[:, :, 2]

    @property
    def bonds(self) -> _T:
        bonds = [system.bonds for system in self._systems]
        return smolF.pad_tensors(bonds)

    @property
    def mask(self) -> _T:
        """Returns a tensor of shape [n_systems, n_atoms in largest system]. 1 for real atoms, 0 for padded atoms."""
        masks = [torch.ones(len(system)) for system in self._systems]
        return smolF.pad_tensors(masks)

    @property
    def ligand_mask(self) -> _T:
        """Returns a padded ligand mask for the batch. 0 for pad atoms, 1 for ligand atoms, 2 for pocket atoms."""
        masks = [system.ligand_mask for system in self._systems]
        return smolF.pad_tensors(masks)

    # *** Sequence methods ***

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, item: int) -> PocketComplex:
        return self._systems[item]

    # *** Helper methods ***
    def remove_hs(self, include_ligand: bool | None = False):
        subset_systems = [
            system.remove_hs(include_ligand=include_ligand)
            for system in self._systems
        ]
        return PocketComplexBatch(subset_systems)

    # *** IO methods ***

    def to_bytes(self) -> bytes:
        system_bytes = [system.to_bytes() for system in self._systems]
        return pickle.dumps(system_bytes)

    @staticmethod
    def from_bytes(data: bytes) -> PocketComplexBatch:
        systems = [
            PocketComplex.from_bytes(system) for system in pickle.loads(data)
        ]
        return PocketComplexBatch(systems)

    # *** Other methods ***

    @staticmethod
    def from_batches(batches: list[PocketComplexBatch]) -> PocketComplexBatch:
        all_systems = [system for batch in batches for system in batch]
        return PocketComplexBatch(all_systems)

    def to_list(self) -> list[PocketComplex]:
        return self._systems

    @staticmethod
    def from_list(complexes: list[PocketComplex]) -> PocketComplexBatch:
        return PocketComplexBatch(complexes)
