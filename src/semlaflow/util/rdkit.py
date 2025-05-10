import random
import threading

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import Mol as RDMol
from rdkit.Chem import AllChem, rdmolops


from .reaction import find_rxn_bonds, find_brics_bonds


ArrT = np.ndarray


# *************************************************************************************************
# ************************************ Periodic Table class ***************************************
# *************************************************************************************************


class PeriodicTable:
    """Singleton class wrapper for the RDKit periodic table providing a neater interface"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self):
        self._table = Chem.GetPeriodicTable()

        # Just to be certain that vocab objects are thread safe
        self._pt_lock = threading.Lock()

    def atomic_from_symbol(self, symbol: str) -> int:
        with self._pt_lock:
            symbol = symbol.upper() if len(symbol) == 1 else symbol
            atomic = self._table.GetAtomicNumber(symbol)

        return atomic

    def symbol_from_atomic(self, atomic_num: int) -> str:
        with self._pt_lock:
            token = self._table.GetElementSymbol(atomic_num)

        return token

    def valence(self, atom: str | int) -> int:
        with self._pt_lock:
            valence = self._table.GetDefaultValence(atom)

        return valence


# *************************************************************************************************
# ************************************* Global Declarations ***************************************
# *************************************************************************************************


PT = PeriodicTable()

IDX_BOND_MAP = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}
BOND_IDX_MAP = {bond: idx for idx, bond in IDX_BOND_MAP.items()}

IDX_CHARGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: -1, 5: -2, 6: -3}
CHARGE_IDX_MAP = {charge: idx for idx, charge in IDX_CHARGE_MAP.items()}

IDX_RESIDUE_MAP = {
    0: "UNK",
    1: "ALA",
    2: "ARG",
    3: "ASN",
    4: "ASP",
    5: "CYS",
    6: "GLN",
    7: "GLU",
    8: "GLY",
    9: "HIS",
    10: "ILE",
    11: "LEU",
    12: "LYS",
    13: "MET",
    14: "PHE",
    15: "PRO",
    16: "SER",
    17: "THR",
    18: "TRP",
    19: "TYR",
    20: "VAL",
}
RESIDUE_IDX_MAP = {residue: idx for idx, residue in IDX_RESIDUE_MAP.items()}


# *************************************************************************************************
# *************************************** Util Functions ******************************************
# *************************************************************************************************

# TODO merge these with check functions in other files


def _check_shape_len(arr, allowed, name="object"):
    num_dims = len(arr.shape)
    allowed = [allowed] if isinstance(allowed, int) else allowed
    if num_dims not in allowed:
        raise RuntimeError(f"Number of dimensions of {name} must be in {str(allowed)}, got {num_dims}")


def _check_dim_shape(arr, dim, allowed, name="object"):
    shape = arr.shape[dim]
    allowed = [allowed] if isinstance(allowed, int) else allowed
    if shape not in allowed:
        raise RuntimeError(f"Shape of {name} for dim {dim} must be in {allowed}, got {shape}")


# *************************************************************************************************
# ************************************* External Functions ****************************************
# *************************************************************************************************


def mol_to_sdf(mol: Chem.rdchem.Mol, filepath: str | Path):
    with Chem.SDWriter(filepath) as writer:
        for cid in range(mol.GetNumConformers()):
            writer.write(mol, confId=cid)


def mol_is_valid(mol: Chem.rdchem.Mol, with_hs: bool = True, connected: bool = True) -> bool:
    """Whether the mol can be sanitised and, optionally, whether it's fully connected

    Args:
        mol (Chem.Mol): RDKit molecule to check
        with_hs (bool): Whether to check validity including hydrogens (if they are in the input mol), default True
        connected (bool): Whether to also assert that the mol must not have disconnected atoms, default True

    Returns:
        bool: Whether the mol is valid
    """

    if mol is None:
        return False

    mol_copy = Chem.Mol(mol)
    if not with_hs:
        mol_copy = Chem.RemoveAllHs(mol_copy)

    try:
        AllChem.SanitizeMol(mol_copy)
    except Exception:
        return False

    n_frags = len(AllChem.GetMolFrags(mol_copy))
    if connected and n_frags != 1:
        return False

    return True


def calc_energy(mol: RDMol, per_atom: bool = False) -> float:
    """Calculate the energy for an RDKit molecule using the MMFF forcefield

    The energy is only calculated for the first (0th index) conformer within the molecule. The molecule is copied so
    the original is not modified.

    Args:
        mol (Chem.Mol): RDKit molecule
        per_atom (bool): Whether to normalise by number of atoms in mol, default False

    Returns:
        float: Energy of the molecule or None if the energy could not be calculated
    """

    mol_copy = Chem.Mol(mol)

    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_copy, mmffVariant="MMFF94")
        ff = AllChem.MMFFGetMoleculeForceField(mol_copy, mmff_props, confId=0)
        energy = ff.CalcEnergy()
        energy = energy / mol.GetNumAtoms() if per_atom else energy
    except Exception:
        energy = None

    return energy


def optimise_mol(mol: RDMol, max_iters: int = 1000) -> RDMol:
    """Optimise the conformation of an RDKit molecule

    Only the first (0th index) conformer within the molecule is optimised. The molecule is copied so the original
    is not modified.

    Args:
        mol (Chem.Mol): RDKit molecule
        max_iters (int): Max iterations for the conformer optimisation algorithm

    Returns:
        Chem.Mol: Optimised molecule or None if the molecule could not be optimised within the given number of
                iterations
    """

    mol_copy = Chem.Mol(mol)
    try:
        exitcode = AllChem.MMFFOptimizeMolecule(mol_copy, maxIters=max_iters)
    except Exception:
        exitcode = -1

    if exitcode == 0:
        return mol_copy

    return None


def conf_distance(
    mol1: Chem.rdchem.Mol,
    mol2: Chem.rdchem.Mol,
    fix_order: bool = True,
    align: bool = True,
) -> float:
    """Approximately align two molecules and then calculate RMSD between them

    Alignment and distance is calculated only between the default conformers of each molecule.

    Args:
        mol1 (Chem.Mol): First molecule to align
        mol2 (Chem.Mol): Second molecule to align
        fix_order (bool): Whether to fix the atom order of the molecules
        align (bool): Whether to align the molecules before calculating the RMSD
    Returns:
        float: RMSD between molecules after approximate alignment
    """

    assert len(mol1.GetAtoms()) == len(mol2.GetAtoms())

    if not fix_order:
        raise NotImplementedError()

    coords1 = np.array(mol1.GetConformer().GetPositions())
    coords2 = np.array(mol2.GetConformer().GetPositions())

    if align:
        # Firstly, centre both molecules
        coords1 = coords1 - (coords1.sum(axis=0) / coords1.shape[0])
        coords2 = coords2 - (coords2.sum(axis=0) / coords2.shape[0])

        try:
            # Find the best rotation alignment between the centred mols
            rotation, _ = Rotation.align_vectors(coords1, coords2)
            aligned_coords2 = rotation.apply(coords2)
        except np.linalg.LinAlgError:
            aligned_coords2 = coords2

    else:
        aligned_coords2 = coords2

    sqrd_dists = (coords1 - aligned_coords2) ** 2
    rmsd = np.sqrt(sqrd_dists.sum(axis=1).mean())
    return rmsd


def centroid_distance(mol1: Chem.rdchem.Mol, mol2: Chem.rdchem.Mol):
    coords1 = np.array(mol1.GetConformer().GetPositions())
    coords2 = np.array(mol2.GetConformer().GetPositions())

    centroid1 = coords1.mean(axis=0)
    centroid2 = coords2.mean(axis=0)
    return np.sqrt(((centroid1 - centroid2) ** 2).sum())


# TODO could allow more args
def smiles_from_mol(mol: Chem.rdchem.Mol, canonical: bool = True, explicit_hs: bool = False) -> str | None:
    """Create a SMILES string from a molecule

    Args:
        mol (Chem.Mol): RDKit molecule object
        canonical (bool): Whether to create a canonical SMILES, default True
        explicit_hs (bool): Whether to embed hydrogens in the mol before creating a SMILES, default False. If True
                this will create a new mol with all hydrogens embedded. Note that the SMILES created by doing this
                is not necessarily the same as creating a SMILES showing implicit hydrogens.

    Returns:
        str: SMILES string which could be None if the SMILES generation failed
    """

    if mol is None:
        return None

    if explicit_hs:
        mol = Chem.AddHs(mol)

    try:
        smiles = Chem.MolToSmiles(mol, canonical=canonical)
    except Exception:
        smiles = None

    return smiles


def mol_from_smiles(smiles: str, explicit_hs: bool = False) -> Chem.rdchem.Mol | None:
    """Create a RDKit molecule from a SMILES string

    Args:
        smiles (str): SMILES string
        explicit_hs (bool): Whether to embed explicit hydrogens into the mol

    Returns:
        Chem.Mol: RDKit molecule object or None if one cannot be created from the SMILES
    """

    if smiles is None:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol) if explicit_hs else mol
    except Exception:
        mol = None

    return mol


def embed_mol(
    mol: Chem.rdchem.Mol,
    num_confs: int = 1,
    random_seed: int = 42,
):
    try:
        AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=random_seed)
    except Exception:
        return None
    return mol


def mol_from_atoms(
    coords: ArrT,
    tokens: list[str],
    bonds: ArrT | None = None,
    charges: ArrT | None = None,
    sanitise=True,
):
    """Create RDKit mol from atom coords and atom tokens (and optionally bonds)

    If any of the atom tokens are not valid atoms (do not exist on the periodic table), None will be returned.

    If bonds are not provided this function will create a partial molecule using the atomics and coordinates and then
    infer the bonds based on the coordinates using OpenBabel. Otherwise the bonds are added to the molecule as they
    are given in the bond array.

    If bonds are provided they must not contain any duplicates.

    If charges are not provided they are assumed to be 0 for all atoms.

    Args:
        coords (np.ndarray): Coordinate tensor, shape [n_atoms, 3]
        atomics (list[str]): Atomic numbers, length must be n_atoms
        bonds (np.ndarray, optional): Bond indices and types, shape [n_bonds, 3]
        charges (np.ndarray, optional): Charge for each atom, shape [n_atoms]
        sanitise (bool): Whether to apply RDKit sanitization to the molecule, default True

    Returns:
        RDMol: RDKit molecule or None if one cannot be created
    """

    _check_shape_len(coords, 2, "coords")
    _check_dim_shape(coords, 1, 3, "coords")

    if coords.shape[0] != len(tokens):
        raise ValueError("coords and atomics tensor must have the same number of atoms.")

    if bonds is not None:
        _check_shape_len(bonds, 2, "bonds")
        _check_dim_shape(bonds, 1, 3, "bonds")

    if charges is not None:
        _check_shape_len(charges, 1, "charges")
        _check_dim_shape(charges, 0, len(tokens), "charges")

    try:
        atomics = [PT.atomic_from_symbol(token) for token in tokens]
    except Exception:
        return None

    charges = charges.tolist() if charges is not None else [0] * len(tokens)

    # Add atom types and charges
    mol = Chem.EditableMol(Chem.Mol())
    for idx, atomic in enumerate(atomics):
        atom = Chem.Atom(atomic)
        atom.SetFormalCharge(charges[idx])
        mol.AddAtom(atom)

    # Add 3D coords
    conf = Chem.Conformer(coords.shape[0])
    for idx, coord in enumerate(coords.tolist()):
        conf.SetAtomPosition(idx, coord)

    mol = mol.GetMol()
    mol.AddConformer(conf)

    if bonds is None:
        return _infer_bonds(mol)

    # Add bonds if they have been provided
    mol = Chem.EditableMol(mol)
    for bond in bonds.astype(np.int32).tolist():
        start, end, b_type = bond

        if b_type not in IDX_BOND_MAP:
            return None

        # Don't add self connections
        if start != end:
            b_type = IDX_BOND_MAP[b_type]
            mol.AddBond(start, end, b_type)

    try:
        mol = mol.GetMol()
        for atom in mol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)
    except Exception:
        return None

    if sanitise:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

    return mol


def _infer_bonds(mol: RDMol):
    coords = mol.GetConformer().GetPositions().tolist()
    coord_strs = ["\t".join([f"{c:.6f}" for c in cs]) for cs in coords]
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    xyz_str_header = f"{str(mol.GetNumAtoms())}\n\n"
    xyz_strs = [f"{str(atom)}\t{coord_str}" for coord_str, atom in zip(coord_strs, atom_symbols, strict=False)]
    xyz_str = xyz_str_header + "\n".join(xyz_strs)

    try:
        pybel_mol = pybel.readstring("xyz", xyz_str)
    except Exception:
        pybel_mol = None

    if pybel_mol is None:
        return None

    mol_str = pybel_mol.write("mol")
    mol = Chem.MolFromMolBlock(mol_str, removeHs=False, sanitize=True)
    return mol


def get_brics_assignment(mol: RDMol, max_num_cuts=np.inf):
    raise NotImplementedError("Use get_decompose_assignment with rule='brics' instead")


def get_decompose_assignment(
    mol: RDMol,
    rule: str = "brics",
    max_num_cuts: int | None = None,
    min_group_size: int | None = None,
) -> tuple[dict[int, int], dict[int, list[int]]]:
    """Get decomposed fragment assignments for each atom in the molecule
    Args:
        mol (Chem.Mol): RDKit molecule

    Returns:
        atom_idx_to_group_idx (dict[int, int]): Mapping from atom index to group index
        group_connectivity (dict[int, list[int]]): Connectivity Map btw Groups
    """
    assert rule in ["brics", "reaction", "rotatable"]

    # NOTE: Assign atom map number to preserve original atom index
    # mol.GetIdx() == mol.GetAtomMapNum() == mol_no_hs.GetAtomMapNum()
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i)
    mol_no_hs: RDMol = Chem.RemoveHs(mol)  # NOTE: Atom index changes

    # NOTE: Find bond to break down
    if rule == "brics":
        decompose_func = find_brics_bonds
    elif rule == "reaction":
        decompose_func = find_rxn_bonds
    else:
        # TODO: Implement Rotatable Bond
        raise NotImplementedError
    bond_idcs = [mol_no_hs.GetBondBetweenAtoms(b[0][0], b[0][1]).GetIdx() for b in decompose_func(mol_no_hs)]
    # NOTE: Remove duplicated bonds
    bond_idcs = list(set(bond_idcs))

    num_to_break = len(bond_idcs)
    if max_num_cuts is not None:
        num_to_break = min(max_num_cuts, num_to_break)

    # NOTE: Break down molecule to fragment groups
    frag_list: tuple[RDMol, ...]
    if num_to_break == 0:
        # NOTE: Return complete molecule
        frag_list = (mol_no_hs,)
    else:
        # NOTE: Get all permutation of bond cuts and choose one among them
        default_outcomes = rdmolops.FragmentOnSomeBonds(mol_no_hs, bond_idcs, num_to_break)

        # NOTE: Filter out outcomes with too small fragment
        if min_group_size is not None:
            # NOTE: Find the most groups satisfying min_group_size
            for i in range(num_to_break, 0, -1):
                possible_outcomes = rdmolops.FragmentOnSomeBonds(mol_no_hs, bond_idcs, i)
                valid_outcomes = [
                    outcome
                    for outcome in possible_outcomes
                    if np.all([len(group) > min_group_size for group in Chem.GetMolFrags(outcome)])
                ]
                if len(valid_outcomes) > 0:
                    break
            valid_outcomes = default_outcomes if len(valid_outcomes) == 0 else valid_outcomes
        else:
            valid_outcomes = default_outcomes

        broken_mol: RDMol = random.choice(valid_outcomes)
        frag_list = Chem.GetMolFrags(broken_mol, asMols=True)

    # NOTE: Get atom index to group index mapping
    atomidx_to_groupidx: dict[int, int] = {}
    for group_idx, frag in enumerate(frag_list):
        for atom in frag.GetAtoms():
            if atom.GetSymbol() != "*":
                atomidx_to_groupidx[atom.GetAtomMapNum()] = group_idx

    # NOTE: Assign group index to hydrogen atoms
    for atom in mol.GetAtoms():
        atom_map_num = atom.GetAtomMapNum()
        if atom_map_num not in atomidx_to_groupidx:
            # NOTE: Assign group index to hydrogen atom's heavy atom neighbor
            assert atom.GetSymbol() == "H"
            heavy_atom = atom.GetNeighbors()[0]
            heavy_atom_map_num = heavy_atom.GetAtomMapNum()
            atomidx_to_groupidx[atom_map_num] = atomidx_to_groupidx[heavy_atom_map_num]

    # NOTE: Get connectivity map between fragment
    group_connectivity: dict[int, list[int]] = {gidx: [] for gidx in atomidx_to_groupidx.values()}
    for bond in mol.GetBonds():
        atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
        group1: int = atomidx_to_groupidx[atom1.GetAtomMapNum()]
        group2: int = atomidx_to_groupidx[atom2.GetAtomMapNum()]
        if group1 != group2:
            group_connectivity[group1].append(group2)
            group_connectivity[group2].append(group1)
    group_connectivity = {k: list(set(v)) for k, v in group_connectivity.items()}

    return atomidx_to_groupidx, group_connectivity
