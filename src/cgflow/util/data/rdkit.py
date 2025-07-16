import itertools
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Mol as RDMol
from scipy.spatial.transform.rotation import Rotation

from cgflow.util.data.reaction import break_bonds, find_brics_bonds, find_rxn_bonds


def mol_to_sdf(mol: Chem.Mol, filepath: str | Path):
    with Chem.SDWriter(filepath) as writer:
        for cid in range(mol.GetNumConformers()):
            writer.write(mol, confId=cid)


def calc_energy(mol: RDMol, per_atom: bool = False) -> float | None:
    """Calculate the energy for an RDKit molecule using the MMFF forcefield

    The energy is only calculated for the first (0th index) conformer within the molecule. The molecule is copied so
    the original is not modified.

    Args:
        mol (Chem.Mol): RDKit molecule
        per_atom (bool): Whether to normalise by number of atoms in mol, default False

    Returns:
        float: Energy of the molecule or None if the energy could not be calculated
    """
    # mol = Chem.Mol(mol)
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    try:
        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=0)
    except Exception:
        return None
    if ff is None:
        return None
    energy = ff.CalcEnergy()
    if per_atom:
        energy = energy / mol.GetNumAtoms()
    if energy in (float("inf"), float("-inf"), float("nan")):
        return None
    return energy


def optimise_mol(mol: RDMol, max_iters: int = 1000) -> RDMol | None:
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
    mol = Chem.Mol(mol)
    try:
        exitcode = AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
    except Exception:
        return None
    if exitcode == 0:
        return mol
    return None


def calc_rmsd(
    mol1: Chem.Mol,
    mol2: Chem.Mol,
    align: bool = True,
    isomorphism: bool = True,
) -> float:
    """Approximately align two molecules and then calculate RMSD between them

    Alignment and distance is calculated only between the default conformers of each molecule.

    Args:
        mol1 (Chem.Mol): First molecule to align
        mol2 (Chem.Mol): Second molecule to align
        align (bool): Whether to align the molecules before calculating the RMSD
        isomorphism (bool): Whether to consider atom isomorphisms (i.e., different atom mappings)
    Returns:
        float: RMSD between molecules after approximate alignment
    """

    assert len(mol1.GetAtoms()) == len(mol2.GetAtoms())

    coords1 = np.array(mol1.GetConformer().GetPositions())
    coords2 = np.array(mol2.GetConformer().GetPositions())

    if isomorphism:
        matches: list[tuple[int, ...]] = mol1.GetSubstructMatches(mol2, uniquify=False)
        if len(matches) == 0:
            return _conf_rmsd(coords1, coords2, align=align)

        min_rmsd = float("inf")
        for match in matches:
            # remapping the coordinates of mol1 to match the indices of mol2
            remapped_coords1 = coords1[list(match)]
            rmsd = _conf_rmsd(remapped_coords1, coords2, align=align)
            min_rmsd = min(rmsd, min_rmsd)
        return min_rmsd
    else:
        return _conf_rmsd(coords1, coords2, align=align)


def _conf_rmsd(coords1: np.ndarray, coords2: np.ndarray, align: bool = True) -> float:
    if align:
        # Firstly, center both molecules
        coords1 = coords1 - (coords1.sum(axis=0) / coords1.shape[0])
        coords2 = coords2 - (coords2.sum(axis=0) / coords2.shape[0])
        try:
            # Find the best rotation alignment between the centred mols
            rotation, _ = Rotation.align_vectors(coords1, coords2)
            aligned_coords2 = rotation.apply(coords2)
        except Exception:
            aligned_coords2 = coords2
        coords2 = aligned_coords2

    sqrd_dists = (coords1 - coords2) ** 2
    return float(np.sqrt(sqrd_dists.sum(axis=1).mean()))


def get_mol_centroid(mol: Chem.Mol) -> np.ndarray:
    coords = np.array(mol.GetConformer().GetPositions())
    centroid = coords.mean(axis=0)
    return centroid


def centroid_distance(mol1: Chem.Mol, mol2: Chem.Mol) -> np.ndarray:
    coords1 = np.array(mol1.GetConformer().GetPositions())
    coords2 = np.array(mol2.GetConformer().GetPositions())

    centroid1 = coords1.mean(axis=0)
    centroid2 = coords2.mean(axis=0)
    return np.sqrt(((centroid1 - centroid2) ** 2).sum())


# TODO could allow more args
def smiles_from_mol(mol: Chem.Mol, canonical: bool = True, explicit_hs: bool = False) -> str | None:
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


def mol_from_smiles(smiles: str, explicit_hs: bool = False) -> Chem.Mol | None:
    """Create a RDKit molecule from a SMILES string

    Args:
        smiles (str): SMILES string
        explicit_hs (bool): Whether to embed explicit hydrogens into the mol

    Returns:
        Chem.Mol: RDKit molecule object or None if one cannot be created from the SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol) if explicit_hs else mol
    except Exception:
        mol = None
    return mol


def construct_mol(
    atoms: list[str],
    charges: list[int],
    bond_indices: list[tuple[int, int]],
    bond_types: list[Chem.BondType],
    coords: NDArray[np.float64],
    sanitise=True,
) -> RDMol | None:
    """Create RDKit mol"""
    # check shape
    Natom = len(atoms)
    Nbond = len(bond_indices)
    assert coords.shape == (Natom, 3), f"coords shape ({coords.shape}) must be ({Natom}, 3)."
    assert len(charges) == Natom, f"charges ({len(charges)}) should have a same length as atoms ({Natom})."
    assert len(bond_types) == Nbond, f"bond_types ({len(bond_types)}) should have a same length as bonds ({Nbond})."

    # Create molecule from atom types, charges, and bond types
    emol = Chem.EditableMol(Chem.Mol())
    for idx in range(Natom):
        atom = Chem.Atom(atoms[idx])
        atom.SetFormalCharge(charges[idx])
        emol.AddAtom(atom)
    for idx in range(Nbond):
        st, end = bond_indices[idx]
        assert st != end
        emol.AddBond(st, end, bond_types[idx])

    mol = emol.GetMol()
    try:
        for atom in mol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)
    except Exception:
        return None

    if sanitise:
        # We didn't infer the explicit hydrogens for the aromatic nitrogens.
        # For sanitisation, we need to add hydrogens to aromatic nitrogens

        # fmt: off
        arom_n_list: list[int] = [
            atom.GetIdx() for atom in mol.GetAtoms()
            if atom.GetIsAromatic() and atom.GetSymbol() == "N" and atom.GetDegree() == 2
        ]
        all_subset: list[tuple[int, ...]] = [
            subset
            for i in range(len(arom_n_list) + 1)
            for subset in itertools.combinations(arom_n_list, i)
        ]
        # fmt: on

        # Iterate all possible combinations of adding hydrogens to aromatic nitrogens
        for nitrogen_indices in all_subset:
            # convert n to [nH]
            sanitized_mol = Chem.Mol(mol)  # copy
            for i in nitrogen_indices:
                sanitized_mol.GetAtomWithIdx(i).SetNumExplicitHs(1)
            # check sanitize test
            try:
                Chem.SanitizeMol(sanitized_mol)
                assert sanitized_mol is not None
            except Exception:
                continue
            break
        else:
            return None
        mol = sanitized_mol

    # add conformer
    conf = Chem.Conformer(Natom)
    conf.SetPositions(coords)
    mol.AddConformer(conf)
    return mol


def get_brics_assignment(mol: RDMol, max_num_cuts=np.inf):
    raise NotImplementedError("Use get_decompose_assignment with rule='brics' instead")


def get_decompose_assignment(
    mol: RDMol,
    rule: str = "brics",
    max_num_cuts: int | None = None,
    min_group_size: int = 1,
) -> tuple[set[int], dict[int, int], dict[int, list[int]]]:
    """Get decomposed fragment assignments for each atom in the molecule
    Args:
        mol (Chem.Mol): RDKit molecule

    Returns:
        atom_idx_to_group_idx (dict[int, int]): Mapping from atom index to group index
        group_connectivity (dict[int, list[int]]): Connectivity Map btw Groups
    """
    assert rule in ["brics", "reaction", "rotatable"]
    random.seed(0)

    # NOTE: Assign atom map number to preserve original atom index
    # mol.GetIdx() == mol.GetAtomMapNum() == mol_no_hs.GetAtomMapNum()
    mol = Chem.Mol(mol)
    mol.RemoveAllConformers()
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i + 1)  # 0 might occurs error.

    # NOTE: Find bond to break down
    if rule == "brics":
        decompose_func = find_brics_bonds
    elif rule == "reaction":
        decompose_func = find_rxn_bonds
    else:
        # TODO: Implement Rotatable Bond
        raise NotImplementedError

    # get bond indices
    bond_idcs: list[tuple[int, int]] = [tuple(sorted((b[0][0], b[0][1]))) for b in decompose_func(mol)]  # pyright: ignore
    # Remove duplicated bonds
    bond_idcs = list(set(bond_idcs))
    num_break_bonds = len(bond_idcs)
    if max_num_cuts is not None:
        num_break_bonds = min(max_num_cuts, num_break_bonds)

    # randomly select a number of bonds to break
    if random.random() < 0.5:
        num_break_bonds = random.randint(0, num_break_bonds)

    # NOTE: Break down molecule to fragment groups
    frag_list: tuple[RDMol, ...]
    for num_to_cut in range(num_break_bonds, 0, -1):
        possible_breaks = list(itertools.permutations(bond_idcs, num_to_cut))
        random.shuffle(possible_breaks)
        for break_bond_idcs in possible_breaks:
            outcome = _break_mol_with_bonds(mol, break_bond_idcs, min_group_size)
            if outcome is not None:
                break
        else:
            # If there is no valid outcome, continue
            continue
        # If there is a valid outcome, break
        frag_list = outcome
        break
    else:
        # if there is no break, just return the original molecule (for-else statement)
        frag_list = (mol,)

    # NOTE: Get stem atom index
    stem_atoms: set[int] = set()
    for frag in frag_list:
        for atom in frag.GetAtoms():
            if atom.GetSymbol() == "*":
                neighbors = list(atom.GetNeighbors())
                assert len(neighbors) == 1
                neighbor_atom = neighbors[0]
                stem_atoms.add(neighbor_atom.GetAtomMapNum() - 1)

    # NOTE: Get atom index to group index mapping
    atomidx_to_groupidx: dict[int, int] = {}
    for group_idx, frag in enumerate(frag_list):
        for atom in frag.GetAtoms():
            if atom.GetSymbol() != "*":
                atomidx_to_groupidx[atom.GetAtomMapNum() - 1] = group_idx
            else:
                neighbors = list(atom.GetNeighbors())
                assert len(neighbors) == 1
                neighbor_atom = neighbors[0]
                # label connecting atom
                atomidx_to_groupidx[neighbor_atom.GetAtomMapNum() - 1] = group_idx

    # NOTE: Get connectivity map between fragment
    group_connectivity: dict[int, list[int]] = {gidx: [] for gidx in atomidx_to_groupidx.values()}
    for bond in mol.GetBonds():
        atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
        group1 = atomidx_to_groupidx[atom1.GetAtomMapNum() - 1]
        group2 = atomidx_to_groupidx[atom2.GetAtomMapNum() - 1]
        if group1 != group2:
            group_connectivity[group1].append(group2)
            group_connectivity[group2].append(group1)
    group_connectivity = {k: list(set(v)) for k, v in group_connectivity.items()}
    return stem_atoms, atomidx_to_groupidx, group_connectivity


def _break_mol_with_bonds(
    mol: Chem.Mol,
    bond_idcs: Sequence[tuple[int, int]],
    min_group_size: int = 1,
) -> tuple[Chem.Mol, ...] | None:
    num_to_cut = len(bond_idcs)
    outcomes = break_bonds(mol, bond_idcs, sanitize=True)
    # check the fragment counts; prevent ring-break
    if len(outcomes) != num_to_cut + 1:
        return None
    # check the fragments are valid
    for frag in outcomes:
        num_attachments = sum(atom.GetAtomicNum() == 0 for atom in frag.GetAtoms())
        # check the number of attachments
        # TODO: should we only consider arm/linker?
        if num_attachments > 2:
            return None
        # check the fragment size
        if frag.GetNumAtoms() - num_attachments < min_group_size:
            return None
    return outcomes
