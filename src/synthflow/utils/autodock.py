import contextlib
import os
import subprocess
import tempfile
from pathlib import Path

import AutoDockTools
import numpy as np
import rdkit.Chem as Chem
from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate
from openbabel import pybel
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from vina import Vina


def get_mol_coords(mol_path: str | Path) -> np.ndarray:
    format = Path(mol_path).suffix
    if format == ".sdf":
        mol = next(Chem.SDMolSupplier(str(mol_path), sanitize=False))
    elif format == ".mol2":
        mol = Chem.MolFromMol2File(str(mol_path), sanitize=False)
    elif format == ".pdb":
        mol = Chem.MolFromPDBFile(str(mol_path), sanitize=False)
    else:
        raise ValueError(mol_path)
    coords = mol.GetConformer().GetPositions()
    return np.array(coords)


def get_mol_center(mol_path: str | Path) -> tuple[float, float, float]:
    coords = get_mol_coords(mol_path)
    x, y, z = np.mean(coords, 0).tolist()
    return x, y, z


def create_vina_from_protein(
    protein_path: str | Path,
    ref_ligand_path: str | Path | None = None,
    center: tuple[float, float, float] | None = None,
    size: tuple[float, float, float] = (30, 30, 30),
    num_cpus: int = 8,
    seed: int = 1,
    verbose: bool = False,
) -> Vina:

    protein_path = Path(protein_path)
    
    if protein_path.suffix == ".pdbqt":
        protein_pdbqt_path = protein_path
    elif protein_path.suffix == ".pdb":
        protein_pdb_path = protein_path
        protein_pdbqt_path = protein_pdb_path.parent / f"{protein_pdb_path.stem}_autodock.pdbqt"
        if not protein_pdbqt_path.exists():
            protein_pdb_to_pdbqt(protein_pdb_path, protein_pdbqt_path)
    else:
        raise ValueError(protein_path)

    v = Vina(cpu=num_cpus, seed=seed, verbosity=verbose)
    v.set_receptor(str(protein_pdbqt_path))

    if center is None:
        assert ref_ligand_path is not None
        x, y, z = get_mol_center(ref_ligand_path)
        center = round(x, 3), round(y, 3), round(z, 3)
    else:
        x, y, z = center
        center = round(x, 3), round(y, 3), round(z, 3)
    v.compute_vina_maps(center, size)
    return v


def suppress_stdout(func):

    def wrapper(*a, **ka):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)

    return wrapper


@suppress_stdout
def ligand_rdmol_to_pdbqt_string(
    rdmol: Chem.Mol,
    run_etkdg: bool = False,
    run_uff: bool = False,
    use_meeko: bool = True,
) -> str:
    # construct/refine molecular structure
    if run_etkdg or run_uff:
        rdmol = Chem.Mol(rdmol)
    if run_etkdg:
        assert rdmol.GetNumConformers() == 0
        param = rdDistGeom.srETKDGv3()
        param.randomSeed = 1
        param.numThreads = 1
        rdDistGeom.EmbedMolecule(rdmol, param)
    if run_uff:
        assert rdmol.GetNumConformers() == 1
        rdForceFieldHelpers.UFFOptimizeMolecule(rdmol)

    # pdbqt conversion
    if use_meeko:
        """Meeko molecular preparation"""
        preparator = MoleculePreparation()
        setup, *_ = preparator.prepare(rdmol)
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
        return pdbqt_string
    else:
        """Simple pdbqt conversion with obabel"""
        # TODO: check whether following code do work or not.
        pbmol: pybel.Molecule = pybel.readstring("sdf",
                                                 Chem.MolToMolBlock(rdmol))
        return pbmol.write("pdbqt")


def ligand_pdbqt_string_to_rdmol(pdbqt_string: str) -> Chem.Mol:
    """
    Read a pdbqt string and return the RDKit molecule object.
    Args:
        - pdbqt_string (str): pdbqt string
    Returns:
        - mol (rdkit.Chem.Mol): RDKit molecule.
    """
    pdbqt_mol = PDBQTMolecule(pdbqt_string, is_dlg=False, skip_typing=True)
    return RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)[0]


def protein_pdb_to_pdbqt(
    pdb_path: str | Path,
    pdbqt_path: str | Path,
    run_pdb2pqr: bool = True,
):
    prepare_receptor = os.path.join(AutoDockTools.__path__[0],
                                    "Utilities24/prepare_receptor4.py")
    if run_pdb2pqr:
        with tempfile.TemporaryDirectory() as dir:
            pqr_path = Path(dir) / (Path(pdb_path).stem + ".pqr")
            subprocess.Popen(
                ["pdb2pqr30", "--ff=AMBER", pdb_path, pqr_path],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            ).communicate()
            subprocess.Popen(
                [
                    "python3", prepare_receptor, "-r", pqr_path, "-o",
                    pdbqt_path
                ],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            ).communicate()
    else:
        subprocess.Popen(
            ["python3", prepare_receptor, "-r", pdb_path, "-o", pdbqt_path],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        ).communicate()


def set_ligand_from_rdmol(v: Vina, mol: Chem.Mol):
    mol_h = Chem.AddHs(mol, addCoords=True)
    pdbqt = ligand_rdmol_to_pdbqt_string(mol_h)
    v.set_ligand_from_string(pdbqt)


def score_only(v: Vina) -> float:
    return v.score()[0]


def local_opt(v: Vina, remove_h=True) -> tuple[Chem.Mol, float]:
    opt_score = v.optimize()[0]
    with tempfile.NamedTemporaryFile() as tmp:
        # save pose
        with open(tmp.name, "w") as f:
            v.write_pose(tmp.name, overwrite=True)
        with open(tmp.name) as f:
            pose = f.read()
    docked_mol = ligand_pdbqt_string_to_rdmol(pose)
    if remove_h:
        docked_mol = Chem.RemoveHs(docked_mol)
    return docked_mol, opt_score


def docking(v: Vina,
            exhaustiveness: int = 8,
            remove_h=True) -> tuple[Chem.Mol, float]:
    v.dock(8, 1)
    docking_score = float(v.energies(1)[0][0])
    pose = v.poses(1)
    docked_mol = ligand_pdbqt_string_to_rdmol(pose)
    if remove_h:
        docked_mol = Chem.RemoveHs(docked_mol)
    return docked_mol, docking_score
