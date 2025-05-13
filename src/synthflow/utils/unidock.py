import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Mol as RDMol
from unidock_tools.application.unidock_pipeline import UniDock
from unidock_tools.modules.protein_prep.pdb2pdbqt import pdb2pdbqt


def run_etkdg(mol: RDMol, sdf_path: Path | str, seed: int = 1) -> bool:
    if mol.GetNumAtoms() == 0:
        return False
    try:
        param = AllChem.srETKDGv3()
        param.randomSeed = seed
        mol = Chem.Mol(mol)
        mol.RemoveAllConformers()
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, param)
        mol = Chem.RemoveHs(mol)
        assert mol.GetNumConformers() > 0
        with Chem.SDWriter(str(sdf_path)) as w:
            w.write(mol)
    except Exception:
        return False
    else:
        return True


def docking(
    rdmols: list[RDMol],
    protein_path: str | Path,
    center: tuple[float, float, float],
    seed: int = 1,
    size: float = 20.0,
    search_mode: str = "balance",
):
    protein_path = Path(protein_path)
    # create pdbqt file
    protein_pdbqt_path: Path = protein_path.parent / (protein_path.stem + "_unidock.pdbqt")
    if not protein_pdbqt_path.exists():
        pdb2pdbqt(protein_path, protein_pdbqt_path)

    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        sdf_list = []
        for i, mol in enumerate(rdmols):
            ligand_file = out_dir / f"{i}.sdf"
            flag = run_etkdg(mol, ligand_file)
            if flag:
                sdf_list.append(ligand_file)
        if len(sdf_list) > 0:
            runner = UniDock(
                protein_pdbqt_path,
                sdf_list,
                round(center[0], 3),
                round(center[1], 3),
                round(center[2], 3),
                round(size, 3),
                round(size, 3),
                round(size, 3),
                out_dir / "workdir",
            )
            runner.docking(
                out_dir / "savedir",
                search_mode=search_mode,
                num_modes=1,
                seed=seed,
            )

        res: list[tuple[None, float] | tuple[RDMol, float]] = []
        for i in range(len(rdmols)):
            try:
                docked_file = out_dir / "savedir" / f"{i}.sdf"
                docked_rdmol: Chem.Mol = list(Chem.SDMolSupplier(str(docked_file)))[0]
                assert docked_rdmol is not None
                docking_score = float(docked_rdmol.GetProp("docking_score"))
            except Exception:
                docked_rdmol, docking_score = None, 0.0
            res.append((docked_rdmol, docking_score))
        return res


def scoring(
    rdmols: list[RDMol],
    protein_path: str | Path,
    center: tuple[float, float, float],
    size: float = 25.0,
):
    protein_path = Path(protein_path)
    # create pdbqt file
    protein_pdbqt_path: Path = protein_path.parent / (protein_path.name + "qt")
    if not protein_pdbqt_path.exists():
        pdb2pdbqt(protein_path, protein_pdbqt_path)

    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        sdf_list = []
        for i, mol in enumerate(rdmols):
            ligand_file = out_dir / f"{i}.sdf"
            try:
                lig_pos = np.array(mol.GetConformer().GetPositions())
                min_x, min_y, min_z = lig_pos.min(0)
                assert center[0] - size / 2 < min_x
                assert center[1] - size / 2 < min_y
                assert center[2] - size / 2 < min_z
                max_x, max_y, max_z = lig_pos.max(0)
                assert center[0] + size / 2 > max_x
                assert center[1] + size / 2 > max_y
                assert center[2] + size / 2 > max_z
                with Chem.SDWriter(str(ligand_file)) as w:
                    w.write(mol)
            except Exception:
                pass
            else:
                sdf_list.append(ligand_file)
        if len(sdf_list) > 0:
            runner = UniDock(
                protein_pdbqt_path,
                sdf_list,
                round(center[0], 3),
                round(center[1], 3),
                round(center[2], 3),
                round(size, 3),
                round(size, 3),
                round(size, 3),
                out_dir / "workdir",
            )
            try:
                runner.docking(out_dir / "savedir", score_only=True)
            except Exception:
                pass

        res: list[tuple[None, float] | tuple[RDMol, float]] = []
        for i in range(len(rdmols)):
            docked_file = str(out_dir / "savedir" / f"{i}.sdf")
            try:
                docked_rdmol: Chem.Mol = list(Chem.SDMolSupplier(str(docked_file), sanitize=False))[0]
                assert docked_rdmol is not None
                docking_score = float(docked_rdmol.GetProp("docking_score"))
            except Exception:
                docked_rdmol, docking_score = None, 0.0
            res.append((docked_rdmol, docking_score))
        return res
