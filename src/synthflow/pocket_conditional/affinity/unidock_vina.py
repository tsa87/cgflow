import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Mol as RDMol
from unidock_tools.application.unidock_pipeline import UniDock


def run_etkdg(mol: RDMol, sdf_path: Path | str, seed: int = 1) -> bool:
    if mol.GetNumAtoms() == 0:
        return False
    try:
        param = AllChem.srETKDGv3()
        param.randomSeed = seed
        mol = Chem.Mol(mol)  # copy
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
                Path(protein_path),
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
