import tempfile
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Mol as RDMol
from rdkit.Chem import AllChem

from unidock_tools.application.unidock_pipeline import UniDock
from unidock_tools.modules.protein_prep.pdb2pdbqt import pdb2pdbqt


def run_etkdg(mol: RDMol, sdf_path: Path | str, seed: int = 1) -> bool:
    if mol.GetNumAtoms() == 0:
        return False
    try:
        param = AllChem.srETKDGv3()
        param.randomSeed = seed

        # NOTE: '*' -> 'C'
        rwmol = Chem.RWMol(mol)
        for atom in rwmol.GetAtoms():
            if atom.GetSymbol() == "*":
                rwmol.ReplaceAtom(atom.GetIdx(), Chem.Atom("C"))
        mol = rwmol.GetMol()
        mol.UpdatePropertyCache()

        # NOTE: get etkdg structure
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
    protein_pdbqt_path: Path = protein_path.parent / (protein_path.name + "qt")
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
                center[0],
                center[1],
                center[2],
                size,
                size,
                size,
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
            except Exception as e:
                docked_rdmol, docking_score = None, 0.0
            res.append((docked_rdmol, docking_score))
        return res
