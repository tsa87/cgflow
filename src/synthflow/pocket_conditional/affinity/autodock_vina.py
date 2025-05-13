import contextlib
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import AutoDockTools
import rdkit.Chem as Chem
from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate
from openbabel import pybel
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from vina import Vina


class VinaDocking:
    protein_path: Path

    def __init__(
        self,
        pocket_path: str | Path,
        scoring_function: str = "vina",
        **vina_params,
    ):
        self.tmp_fd = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp_fd.name)

        self.scoring_function = scoring_function
        self.pocket_path = self.tmp_dir / "pocket.pdbqt"

        if Path(pocket_path).suffix == ".pdb":
            pocket_pdb_to_pdbqt(pocket_path, self.pocket_path)
        else:
            os.system(f"cp {pocket_path} {self.pocket_path}")

    def docking(self, ligand_rdmol: Chem.Mol, exhaustiveness: int = 8, cpu: int = 0) -> tuple[float, Chem.Mol]:
        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)
        center, size = self.get_search_box(ligand_rdmol)
        pdbqt_string_list = ligand_rdmol_to_pdbqt_string(ligand_rdmol)

        runner = Vina(self.scoring_function, cpu, seed=1, verbosity=0)
        runner.set_receptor(str(self.pocket_path))
        runner.set_ligand_from_string(pdbqt_string_list)
        runner.compute_vina_maps(center=center, box_size=size)
        runner.dock(exhaustiveness, 1)
        score = float(runner.energies(1)[0][0])
        pose = runner.poses(1)
        docked_mol = ligand_pdbqt_string_to_rdmol(pose)
        return score, docked_mol

    def local_opt_batch(self, ligand_rdmols: list[Chem.Mol], cpu: int = 0) -> list[tuple[float, Chem.Mol | None]]:
        if cpu == 0:
            cpu = len(os.sched_getaffinity(0))
        if cpu == 1:
            res = list(map(self.local_opt, ligand_rdmols))
        else:
            with ProcessPoolExecutor(cpu) as pool:
                res = list(pool.map(self.local_opt, ligand_rdmols))
        return res

    def local_opt(self, ligand_rdmol: Chem.Mol) -> tuple[float, Chem.Mol | None]:
        try:
            ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)
            center, size = self.get_search_box(ligand_rdmol)
            pdbqt_string_list = ligand_rdmol_to_pdbqt_string(ligand_rdmol)

            runner = Vina(self.scoring_function, 1, seed=1, verbosity=0)
            runner.set_receptor(str(self.pocket_path))
            runner.set_ligand_from_string(pdbqt_string_list)
            runner.compute_vina_maps(center=center, box_size=size)

            score = runner.optimize()[0]
            with tempfile.NamedTemporaryFile() as tmp:
                with open(tmp.name, "w") as f:
                    runner.write_pose(tmp.name, overwrite=True)
                with open(tmp.name) as f:
                    pose = f.read()
        except KeyboardInterrupt as e:
            raise e
        except Exception:
            return 0, None

        try:
            docked_mol = ligand_pdbqt_string_to_rdmol(pose)
        except TypeError:
            docked_mol = None
        return score, docked_mol

    @staticmethod
    def get_search_box(rdmol: Chem.Mol) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        pose = rdmol.GetConformer(0).GetPositions()
        cx, cy, cz = ((pose.max(0) + pose.min(0)) / 2).tolist()
        sx, sy, sz = ((pose.max(0) - pose.min(0)) + 5.0).tolist()
        return (cx, cy, cz), (sx, sy, sz)

    def __del__(self):
        self.tmp_fd.cleanup()


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
        pbmol: pybel.Molecule = pybel.readstring("sdf", Chem.MolToMolBlock(rdmol))
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


def pocket_pdb_to_pdbqt(
    pdb_path: str | Path,
    pdbqt_path: str | Path,
    run_pdb2pqr: bool = True,
):
    prepare_receptor = os.path.join(AutoDockTools.__path__[0], "Utilities24/prepare_receptor4.py")
    if run_pdb2pqr:
        with tempfile.TemporaryDirectory() as dir:
            pqr_path = Path(dir) / (Path(pdb_path).stem + ".pqr")
            subprocess.Popen(
                ["pdb2pqr30", "--ff=AMBER", pdb_path, pqr_path],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            ).communicate()
            subprocess.Popen(
                ["python3", prepare_receptor, "-r", pqr_path, "-o", pdbqt_path],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            ).communicate()
    else:
        subprocess.Popen(
            ["python3", prepare_receptor, "-r", pdb_path, "-o", pdbqt_path],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        ).communicate()
