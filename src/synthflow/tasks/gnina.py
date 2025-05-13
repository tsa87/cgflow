import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule

from synthflow.config import Config
from synthflow.tasks.autodock_vina import AutoDockVina_MOGFNTrainer, AutoDockVina_MOOTrainer
from synthflow.tasks.docking import BaseDockingMOGFNTask, BaseDockingMOOTask, BaseDockingTask
from synthflow.utils import gnina, unidock


def get_protein_pdbqt(protein_pdb_path: str | Path) -> Path:
    protein_pdb_path = Path(protein_pdb_path)
    protein_pdbqt_path = protein_pdb_path.parent / (protein_pdb_path.stem + "_unidock.pdbqt")
    if not protein_pdbqt_path.exists():
        unidock.pdb2pdbqt(protein_pdb_path, protein_pdbqt_path)
    return protein_pdbqt_path


def _run_redocking_caching(self: BaseDockingTask, mols: list[Chem.Mol]) -> list[float]:
    # pdbqt conversion
    protein_pdbqt_path = get_protein_pdbqt(self.protein_path)
    smiles_list = [Chem.MolToSmiles(obj) for obj in mols]
    scores = [self.topn_affinity.get(smi, 0.0) for obj, smi in zip(mols, smiles_list, strict=True)]
    unique_indices = [i for i, v in enumerate(scores) if v >= 0.0]
    if len(unique_indices) > 0:
        unique_mols = [Chem.Mol(mols[i]) for i in unique_indices]
        print(f"run docking for {len(unique_mols)} molecules among {len(smiles_list)} molecules")
        res = unidock.docking(unique_mols, self.protein_path, self.center, search_mode="balance")
        for j, (r, v) in zip(unique_indices, res, strict=True):
            if r is not None:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    writer = Chem.SDWriter(tmp_dir + "/molecule.sdf")
                    writer.write(r)
                    writer.close()
                    scores[j] = gnina.local_opt(
                        tmp_dir + "/molecule.sdf", protein_pdbqt_path, tmp_dir + "/result.sdf", num_workers=8
                    )
            else:
                scores[j] = 0.0
            assert scores[j] >= 0.0
            scores[j] = min(v, 0.0)
    return scores


def _run_localopt(self: BaseDockingTask, mols: list[Chem.Mol]) -> list[float]:
    # pdbqt conversion
    protein_pdbqt_path = get_protein_pdbqt(self.protein_path)
    # uff opt
    input_ligand_path = self.save_dir / f"oracle{self.oracle_idx}_uff.sdf"
    with Chem.SDWriter(str(input_ligand_path)) as w:
        for obj in mols:
            try:
                UFFOptimizeMolecule(obj, maxIters=100)  # use 100 instead of 200 to preserve global structures
                w.write(obj)
            except Exception:
                print(f"UFF optimization failed for molecule {Chem.MolToSmiles(obj)}")
                with open(self.save_dir / f"oracle{self.oracle_idx}_uff_failed.sdf", "a") as w2:
                    w2.write(Chem.MolToMolBlock(obj))
                w.write(obj)

    # unidock local opt
    output_result_path = self.save_dir / f"oracle{self.oracle_idx}_localopt.sdf"
    scores = gnina.local_opt(input_ligand_path, protein_pdbqt_path, output_result_path, num_workers=8)
    return [min(v, 0.0) for v in scores]


def _calc_gnina_score_batch(self, mols: list[Chem.Mol]) -> list[float]:
    if self.redocking:
        raise NotImplementedError("Redocking is not implemented for GninaTask")
        # return _run_redocking(self, objs)
    else:
        return _run_localopt(self, mols)


class Gnina_Task(BaseDockingTask):
    _calc_affinity_batch = _calc_gnina_score_batch


class Gnina_MOOTask(BaseDockingMOOTask):
    _calc_affinity_batch = _calc_gnina_score_batch


class Gnina_MOGFNTask(BaseDockingMOGFNTask):
    _calc_affinity_batch = _calc_gnina_score_batch


class Gnina_MOOTrainer(AutoDockVina_MOOTrainer):
    task: Gnina_MOOTask

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.task.moo.objectives = ["vina", "qed"]

    def setup_task(self):
        self.task = Gnina_MOOTask(cfg=self.cfg)


class Gnina_MOGFNTrainer(AutoDockVina_MOGFNTrainer):
    task: Gnina_MOGFNTask

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.task.moo.objectives = ["vina", "qed"]

    def setup_task(self):
        self.task = Gnina_MOGFNTask(cfg=self.cfg)
