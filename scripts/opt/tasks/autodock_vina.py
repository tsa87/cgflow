from collections import OrderedDict

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED, rdDistGeom
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule, UFFOptimizeMolecule
from rdkit.Chem.rdMolAlign import AlignMol
from tqdm import tqdm
from vina import Vina

from gflownet.utils import sascore

from synthflow.config import Config
from synthflow.pocket_specific.trainer import RxnFlow3DTrainer_single
from synthflow.utils import autodock

from .docking import BaseDockingTask


class AutoDockVinaTask(BaseDockingTask):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.vina_module: Vina = autodock.create_vina_from_protein(
            self.protein_path, center=self.center, size=self.size
        )
        self.exhaustiveness: int = cfg.task.docking.exhaustiveness

    def calc_affinities(self, mols: list[Chem.Mol]) -> list[float]:
        if self.redocking:
            return self.run_redocking(mols)
        else:
            return self.run_localopt(mols)

    def run_localopt(self, mols: list[Chem.Mol]) -> list[float]:
        ff_opt_mols: list[Chem.Mol] = []
        docked_mols: list[Chem.Mol] = []
        scores: list[float] = []
        for mol in tqdm(mols, desc="Vina Local Opt", unit="mol", leave=False):
            try:
                mol: Chem.Mol = Chem.AddHs(mol, addCoords=True)
                if self.ff_opt != "none":
                    ref_mol = Chem.Mol(mol)
                    if self.ff_opt == "uff":
                        UFFOptimizeMolecule(mol, maxIters=200)
                    elif self.ff_opt == "mmff":
                        MMFFOptimizeMolecule(mol, maxIters=200)
                    AlignMol(mol, ref_mol)
                    ff_opt_mols.append(mol)

                mol_pdbqt_string = autodock.ligand_rdmol_to_pdbqt_string(mol)
                self.vina_module.set_ligand_from_string(mol_pdbqt_string)
                docked_mol, score = autodock.local_opt(self.vina_module, remove_h=False)
            except Exception:
                docked_mol = Chem.Mol()
                score = 0.0
            docked_mol.SetIntProp("sample_idx", mol.GetIntProp("sample_idx"))
            docked_mol.SetDoubleProp("docking_score", score)
            docked_mols.append(docked_mol)
            scores.append(score)

        if self.ff_opt != "none":
            out_result_path = self.save_dir / f"oracle{self.oracle_idx}_ff.sdf"
            with Chem.SDWriter(str(out_result_path)) as w:
                for mol in ff_opt_mols:
                    w.write(mol)

        out_result_path = self.save_dir / f"oracle{self.oracle_idx}_opt.sdf"
        with Chem.SDWriter(str(out_result_path)) as w:
            for mol in docked_mols:
                w.write(mol)
        return [min(v, 0.0) for v in scores]

    def run_redocking(self, mols: list[Chem.Mol]) -> list[float]:
        param = rdDistGeom.srETKDGv3()
        param.randomSeed = 1
        param.numThreads = 1

        docked_mols: list[Chem.Mol] = []
        scores: list[float] = []

        for mol in tqdm(mols, desc="Vina Redocking", unit="mol", leave=False):
            try:
                mol: Chem.Mol = Chem.Mol(mol)
                mol.RemoveAllConformers()
                mol = Chem.AddHs(mol)
                rdDistGeom.EmbedMolecule(mol, param)
                if self.ff_opt == "uff":
                    UFFOptimizeMolecule(mol)
                elif self.ff_opt == "mmff":
                    UFFOptimizeMolecule(mol)
                mol_pdbqt_string = autodock.ligand_rdmol_to_pdbqt_string(mol)
                self.vina_module.set_ligand_from_string(mol_pdbqt_string)
                docked_mol, score = autodock.docking(
                    self.vina_module, exhaustiveness=self.exhaustiveness, remove_h=False
                )
            except Exception:
                docked_mol = Chem.Mol()
                score = 0.0
            docked_mol.SetIntProp("sample_idx", mol.GetIntProp("sample_idx"))
            docked_mol.SetDoubleProp("docking_score", score)
            docked_mols.append(docked_mol)
            scores.append(score)

        out_result_path = self.save_dir / f"oracle{self.oracle_idx}_dock.sdf"
        with Chem.SDWriter(str(out_result_path)) as w:
            for mol in docked_mols:
                w.write(mol)
        return [min(v, 0.0) for v in scores]


class AutoDockVinaMOOTask(AutoDockVinaTask):
    avg_reward_info: OrderedDict[str, float]

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.objectives = self.cfg.task.moo.objectives
        assert set(self.objectives) <= {"vina", "qed", "sa"}

    def compute_rewards(self, mols: list[Chem.Mol]) -> torch.Tensor:
        self.save_pose(mols)
        flat_r: list[torch.Tensor] = []
        self.avg_reward_info = OrderedDict()
        for prop in self.objectives:
            if prop == "vina":
                fr = self.calc_affinity_reward(mols)
            elif prop == "qed":
                fr = self.calc_qed_reward(mols)
            elif prop == "sa":
                fr = self.calc_sa_reward(mols)
            else:
                raise NotImplementedError(f"Objective {prop} is not implemented")
            flat_r.append(fr)
            self.avg_reward_info[prop] = fr.mean().item()
        flat_rewards = torch.stack(flat_r, dim=1).prod(dim=1, keepdim=True)
        assert flat_rewards.shape[0] == len(mols)
        return flat_rewards

    def calc_qed_reward(self, mols: list[Chem.Mol]) -> torch.Tensor:
        def calc_score(mol: Chem.Mol) -> float:
            try:
                return QED.qed(mol)
            except Exception:
                return 0.0

        return torch.tensor([calc_score(mol) for mol in mols])

    def calc_sa_reward(self, mols: list[Chem.Mol]) -> torch.Tensor:
        def calc_score(mol: Chem.Mol) -> float:
            try:
                return (10 - sascore.calculateScore(mol)) / 9
            except Exception:
                return 0.0

        return torch.tensor([calc_score(mol) for mol in mols])

    def update_storage(self, mols: list[Chem.Mol], scores: list[float]):
        def _filter(mol: Chem.Mol) -> bool:
            """Check the object passes a property filter"""
            return QED.qed(mol) > 0.5

        pass_idcs = [i for i, mol in enumerate(mols) if _filter(mol)]
        pass_mols = [mols[i] for i in pass_idcs]
        pass_scores = [scores[i] for i in pass_idcs]
        super().update_storage(pass_mols, pass_scores)


class AutoDockVinaTrainer(RxnFlow3DTrainer_single[AutoDockVinaTask]):
    def setup_task(self):
        self.task: AutoDockVinaTask = AutoDockVinaTask(cfg=self.cfg)

    def log(self, info, index, key):
        self.add_extra_info(info)
        super().log(info, index, key)

    def add_extra_info(self, info):
        if len(self.task.batch_affinity) > 0:
            info["sample_vina_avg"] = np.mean(self.task.batch_affinity)
        best_vinas = list(self.task.topn_affinity.values())
        for topn in [10, 100, 1000]:
            if len(best_vinas) > topn:
                info[f"top{topn}_vina"] = np.mean(best_vinas[:topn])


class AutoDockVinaMOOTrainer(RxnFlow3DTrainer_single[AutoDockVinaMOOTask]):
    def setup_task(self):
        self.task: AutoDockVinaMOOTask = AutoDockVinaMOOTask(cfg=self.cfg)

    def log(self, info, index, key):
        self.add_extra_info(info)
        super().log(info, index, key)

    def add_extra_info(self, info):
        for prop, fr in self.task.avg_reward_info.items():
            info[f"sample_r_{prop}_avg"] = fr
        if len(self.task.batch_affinity) > 0:
            info["sample_vina_avg"] = np.mean(self.task.batch_affinity)
        best_vinas = list(self.task.topn_affinity.values())
        for topn in [10, 100, 1000]:
            if len(best_vinas) > topn:
                info[f"top{topn}_vina"] = np.mean(best_vinas[:topn])
