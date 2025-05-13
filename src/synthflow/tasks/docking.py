from collections import OrderedDict
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import QED, Crippen, rdMolDescriptors
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from rxnflow.base import BaseTask
from rxnflow.utils.chem_metrics import mol2qed, mol2sascore
from synthflow.config import Config
from synthflow.utils.extract_pocket import get_mol_center

aux_tasks = {"qed": mol2qed, "sa": mol2sascore}


class BaseDockingTask(BaseTask):
    cfg: Config

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        # binding affinity estimation
        self.redocking = cfg.task.docking.redocking
        self.ff_opt = cfg.task.docking.ff_opt

        # docking
        self.protein_path: Path = Path(cfg.task.docking.protein_path)

        x, y, z = get_mol_center(cfg.task.docking.ref_ligand_path)
        self.center: tuple[float, float, float] = round(x, 3), round(y, 3), round(z, 3)

        self.filter: str | None = cfg.task.constraint.rule
        assert self.filter in [None, "lipinski", "veber"]

        self.save_dir: Path = Path(cfg.log_dir) / "pose"
        self.save_dir.mkdir()

        self.topn_affinity: OrderedDict[str, float] = OrderedDict()
        self.batch_affinity: list[float] = []

    def compute_rewards(self, mols: list[RDMol]) -> Tensor:
        self.save_pose(mols)
        fr = self.calc_affinity_reward(mols)
        return fr.reshape(-1, 1)

    def _calc_affinity(self, mol: RDMol) -> float:
        raise NotImplementedError

    def __calc_affinity(self, mol: RDMol) -> float:
        try:
            return self._calc_affinity(mol)
        except Exception:
            return 0.0

    def _calc_affinity_batch(self, mols: list[RDMol]) -> list[float]:
        return [self.__calc_affinity(mol) for mol in mols]

    def calc_affinity_reward(self, mols: list[RDMol]) -> Tensor:
        affinities = self._calc_affinity_batch(mols)
        self.batch_affinity = affinities
        self.update_storage(mols, affinities)
        fr = torch.tensor(affinities, dtype=torch.float32) * -0.1
        return fr.clip(min=1e-5)

    def filter_object(self, mol: RDMol) -> bool:
        if self.filter is None:
            pass
        elif self.filter in ("lipinski", "veber"):
            if rdMolDescriptors.CalcExactMolWt(mol) > 500:
                return False
            if rdMolDescriptors.CalcNumHBD(mol) > 5:
                return False
            if rdMolDescriptors.CalcNumHBA(mol) > 10:
                return False
            if Crippen.MolLogP(mol) > 5:
                return False
            if self.filter == "veber":
                if rdMolDescriptors.CalcTPSA(mol) > 140:
                    return False
                if rdMolDescriptors.CalcNumRotatableBonds(mol) > 10:
                    return False
        else:
            raise ValueError(self.filter)
        return True

    def save_pose(self, mols: list[RDMol]):
        out_path = self.save_dir / f"oracle{self.oracle_idx}.sdf"
        with Chem.SDWriter(str(out_path)) as w:
            for mol in mols:
                # NOTE: see env_ctx.graph_to_obj
                assert mol.HasProp("sample_idx")
                w.write(mol)

    def update_storage(self, mols: list[RDMol], scores: list[float]):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        self.topn_affinity.update(zip(smiles_list, scores, strict=True))
        topn = sorted(list(self.topn_affinity.items()), key=lambda v: v[1])[:2000]
        self.topn_affinity = OrderedDict(topn)


class BaseDockingMOGFNTask(BaseDockingTask):
    """Sets up a task where the reward is computed using a Docking, QED."""

    is_moo = True

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        assert set(self.objectives) <= {"vina", "qed", "sa"}

    def compute_rewards(self, mols: list[RDMol]) -> Tensor:
        self.save_pose(mols)
        flat_r: list[Tensor] = []
        self.avg_reward_info = OrderedDict()
        for prop in self.objectives:
            if prop == "vina":
                fr = self.calc_affinity_reward(mols)
            else:
                fr = aux_tasks[prop](mols)
            flat_r.append(fr)
            self.avg_reward_info[prop] = fr.mean().item()
        flat_rewards = torch.stack(flat_r, dim=1)
        assert flat_rewards.shape[0] == len(mols)
        return flat_rewards

    def update_storage(self, mols: list[RDMol], scores: list[float]):
        def _filter(mol: RDMol) -> bool:
            """Check the object passes a property filter"""
            return QED.qed(mol) > 0.5

        pass_idcs = [i for i, mol in enumerate(mols) if _filter(mol)]
        pass_mols = [mols[i] for i in pass_idcs]
        pass_scores = [scores[i] for i in pass_idcs]
        super().update_storage(pass_mols, pass_scores)


class BaseDockingMOOTask(BaseDockingTask):
    """Sets up a task where the reward is computed using a Docking, QED."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.objectives = self.cfg.task.moo.objectives
        assert set(self.objectives) <= {"vina", "qed", "sa"}

    def compute_rewards(self, mols: list[RDMol]) -> Tensor:
        self.save_pose(mols)
        flat_r: list[Tensor] = []
        self.avg_reward_info = OrderedDict()
        for prop in self.objectives:
            if prop == "vina":
                fr = self.calc_affinity_reward(mols)
            else:
                fr = aux_tasks[prop](mols)
            flat_r.append(fr)
            self.avg_reward_info[prop] = fr.mean().item()
        flat_rewards = torch.stack(flat_r, dim=1).prod(dim=1, keepdim=True)
        assert flat_rewards.shape[0] == len(mols)
        return flat_rewards

    def update_storage(self, mols: list[RDMol], scores: list[float]):
        def _filter(mol: RDMol) -> bool:
            """Check the object passes a property filter"""
            return QED.qed(mol) > 0.5

        pass_idcs = [i for i, mol in enumerate(mols) if _filter(mol)]
        pass_mols = [mols[i] for i in pass_idcs]
        pass_scores = [scores[i] for i in pass_idcs]
        super().update_storage(pass_mols, pass_scores)
