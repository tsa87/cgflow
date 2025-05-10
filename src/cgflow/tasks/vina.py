from pathlib import Path
import torch
import torch.nn as nn
from torch import Tensor

from rdkit import Chem
from rdkit.Chem import Mol as RDMol
from rdkit.Chem import rdMolDescriptors, Crippen, QED

from collections import OrderedDict
from collections.abc import Callable

from rxnflow.base import BaseTask
from rxnflow.utils.chem_metrics import mol2qed, mol2sascore

from cgflow.config import Config
from cgflow.utils.extract_pocket import get_mol_center
from cgflow.utils.unidock import pdb2pdbqt


aux_tasks = {"qed": mol2qed, "sa": mol2sascore}


class VinaTask(BaseTask):
    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)

        # binding affinity estimation
        self.redocking = cfg.task.docking.redocking
        self.ff_opt = cfg.task.docking.ff_opt
        self.protein_path: Path = Path(cfg.task.docking.protein_path)
        x, y, z = get_mol_center(cfg.task.docking.ref_ligand_path)
        self.center: tuple[float, float, float] = round(x, 3), round(y, 3), round(z, 3)

        # create pdbqt file
        self.protein_pdbqt_path: Path = self.protein_path.parent / (self.protein_path.name + "qt")
        if not self.protein_pdbqt_path.exists():
            pdb2pdbqt(self.protein_path, self.protein_pdbqt_path)

        self.filter: str | None = cfg.task.constraint.rule
        assert self.filter in [None, "lipinski", "veber"]

        self.save_dir: Path = Path(cfg.log_dir) / "pose"
        self.save_dir.mkdir()

        self.topn_vina: OrderedDict[str, float] = OrderedDict()
        self.batch_vina: list[float] = []

    def save_pose(self, objs: list[RDMol]):
        # save pose
        out_path = self.save_dir / f"oracle{self.oracle_idx}.sdf"
        with Chem.SDWriter(str(out_path)) as w:
            for i, obj in enumerate(objs):
                obj.SetIntProp("sample_idx", i)
                w.write(obj)

    def compute_rewards(self, objs: list[RDMol]) -> Tensor:
        self.save_pose(objs)
        fr = self.calc_vina_reward(objs)
        return fr.reshape(-1, 1)

    def _calc_vina_score(self, obj: RDMol) -> float:
        raise NotImplementedError

    def __calc_vina_score(self, obj: RDMol) -> float:
        try:
            return self._calc_vina_score(obj)
        except Exception:
            return 0.0

    def _calc_vina_score_batch(self, objs: list[RDMol]) -> list[float]:
        return [self.__calc_vina_score(obj) for obj in objs]

    def calc_vina_reward(self, objs: list[RDMol]) -> Tensor:
        vina_scores = self._calc_vina_score_batch(objs)
        self.update_storage(objs, vina_scores)
        fr = torch.tensor(vina_scores, dtype=torch.float32) * -0.1
        return fr.clip(min=1e-5)

    def filter_object(self, obj: RDMol) -> bool:
        if self.filter is None:
            pass
        elif self.filter in ("lipinski", "veber"):
            if rdMolDescriptors.CalcExactMolWt(obj) > 500:
                return False
            if rdMolDescriptors.CalcNumHBD(obj) > 5:
                return False
            if rdMolDescriptors.CalcNumHBA(obj) > 10:
                return False
            if Crippen.MolLogP(obj) > 5:
                return False
            if self.filter == "veber":
                if rdMolDescriptors.CalcTPSA(obj) > 140:
                    return False
                if rdMolDescriptors.CalcNumRotatableBonds(obj) > 10:
                    return False
        else:
            raise ValueError(self.filter)
        return True

    def update_storage(self, objs: list[RDMol], scores: list[float]):
        self.batch_vina = scores
        smiles_list = [Chem.MolToSmiles(obj) for obj in objs]
        self.topn_vina.update(zip(smiles_list, scores, strict=True))
        topn = sorted(list(self.topn_vina.items()), key=lambda v: v[1])[:2000]
        self.topn_vina = OrderedDict(topn)


class VinaMOOTask(VinaTask):
    """Sets up a task where the reward is computed using a Vina, QED."""

    is_moo = True

    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        assert set(self.objectives) <= {"vina", "qed", "sa"}

    def compute_rewards(self, objs: list[RDMol]) -> Tensor:
        self.save_pose(objs)

        flat_r: list[Tensor] = []
        self.avg_reward_info = OrderedDict()
        for prop in self.objectives:
            if prop == "vina":
                fr = self.calc_vina_reward(objs)
            else:
                fr = aux_tasks[prop](objs)
            flat_r.append(fr)
            self.avg_reward_info[prop] = fr.mean().item()
        flat_rewards = torch.stack(flat_r, dim=1)
        assert flat_rewards.shape[0] == len(objs)
        return flat_rewards

    def update_storage(self, objs: list[RDMol], scores: list[float]):
        def _filter(obj: RDMol) -> bool:
            """Check the object passes a property filter"""
            return QED.qed(obj) > 0.5

        pass_idcs = [i for i, obj in enumerate(objs) if _filter(obj)]
        pass_objs = [objs[i] for i in pass_idcs]
        pass_scores = [scores[i] for i in pass_idcs]
        super().update_storage(pass_objs, pass_scores)
