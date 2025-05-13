from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED, Crippen, rdMolDescriptors
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from cgflow.utils.extract_pocket import get_mol_center
from cgflow.utils.unidock import docking
from rxnflow.base import BaseTask, RxnFlowTrainer
from rxnflow.config import Config, init_empty
from rxnflow.utils.chem_metrics import mol2qed, mol2sascore

aux_tasks = {"qed": mol2qed, "sa": mol2sascore}


class VinaTask(BaseTask):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.protein_path: Path = Path(cfg.task.docking.protein_path)
        x, y, z = get_mol_center(cfg.task.docking.ref_ligand_path)
        self.center: tuple[float, float, float] = round(x, 3), round(y, 3), round(z, 3)
        self.filter: str | None = cfg.task.constraint.rule
        self.ff_optimization: None | str = None  # None, UFF, MMFF

        self.search_mode: str = "balance"  # fast, balance, detail
        assert self.filter in [None, "lipinski", "veber"]

        self.save_dir: Path = Path(cfg.log_dir) / "docking"
        self.save_dir.mkdir()

        self.topn_vina: OrderedDict[str, float] = OrderedDict()
        self.batch_vina: list[float]

    def compute_rewards(self, mols: list[RDMol]) -> Tensor:
        fr = self.calc_vina_reward(mols)
        return fr.reshape(-1, 1)

    def calc_vina_reward(self, mols: list[RDMol]) -> Tensor:
        vina_scores = self.run_docking(mols)
        self.batch_vina = vina_scores
        self.update_storage(mols, vina_scores)
        fr = torch.tensor(vina_scores, dtype=torch.float32) * -0.1
        return fr.clip(min=1e-5)

    def run_docking(self, mols: list[RDMol]) -> list[float]:
        # docking reuslt
        res = docking(
            [Chem.Mol(mol) for mol in mols],
            self.protein_path,
            self.center,
            seed=1,
            search_mode=self.search_mode,
        )
        # save pose
        out_path = self.save_dir / f"oracle{self.oracle_idx}.sdf"
        with Chem.SDWriter(str(out_path)) as w:
            for i, (docked_mol, _) in enumerate(res):
                if docked_mol is not None:
                    docked_mol.SetIntProp("sample_idx", i)
                    w.write(docked_mol)
        return [min(v, 0.0) for _, v in res]

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

    def update_storage(self, mols: list[RDMol], scores: list[float]):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        self.topn_vina.update(zip(smiles_list, scores, strict=True))
        topn = sorted(list(self.topn_vina.items()), key=lambda v: v[1])[:2000]
        self.topn_vina = OrderedDict(topn)


class VinaMOOTask(VinaTask):
    """Sets up a task where the reward is computed using a Vina, QED."""

    is_moo = True  # turn on moo

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        assert set(self.objectives) <= {"vina", "qed", "sa"}

    def compute_rewards(self, mols: list[RDMol]) -> Tensor:
        flat_r: list[Tensor] = []
        self.avg_reward_info = OrderedDict()
        for prop in self.objectives:
            if prop == "vina":
                fr = self.calc_vina_reward(mols)
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


class VinaTrainer(RxnFlowTrainer):
    task: VinaTask

    def set_default_hps(self, base: Config):
        """find the parameter settings from RxnFlowTrainer.set_default_hps().
        Most parameters are same to SEHFragTrainer and SEHFragMOOTrainer"""
        super().set_default_hps(base)
        base.num_training_steps = 1000
        base.task.constraint.rule = None

        # hparams
        base.algo.action_subsampling.sampling_ratio = 0.01
        base.algo.train_random_action_prob = 0.05
        base.algo.sampling_tau = 0.9
        base.cond.temperature.sample_dist = "uniform"
        base.cond.temperature.dist_params = [0, 64]

    def setup_task(self):
        self.task = VinaTask(cfg=self.cfg)

    def log(self, info, index, key):
        self.add_extra_info(info)
        super().log(info, index, key)

    def add_extra_info(self, info):
        if len(self.task.batch_vina) > 0:
            info["sample_vina_avg"] = np.mean(self.task.batch_vina)
        best_vinas = list(self.task.topn_vina.values())
        for topn in [10, 100, 1000]:
            if len(best_vinas) > topn:
                info[f"top{topn}_vina"] = np.mean(best_vinas[:topn])


class VinaMOOTrainer(VinaTrainer):
    task: VinaMOOTask

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.task.moo.objectives = ["vina", "qed"]

    def setup_task(self):
        self.task = VinaMOOTask(cfg=self.cfg)

    def add_extra_info(self, info):
        for prop, fr in self.task.avg_reward_info.items():
            info[f"sample_r_{prop}_avg"] = fr
        super().add_extra_info(info)


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = "./logs/debug-unidock/"
    config.env_dir = "./data/envs/catalog"
    config.task.constraint.rule = "veber"
    config.overwrite_existing_exp = True

    config.task.docking.protein_path = "./data/experiments/examples/6oim_protein.pdb"
    config.task.docking.ref_ligand_path = "./data/experiments/examples/6oim_ligand.pdb"

    trial = VinaMOOTrainer(config)
    trial.run()
