import numpy as np
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule

from rxnflow.base.trainer import mogfn_trainer
from cgflow.config import Config, init_empty
from cgflow.tasks.vina import VinaTask, VinaMOOTask
from cgflow.base_trainer import RxnFlow3DTrainer_semla
from cgflow.utils import unidock
from cgflow.utils import vina


def _run_redocking_caching(self: VinaTask, objs: list[Chem.Mol]) -> list[float]:
    smiles_list = [Chem.MolToSmiles(obj) for obj in objs]
    scores = [self.topn_vina.get(smi, 0.0) for obj, smi in zip(objs, smiles_list, strict=True)]
    unique_indices = [i for i, v in enumerate(scores) if v >= 0.0]
    if len(unique_indices) > 0:
        unique_objs = [Chem.Mol(objs[i]) for i in unique_indices]
        print(f"run docking for {len(unique_objs)} molecules among {len(smiles_list)} molecules")
        res = unidock.docking(unique_objs, self.protein_path, self.center, search_mode="balance")
        for j, (_, v) in zip(unique_indices, res, strict=True):
            assert scores[j] >= 0.0
            scores[j] = min(v, 0.0)
    return scores


def _run_redocking(self, objs: list[Chem.Mol]) -> list[float]:
    # unidock redocking
    res = unidock.docking(objs, self.protein_path, self.center, search_mode="balance")

    output_result_path = self.save_dir / f"oracle{self.oracle_idx}_redock.sdf"
    with Chem.SDWriter(str(output_result_path)) as w:
        for docked_obj, _ in res:
            if docked_obj is not None:
                w.write(docked_obj)
            else:
                w.write(Chem.Mol())
    scores = [v for mol, v in res]
    return [min(v, 0.0) for v in scores]


def _run_localopt(self, objs: list[Chem.Mol]) -> list[float]:
    # uff opt
    input_ligand_path = self.save_dir / f"oracle{self.oracle_idx}_uff.sdf"
    with Chem.SDWriter(str(input_ligand_path)) as w:
        for obj in objs:
            UFFOptimizeMolecule(obj, maxIters=100)  # use 100 instead of 200 to preserve global structures
            w.write(obj)

    # unidock local opt
    output_result_path = self.save_dir / f"oracle{self.oracle_idx}_localopt.sdf"
    scores = vina.local_opt(input_ligand_path, self.protein_pdbqt_path, output_result_path, num_workers=8)
    return [min(v, 0.0) for v in scores]


def _calc_vina_score_batch(self, objs: list[Chem.Mol]) -> list[float]:
    if self.redocking:
        return _run_redocking_caching(self, objs)
        # return _run_redocking(self, objs)
    else:
        return _run_localopt(self, objs)


class VinaTask_semla(VinaTask):
    _calc_vina_score_batch = _calc_vina_score_batch


class VinaMOOTask_semla(VinaMOOTask):
    _calc_vina_score_batch = _calc_vina_score_batch


class VinaTrainer_semla(RxnFlow3DTrainer_semla):
    task: VinaTask_semla

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.num_training_steps = 1000
        base.task.constraint.rule = None

        # hparams
        base.model.num_emb = 64
        base.algo.action_subsampling.sampling_ratio = 0.01
        base.algo.train_random_action_prob = 0.05
        base.algo.sampling_tau = 0.9
        base.cond.temperature.sample_dist = "uniform"
        base.cond.temperature.dist_params = [0, 64]

    def setup_task(self):
        self.task = VinaTask_semla(cfg=self.cfg, wrap_model=self._wrap_for_mp)

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


@mogfn_trainer
class VinaMOOTrainer_semla(VinaTrainer_semla):
    task: VinaMOOTask_semla

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.task.moo.objectives = ["vina", "qed"]

    def setup_task(self):
        self.task = VinaMOOTask_semla(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def add_extra_info(self, info):
        for prop, fr in self.task.avg_reward_info.items():
            info[f"sample_r_{prop}_avg"] = fr
        super().add_extra_info(info)


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = "./logs/debug-semla/"
    config.env_dir = "./data/envs/stock"
    config.overwrite_existing_exp = True

    config.semlaflow.ckpt_path = "./weights/semlaflow.ckpt"
    config.task.docking.protein_path = "./data/examples/6oim_protein.pdb"
    config.task.docking.ref_ligand_path = "./data/examples/6oim_ligand.pdb"

    trial = VinaMOOTrainer_semla(config)
    trial.run()
