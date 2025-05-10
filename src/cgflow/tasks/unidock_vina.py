import numpy as np
from rdkit.Chem import Mol as RDMol

from rxnflow.base.trainer import mogfn_trainer
from cgflow.config import Config, init_empty
from cgflow.base_trainer import RxnFlow3DTrainer_unidock
from cgflow.tasks.vina import VinaTask, VinaMOOTask


class VinaTask_unidock(VinaTask):
    def _calc_vina_score(self, obj: RDMol) -> float:
        return min(0, obj.GetDoubleProp("docking_score"))


class VinaMOOTask_unidock(VinaMOOTask):
    def _calc_vina_score(self, obj: RDMol) -> float:
        return min(0, obj.GetDoubleProp("docking_score"))


class VinaTrainer_unidock(RxnFlow3DTrainer_unidock):
    task: VinaTask_unidock

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
        self.task = VinaTask_unidock(cfg=self.cfg, wrap_model=self._wrap_for_mp)

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
class VinaMOOTrainer_unidock(VinaTrainer_unidock):
    task: VinaMOOTask_unidock

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.task.moo.objectives = ["vina", "qed"]

    def setup_task(self):
        self.task = VinaMOOTask_unidock(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def add_extra_info(self, info):
        for prop, fr in self.task.avg_reward_info.items():
            info[f"sample_r_{prop}_avg"] = fr
        super().add_extra_info(info)


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = "./logs/debug-unidock/"
    config.env_dir = "./data/envs/stock"
    config.overwrite_existing_exp = True

    config.task.docking.protein_path = "./data/experiments/examples/6oim_protein.pdb"
    config.task.docking.ref_ligand_path = "./data/experiments/examples/6oim_ligand.pdb"

    trial = VinaMOOTrainer_unidock(config)
    trial.run()
