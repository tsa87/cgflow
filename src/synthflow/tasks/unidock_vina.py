import numpy as np
from rdkit import Chem

from synthflow.config import Config, init_empty
from synthflow.pocket_specific.trainer import RxnFlow3DTrainer_single
from synthflow.tasks.docking import BaseDockingMOGFNTask, BaseDockingMOOTask, BaseDockingTask
from synthflow.utils import unidock


def _run_redocking(self: BaseDockingTask, mols: list[Chem.Mol]) -> list[float]:
    # unidock redocking
    try:
        res = unidock.docking(mols,
                              self.protein_path,
                              self.center,
                              search_mode="balance")
    except Exception:
        return [0.0] * len(mols)
    output_result_path = self.save_dir / f"oracle{self.oracle_idx}_redock.sdf"
    with Chem.SDWriter(str(output_result_path)) as w:
        for docked_mol, _ in res:
            docked_mol = Chem.Mol() if docked_mol is None else docked_mol
            w.write(docked_mol)
    scores = [v for mol, v in res]
    return [min(v, 0.0) for v in scores]


class UniDockVina_Task(BaseDockingTask):
    _calc_affinity_batch = _run_redocking


class UniDockVina_MOOTask(BaseDockingMOOTask):
    _calc_affinity_batch = _run_redocking


class UniDockVina_MOGFNTask(BaseDockingMOGFNTask):
    _calc_affinity_batch = _run_redocking


class UniDockVina_Trainer(RxnFlow3DTrainer_single):
    task: UniDockVina_Task

    def set_default_hps(self, base: Config):
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
        self.task = UniDockVina_Task(cfg=self.cfg)

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


class UniDockVina_MOOTrainer(UniDockVina_Trainer):
    task: UniDockVina_MOOTask

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.task.moo.objectives = ["vina", "qed"]

    def setup_task(self):
        self.task = UniDockVina_MOOTask(cfg=self.cfg)

    def add_extra_info(self, info):
        for prop, fr in self.task.avg_reward_info.items():
            info[f"sample_r_{prop}_avg"] = fr
        super().add_extra_info(info)


class UniDockVina_MOGFNTrainer(UniDockVina_MOOTrainer):
    task: UniDockVina_MOGFNTask

    def setup_task(self):
        self.task = UniDockVina_MOGFNTask(cfg=self.cfg)

    def add_extra_info(self, info):
        for prop, fr in self.task.avg_reward_info.items():
            info[f"sample_r_{prop}_avg"] = fr
        super().add_extra_info(info)


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = "./logs/debug-cl/"
    config.env_dir = "./data/stock"
    config.overwrite_existing_exp = True

    config.cgflow.ckpt_path = "./weights/crossdocked_till_end.ckpt"
    config.task.docking.protein_path = "./experiments/data/examples/6oim_protein.pdb"
    config.task.docking.ref_ligand_path = "./experiments/data/examples/6oim_ligand.pdb"

    trial = UniDockVina_MOOTrainer(config)
    trial.run()
