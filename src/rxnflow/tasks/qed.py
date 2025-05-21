from rdkit.Chem import Mol as RDMol
from torch import Tensor

from rxnflow.base import BaseTask, RxnFlowTrainer
from rxnflow.config import Config, init_empty
from rxnflow.utils.chem_metrics import mol2qed


class QEDTask(BaseTask):
    def compute_rewards(self, mols: list[RDMol]) -> Tensor:
        return mol2qed(mols).reshape(-1, 1)


class QEDTrainer(RxnFlowTrainer):  # For online training
    def setup_task(self):
        self.task: QEDTask = QEDTask(cfg=self.cfg)


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.num_training_steps = 100
    config.log_dir = "./logs/debug-qed/"
    config.env_dir = "./data/envs/stock"
    config.overwrite_existing_exp = True
    config.algo.action_subsampling.sampling_ratio = 0.01

    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [16]
    config.algo.train_random_action_prob = 0.1

    trial = QEDTrainer(config)
    trial.run()
