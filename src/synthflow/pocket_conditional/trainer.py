from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from rxnflow.base.task import BaseTask
from rxnflow.utils import chem_metrics
from rxnflow.utils.misc import get_worker_env
from synthflow.base.trainer import RxnFlow3DTrainer
from synthflow.config import Config
from synthflow.pocket_conditional.env import SynthesisEnvContext3D_pocket_conditional


class PocketConditionalTask(BaseTask):
    pocket_key: str

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.objectives = cfg.task.moo.objectives
        self.last_reward: dict[str, Tensor] = {}  # For Logging
        self.save_dir = Path(cfg.log_dir) / "pose/"
        self.save_dir.mkdir(exist_ok=True)

    def compute_rewards(self, mols: list[RDMol]) -> Tensor:
        """TacoGFN Reward Function"""
        self.save_pose(mols)
        self.sample_reward_info: dict[str, Tensor] = {}
        flat_r = []
        for prop in self.objectives:
            if prop == "vina":
                r = self.calculate_affinity(mols)
                num_heavy_atoms = torch.tensor([mol.GetNumHeavyAtoms() for mol in mols], dtype=torch.float32)
                r_norm = -1 * r * (num_heavy_atoms ** (-1 / 3))
            elif prop == "qed":
                r = chem_metrics.mol2qed(mols)
                r_norm = (r / 0.7).clip(0, 1)
            elif prop == "sa":
                r = chem_metrics.mol2sascore(mols)
                r_norm = (r / 0.8).clip(0, 1)
            else:
                raise ValueError(prop)
            self.sample_reward_info[prop] = r
            flat_r.append(r_norm)
        flat_rewards = torch.stack(flat_r, dim=1).prod(dim=1, keepdim=True)
        assert flat_rewards.shape[0] == len(mols)
        return flat_rewards

    def calculate_affinity(self, mols: list[RDMol]) -> Tensor:
        raise NotImplementedError

    def sample_conditional_information(self, n: int, train_it: int) -> dict[str, Tensor]:
        ctx: SynthesisEnvContext3D_pocket_conditional = get_worker_env("ctx")
        cond_info = self.temperature_conditional.sample(n)
        pocket_cond = ctx._tmp_pocket_cond.reshape(1, -1).repeat(n, 1)
        cond_info["encoding"] = torch.cat([cond_info["encoding"], pocket_cond], dim=1)
        return cond_info

    def save_pose(self, mols: list[RDMol]):
        out_path = self.save_dir / f"oracle{self.oracle_idx}.sdf"
        with Chem.SDWriter(str(out_path)) as w:
            for i, mol in enumerate(mols):
                mol.SetIntProp("sample_idx", i)
                w.write(mol)


class PocketConditionalTrainer(RxnFlow3DTrainer):
    task: PocketConditionalTask

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.desc = "Proxy-QED-SA optimization for multiple targets"
        base.task.moo.objectives = ["vina", "qed"]
        base.validate_every = 0
        base.num_training_steps = 200_000
        base.checkpoint_every = 1_000

        base.model.num_emb = 64

        # GFN parameters
        base.cond.temperature.sample_dist = "uniform"
        base.cond.temperature.dist_params = [1, 64]

        # model training
        base.algo.train_random_action_prob = 0.2
        base.algo.num_from_policy = 16
        base.replay.use = True
        base.replay.capacity = 16 * 500
        base.replay.warmup = 16 * 20
        base.replay.num_from_replay = 16 * 3
        base.num_workers_retrosynthesis = 4

        # training learning rate
        base.opt.learning_rate = 1e-4
        base.opt.lr_decay = 10_000
        base.algo.tb.Z_learning_rate = 1e-2
        base.algo.tb.Z_lr_decay = 20_000

    def setup_env_context(self):
        ckpt_path = self.cfg.cgflow.ckpt_path
        use_predicted_pose = self.cfg.cgflow.use_predicted_pose
        num_inference_steps = self.cfg.cgflow.num_inference_steps
        self.ctx = SynthesisEnvContext3D_pocket_conditional(
            self.env,
            self.task.num_cond_dim,
            ckpt_path,
            use_predicted_pose,
            num_inference_steps,
        )

    def setup_task(self):
        self.task = PocketConditionalTask(cfg=self.cfg)

    def log(self, info, index, key):
        for obj in self.task.objectives:
            info[f"sampled_{obj}_avg"] = self.task.sample_reward_info[obj].mean().item()
        super().log(info, index, key)
