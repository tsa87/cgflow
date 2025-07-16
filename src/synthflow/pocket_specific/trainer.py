from typing import Generic

from synthflow.base.env_ctx_cgflow import SynthesisEnvContext3D_cgflow
from synthflow.base.trainer import BaseTaskT, RxnFlow3DTrainer
from synthflow.config import Config


class RxnFlow3DTrainer_single(RxnFlow3DTrainer[BaseTaskT], Generic[BaseTaskT]):
    ctx: SynthesisEnvContext3D_cgflow

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        # Model
        base.model.num_emb = 64
        base.model.num_mlp_layers = 1
        base.model.num_mlp_layers_block = 1

        # GFN parameters
        base.cond.temperature.sample_dist = "uniform"
        base.cond.temperature.dist_params = [1, 64]

        # optimizer
        base.opt.opt = "adamw"
        base.opt.weight_decay = 1e-2
        base.opt.momentum = 0.9
        base.opt.eps = 1e-8
        base.opt.lr_decay = 20_000
        base.opt.clip_grad_type = "norm"
        base.opt.clip_grad_param = 10

        base.algo.tb.Z_learning_rate = 1e-2
        base.algo.tb.Z_lr_decay = 50_000

        # Online Training Parameters
        base.algo.num_from_policy = 32
        base.algo.sampling_tau = 0.9
        base.algo.train_random_action_prob = 0.05  # suggest to set positive value

        base.replay.use = True
        base.replay.warmup = 32 * 10
        base.replay.capacity = 32 * 100
        base.replay.num_from_replay = 32

        base.num_workers_retrosynthesis = 4

    def setup_env_context(self):
        # protein binding site
        protein_path = self.cfg.task.docking.protein_path
        center = self.cfg.task.docking.center
        ref_ligand_path = self.cfg.task.docking.ref_ligand_path

        # cgflow
        ckpt_path = self.cfg.cgflow.ckpt_path
        use_predicted_pose = self.cfg.cgflow.use_predicted_pose
        num_inference_steps = self.cfg.cgflow.num_inference_steps

        if center is None:
            assert ref_ligand_path is not None, (
                "Either `center` or `ref_ligand_path` must be provided to identify the binding site."
            )

        self.ctx = SynthesisEnvContext3D_cgflow(
            self.env,
            self.task.num_cond_dim,
            ckpt_path,
            use_predicted_pose,
            num_inference_steps,
        )
        self.ctx.set_pocket(protein_path, ref_ligand_path=ref_ligand_path)
