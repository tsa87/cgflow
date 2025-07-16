from rxnflow.base.generator import RxnFlowSampler

from synthflow.base.algo import SynthesisTB3D
from synthflow.base.env import SynthesisEnv3D, SynthesisEnvContext3D
from synthflow.base.env_ctx_cgflow import SynthesisEnvContext3D_cgflow
from synthflow.config import Config


class RxnFlow3DSampler(RxnFlowSampler):
    cfg: Config
    env: SynthesisEnv3D
    ctx: SynthesisEnvContext3D

    def setup_env(self):
        self.env = SynthesisEnv3D(self.cfg.env_dir)

    def setup_env_context(self):
        self.ctx = SynthesisEnvContext3D(self.env, self.task.num_cond_dim)

    def setup_algo(self):
        assert self.cfg.algo.method == "TB"
        self.num_workers_retrosynthesis = 0  # no retro analysis
        self.algo = SynthesisTB3D(self.env, self.ctx, self.cfg)


class SynthFlowSampler(RxnFlow3DSampler):
    ctx: SynthesisEnvContext3D_cgflow

    def setup_env_context(self):
        ckpt_path = self.cfg.cgflow.ckpt_path
        use_predicted_pose = self.cfg.cgflow.use_predicted_pose
        num_inference_steps = self.cfg.cgflow.num_inference_steps
        self.ctx = SynthesisEnvContext3D_cgflow(
            self.env,
            self.task.num_cond_dim,
            ckpt_path,
            use_predicted_pose,
            num_inference_steps,
        )
