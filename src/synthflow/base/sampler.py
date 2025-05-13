from rxnflow.base.generator import RxnFlowSampler
from synthflow.base.algo import SynthesisTB3D
from synthflow.base.env import SynthesisEnv3D, SynthesisEnvContext3D
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
        self.algo = SynthesisTB3D(self.env, self.ctx, self.cfg)
