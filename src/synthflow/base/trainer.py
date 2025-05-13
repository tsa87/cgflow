from rxnflow.base.trainer import RxnFlowTrainer
from synthflow.base.algo import SynthesisTB3D
from synthflow.base.env import SynthesisEnv3D, SynthesisEnvContext3D
from synthflow.config import Config


class RxnFlow3DTrainer(RxnFlowTrainer):
    cfg: Config
    env: SynthesisEnv3D

    def set_default_hps(self, base: Config):
        """rxnflow.config.Config -> cgflow.config.Config"""
        super().set_default_hps(base)
        base.model.num_emb = 64

    def get_default_cfg(self):
        """rxnflow.config.Config -> cgflow.config.Config"""
        return Config()

    def setup_env(self):
        self.env = SynthesisEnv3D(self.cfg.env_dir)

    def setup_env_context(self):
        self.ctx = SynthesisEnvContext3D(self.env, self.task.num_cond_dim)

    def setup_algo(self):
        assert self.cfg.algo.method == "TB"
        self.algo = SynthesisTB3D(self.env, self.ctx, self.cfg)
