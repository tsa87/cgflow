from abc import ABC, abstractmethod
from typing import Generic

from rxnflow.base.trainer import BaseTaskT, RxnFlowTrainer

from synthflow.base.algo import SynthesisTB3D
from synthflow.base.env import SynthesisEnv3D, SynthesisEnvContext3D
from synthflow.config import Config


class RxnFlow3DTrainer(RxnFlowTrainer[BaseTaskT], ABC, Generic[BaseTaskT]):
    cfg: Config
    env: SynthesisEnv3D
    ctx: SynthesisEnvContext3D
    algo: SynthesisTB3D

    @abstractmethod
    def setup_env_context(self): ...

    # type mapping
    def set_default_hps(self, base: Config):
        """rxnflow.config.Config -> cgflow.config.Config"""
        super().set_default_hps(base)

    def get_default_cfg(self):
        """rxnflow.config.Config -> cgflow.config.Config"""
        return Config()

    def setup_env(self):
        self.env = SynthesisEnv3D(self.cfg.env_dir)

    def setup_algo(self):
        assert self.cfg.algo.method == "TB"
        self.algo = SynthesisTB3D(self.env, self.ctx, self.cfg)
