from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from typing_extensions import override

from synthflow.base.sampler import RxnFlow3DSampler
from synthflow.pocket_conditional.env import SynthesisEnvContext3D_pocket_conditional

"""
Summary
- ProxyTask: Base Class
- ProxyTask_MultiPocket & ProxyTrainer_MultiPocket: Train Pocket-Conditioned RxnFlow.
"""


class PocketConditionalSampler(RxnFlow3DSampler):
    ctx: SynthesisEnvContext3D_pocket_conditional

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

    def set_pocket(self, pocket_path: str | Path):
        """setup pose prediction model"""
        self.ctx.set_pocket(pocket_path)
        self.pocket_cond = self.ctx._tmp_pocket_cond

    def sample_conditional_information(self, n: int, train_it: int) -> dict[str, Tensor]:
        cond_info = super().sample_conditional_information(n, train_it)
        pocket_cond = self.pocket_cond.reshape(1, -1).repeat(n, 1)
        cond_info["encoding"] = torch.cat([cond_info["encoding"], pocket_cond], dim=1)
        return cond_info

    def sample_against_pocket(self, pocket_path: str | Path, n: int):
        self.set_pocket(pocket_path)
        return self.sample(n)

    @override
    def sample(self, n: int, calc_reward: bool = False) -> list[dict[str, Any]]:
        assert calc_reward is False
        return super().sample(n, calc_reward)
