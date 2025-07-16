from torch import Tensor

from gflownet.algo.trajectory_balance import TrajectoryBalance
from rxnflow.algo.synthetic_path_sampling import SyntheticPathSampler
from rxnflow.config import Config
from rxnflow.envs import SynthesisEnv, SynthesisEnvContext
from rxnflow.models.gfn import RxnFlow
from rxnflow.policy import SubsamplingPolicy
from rxnflow.utils.misc import set_worker_env


class SynthesisTB(TrajectoryBalance):
    env: SynthesisEnv
    ctx: SynthesisEnvContext
    global_cfg: Config
    graph_sampler: SyntheticPathSampler

    def __init__(self, env: SynthesisEnv, ctx: SynthesisEnvContext, cfg: Config):
        self.action_subsampler: SubsamplingPolicy = SubsamplingPolicy(env, cfg)
        self.importance_temp = cfg.algo.action_subsampling.importance_temp
        set_worker_env("action_subsampler", self.action_subsampler)
        super().__init__(env, ctx, cfg)

    def setup_graph_sampler(self):
        self.graph_sampler = SyntheticPathSampler(
            self.ctx,
            self.env,
            self.action_subsampler,
            max_len=self.max_len,
            max_nodes=self.max_nodes,
            importance_temp=self.importance_temp,
            sample_temp=self.sample_temp,
            num_workers=self.global_cfg.num_workers_retrosynthesis,
        )

    def create_training_data_from_graphs(
        self,
        graphs,
        model: RxnFlow | None = None,
        cond_info: Tensor | None = None,
        random_action_prob: float | None = 0.0,
    ):
        # TODO: implement here
        assert len(graphs) == 0
        return []
