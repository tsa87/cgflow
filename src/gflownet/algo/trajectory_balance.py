from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch import Tensor
from torch_scatter import scatter

from gflownet.algo.config import LossFN
from gflownet.algo.graph_sampling import GraphSampler
from gflownet.config import Config
from gflownet.envs.graph_building_env import (
    Graph,
    GraphActionCategorical,
    GraphBuildingEnv,
    GraphBuildingEnvContext,
    generate_forward_trajectory,
)
from gflownet.trainer import GFNAlgorithm
from gflownet.utils.misc import get_worker_device


class TrajectoryBalanceModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch: gd.Batch, cond_info: Tensor) -> tuple[GraphActionCategorical, Tensor]: ...

    @abstractmethod
    def logZ(self, cond_info: Tensor) -> Tensor: ...


class TrajectoryBalance(GFNAlgorithm):
    """Trajectory-based GFN loss implementations. Implements
    - TB: Trajectory Balance: Improved Credit Assignment in GFlowNets Nikolay Malkin, Moksh Jain,
    Emmanuel Bengio, Chen Sun, Yoshua Bengio
    https://arxiv.org/abs/2201.13259
    """

    def __init__(
        self,
        env: GraphBuildingEnv,
        ctx: GraphBuildingEnvContext,
        cfg: Config,
    ) -> None:
        """Instanciate a TB algorithm.

        Parameters
        ----------
        env: GraphBuildingEnv
            A graph environment.
        ctx: GraphBuildingEnvContext
            A context.
        cfg: Config
            Hyperparameters
        """
        self.ctx = ctx
        self.env = env
        self.global_cfg = cfg
        self.cfg = cfg.algo.tb
        self.max_nodes = cfg.algo.max_nodes
        self.max_len = cfg.algo.max_len
        self.length_normalize_losses = cfg.algo.tb.do_length_normalize

        # Experimental flags
        self.reward_loss = self.cfg.loss_fn
        self.tb_loss = self.cfg.loss_fn
        self.mask_invalid_rewards = False
        self.reward_normalize_losses = False
        self.sample_temp = 1
        self.bootstrap_own_reward = self.cfg.bootstrap_own_reward
        self.setup_graph_sampler()

    def set_is_eval(self, is_eval: bool):
        self.is_eval = is_eval

    def setup_graph_sampler(self):
        self.graph_sampler = GraphSampler(
            self.ctx,
            self.env,
            self.global_cfg.algo.max_len,
            self.max_nodes,
            self.sample_temp,
            correct_idempotent=False,
            pad_with_terminal_state=False,
        )

    def create_training_data_from_own_samples(
        self,
        model: TrajectoryBalanceModel,
        n: int,
        cond_info: Tensor | None = None,
        random_action_prob: float = 0.0,
    ):
        """Generate trajectories by sampling a model

        Parameters
        ----------
        model: TrajectoryBalanceModel
           The model being sampled
        n: int
            Number of trajectories to sample
        cond_info: torch.tensor
            Conditional information, shape (N, n_info)
        random_action_prob: float
            Probability of taking a random action
        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]]
           - reward_pred: float, -100 if an illegal action is taken, predicted R(x) if bootstrapping, None otherwise
           - fwd_logprob: log Z + sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - logZ: predicted log Z
           - loss: predicted loss (if bootstrapping)
           - is_valid: is the generated graph valid according to the env & ctx
        """
        dev = get_worker_device()
        cond_info = cond_info.to(dev) if cond_info is not None else None
        data = self.graph_sampler.sample_from_model(model, n, cond_info, random_action_prob)
        if cond_info is not None:
            logZ_pred = model.logZ(cond_info)
            for i in range(n):
                data[i]["logZ"] = logZ_pred[i].item()
        return data

    def create_training_data_from_graphs(self, graphs: list[Graph]) -> list[dict]:
        """Generate trajectories from known endpoints

        Parameters
        ----------
        graphs: List[Graph]
            List of Graph endpoints
        model: TrajectoryBalanceModel
           The model being sampled
        cond_info: torch.tensor
            Conditional information, shape (N, n_info)
        random_action_prob: float
            Probability of taking a random action

        Returns
        -------
        trajs: List[Dict{'traj': List[tuple[Graph, GraphAction]]}]
           A list of trajectories.
        """
        trajs: list[dict[str, Any]] = [{"traj": generate_forward_trajectory(i)} for i in graphs]
        for traj in trajs:
            n_back = [self.env.count_backward_transitions(gp, check_idempotent=False) for gp, _ in traj["traj"][1:]] + [
                1
            ]
            traj["bck_logprobs"] = (1 / torch.tensor(n_back).float()).log().to(get_worker_device())
            traj["result"] = traj["traj"][-1][0]
        return trajs

    def construct_batch(self, trajs, cond_info, log_rewards):
        """Construct a batch from a list of trajectories and their information

        Parameters
        ----------
        trajs: List[List[tuple[Graph, GraphAction]]]
            A list of N trajectories.
        cond_info: Tensor
            The conditional info that is considered for each trajectory. Shape (N, n_info)
        log_rewards: Tensor
            The transformed log-reward (e.g. torch.log(R(x) ** beta) ) for each trajectory. Shape (N,)
        Returns
        -------
        batch: gd.Batch
             A (CPU) Batch object with relevant attributes added
        """
        torch_graphs = [self.ctx.graph_to_Data(i[0]) for tj in trajs for i in tj["traj"]]
        actions = [
            self.ctx.GraphAction_to_ActionIndex(g, a)
            for g, a in zip(torch_graphs, [i[1] for tj in trajs for i in tj["traj"]], strict=False)
        ]
        batch = self.ctx.collate(torch_graphs)
        batch.traj_lens = torch.tensor([len(i["traj"]) for i in trajs])
        batch.log_p_B = torch.cat([i["bck_logprobs"] for i in trajs], 0)
        batch.actions = torch.tensor(actions)
        batch.log_rewards = log_rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get("is_valid", True) for i in trajs]).float()
        # compute_batch_losses expects these two optional values, if someone else doesn't fill them in, default to 0
        batch.num_offline = 0
        batch.num_online = 0
        return batch

    def estimate_policy(
        self,
        model: TrajectoryBalanceModel,
        batch: gd.Batch,
        cond_info: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> tuple[GraphActionCategorical, Tensor]:
        batched_cond_info = cond_info[batch_idx]
        fwd_cat, per_graph_out = model.forward(batch, batched_cond_info)
        return fwd_cat, per_graph_out

    def compute_batch_losses(
        self,
        model: TrajectoryBalanceModel,
        batch: gd.Batch,
        num_bootstrap: int = 0,  # type: ignore[override]
    ):
        """Compute the losses over trajectories contained in the batch

        Parameters
        ----------
        model: TrajectoryBalanceModel
           A GNN taking in a batch of graphs as input as per constructed by `self.construct_batch`.
           Must have a `logZ` attribute, itself a model, which predicts log of Z(cond_info)
        batch: gd.Batch
          batch of graphs inputs as per constructed by `self.construct_batch`
        num_bootstrap: int
          the number of trajectories for which the reward loss is computed. Ignored if 0."""
        dev = batch.x.device
        # A single trajectory is comprised of many graphs
        num_trajs = int(batch.traj_lens.shape[0])
        log_rewards = batch.log_rewards
        # Clip rewards
        assert log_rewards.ndim == 1
        clip_log_R = torch.maximum(
            log_rewards, torch.tensor(self.global_cfg.algo.illegal_action_logreward, device=dev)
        ).float()
        cond_info = batch.cond_info
        invalid_mask = 1 - batch.is_valid

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        # The position of the last graph of each trajectory
        traj_cumlen = torch.cumsum(batch.traj_lens, 0)
        final_graph_idx = traj_cumlen - 1

        fwd_cat: GraphActionCategorical  # The per-state cond_info
        batched_cond_info = cond_info[batch_idx]
        fwd_cat, per_graph_out = model.forward(batch, batched_cond_info)

        # Retreive the reward predictions for the full graphs,
        # i.e. the final graph of each trajectory
        log_reward_preds = per_graph_out[final_graph_idx, 0]

        # Compute trajectory balance objective
        log_Z = model.logZ(cond_info)[:, 0]
        # Compute the log prob of each action in the trajectory
        # Else just naively take the logprob of the actions we took
        log_p_F = fwd_cat.log_prob(batch.actions)
        log_p_B = batch.log_p_B
        assert log_p_F.shape == log_p_B.shape

        # This is the log probability of each trajectory
        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")

        # Compute log numerator and denominator of the TB objective
        numerator = log_Z + traj_log_p_F
        denominator = clip_log_R + traj_log_p_B

        if self.mask_invalid_rewards:
            # Instead of being rude to the model and giving a
            # logreward of -100 what if we say, whatever you think the
            # logprobablity of this trajetcory is it should be smaller
            # (thus the `numerator - 1`). Why 1? Intuition?
            denominator = denominator * (1 - invalid_mask) + invalid_mask * (numerator.detach() - 1)

        if self.cfg.epsilon is not None:
            # Numerical stability epsilon
            epsilon = torch.tensor([self.cfg.epsilon], device=dev).float()
            numerator = torch.logaddexp(numerator, epsilon)
            denominator = torch.logaddexp(denominator, epsilon)
        traj_losses = self._loss(numerator - denominator, self.tb_loss)

        # Normalize losses by trajectory length
        if self.length_normalize_losses:
            traj_losses = traj_losses / batch.traj_lens
        if self.reward_normalize_losses:
            # multiply each loss by how important it is, using R as the importance factor
            # factor = Rp.exp() / Rp.exp().sum()
            factor = -clip_log_R.min() + clip_log_R + 1
            factor = factor / factor.sum()
            assert factor.shape == traj_losses.shape
            # * num_trajs because we're doing a convex combination, and a .mean() later, which would
            # undercount (by 2N) the contribution of each loss
            traj_losses = factor * traj_losses * num_trajs

        if self.cfg.bootstrap_own_reward:
            num_bootstrap = num_bootstrap or len(log_rewards)
            reward_losses = self._loss(log_rewards[:num_bootstrap] - log_reward_preds[:num_bootstrap], self.reward_loss)

            reward_loss = reward_losses.mean() * self.cfg.reward_loss_multiplier
        else:
            reward_loss = 0

        tb_loss = traj_losses.mean()
        loss = tb_loss + reward_loss
        info = {
            # "offline_loss": traj_losses[: batch.num_offline].mean() if batch.num_offline > 0 else 0,
            # "online_loss": traj_losses[batch.num_offline :].mean() if batch.num_online > 0 else 0,
            # "reward_loss": reward_loss,
            "invalid_trajectories": invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
            # "invalid_logprob": (invalid_mask * traj_log_p_F).sum() / (invalid_mask.sum() + 1e-4),
            # "invalid_losses": (invalid_mask * traj_losses).sum() / (invalid_mask.sum() + 1e-4),
            "logZ": log_Z.mean(),
            "loss": loss.item(),
            # "tb_loss": tb_loss.item(),
            "batch_entropy": -traj_log_p_F.mean(),
            "traj_lens": batch.traj_lens.float().mean(),
        }

        return loss, info

    def _loss(self, x, loss_fn=None):
        if loss_fn is None:
            loss_fn = self.cfg.loss_fn
        if loss_fn == LossFN.MSE:
            return x * x
        elif loss_fn == LossFN.MAE:
            return torch.abs(x)
        elif loss_fn == LossFN.HUB:
            ax = torch.abs(x)
            d = self.cfg.loss_fn_par
            return torch.where(ax < 1, 0.5 * x * x / d, ax / d - 0.5 / d)
        elif loss_fn == LossFN.GHL:
            ax = self.cfg.loss_fn_par * x
            return torch.logaddexp(ax, -ax) - np.log(2)
        else:
            raise NotImplementedError()
