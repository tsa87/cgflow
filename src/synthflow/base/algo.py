import numpy as np
import torch
from torch import Tensor

from gflownet.envs.graph_building_env import ActionIndex
from rxnflow.algo.synthetic_path_sampling import SyntheticPathSampler
from rxnflow.algo.trajectory_balance import SynthesisTB
from rxnflow.envs import MolGraph, RxnAction, RxnActionType
from rxnflow.envs.retrosynthesis import RetroSynthesisTree
from rxnflow.models.gfn import RxnFlow
from rxnflow.policy.action_categorical import RxnActionCategorical
from rxnflow.policy.action_space_subsampling import SubsamplingPolicy
from synthflow.base.env import SynthesisEnv3D, SynthesisEnvContext3D


class SynthesisTB3D(SynthesisTB):
    env: SynthesisEnv3D
    ctx: SynthesisEnvContext3D

    def setup_graph_sampler(self):
        self.graph_sampler = SyntheticPathSampler3D(
            self.ctx,
            self.env,
            self.action_subsampler,
            max_len=self.max_len,
            importance_temp=self.importance_temp,
            sample_temp=self.sample_temp,
            correct_idempotent=self.cfg.do_correct_idempotent,
            pad_with_terminal_state=self.cfg.do_parameterize_p_b,
            num_workers=self.global_cfg.num_workers_retrosynthesis,
        )


class SyntheticPathSampler3D(SyntheticPathSampler):
    """A helper class to sample from ActionCategorical-producing models"""

    env: SynthesisEnv3D
    ctx: SynthesisEnvContext3D

    def __init__(
        self,
        ctx: SynthesisEnvContext3D,
        env: SynthesisEnv3D,
        action_subsampler: SubsamplingPolicy,
        max_len: int = 4,
        importance_temp: float = 1,
        sample_temp: float = 1,
        correct_idempotent: bool = False,
        pad_with_terminal_state: bool = False,
        num_workers: int = 4,
    ):
        super().__init__(
            ctx,
            env,
            action_subsampler,
            max_len,
            importance_temp,
            sample_temp,
            correct_idempotent,
            pad_with_terminal_state,
            num_workers,
        )

    def sample_from_model(
        self,
        model: RxnFlow,
        n: int,
        cond_info: Tensor,
        random_action_prob: float = 0.0,
    ) -> list[dict]:
        """Samples a model in a minibatch

        Parameters
        ----------
        model: nn.Module
            Model whose forward() method returns ActionCategorical instances
        n: int
            Number of graphs to sample
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)

        Returns
        -------
        data: list[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: list[Tuple[Graph, RxnAction]], the list of states and actions
           - fwd_logprob: sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """
        # This will be returned
        data = [{"traj": [], "reward_pred": None, "is_valid": True, "is_sink": []} for _ in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        fwd_logprob: list[list[float]] = [[] for _ in range(n)]
        bck_logprob: list[list[float]] = [[] for _ in range(n)]

        fwd_a: list[list[RxnAction]] = [[] for _ in range(n)]
        bck_a: list[list[RxnAction]] = [[RxnAction(RxnActionType.Stop)] for _ in range(n)]

        graphs: list[MolGraph] = [self.env.new() for _ in range(n)]
        retro_trees: list[RetroSynthesisTree] = [RetroSynthesisTree("")] * n
        done: list[bool] = [False] * n

        def not_done(lst) -> list[int]:
            return [e for i, e in enumerate(lst) if not done[i]]

        def set_invalid(i: int):
            done[i] = True
            data[i]["is_valid"] = False
            data[i]["is_sink"].append(1)
            bck_logprob[i].append(0.0)

        for traj_idx in range(self.max_len):
            # Label the state is last or not
            is_last_step = traj_idx == (self.max_len - 1)
            for i in not_done(range(n)):
                graphs[i].graph.update({"is_last_step": is_last_step, "sample_idx": i})

            # Construct graphs for the trajectories that aren't yet done
            torch_graphs = [self.ctx.graph_to_Data(graphs[i]) for i in not_done(range(n))]
            not_done_mask = [not v for v in done]

            # NOTE: forward transition probability (forward policy) estimation
            fwd_cat: RxnActionCategorical = self._estimate_policy(model, torch_graphs, cond_info, not_done_mask)
            actions: list[ActionIndex] = self._sample_action(torch_graphs, fwd_cat, random_action_prob)
            reaction_actions: list[RxnAction] = [
                self.ctx.ActionIndex_to_GraphAction(g, a) for g, a in zip(torch_graphs, actions, strict=True)
            ]
            log_probs = fwd_cat.log_prob(actions)

            # NOTE: Step each trajectory, and accumulate statistics
            for i, j in zip(not_done(range(n)), range(n), strict=False):
                data[i]["traj"].append((graphs[i], reaction_actions[j]))
                fwd_logprob[i].append(log_probs[j].item())
                # bck_logprob[i].append(0.0)  # for cgflow, output depends on trajectory
                fwd_a[i].append(reaction_actions[j])
                bck_a[i].append(self.env.reverse(graphs[i], reaction_actions[j]))
                try:
                    graphs[i] = g = self.env.step(graphs[i], reaction_actions[j])
                    assert g.mol is not None
                    assert g.mol.GetNumHeavyAtoms() < 40  # HACK: is there better way to control this hparam?
                except AssertionError:
                    set_invalid(i)
                    continue
                # NOTE: lazy retrosynthetic analysis
                self.retro_analyzer.submit(i, g.smi, traj_idx, [(bck_a[i][-1], retro_trees[i])])
                if self.is_terminate(g.mol):
                    done[i] = True
                    data[i]["is_sink"].append(1)
                else:
                    data[i]["is_sink"].append(0)

            # NOTE: run binding pose prediction with cgflow
            valid_graphs = [graphs[i] for i in range(n) if data[i]["is_valid"]]
            self.ctx.set_binding_pose_batch(valid_graphs, traj_idx, is_last_step=all(done))
            for g in graphs:
                i = g.graph["sample_idx"]
                pos = np.array(g.mol.GetConformer().GetPositions())
                if not np.all(np.isfinite(pos)):
                    set_invalid(i)

            # NOTE: lazy retrosynthetic analysis
            for i, analysis_res in self.retro_analyzer.result():
                if data[i]["is_valid"] is False:
                    continue
                if analysis_res is None:
                    set_invalid(i)
                else:
                    retro_trees[i] = analysis_res
                    bck_logprob[i].append(self.calc_bck_logprob(bck_a[i][-1], analysis_res))

            if all(done):
                break
        assert all(done)

        for i in range(n):
            data[i]["fwd_logprob"] = sum(fwd_logprob[i])
            data[i]["bck_logprob"] = sum(bck_logprob[i])
            data[i]["bck_logprobs"] = torch.tensor(bck_logprob[i]).reshape(-1)
            data[i]["result"] = graphs[i]
            data[i]["bck_a"] = bck_a[i]
        return data

    def sample_inference(
        self,
        model: RxnFlow,
        n: int,
        cond_info: Tensor,
    ) -> list[dict]:
        """Model Sampling (Inference - Non Retrosynthetic Analysis)

        Parameters
        ----------
        model: RxnFlow
            Model whose forward() method returns RxnActionCategorical instances
        n: int
            Number of samples
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)

        Returns
        -------
        data: list[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: list[Tuple[Chem.Mol, RxnAction]], the list of states and actions
           - fwd_logprob: P_F(tau)
           - is_valid: is the generated graph valid according to the env & ctx
        """
        # This will be returned
        data = [{"traj": [], "reward_pred": None, "is_valid": True} for _ in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        fwd_logprob: list[list[float]] = [[] for _ in range(n)]

        fwd_a: list[list[RxnAction]] = [[] for _ in range(n)]

        graphs: list[MolGraph] = [self.env.new() for _ in range(n)]
        done: list[bool] = [False] * n

        def not_done(lst) -> list[int]:
            return [e for i, e in enumerate(lst) if not done[i]]

        def set_invalid(i: int):
            done[i] = True
            data[i]["is_valid"] = False

        for traj_idx in range(self.max_len):
            # Label the state is last or not
            is_last_step = traj_idx == (self.max_len - 1)
            for i in not_done(range(n)):
                graphs[i].graph.update({"is_last_step": is_last_step, "sample_idx": i})

            # Construct graphs for the trajectories that aren't yet done
            torch_graphs = [self.ctx.graph_to_Data(graphs[i]) for i in not_done(range(n))]
            not_done_mask = [not v for v in done]

            # NOTE: forward transition probability (forward policy) estimation
            fwd_cat: RxnActionCategorical = self._estimate_policy(model, torch_graphs, cond_info, not_done_mask)
            actions: list[ActionIndex] = self._sample_action(torch_graphs, fwd_cat, random_action_prob=0.0)
            reaction_actions: list[RxnAction] = [
                self.ctx.ActionIndex_to_GraphAction(g, a) for g, a in zip(torch_graphs, actions, strict=True)
            ]
            log_probs = fwd_cat.log_prob(actions)

            # NOTE: Step each trajectory, and accumulate statistics
            for i, j in zip(not_done(range(n)), range(n), strict=False):
                data[i]["traj"].append((graphs[i], reaction_actions[j]))
                fwd_logprob[i].append(log_probs[j].item())
                fwd_a[i].append(reaction_actions[j])
                try:
                    graphs[i] = g = self.env.step(graphs[i], reaction_actions[j])
                    assert g.mol is not None
                except AssertionError:
                    set_invalid(i)
                else:
                    if self.is_terminate(g.mol):
                        done[i] = True

            # NOTE: run binding pose prediction with cgflow
            valid_graphs = [graphs[i] for i in range(n) if data[i]["is_valid"]]
            self.ctx.set_binding_pose_batch(valid_graphs, traj_idx, is_last_step=all(done))
            for g in graphs:
                i = g.graph["sample_idx"]
                pos = np.array(g.mol.GetConformer().GetPositions())
                if not np.all(np.isfinite(pos)):
                    set_invalid(i)

            if all(done):
                break
        assert all(done)

        for i in range(n):
            data[i]["fwd_logprob"] = sum(fwd_logprob[i])
            data[i]["result"] = graphs[i]
            data[i]["result_rdmol"] = self.ctx.graph_to_obj(graphs[i])
        return data
