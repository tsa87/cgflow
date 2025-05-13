import math
from collections import OrderedDict

import torch
import torch_geometric.data as gd
from torch import Tensor

from gflownet.envs.graph_building_env import ActionIndex, GraphActionCategorical
from rxnflow.envs.action import Protocol, RxnActionType
from rxnflow.envs.env_context import SynthesisEnvContext
from rxnflow.policy.action_space_subsampling import SubsamplingPolicy
from rxnflow.utils.misc import get_worker_env


def placeholder(size: tuple[int, ...], device: torch.device) -> Tensor:
    return torch.empty(size, dtype=torch.float32, device=device)


def neginf(size: tuple[int, ...], device: torch.device) -> Tensor:
    return torch.full(size, -torch.inf, dtype=torch.float32, device=device)


def falsetensor(size: tuple[int, ...], device: torch.device) -> Tensor:
    return torch.zeros(size, dtype=torch.bool, device=device)


class RxnActionCategorical(GraphActionCategorical):
    def __init__(
        self,
        graphs: gd.Batch,
        emb: Tensor,
        logit_scale: Tensor,
        protocol_masks: list[Tensor],
        model: torch.nn.Module,
    ):
        self.ctx: SynthesisEnvContext = get_worker_env("ctx")
        self.model = model
        self.graphs = graphs
        self.num_graphs = graphs.num_graphs
        self.emb: Tensor = emb
        self.logit_scale: Tensor = logit_scale

        self._epsilon = 1e-38
        self._log_epsilon = math.log(self._epsilon)
        self._protocol_masks: list[Tensor] = protocol_masks
        self.dev = dev = self.emb.device

        # NOTE: action subsampling
        sampler: SubsamplingPolicy = get_worker_env("action_subsampler")
        self.subsamples: list[OrderedDict[str, torch.Tensor]] = []
        self.subsample_size: list[int] = []
        self._weights: list[Tensor] = []  # importance weight
        for protocol in self.ctx.protocols:
            subsample, weights = sampler.sampling(protocol)
            self.subsamples.append(subsample)
            self.subsample_size.append(sum(v.shape[0] for v in subsample.values()))
            self._weights.append(weights)

        self._masked_logits: list[Tensor] = self._calculate_logits()

        self.raw_logits: list[Tensor] = self._masked_logits
        self.weighted_logits: list[Tensor] = self.importance_weighting(1.0)

        self.batch = [torch.arange(self.num_graphs, device=dev)] * self.ctx.num_protocols
        self.slice = [torch.arange(self.num_graphs + 1, device=dev)] * self.ctx.num_protocols

    def _calculate_logits(self) -> list[Tensor]:
        # TODO: add descriptors
        masked_logits: list[Tensor] = []
        for protocol_idx, protocol in enumerate(self.ctx.protocols):
            subsample = self.subsamples[protocol_idx]
            num_actions = self.subsample_size[protocol_idx]
            protocol_mask = self._protocol_masks[protocol_idx]
            if protocol_mask.all():
                if protocol.action in (RxnActionType.FirstBlock, RxnActionType.BiRxn):
                    # collect block data; [Nblock,], [Nblock, F], [Nblock, F]
                    block_data = self.get_block_data(subsample)
                    # calculate the logit for each action - (state, action)
                    # shape: [Nstate, Nblock]
                    logits = self.model_hook(protocol, self.emb, block_data)
                    # logit scaling
                    logits = logits * self.logit_scale.view(-1, 1)
                else:
                    raise ValueError(protocol.action)

            elif protocol_mask.any():
                if protocol.action in (RxnActionType.FirstBlock, RxnActionType.BiRxn):
                    # collect block data; [Nblock,], [Nblock, F], [Nblock, F]
                    block_data = self.get_block_data(subsample)
                    # calculate the logit for each action - (state, action)
                    # shape: [Nstate', Nblock]
                    allowed_logits = self.model_hook(protocol, self.emb[protocol_mask], block_data)
                    # logit scaling
                    allowed_logits = allowed_logits * self.logit_scale[protocol_mask].view(-1, 1)
                else:
                    raise ValueError(protocol.action)

                # create placeholder first and then insert the calculated.
                logits = neginf((self.num_graphs, num_actions), device=self.dev)
                logits[protocol_mask] = allowed_logits
            else:
                logits = neginf((self.num_graphs, num_actions), device=self.dev)

            masked_logits.append(logits)
        return masked_logits

    def get_block_data(self, subsample: OrderedDict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        block_data_list = [self.ctx.get_block_data(typ, indices) for typ, indices in subsample.items()]
        typs, props, fps = list(zip(*block_data_list, strict=True))
        typ = torch.cat(typs).to(self.dev, non_blocking=True)
        prop = torch.cat(props).to(self.dev, non_blocking=True)
        fp = torch.cat(fps).to(dtype=torch.float32, device=self.dev, non_blocking=True)
        return (typ, prop, fp)

    def model_hook(
        self,
        protocol: Protocol,
        emb: Tensor,
        block_data: tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        if protocol.action is RxnActionType.FirstBlock:
            return self.model.hook_firstblock(emb, block_data, protocol.name)
        else:
            return self.model.hook_birxn(emb, block_data, protocol.name)

    def _cal_action_logits(self, actions: list[ActionIndex]) -> Tensor:
        """Calculate the logit values for sampled actions"""
        action_logits = placeholder((len(actions),), device=self.dev)
        for i, action in enumerate(actions):
            protocol_idx, block_type_idx, block_idx = action
            protocol = self.ctx.protocols[protocol_idx]
            if protocol.action in (RxnActionType.FirstBlock, RxnActionType.BiRxn):
                block_type = protocol.block_types[int(block_type_idx)]
                typ, prop, fp = self.ctx.get_block_data(block_type, block_idx)
                block_data = (typ.to(self.dev), prop.to(self.dev), fp.to(dtype=torch.float32, device=self.dev))
                logit = self.model_hook(protocol, self.emb[i], block_data).view(-1)
            else:
                raise ValueError(protocol.action)
            action_logits[i] = logit
        # logit scaling
        action_logits = action_logits * self.logit_scale.view(len(actions))
        return action_logits

    # NOTE: Function override
    def sample(self) -> list[ActionIndex]:
        """Sample the action
        Since we perform action space subsampling, the indices of block is from the partial space.
        Therefore, we reassign the block indices on the entire block library.
        """
        action_list = super().sample()
        reindexed_actions: list[ActionIndex] = []
        for action in action_list:
            protocol_idx, row_idx, action_idx = action
            assert row_idx == 0
            action_type = self.ctx.protocols[protocol_idx].action
            if action_type in (RxnActionType.FirstBlock, RxnActionType.BiRxn):
                ofs = action_idx
                _action = None
                for block_type_idx, use_block_idcs in enumerate(self.subsamples[protocol_idx].values()):
                    assert ofs >= 0
                    if ofs < len(use_block_idcs):
                        block_idx = int(use_block_idcs[ofs])
                        _action = ActionIndex(protocol_idx, block_type_idx, block_idx)
                        break
                    else:
                        ofs -= len(use_block_idcs)
                assert _action is not None
                action = _action
            else:
                raise ValueError(action)
            reindexed_actions.append(action)
        return reindexed_actions

    def log_prob(
        self,
        actions: list[ActionIndex],
        logprobs: Tensor | None = None,
        batch: Tensor | None = None,
    ) -> Tensor:
        """The log-probability of a list of action tuples, effectively indexes `logprobs` using internal
        slice indices.

        Parameters
        ----------
        actions: List[ActionIndex]
            A list of n action tuples denoting indices
        logprobs: None (dummy)
        batch: None (dummy)

        Returns
        -------
        action_logprobs: Tensor
            The log probability of each action.
        """
        assert logprobs is None
        assert batch is None

        # when graph-wise prediction is only performed
        logits = self.weighted_logits  # use logit from importance weighting
        maxl: Tensor = self._compute_batchwise_max(logits).values  # [Ngraph,]
        corr_logits: list[Tensor] = [(i - maxl.unsqueeze(1)) for i in logits]
        exp_logits: list[Tensor] = [i.exp().clamp(self._epsilon) for i in corr_logits]
        logZ: Tensor = sum([i.sum(1) for i in exp_logits]).log()

        action_logits = self._cal_action_logits(actions) - maxl
        action_logprobs = (action_logits - logZ).clamp(max=0.0)
        return action_logprobs

    def importance_weighting(self, alpha: float = 1.0) -> list[Tensor]:
        if alpha == 0.0:
            return self.logits
        else:
            return [logits + alpha * w.view(1, -1) for logits, w in zip(self.logits, self._weights, strict=True)]

    def _apply_action_masks(self):
        def _mask(logits, protocol_mask):
            logits.masked_fill_(~protocol_mask.view(-1, 1), -torch.inf)
            return logits

        self._masked_logits = [
            _mask(logits, pm) for logits, pm in zip(self.raw_logits, self._protocol_masks, strict=True)
        ]

    # NOTE: same but 10x faster (optimized for graph-wise predictions)
    def argmax(
        self,
        x: list[Tensor],
        batch: list[Tensor] | None = None,
        dim_size: int | None = None,
    ) -> list[ActionIndex]:
        # Find protocol type
        max_per_type = [torch.max(tensor, dim=1) for tensor in x]
        max_values_per_type = [pair[0] for pair in max_per_type]
        type_max: list[int] = torch.max(torch.stack(max_values_per_type), dim=0)[1].tolist()
        assert len(type_max) == self.num_graphs

        # find action indexes
        col_max_per_type = [pair[1] for pair in max_per_type]
        col_max: list[int] = [int(col_max_per_type[t][i]) for i, t in enumerate(type_max)]

        # return argmaxes
        argmaxes = [ActionIndex(i, 0, j) for i, j in zip(type_max, col_max, strict=True)]
        return argmaxes

    # NOTE: same but faster (optimized for graph-wise predictions)
    def _compute_batchwise_max(
        self,
        x: list[Tensor],
        detach: bool = True,
        batch: list[Tensor] | None = None,
        reduce_columns: bool = True,
    ):
        if detach:
            x = [i.detach() for i in x]
        if batch is None:
            batch = self.batch
        if reduce_columns:
            return torch.cat(x, dim=1).max(1)
        return [(i, b.view(-1, 1).repeat(1, i.shape[1])) for i, b in zip(x, batch, strict=True)]
