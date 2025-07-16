import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch

from gflownet.utils.misc import get_worker_device, get_worker_rng
from rxnflow.config import Config
from rxnflow.envs.action import Protocol, RxnActionType
from rxnflow.envs.env import SynthesisEnv


class ActionSpace:
    def __init__(self, num_actions: int, sampling_ratio: float, min_sampling: int):
        assert sampling_ratio <= 1
        min_sampling = min(num_actions, min_sampling)
        self.num_actions: int = num_actions
        self.num_sampling = max(int(num_actions * sampling_ratio), min_sampling)
        self.sampling_ratio: float = self.num_sampling / self.num_actions
        self.rng: np.random.RandomState = get_worker_rng()

    def sampling(self) -> torch.Tensor:
        # TODO: introduce importance subsampling instead of uniform subsampling
        if self.sampling_ratio < 1:
            indices = self.rng.choice(self.num_actions, self.num_sampling, replace=False)
            np.sort(indices)
            return torch.from_numpy(indices).to(torch.long)
        else:
            return torch.arange((self.num_actions), dtype=torch.long)


@dataclass(frozen=True, slots=True)
class ActionSubspaceForProtocol:
    protocol: str
    subsamples: OrderedDict[str, torch.Tensor]  # [protocol, num_actions]
    weights: torch.Tensor  # [num_total_actions,]

    def items(self):
        return self.subsamples.items()

    def values(self):
        return self.subsamples.values()

    def keys(self):
        return self.subsamples.keys()

    @property
    def num_clusters(self) -> int:
        return len(self.subsamples)

    @property
    def num_actions(self) -> int:
        return self.weights.shape[0]


@dataclass(frozen=True, slots=True)
class ActionSubspace:
    _subspaces: OrderedDict[str, ActionSubspaceForProtocol]

    def __getitem__(self, protocol: str | int) -> ActionSubspaceForProtocol:
        if isinstance(protocol, int):
            protocol = list(self._subspaces.keys())[protocol]
        return self._subspaces[protocol]

    def items(self):
        return self._subspaces.items()

    def keys(self):
        return self._subspaces.keys()

    def values(self):
        return self._subspaces.values()


class SubsamplingPolicy:
    def __init__(self, env: SynthesisEnv, cfg: Config):
        self.global_cfg = cfg
        self.cfg = cfg.algo.action_subsampling
        self.protocols = env.protocols

        self.block_spaces: dict[str, ActionSpace] = {}
        self.num_blocks: dict[str, int] = {}

        # n-total-linker-blocks = O(n-brick-types * n-total-brick-blocks)
        # therefore, we use lower sampling ratio for linkers
        is_linker = lambda t: "-" in t  # noqa
        num_brick_types = sum(not is_linker(t) for t in env.blocks.keys())
        num_linker_types = sum(is_linker(t) for t in env.blocks.keys())
        linker_coeff = num_linker_types / num_brick_types

        # sampling ratio
        sr = self.cfg.sampling_ratio
        nmin = int(self.cfg.min_sampling)

        for block_type, blocks in env.blocks.items():
            if is_linker(block_type):
                sampling_ratio = sr / linker_coeff
            else:
                sampling_ratio = sr
            self.block_spaces[block_type] = ActionSpace(len(blocks), sampling_ratio, nmin)

        dev = get_worker_device()
        self.protocol_weights: dict[str, torch.Tensor] = {}
        for protocol in env.protocols:
            weight_list: list[float] = []
            for block_type in protocol.block_types:
                # importance weight
                space = self.block_spaces[block_type]
                weight = math.log(1 / space.sampling_ratio)
                weight_list += [weight] * space.num_sampling
            weights = torch.tensor(weight_list, dtype=torch.float32, device=dev)
            self.protocol_weights[protocol.name] = weights

    def sample(self) -> ActionSubspace:
        subspace: OrderedDict[str, ActionSubspaceForProtocol] = OrderedDict()
        for protocol in self.protocols:
            if protocol.action not in [RxnActionType.FirstBlock, RxnActionType.BiRxn]:
                continue
            subspace[protocol.name] = self.sample_protocol(protocol)
        return ActionSubspace(subspace)

    def sample_protocol(self, protocol: Protocol) -> ActionSubspaceForProtocol:
        assert protocol.action in [RxnActionType.FirstBlock, RxnActionType.BiRxn]
        subsample: OrderedDict[str, torch.Tensor] = OrderedDict()
        for block_type in protocol.block_types:
            # subsampling
            action_space = self.block_spaces[block_type]
            subsample[block_type] = action_space.sampling()
        weights = self.protocol_weights[protocol.name]
        return ActionSubspaceForProtocol(protocol.name, subsample, weights)
