from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch

from cgflow.util.registry import PRIOR_DISTRIBUTION


@dataclass
class PriorDistributionConfig:
    _registry_: ClassVar[str] = "prior_distribution"
    _type_: str


@dataclass
class GaussianPriorDistributionConfig(PriorDistributionConfig):
    _type_: str = "GaussianPriorDistribution"
    noise_std: float = 1.0


@PRIOR_DISTRIBUTION.register(config=PriorDistributionConfig)
class PriorDistribution(ABC):
    def __init__(self, config: PriorDistributionConfig):
        self.config = config

    @abstractmethod
    def sample(self, coords: torch.Tensor) -> torch.Tensor: ...

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        return self.sample(coords)


@PRIOR_DISTRIBUTION.register(config=GaussianPriorDistributionConfig)
class GaussianPriorDistribution(PriorDistribution):
    def __init__(self, config: GaussianPriorDistributionConfig):
        self.config = config
        self.noise_std: float = config.noise_std

    def sample(self, coords: torch.Tensor) -> torch.Tensor:
        prior = torch.randn_like(coords)
        if self.noise_std != 1.0:
            prior = prior * self.noise_std
        return prior
