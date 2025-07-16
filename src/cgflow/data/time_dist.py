from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch

from cgflow.util.registry import TIME_DISTRIBUTION


@dataclass
class TimeDistributionConfig:
    _registry_: ClassVar[str] = "time_distribution"
    _type_: str


@dataclass
class ConstantTimeDistributionConfig(TimeDistributionConfig):
    _type_: str = "ConstantTimeDistribution"
    time: float = 0.9


@dataclass
class UniformTimeDistributionConfig(TimeDistributionConfig):
    _type_: str = "UniformTimeDistribution"


@dataclass
class BetaTimeDistributionConfig(TimeDistributionConfig):
    _type_: str = "UniformTimeDistribution"
    alpha: float = 1.0
    beta: float = 1.0


@TIME_DISTRIBUTION.register(config=TimeDistributionConfig)
class TimeDistribution(ABC):
    def __init__(self, config: TimeDistributionConfig):
        self.config = config

    def __call__(self) -> float:
        return self.sample()

    @abstractmethod
    def sample(self) -> float: ...


@TIME_DISTRIBUTION.register(config=ConstantTimeDistributionConfig)
class ConstantTimeDistribution(TimeDistribution):
    def __init__(self, config: ConstantTimeDistributionConfig):
        self.time = config.time
        if config.time is not None and (config.time < 0 or config.time > 1):
            raise ValueError("time must be between 0 and 1 if provided.")

    def sample(self) -> float:
        return self.time


@TIME_DISTRIBUTION.register(config=BetaTimeDistributionConfig)
class BetaTimeDistribution(TimeDistribution):
    def __init__(self, config: BetaTimeDistributionConfig):
        self.dist = torch.distributions.Beta(config.alpha, config.beta)

    def sample(self) -> float:
        return self.dist.sample().item()


@TIME_DISTRIBUTION.register(config=UniformTimeDistributionConfig)
class UniformTimeDistribution(TimeDistribution):
    def __init__(self, config: UniformTimeDistributionConfig):
        self.dist = torch.distributions.Uniform(0, 1)

    def sample(self) -> float:
        return self.dist.sample().item()
