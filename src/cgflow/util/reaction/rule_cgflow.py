from omegaconf import DictConfig

from .rule import Rule


class CGFlowRule(Rule):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.order: list[tuple[int, int]] = config.order
