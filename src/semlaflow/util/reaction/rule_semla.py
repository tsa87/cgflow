from .rule import Rule

from omegaconf import DictConfig


class SemlaRule(Rule):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.order: list[tuple[int, int]] = config.order
