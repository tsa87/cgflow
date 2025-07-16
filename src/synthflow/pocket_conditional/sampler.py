from pathlib import Path
from typing import Any

from typing_extensions import override

from synthflow.base.sampler import SynthFlowSampler


class PocketConditionalSampler(SynthFlowSampler):
    def set_pocket(
        self,
        protein_path: str | Path,
        center: tuple[float, float, float] | None = None,
        ref_ligand_path: str | Path | None = None,
        extract: bool = True,
    ):
        """setup pose prediction model"""
        self.ctx.set_pocket(protein_path, center, ref_ligand_path, extract)

    def sample_against_pocket(
        self,
        n: int,
        protein_path: str | Path,
        center: tuple[float, float, float] | None = None,
        ref_ligand_path: str | Path | None = None,
    ):
        """setup pose prediction model"""
        self.set_pocket(protein_path, center, ref_ligand_path)
        return self.sample(n)

    @override
    def sample(self, n: int, calc_reward: bool = False) -> list[dict[str, Any]]:
        assert calc_reward is False
        return super().sample(n, calc_reward)
