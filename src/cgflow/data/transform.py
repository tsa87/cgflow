from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from cgflow.util.data.pocket import PocketComplex
from cgflow.util.registry import TRANSFORM


@dataclass
class TransformConfig:
    _registry_: ClassVar[str] = "transform"
    _type_: str


@dataclass
class IdenticalTransformConfig(TransformConfig):
    _type_: str = "IdenticalTransform"


@dataclass
class ComplexTransformConfig(TransformConfig):
    _type_: str = "ComplexTransform"
    radius: float | None = 15.0  # radius for residue filtering
    rotate: bool = True
    zero_com: str | None = "pocket"  # pocket, ligand, None (no zero com)
    center_noise: float = 0.0


@TRANSFORM.register(config=TransformConfig)
class Transform(ABC):
    def __init__(self, config: TransformConfig, mode: str = "train"):
        self.config: TransformConfig = config
        self.mode: str = mode

    def __call__(self, data: Any) -> Any:
        return self.transform(data)

    @abstractmethod
    def transform(self, data: Any) -> Any:
        """transform input data"""


@TRANSFORM.register(config=IdenticalTransformConfig)
class IdenticalTransform(Transform):
    def transform(self, data: Any) -> Any:
        return data


@TRANSFORM.register(config=ComplexTransformConfig)
class ComplexTransform(Transform):
    """
    Applies transformations to a protein-ligand complex:
    1. Normalizes coordinate values by dividing by coord_std.
    2. If fix_pos is False, applies a random 3D rotation to the entire complex.
    3. Removes the center of mass of the holo pocket from the complex.
    6. Converts the holo pocket into a geometric molecular representation.
    7. Converts the holo's atomic numbers and charges to categorical indices.
    8. Restores the holo pocket's original scale (for evaluating PoseCheck metrics).
    """

    def __init__(self, config: ComplexTransformConfig, mode: str = "train"):
        super().__init__(config, mode)
        self.radius: float | None = config.radius
        self.rotate: bool = config.rotate
        self.zero_com: str | None = config.zero_com
        self.center_noise: float = config.center_noise
        assert self.zero_com in ("pocket", "ligand", None)

    def transform(self, data: PocketComplex) -> PocketComplex:
        # rotate
        if self.rotate and self.mode == "train":
            rotation = np.random.rand(3) * np.pi * 2
            data = data.rotate(rotation)

        # set com
        if self.zero_com == "pocket":
            data = data.zero_holo_com()
        elif self.zero_com == "ligand":
            data = data.zero_ligand_com()

        # add center noise
        if self.center_noise > 0.0 and self.mode == "train":
            shift = np.random.randn(3).astype(np.float32) * self.center_noise
            data = data.shift(shift)

        # residue filtering
        if self.radius is not None:
            centriod = np.zeros(3, dtype=np.float32)
            holo_subset = data.holo.select_residues_by_distance(centriod, self.radius)
            assert len(holo_subset) > 0, "No holo subset found"
            data = data.copy_with(holo=holo_subset)

        # get holo-mol(geomol)
        holo_mol = data.holo.to_geometric_mol()
        data._holo_mol = holo_mol
        return data
