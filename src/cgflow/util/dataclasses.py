from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Generic, Self, TypeVar

import torch

# Define a TypeVar for the specific BaseTensor subtype
_TensorType = TypeVar("_TensorType", bound="BaseTensor")


def _check_tensor(t: torch.Tensor, shape: tuple[int, ...], dtype: Sequence[torch.dtype]) -> None:
    """Check if the tensor has the expected shape."""
    assert t.ndim == len(shape), f"Expected tensor ndim {len(shape)}, but got {t.ndim}."
    shape = tuple(shape[i] if v == -1 else v for i, v in enumerate(t.shape))  # broadcasting
    assert t.shape == shape, f"Expected tensor shape {shape}, but got {t.shape}."
    assert t.dtype in dtype, f"Expected tensor dtype in {dtype}, but got {t.dtype}."


@dataclass
class BaseTensor(ABC):
    def to_dict(self) -> dict[str, torch.Tensor]:
        """return a dictionary representation of the tensor."""
        # NOTE: dataclasses.asdict and dataclasses.astuple use deep-copy instead of shallow-copy
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def to(self, device: str | torch.device) -> Self:
        """Convert the tensor to a specific device and/or dtype."""
        return self.__class__(**{k: v.to(device) for k, v in self.to_dict().items()})

    def __post_init__(self):
        """Post-initialization hook to ensure all tensors are on the same device."""
        dev = self.device
        devices = {k: v.device for k, v in self.to_dict().items()}
        assert all(dev == v for v in devices.values()), f"All tensors must be on the same device. {devices}"
        self.check_tensors()

    def __len__(self) -> int:
        """Return the length of the tensor."""
        return self.length

    def check_tensors(self):  # noqa
        """cehck tensors for consistency"""
        pass

    @property
    @abstractmethod
    def length(self) -> int:
        """Return the length of the tensor."""
        raise NotImplementedError("Subclasses must implement the length property.")

    @property
    def device(self) -> torch.device:
        """Return the device of the tensor."""
        return next(iter(self.to_dict().values())).device

    def copy_with(self, **kwargs: torch.Tensor) -> Self:
        """shallow copy with given attributes"""
        attrs = self.to_dict()
        assert kwargs.keys() <= attrs.keys(), "Cannot set new attributes in copy_with."
        attrs = attrs | kwargs
        return self.__class__(**attrs)


@dataclass
class BatchTensor(BaseTensor, Generic[_TensorType], ABC):
    @property
    @abstractmethod
    def batch_size(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> _TensorType: ...

    @classmethod
    @abstractmethod
    def from_tensors(cls, tensors: Sequence[_TensorType], max_length: int | None) -> Self:
        """Construct from Tensors."""

    @abstractmethod
    def to_tensors(self) -> list[_TensorType]:
        """Construct from Tensors."""

    @staticmethod
    def get_stacked_length(tensors: Sequence[_TensorType], max_length: int | None) -> int:
        """Get the maximum length of the tensors."""
        L = max(len(tensor) for tensor in tensors)
        if max_length is not None:
            assert L <= max_length, f"Tensor length {L} exceeds max_length {max_length}."
            L = min(L, max_length)
        return L


@dataclass
class LigandTensor(BaseTensor):
    atoms: torch.Tensor  # [L,]
    charges: torch.Tensor  # [L,]
    attachments: torch.Tensor  # [L,]
    adjacency: torch.Tensor  # [L, L]
    coords: torch.Tensor  # [L, 3]

    @property
    def length(self) -> int:
        return self.atoms.shape[0]

    def check_tensors(self):
        L = len(self)
        _check_tensor(self.atoms, (L,), [torch.int32, torch.int64])
        _check_tensor(self.charges, (L,), [torch.int32, torch.int64])
        _check_tensor(self.adjacency, (L, L), [torch.int32, torch.int64])
        _check_tensor(self.coords, (L, 3), [torch.bfloat16, torch.float16, torch.float32])


@dataclass
class PocketTensor(BaseTensor):
    atoms: torch.Tensor  # [L,]
    charges: torch.Tensor  # [L,]
    residues: torch.Tensor  # [L,]
    residue_ids: torch.Tensor  # [L,]
    adjacency: torch.Tensor  # [L, L]
    coords: torch.Tensor  # [L, 3]

    @property
    def length(self) -> int:
        return self.atoms.shape[0]

    def check_tensors(self):
        L = len(self)
        _check_tensor(self.atoms, (L,), [torch.int32, torch.int64])
        _check_tensor(self.charges, (L,), [torch.int32, torch.int64])
        _check_tensor(self.residues, (L,), [torch.int32, torch.int64])
        _check_tensor(self.residue_ids, (L,), [torch.int32, torch.int64])
        _check_tensor(self.adjacency, (L, L), [torch.int32, torch.int64])
        _check_tensor(self.coords, (L, 3), [torch.bfloat16, torch.float16, torch.float32])


@dataclass
class ConditionTensor(BaseTensor):
    time_cond: torch.Tensor  # [L, 3] # time(identical), gen-time, rel-time
    self_cond: torch.Tensor  # [L, 3]

    @property
    def length(self) -> int:
        return self.self_cond.shape[0]

    def check_tensors(self):
        L = len(self)
        _check_tensor(self.time_cond, (L, -1), [torch.bfloat16, torch.float16, torch.float32])
        _check_tensor(self.self_cond, (L, 3), [torch.bfloat16, torch.float16, torch.float32])


@dataclass
class LigandBatch(BatchTensor[LigandTensor]):
    atoms: torch.Tensor  # [B, L]
    charges: torch.Tensor  # [B, L]
    attachments: torch.Tensor  # [B, L]
    adjacency: torch.Tensor  # [B, L, L]
    coords: torch.Tensor  # [B, L, 3]
    mask: torch.Tensor  # [B, L]

    @property
    def batch_size(self) -> int:
        return self.atoms.shape[0]

    @property
    def length(self) -> int:
        return self.atoms.shape[1]

    def check_tensors(self):
        B = self.batch_size
        L = len(self)
        _check_tensor(self.atoms, (B, L), [torch.int32, torch.int64])
        _check_tensor(self.charges, (B, L), [torch.int32, torch.int64])
        _check_tensor(self.attachments, (B, L), [torch.bool])
        _check_tensor(self.adjacency, (B, L, L), [torch.int32, torch.int64])
        _check_tensor(self.coords, (B, L, 3), [torch.bfloat16, torch.float16, torch.float32])
        _check_tensor(self.mask, (B, L), [torch.bool])

    def __getitem__(self, index: int) -> LigandTensor:
        length = self.mask[index].sum().item()  # Get the length of the tensor at index
        atoms = self.atoms[index, :length]
        charges = self.charges[index, :length]
        attachments = self.attachments[index, :length]
        adjacency = self.adjacency[index, :length, :length]
        coords = self.coords[index, :length]
        return LigandTensor(atoms, charges, attachments, adjacency, coords)

    @classmethod
    def from_tensors(cls, tensors: Sequence[LigandTensor], max_length: int | None = None) -> Self:
        B = len(tensors)
        assert B > 0, "Cannot create a batch with zero tensors."
        L = cls.get_stacked_length(tensors, max_length)
        dev = tensors[0].device
        base_t = tensors[0]
        atoms = torch.zeros((B, L), dtype=base_t.atoms.dtype, device=dev)
        charges = torch.zeros((B, L), dtype=base_t.charges.dtype, device=dev)
        attachments = torch.zeros((B, L), dtype=base_t.attachments.dtype, device=dev)
        adjacency = torch.zeros((B, L, L), dtype=base_t.adjacency.dtype, device=dev)
        coords = torch.zeros((B, L, 3), dtype=base_t.coords.dtype, device=dev)
        mask = torch.zeros((B, L), dtype=torch.bool, device=dev)
        for i, t in enumerate(tensors):
            length = len(t)
            atoms[i, :length] = t.atoms
            charges[i, :length] = t.charges
            attachments[i, :length] = t.attachments
            adjacency[i, :length, :length] = t.adjacency
            coords[i, :length, :] = t.coords
            mask[i, :length] = True
        return cls(atoms, charges, attachments, adjacency, coords, mask)

    def to_tensors(self) -> list[LigandTensor]:
        tensors: list[LigandTensor] = []
        length_list = self.mask.sum(dim=1).tolist()  # [B,]
        for i, length in enumerate(length_list):
            atoms = self.atoms[i, :length]
            charges = self.charges[i, :length]
            attachments = self.attachments[i, :length]
            adjacency = self.adjacency[i, :length, :length]
            coords = self.coords[i, :length]
            tensors.append(LigandTensor(atoms, charges, attachments, adjacency, coords))
        return tensors


@dataclass
class PocketBatch(BatchTensor[PocketTensor]):
    atoms: torch.Tensor  # [B, L,]
    charges: torch.Tensor  # [B, L,]
    residues: torch.Tensor  # [B, L,]
    residue_ids: torch.Tensor  # [B, L,]
    adjacency: torch.Tensor  # [B, L, L]
    coords: torch.Tensor  # [B, L, 3]
    mask: torch.Tensor  # [B, L]

    @property
    def batch_size(self) -> int:
        return self.atoms.shape[0]

    @property
    def length(self) -> int:
        return self.atoms.shape[1]

    def check_tensors(self):
        B = self.batch_size
        L = len(self)
        _check_tensor(self.atoms, (B, L), [torch.int32, torch.int64])
        _check_tensor(self.charges, (B, L), [torch.int32, torch.int64])
        _check_tensor(self.residues, (B, L), [torch.int32, torch.int64])
        _check_tensor(self.residue_ids, (B, L), [torch.int32, torch.int64])
        _check_tensor(self.adjacency, (B, L, L), [torch.int32, torch.int64])
        _check_tensor(self.coords, (B, L, 3), [torch.bfloat16, torch.float16, torch.float32])
        _check_tensor(self.mask, (B, L), [torch.bool])

    def __getitem__(self, index: int) -> PocketTensor:
        length = self.mask[index].sum().item()  # Get the length of the tensor at index
        atoms = self.atoms[index, :length]
        charges = self.charges[index, :length]
        residues = self.residues[index, :length]
        residue_ids = self.residue_ids[index, :length]
        adjacency = self.adjacency[index, :length, :length]
        coords = self.coords[index, :length]
        return PocketTensor(atoms, charges, residues, residue_ids, adjacency, coords)

    @classmethod
    def from_tensors(cls, tensors: Sequence[PocketTensor], max_length: int | None = None) -> Self:
        B = len(tensors)
        assert B > 0, "Cannot create a batch with zero tensors."
        L = cls.get_stacked_length(tensors, max_length)
        dev = tensors[0].device
        base_t = tensors[0]
        atoms = torch.zeros((B, L), dtype=base_t.atoms.dtype, device=dev)
        charges = torch.zeros((B, L), dtype=base_t.charges.dtype, device=dev)
        residues = torch.zeros((B, L), dtype=base_t.residues.dtype, device=dev)
        residue_ids = torch.zeros((B, L), dtype=base_t.residue_ids.dtype, device=dev)
        adjacency = torch.zeros((B, L, L), dtype=base_t.adjacency.dtype, device=dev)
        coords = torch.zeros((B, L, 3), dtype=base_t.coords.dtype, device=dev)
        mask = torch.zeros((B, L), dtype=torch.bool, device=dev)
        for i, t in enumerate(tensors):
            length = len(t)
            atoms[i, :length] = t.atoms
            charges[i, :length] = t.charges
            residues[i, :length] = t.residues
            residue_ids[i, :length] = t.residue_ids
            adjacency[i, :length, :length] = t.adjacency
            coords[i, :length, :] = t.coords
            mask[i, :length] = True
        return cls(atoms, charges, residues, residue_ids, adjacency, coords, mask)

    def to_tensors(self) -> list[PocketTensor]:
        tensors: list[PocketTensor] = []
        length_list = self.mask.sum(dim=1).tolist()  # [B,]
        for i, length in enumerate(length_list):
            atoms = self.atoms[i, :length]
            charges = self.charges[i, :length]
            residues = self.residues[i, :length]
            residue_ids = self.residue_ids[i, :length]
            adjacency = self.adjacency[i, :length, :length]
            coords = self.coords[i, :length]
            tensors.append(PocketTensor(atoms, charges, residues, residue_ids, adjacency, coords))
        return tensors


@dataclass
class ConditionBatch(BatchTensor[ConditionTensor]):
    time_cond: torch.Tensor  # [B, L, 3] # time, gen-time, rel-time
    self_cond: torch.Tensor  # [B, L, 3]
    mask: torch.Tensor  # [B, L]

    @property
    def batch_size(self) -> int:
        return self.self_cond.shape[0]

    @property
    def length(self) -> int:
        return self.self_cond.shape[1]

    def check_tensors(self):
        B = self.batch_size
        L = len(self)
        _check_tensor(self.time_cond, (B, L, -1), [torch.bfloat16, torch.float16, torch.float32])
        _check_tensor(self.self_cond, (B, L, 3), [torch.bfloat16, torch.float16, torch.float32])
        _check_tensor(self.mask, (B, L), [torch.bool])

    def __getitem__(self, index: int) -> ConditionTensor:
        length = self.mask[index].sum().item()  # Get the length of the tensor at index
        time_cond = self.time_cond[index, :length]
        self_cond = self.self_cond[index, :length]
        return ConditionTensor(time_cond, self_cond)

    @classmethod
    def from_tensors(cls, tensors: Sequence[ConditionTensor], max_length: int | None = None) -> Self:
        B = len(tensors)
        assert B > 0, "Cannot create a batch with zero tensors."
        L = cls.get_stacked_length(tensors, max_length)
        dev = tensors[0].device
        base_t = tensors[0]
        time_cond = torch.zeros((B, *base_t.time_cond.shape), dtype=base_t.time_cond.dtype, device=dev)
        self_cond = torch.zeros((B, *base_t.self_cond.shape), dtype=base_t.self_cond.dtype, device=dev)
        mask = torch.zeros((B, L), dtype=torch.bool, device=dev)
        for i, t in enumerate(tensors):
            length = len(t)
            time_cond[i] = t.time_cond
            self_cond[i, :length] = t.self_cond
            mask[i, :length] = True
        return cls(time_cond, self_cond, mask)

    def to_tensors(self) -> list[ConditionTensor]:
        tensors: list[ConditionTensor] = []
        length_list = self.mask.sum(dim=1).tolist()  # [B,]
        for i, length in enumerate(length_list):
            time_cond = self.time_cond[i, :length]
            self_cond = self.self_cond[i, :length]
            tensors.append(ConditionTensor(time_cond, self_cond))
        return tensors
