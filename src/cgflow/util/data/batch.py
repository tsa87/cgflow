import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Generic, Self, TypeVar

import numpy as np
import torch
from scipy.spatial.transform.rotation import Rotation

import cgflow.util.misc.functional as smolF
from cgflow.util.data.molrepr import GeometricMol, SmolMol
from cgflow.util.data.pocket import PocketComplex

# Type aliases
_T = torch.Tensor
TCoord = tuple[float, float, float]

# Generics
TSmolMol = TypeVar("TSmolMol", bound=SmolMol)

# **********************
# *** Util functions ***
# **********************


def _check_type(obj, obj_type, name="object"):
    if not isinstance(obj, obj_type):
        raise TypeError(f"{name} must be an instance of {obj_type} or one of its subclasses, got {type(obj)}")


def _check_shape_len(tensor, allowed, name="object"):
    num_dims = len(tensor.size())
    allowed = [allowed] if isinstance(allowed, int) else allowed
    if num_dims not in allowed:
        raise RuntimeError(f"Number of dimensions of {name} must be in {str(allowed)}, got {num_dims}")


def _check_shapes_equal(t1, t2, dims=None):
    if dims is None:
        if t1.size() != t2.size():
            raise RuntimeError(f"objects must have the same shape, got {t1.shape} and {t2.shape}")
        else:
            return

    if isinstance(dims, int):
        dims = [dims]

    t1_dims = [t1.size(dim) for dim in dims]
    t2_dims = [t2.size(dim) for dim in dims]
    if t1_dims != t2_dims:
        raise RuntimeError(f"Expected dimensions {str(dims)} to match, got {t1.size()} and {t2.size()}")


class SmolBatch(ABC, Sequence, Generic[TSmolMol]):
    """Abstract class for molecule batch representations for the Smol library"""

    # All subclasses must call super init
    def __init__(self, mols: list[TSmolMol]):
        if len(mols) == 0:
            raise RuntimeError("Batch must be non-empty")
        self.mols: list[TSmolMol] = mols

    # *** Sequence methods ***

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, item: int) -> TSmolMol:
        return self.mols[item]

    # *** Properties for molecular batches ***

    @property
    def seq_length(self) -> list[int]:
        return [mol.seq_length for mol in self.mols]

    @property
    def batch_size(self) -> int:
        return len(self.mols)

    # *** Default methods which may need overriden ***

    def to_bytes(self) -> bytes:
        mol_bytes = [mol.to_bytes() for mol in self.mols]
        return pickle.dumps(mol_bytes)

    def to_list(self) -> list[TSmolMol]:
        return self.mols

    def apply(self, fn: Callable[[TSmolMol, int], TSmolMol]) -> Self:
        applied = [fn(mol, idx) for idx, mol in enumerate(self.mols)]
        [_check_type(mol, SmolMol, "apply result") for mol in applied]
        return self.from_list(applied)

    def copy(self) -> Self:
        # Only performs shallow copy on individual mols
        mol_copies = [mol.copy() for mol in self.mols]
        return self.from_list(mol_copies)

    @classmethod
    def collate(cls, batches: list["SmolBatch"]) -> Self:
        allmols = [mol for batch in batches for mol in batch]
        return cls.from_list(allmols)

    # *** Abstract methods for batches ***
    @property
    @abstractmethod
    def mask(self) -> _T: ...

    @classmethod
    @abstractmethod
    def from_bytes(cls, data: bytes) -> Self: ...

    @classmethod
    @abstractmethod
    def from_list(cls, mols: list[TSmolMol]) -> Self: ...

    @classmethod
    @abstractmethod
    def from_tensors(cls, *tensors: _T) -> Self: ...

    @classmethod
    @abstractmethod
    def load(cls, save_dir: str, lazy: bool = False) -> Self: ...

    @abstractmethod
    def save(self, save_dir: str, shards: int = 0, threads: int = 0) -> None: ...


class GeometricMolBatch(SmolBatch[GeometricMol]):
    def __init__(self, mols: list[GeometricMol]):
        self.mols: list[GeometricMol]
        for mol in mols:
            _check_type(mol, GeometricMol, "molecule object")

        super().__init__(mols)

        # Cache for batched tensors
        self._cache: dict[str, torch.Tensor] = {}

    # *** Properties ***

    @property
    def mask(self) -> _T:
        if "mask" not in self._cache:
            mask = [torch.ones(mol.seq_length) for mol in self.mols]
            self._cache["mask"] = smolF.pad_tensors(mask)
        return self._cache["mask"]

    @property
    def adjacency(self) -> _T:
        n_atoms = max(self.seq_length)
        adjs = [smolF.adj_from_edges(mol.bond_indices, mol.bond_types, n_atoms, symmetric=True) for mol in self.mols]
        return torch.stack(adjs)

    @property
    def com(self) -> _T:
        return smolF.calc_com(self.coords, node_mask=self.mask)

    def get_attr(self, key: str) -> _T:
        assert key in self[0].attr_keys, "Attribute not found in GeometricMol"
        if key not in self._cache:
            values = [getattr(mol, key) for mol in self.mols]
            self._cache[key] = smolF.pad_tensors(values)
        return self._cache[key]

    @property
    def coords(self) -> _T:
        return self.get_attr("coords")

    @property
    def atoms(self) -> _T:
        return self.get_attr("atoms")

    @property
    def bond_indices(self) -> _T:
        return self.get_attr("bond_indices")

    @property
    def bond_types(self) -> _T:
        return self.get_attr("bond_types")

    @property
    def charges(self) -> _T:
        return self.get_attr("charges")

    # *** Interface Methods ***

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        mols = [GeometricMol.from_bytes(mol_bytes) for mol_bytes in pickle.loads(data)]
        return cls.from_list(mols)

    @classmethod
    def from_list(cls, mols: list[GeometricMol]) -> Self:
        return cls(mols)

    # TODO add bonds and charges
    @classmethod
    def from_tensors(
        cls,
        coords: _T,
        atoms: _T | None = None,
        num_atoms: _T | None = None,
    ) -> Self:
        raise NotImplementedError("from_tensors is not implemented for GeometricMolBatch")

    @classmethod
    def load(cls, save_dir: str, lazy: bool = False) -> Self:
        save_path = Path(save_dir)

        if not save_path.exists() or not save_path.is_dir():
            raise RuntimeError(f"Folder {save_dir} does not exist.")

        batches = []
        curr_folders = [save_path]

        while len(curr_folders) != 0:
            curr_path = curr_folders[0]
            if (curr_path / "atoms.npy").exists():
                batch = cls._load_batch(curr_path, lazy=lazy)
                batches.append(batch)

            children = [path for path in curr_path.iterdir() if path.is_dir()]
            curr_folders = curr_folders[1:]
            curr_folders.extend(children)

        collated = cls.collate(batches)
        return collated

    def save(self, save_dir: str | Path, shards: int = 0, threads: int = 0) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if shards is None or shards <= 0:
            return self._save_batch(self, save_path)

        items_per_shard = (len(self) // shards) + 1
        start_idxs = [idx * items_per_shard for idx in range(shards)]
        end_idxs = [(idx + 1) * items_per_shard for idx in range(shards)]
        end_idxs[-1] = len(self)

        batches = [self.mols[s_idx:e_idx] for s_idx, e_idx in zip(start_idxs, end_idxs, strict=False)]
        batches = [GeometricMolBatch.from_list(batch_list) for batch_list in batches]

        f_len = len(str(shards - 1))
        dir_names = [f"{str(b_idx):0{f_len}}_n{str(b.batch_size)}" for b_idx, b in enumerate(batches)]
        save_paths = [save_path / name for name in dir_names]

        if threads is not None and threads > 0:
            executor = ThreadPoolExecutor(threads)
            futures = [executor.submit(self._save_batch, b, path) for b, path in zip(batches, save_paths, strict=False)]
            [future.result() for future in futures]

        else:
            [self._save_batch(batch, path) for batch, path in zip(batches, save_paths, strict=False)]

    # *** Geometric Specific Methods ***

    def zero_com(self) -> Self:
        shifted = self.coords - self.com
        shifted = shifted * self.mask.unsqueeze(2)
        return self._from_coords(shifted)

    def rotate(self, rotation: Rotation | TCoord) -> Self:
        return self.apply(lambda mol, idx: mol.rotate(rotation))

    def shift(self, shift: TCoord) -> Self:
        shift_tensor = torch.tensor(shift).view(1, 1, -1)
        shifted = (self.coords + shift_tensor) * self.mask.unsqueeze(2)
        return self._from_coords(shifted)

    def scale(self, scale: float) -> Self:
        scaled = (self.coords * scale) * self.mask.unsqueeze(2)
        return self._from_coords(scaled)

    # *** Util Methods ***

    def _from_coords(self, coords: _T) -> Self:
        _check_shape_len(coords, 3, "coords")
        _check_shapes_equal(coords, self.coords, [0, 1, 2])

        if coords.size(0) != self.batch_size:
            raise RuntimeError("coords batch size must be the same as self batch size")

        if coords.size(1) != max(self.seq_length):
            raise RuntimeError("coords num atoms must be the same as largest molecule")

        coords = coords.float()

        mol_coords = [cs[:num_atoms, :] for cs, num_atoms in zip(list(coords), self.seq_length, strict=False)]
        mols = [mol.copy_with(coords=cs) for mol, cs in zip(self.mols, mol_coords, strict=False)]
        batch = self.__class__(mols)

        # Set the cache for the tensors that have already been created
        batch._cache = self._cache.copy()
        batch._cache["coords"] = coords
        return batch

    # TODO add bonds and charges
    @classmethod
    def _load_batch(cls, batch_dir: Path, lazy: bool) -> Self:
        raise NotImplementedError

    @staticmethod
    def _save_batch(batch, save_path: Path) -> None:
        save_path.mkdir(exist_ok=True, parents=True)

        coords = batch.coords.cpu().numpy()
        np.save(save_path / "coords.npy", coords)

        num_atoms = np.array(batch.seq_length).astype(np.int16)
        np.save(save_path / "atoms.npy", num_atoms)

        atoms = batch.atoms.cpu().numpy()
        np.save(save_path / "atoms.npy", atoms)

        bonds = batch.bonds.cpu().numpy()
        if bonds.shape[1] != 0:
            np.save(save_path / "bonds.npy", bonds)


class PocketComplexBatch:
    def __init__(self, systems: list[PocketComplex]):
        for system in systems:
            _check_type(system, PocketComplex, "system")
        self._systems: list[PocketComplex] = systems

    # *** Sequence methods ***

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, item: int) -> PocketComplex:
        return self._systems[item]

    # *** Useful properties ***
    @property
    def seq_length(self) -> list[int]:
        return [system.seq_length for system in self._systems]

    @property
    def batch_size(self) -> int:
        return len(self._systems)

    @property
    def mask(self) -> _T:
        """Returns a tensor of shape [n_systems, n_atoms in largest system]. 1 for real atoms, 0 for padded atoms."""
        masks = [torch.ones(len(system)) for system in self._systems]
        return smolF.pad_tensors(masks)

    @property
    def ligand_mask(self) -> _T:
        """Returns a padded ligand mask for the batch. 0 for pad atoms, 1 for ligand atoms, 2 for pocket atoms."""
        masks = [system.ligand_mask for system in self._systems]
        return smolF.pad_tensors(masks)

    # *** IO methods ***

    def to_bytes(self) -> bytes:
        system_bytes = [system.to_bytes() for system in self._systems]
        return pickle.dumps(system_bytes)

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        systems = [PocketComplex.from_bytes(system) for system in pickle.loads(data)]
        return cls(systems)

    # *** Other methods ***

    @classmethod
    def from_batches(cls, batches: list[Self]) -> Self:
        all_systems = [system for batch in batches for system in batch]
        return cls(all_systems)

    def to_list(self) -> list[PocketComplex]:
        return self._systems

    @classmethod
    def from_list(cls, complexes: list[PocketComplex]) -> Self:
        return cls(complexes)
