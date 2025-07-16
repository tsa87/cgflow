import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import lmdb
import numpy as np
import torch
from omegaconf import MISSING
from tqdm import tqdm

from cgflow.data.interpolate import Interpolant, InterpT
from cgflow.data.transform import Transform
from cgflow.util.data.batch import PocketComplexBatch
from cgflow.util.data.molrepr import LigandMol
from cgflow.util.data.pocket import PocketComplex, ProteinPocket
from cgflow.util.registry import DATASET


@dataclass
class DatasetConfig:
    _registry_: ClassVar[str] = "dataset"
    _type_: str


@dataclass
class BatchDatasetConfig(DatasetConfig):
    _type_: str = "BatchDataset"
    data_path: str | None = None


@dataclass
class LMDBDatasetConfig(DatasetConfig):
    _type_: str = "LMDBDataset"
    data_path: str = MISSING  # lmdb path
    key_path: str = MISSING  # path to the file with keys
    dataset_size: int | None = None  # use first N datapoints
    max_length: int | None = None  # ignore datas with size > MAX
    bytes_per_length: float = 0.0015  # NOT-USED; depreciated


@DATASET.register(config=DatasetConfig)
class Dataset(ABC, torch.utils.data.Dataset):
    def __init__(self, config: DatasetConfig, transform: Transform, interpolant: Interpolant):
        super().__init__()
        self.config: DatasetConfig = config
        self.transform: Transform = transform
        self.interpolant: Interpolant = interpolant

    def __getitem__(self, item: int) -> InterpT:
        molecule = self.get_data(item)
        molecule = self.transform(molecule)
        return self.interpolant.interpolate(molecule)

    def get_trajectory(self, item: int, times: list[float]) -> list[InterpT]:
        molecule = self.get_data(item)
        molecule = self.transform(molecule)
        return self.interpolant.interpolate_traj(molecule, times)

    @property
    @abstractmethod
    def lengths(self) -> list[int]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def get_data(self, item: int) -> Any: ...


@DATASET.register(config=BatchDatasetConfig)
class BatchDataset(Dataset):
    config: BatchDatasetConfig

    def __init__(
        self,
        batch: Any,
        config: BatchDatasetConfig,
        transform: Transform,
        interpolant: Interpolant,
    ):
        super().__init__(config, transform, interpolant)
        self.data: Any = batch
        self._lengths: list[int] = self.data.seq_length
        if config.data_path is not None and not os.path.exists(config.data_path):
            self.save(config.data_path)

    @property
    def lengths(self) -> list[int]:
        return self._lengths

    def __len__(self) -> int:
        return len(self.data)

    def get_data(self, item) -> Any:
        return self.data[item]

    # *** File IO ***
    def save(self, data_path: str):
        with open(data_path, "wb") as w:
            w.write(self.data.to_bytes())

    @classmethod
    @abstractmethod
    def load(cls, config: BatchDatasetConfig, transform: Transform, interpolant: Interpolant) -> Self: ...

    @staticmethod
    def _load_data(data_path: str | Path, batch_cls: type):
        data_path = Path(data_path)
        combined_bytes = b""
        if data_path.is_dir():
            for file in sorted(data_path.iterdir()):  # Sort for consistent order
                if file.is_file() and file.suffix == ".smol":
                    combined_bytes += file.read_bytes()
        else:
            # TODO: maybe read in chunks if this is too big
            combined_bytes = data_path.read_bytes()
        return batch_cls.from_bytes(combined_bytes)


@DATASET.register(config=LMDBDatasetConfig)
class LMDBDataset(Dataset):
    config: LMDBDatasetConfig

    def __init__(self, config: LMDBDatasetConfig, transform: Transform, interpolant: Interpolant):
        super().__init__(config, transform, interpolant)
        # load keys
        with open(config.key_path) as f:
            keys = [line.strip() for line in f.readlines()]

        self.keys: list[str] = keys
        self.max_length = config.max_length
        self.bytes_per_length: float = config.bytes_per_length  # This is a placeholder, adjust as needed

        # set lmdb environment
        self.lmdb_path = str(config.data_path)
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, max_readers=512)

        self._lengths: list[int] = self.get_data_length()
        assert len(self.keys) == len(self._lengths), "Keys and lengths must match in size."

        # drop the big data points
        if config.max_length is not None:
            self.drop_longer_than(config.max_length)

        # Take the subset of the keys if length is specified
        if config.dataset_size is not None:
            self.keys = self.keys[: config.dataset_size]
            self._lengths = self._lengths[: config.dataset_size]

    @property
    def lengths(self) -> list[int]:
        return self._lengths

    def __len__(self) -> int:
        return len(self.keys)

    def get_data_length(self) -> list[int]:
        """get data lengths
        Instead of acutally loading the object which takes a long time,
        We'll just directly estimate the length of object based on byte length.
        """
        lengths: list[int]
        length_path = Path(self.lmdb_path) / "lengths.pkl"
        if length_path.exists():
            lengths = pickle.load(open(length_path, "rb"))
        else:
            lengths = []
            with self.env.begin(write=False) as txn:
                for i in tqdm(range(len(self.keys))):
                    value = txn.get(self.keys[i].encode("utf-8"))
                    if True:
                        # this is enough fast to check all objects
                        complex_data = PocketComplex.from_bytes(value)
                        data_length = complex_data.seq_length
                    else:
                        num_bytes = len(value)
                        data_length = int(num_bytes * self.bytes_per_length)
                    lengths.append(data_length)
            # save length
            with length_path.open("wb") as f:
                pickle.dump(lengths, f)
        return lengths

    def drop_longer_than(self, max_length: int):
        new_keys = []
        new_lengths = []
        for key, length in zip(self.keys, self._lengths, strict=True):
            if length < max_length:
                new_keys.append(key)
                new_lengths.append(length)
        self.keys = new_keys
        self._lengths = new_lengths


# *** SmolDataset implementations ***


@DATASET.register(config=BatchDatasetConfig)
class PocketComplexDataset(BatchDataset):
    data: PocketComplexBatch

    def sample(self, n_items: int, replacement: bool = False) -> Self:
        complex_samples = np.random.choice(self.data.to_list(), n_items, replace=replacement)
        batch = PocketComplexBatch.from_list(complex_samples)
        return self.__class__(batch, self.config, self.transform, self.interpolant)

    @classmethod
    def load(cls, config: BatchDatasetConfig, transform: Transform, interpolant: Interpolant):
        assert config.data_path is not None, f"Data path must be specified {config}"
        data = cls._load_data(config.data_path, PocketComplexBatch)
        return PocketComplexDataset(data, config, transform, interpolant)


@DATASET.register(config=LMDBDatasetConfig)
class LMDBPocketComplexDataset(LMDBDataset):
    def get_data(self, item: int) -> PocketComplex:
        try:
            key = self.keys[item]
            with self.env.begin(write=False) as txn:
                value = txn.get(key.encode("utf-8"))
            complex_data = PocketComplex.from_bytes(value)
            if self.max_length is not None:
                holo, ligand = complex_data.holo, complex_data.ligand
                if len(holo) + len(ligand) > self.max_length:
                    print(f"Skipping idx {item} due to length {len(holo) + len(ligand)} > {self.max_length}")
                    return self.get_data(np.random.randint(0, len(self)))
            return complex_data
        except Exception as e:
            print(f"[error] Skipping idx {item} due to: {e}")
            return self.get_data(np.random.randint(0, len(self)))


@DATASET.register(config=LMDBDatasetConfig)
class EfficentLMDBPocketComplexDataset(LMDBDataset):
    def setup(self):
        map_size = 100 * 1024**3
        self.protein_lmdb_path = str(Path(self.lmdb_path) / "protein")
        self.ligand_lmdb_path = str(Path(self.lmdb_path) / "ligand")

        if os.path.exists(self.new_keys_file):
            self.keys = pickle.load(open(self.new_keys_file, "rb"))
            return

        os.makedirs(self.protein_lmdb_path, exist_ok=True)
        os.makedirs(self.ligand_lmdb_path, exist_ok=True)

        self.protein_env = lmdb.open(self.protein_lmdb_path, readonly=False, lock=False, map_size=map_size)
        self.ligand_env = lmdb.open(self.ligand_lmdb_path, readonly=False, lock=False, map_size=map_size)

        new_keys = []
        for protein_key in tqdm(self.keys):
            with self.env.begin(write=False) as txn:
                value = txn.get(protein_key.encode("utf-8"))

            raw_bytes = pickle.loads(value)
            protein_bytes = raw_bytes[0]
            ligands_bytes = raw_bytes[1]

            with self.protein_env.begin(write=True) as txn:
                txn.put(protein_key.encode("utf-8"), protein_bytes)

            # check if type is list
            if isinstance(ligands_bytes, list):
                for i, ligand in enumerate(ligands_bytes):
                    ligand_key = protein_key + "_" + str(i)
                    new_keys.append((protein_key, ligand_key))

                    with self.ligand_env.begin(write=True) as txn:
                        txn.put(ligand_key.encode("utf-8"), ligand)
            else:
                new_keys.append((protein_key, protein_key))
                with self.ligand_env.begin(write=True) as txn:
                    txn.put(protein_key.encode("utf-8"), ligands_bytes)

        self.keys = new_keys
        pickle.dump(self.keys, open(self.new_keys_file, "wb"))

    def __init__(self, key_path, lmdb_path, transform=None):
        super().__init__(key_path, lmdb_path, transform)
        self.new_keys_file = str(Path(lmdb_path) / "new_keys.pkl")
        self.setup()

        map_size = 100 * 1024**3
        self.protein_env = lmdb.open(self.protein_lmdb_path, readonly=True, lock=False, map_size=map_size)
        self.ligand_env = lmdb.open(self.ligand_lmdb_path, readonly=True, lock=False, map_size=map_size)

    def get_data(self, item: int) -> PocketComplex:
        try:
            key = self.keys[item]
            protein_key, ligand_key = key

            with self.protein_env.begin(write=False) as txn:
                protein_bytes = txn.get(protein_key.encode("utf-8"))
                protein = ProteinPocket.from_bytes(protein_bytes)
            with self.ligand_env.begin(write=False) as txn:
                ligand_bytes = txn.get(ligand_key.encode("utf-8"))
                ligand = LigandMol.from_bytes(ligand_bytes)

            complex = PocketComplex(holo=protein, ligand=ligand)
            if self.transform is not None:
                complex = self.transform(complex)
            return complex
        except Exception as e:
            print(f"[error] Skipping idx {item} due to: {e}")
            return self.get_data((item + 1) % len(self))  # try next one
