import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import lightning as L
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from cgflow.data.datasets import Dataset
from cgflow.data.interpolate import InterpT
from cgflow.data.util import BucketBatchSampler, DistributedBucketBatchSampler
from cgflow.util.data.molrepr import GeometricMol, LigandMol, PocketMol
from cgflow.util.dataclasses import BatchTensor, LigandBatch, PocketBatch
from cgflow.util.registry import DATAMODULE

# *** collate function ***


@dataclass
class DataModuleConfig:
    _registry_: ClassVar[str] = "datamodule"
    _type_: str


@dataclass
class SimpleDataModuleConfig(DataModuleConfig):
    _type_: str = "SimpleDataModule"
    batch_size: int = 1
    num_workers: int = 0


@dataclass
class BucketDataModuleConfig(DataModuleConfig):
    _type_: str = "BucketDataModule"
    batch_cost: int = 4000
    bucket_limits: tuple[int, ...] = (64, 96, 128, 160)
    bucket_cost_scale: str | float = "linear"
    num_workers: int = 4


def collate_fn(data_list: list[InterpT]) -> dict[str, Any]:
    """colate interpolation data into a dictionary of collated datas."""
    dict_list = [item.to_dict() for item in data_list]
    keys = data_list[0].to_dict().keys()
    return {k: _collate_objs([data_list[k] for data_list in dict_list]) for k in keys}


def _collate_objs(objs: Sequence[Any]) -> Any:
    # NOTE: here we only consider the data types belong to InterP
    if isinstance(objs[0], GeometricMol):
        batch = _to_batched_tensor(objs)
        return batch.to_dict()
    elif isinstance(objs[0], torch.Tensor):
        return pad_sequence(objs, batch_first=True)
    elif isinstance(objs[0], float | int):
        return torch.tensor(objs)
    else:
        return objs


def _to_batched_tensor(mols: Sequence[GeometricMol]) -> BatchTensor:
    if isinstance(mols[0], LigandMol):
        tensors = [mol.to_tensor() for mol in mols]
        return LigandBatch.from_tensors(tensors)
    elif isinstance(mols[0], PocketMol):
        tensors = [mol.to_tensor() for mol in mols]
        return PocketBatch.from_tensors(tensors)
    else:
        raise ValueError(f"Unsupported molecule type: {type(mols[0])}. Expected LigandMol or PocketMol.")


@DATAMODULE.register(config=DataModuleConfig)
class DataModule(L.LightningDataModule, ABC):
    def __init__(
        self,
        config: DataModuleConfig,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ):
        super().__init__()
        self.config: DataModuleConfig = config
        self.train_dataset: Dataset | None = train_dataset
        self.val_dataset: Dataset | None = val_dataset
        self.test_dataset: Dataset | None = test_dataset

    @abstractmethod
    def train_dataloader(self) -> DataLoader: ...

    @abstractmethod
    def val_dataloader(self) -> DataLoader: ...

    @abstractmethod
    def test_dataloader(self) -> DataLoader: ...

    def collate_fn(self, data_list: list[InterpT]):
        return collate_fn(data_list)

    def train_collate_fn(self, data_list: list[InterpT]):
        return self.collate_fn(data_list)

    def val_collate_fn(self, data_list: list[InterpT]):
        return self.collate_fn(data_list)

    def test_collate_fn(self, data_list: list[InterpT]):
        return self.collate_fn(data_list)


@DATAMODULE.register(config=SimpleDataModuleConfig)
class SimpleDataModule(DataModule):
    def __init__(
        self,
        config: SimpleDataModuleConfig,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ):
        super().__init__(config, train_dataset, val_dataset, test_dataset)

        self.train_dataset: Dataset | None = train_dataset
        self.val_dataset: Dataset | None = val_dataset
        self.test_dataset: Dataset | None = test_dataset

        self.batch_size = config.batch_size
        self.num_workers: int = config.num_workers if config.num_workers >= 0 else len(os.sched_getaffinity(0))

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "Train dataset must be set before calling train_dataloader."
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_collate_fn,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "Validation dataset must be set before calling val_dataloader."
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.val_collate_fn,
        )
        return dataloader

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None, "Test dataset must be set before calling test_dataloader."
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.test_collate_fn,
        )
        return dataloader


@DATAMODULE.register(config=BucketDataModuleConfig)
class BucketDataModule(DataModule):
    def __init__(
        self,
        config: BucketDataModuleConfig,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ):
        super().__init__(config, train_dataset, val_dataset, test_dataset)

        if isinstance(config.bucket_cost_scale, str):
            if config.bucket_cost_scale == "linear":
                bucket_cost_scale = 1
            elif config.bucket_cost_scale == "quadratic":
                bucket_cost_scale = 2
            else:
                raise ValueError('config.bucket_cost_scale must be "linear" or "quadratic".')
        else:
            bucket_cost_scale = config.bucket_cost_scale

        self.batch_cost: int = config.batch_cost
        self.bucket_limits: list[float] = sorted(config.bucket_limits)
        self.bucket_cost_scale: float = bucket_cost_scale

        self.train_dataset: Dataset | None = train_dataset
        self.val_dataset: Dataset | None = val_dataset
        self.test_dataset: Dataset | None = test_dataset

        self.num_workers: int = config.num_workers if config.num_workers >= 0 else len(os.sched_getaffinity(0))

        largest_padding = self.bucket_limits[-1]
        if train_dataset is not None and max(train_dataset.lengths) > largest_padding:
            raise ValueError(
                f"At least one item in train dataset is larger than largest padded size, {max(train_dataset.lengths)}."
            )
        if val_dataset is not None and max(val_dataset.lengths) > largest_padding:
            raise ValueError("At least one item in val dataset is larger than largest padded size.")
        if test_dataset is not None and max(test_dataset.lengths) > largest_padding:
            raise ValueError("At least one item in test dataset is larger than largest padded size.")

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "Train dataset must be set before calling train_dataloader."
        sampler = self._sampler(self.train_dataset, drop_last=False)
        batch_size = self.batch_cost if sampler is None else 1
        shuffle = sampler is None
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_collate_fn,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "Validation dataset must be set before calling val_dataloader."
        sampler = self._sampler(self.val_dataset, drop_last=False)
        batch_size = self.batch_cost if sampler is None else 1
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.val_collate_fn,
        )
        return dataloader

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None, "Test dataset must be set before calling test_dataloader."
        sampler = self._sampler(self.test_dataset, drop_last=False)
        batch_size = self.batch_cost if sampler is None else 1
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.test_collate_fn,
        )
        return dataloader

    def _sampler(self, dataset: Dataset, drop_last=False):
        batch_cost = self.batch_cost**self.bucket_cost_scale
        data_costs = [v**self.bucket_cost_scale for v in dataset.lengths]
        bucket_costs = [v**self.bucket_cost_scale for v in self.bucket_limits]
        if torch.distributed.is_initialized():
            sampler_cls = DistributedBucketBatchSampler
        else:
            sampler_cls = BucketBatchSampler
        sampler = sampler_cls(
            batch_cost,
            data_costs,
            bucket_costs,
            drop_last=drop_last,
            round_batch_to_8=False,
        )
        return sampler
