from dataclasses import dataclass

from cgflow.data.datamodules import DataModule, DataModuleConfig
from cgflow.data.datasets import BatchDataset, BatchDatasetConfig, Dataset, DatasetConfig
from cgflow.data.interpolate import Interpolant, InterpolantConfig
from cgflow.data.prior_dist import PriorDistribution, PriorDistributionConfig
from cgflow.data.time_dist import (
    ConstantTimeDistribution,
    ConstantTimeDistributionConfig,
    TimeDistribution,
    TimeDistributionConfig,
)
from cgflow.data.transform import Transform, TransformConfig
from cgflow.util.data.batch import PocketComplexBatch
from cgflow.util.registry import Registry


@dataclass
class DataConfig:
    transform: TransformConfig
    prior_dist: PriorDistributionConfig
    time_dist: TimeDistributionConfig
    interpolant: InterpolantConfig
    train_dataset: DatasetConfig | None
    val_dataset: DatasetConfig | None
    test_dataset: DatasetConfig | None
    datamodule: DataModuleConfig


def get_transform(config: TransformConfig, mode: str = "train") -> Transform:
    assert config._registry_ == "transform"
    registry = Registry.get_register(config._registry_)
    obj_cls: type[Transform] = registry[config._type_]
    return obj_cls(config, mode)


def get_prior_dist(config: PriorDistributionConfig) -> PriorDistribution:
    assert config._registry_ == "prior_distribution"
    registry = Registry.get_register(config._registry_)
    obj_cls: type[PriorDistribution] = registry[config._type_]
    return obj_cls(config)


def get_time_dist(config: TimeDistributionConfig) -> TimeDistribution:
    assert config._registry_ == "time_distribution"
    registry = Registry.get_register(config._registry_)
    obj_cls: type[TimeDistribution] = registry[config._type_]
    return obj_cls(config)


def get_time_dist_for_evaluation() -> ConstantTimeDistribution:
    return ConstantTimeDistribution(ConstantTimeDistributionConfig(time=0.99))


def get_interpolant(
    config: InterpolantConfig,
    prior_dist: PriorDistribution,
    time_dist: TimeDistribution,
) -> Interpolant:
    assert config._registry_ == "interpolant"
    registry = Registry.get_register(config._registry_)
    dataset_cls: type[Interpolant] = registry[config._type_]
    return dataset_cls(config, prior_dist, time_dist)


def get_dataset(
    config: DatasetConfig,
    transform: Transform,
    interpolant: Interpolant,
) -> Dataset:
    assert config._registry_ == "dataset"
    registry = Registry.get_register(config._registry_)
    dataset_cls: type[Dataset] = registry[config._type_]
    return dataset_cls(config, transform, interpolant)


def get_dataset_from_batch(
    batch: PocketComplexBatch,
    config: BatchDatasetConfig,
    transform: Transform,
    interpolant: Interpolant,
) -> BatchDataset:
    assert config._registry_ == "dataset"
    registry = Registry.get_register(config._registry_)
    dataset_cls: type[BatchDataset] = registry[config._type_]
    return dataset_cls(batch, config, transform, interpolant)


def get_datamodule(
    config: DataModuleConfig,
    train_dataset: Dataset | None,
    val_dataset: Dataset | None,
    test_dataset: Dataset | None,
) -> DataModule:
    assert config._registry_ == "datamodule"
    registry = Registry.get_register(config._registry_)
    dataset_cls: type[DataModule] = registry[config._type_]
    return dataset_cls(config, train_dataset, val_dataset, test_dataset)


def build_dm(config: DataConfig):
    prior_dist = get_prior_dist(config.prior_dist)
    time_dist = get_time_dist(config.time_dist)
    time_dist_for_evaluation = get_time_dist_for_evaluation()  # val, test

    if config.train_dataset is not None:
        transform = get_transform(config.transform, "train")
        interpolant = get_interpolant(config.interpolant, prior_dist, time_dist)
        train_dataset = get_dataset(config.train_dataset, transform, interpolant)
    else:
        train_dataset = None

    if config.val_dataset is not None:
        transform = get_transform(config.transform, "val")
        interpolant = get_interpolant(config.interpolant, prior_dist, time_dist_for_evaluation)
        val_dataset = get_dataset(config.val_dataset, transform, interpolant)
    else:
        val_dataset = None

    if config.test_dataset is not None:
        transform = get_transform(config.transform, "test")
        interpolant = get_interpolant(config.interpolant, prior_dist, time_dist_for_evaluation)
        test_dataset = get_dataset(config.test_dataset, transform, interpolant)
    else:
        test_dataset = None

    return get_datamodule(config.datamodule, train_dataset, val_dataset, test_dataset)


def build_test_dm(config: DataConfig):
    transform = get_transform(config.transform, mode="test")
    prior_dist = get_prior_dist(config.prior_dist)
    time_dist_for_evaluation = get_time_dist_for_evaluation()  # val, test
    assert config.test_dataset is not None, "Test dataset configuration is required."
    interpolant = get_interpolant(config.interpolant, prior_dist, time_dist_for_evaluation)
    test_dataset = get_dataset(config.test_dataset, transform, interpolant)
    return get_datamodule(config.datamodule, None, None, test_dataset)
