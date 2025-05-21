from collections import namedtuple
from functools import partial
from pathlib import Path

import cgflow.scriptutil as util
from cgflow.data.datamodules import GeometricInterpolantDM
from cgflow.data.datasets import GeometricDataset, LMDBGeometricDataset, LMDBPocketComplexDataset, PocketComplexDataset
from cgflow.data.interpolate import (
    ARGeometricComplexInterpolant,
    ARGeometricInterpolant,
    GeometricComplexInterpolant,
    GeometricInterpolant,
    GeometricNoiseSampler,
    PseudoARGeometricComplexInterpolant,
    PseudoGeometricComplexInterpolant,
)

PLINDER_VAL_PATH = "semlaflow/saved/data/plinder/smol/val.smol"

DatasetConfig = namedtuple(
    "DatasetConfig",
    [
        "coord_std", "padded_sizes", "max_length", "dataset_type",
        "train_lmdb_path", "val_lmdb_path", "train_key_path", "val_key_path"
    ],
)
CategoricalStrategyConfig = namedtuple(
    "CategoricalStrategyConfig",
    [
        "type_mask_index",
        "bond_mask_index",
        "categorical_interpolation",
        "categorical_noise",
        "n_bond_types",
    ],
)
OptimalTransportConfig = namedtuple("OptimalTransportConfig",
                                    ["scale_ot", "batch_ot", "equivariant_ot"])


def get_dataset_config(dataset_name, data_path, is_pseudo_complex, max_atoms):
    molecule_dataset_type = "pseudocomplex" if is_pseudo_complex else "molecule"

    train_lmdb_path = data_path / "lmdb" / "train"
    val_lmdb_path = data_path / "lmdb" / "val"
    train_key_path = data_path / "keys" / "train.txt"
    val_key_path = data_path / "keys" / "val.txt"

    complex_bucket_limit = util.PLINDER_BUCKET_LIMITS

    if dataset_name == "qm9":
        return DatasetConfig(
            util.QM9_COORDS_STD_DEV,
            util.QM9_BUCKET_LIMITS,
            max_atoms,
            molecule_dataset_type,
            train_lmdb_path,
            val_lmdb_path,
            train_key_path,
            val_key_path,
        )
    elif dataset_name == "geom-drugs":
        return DatasetConfig(
            util.GEOM_COORDS_STD_DEV,
            util.GEOM_DRUGS_BUCKET_LIMITS,
            max_atoms,
            molecule_dataset_type,
            train_lmdb_path,
            val_lmdb_path,
            train_key_path,
            val_key_path,
        )
    elif dataset_name == "plinder-ligand":
        return DatasetConfig(
            util.PLINDER_COORDS_STD_DEV,
            util.GEOM_DRUGS_BUCKET_LIMITS,
            max_atoms,
            molecule_dataset_type,
            train_lmdb_path,
            val_lmdb_path,
            train_key_path,
            val_key_path,
        )
    elif dataset_name == "plinder":
        return DatasetConfig(
            util.PLINDER_COORDS_STD_DEV,
            complex_bucket_limit,
            max_atoms,
            "complex",
            train_lmdb_path,
            val_lmdb_path,
            train_key_path,
            val_key_path,
        )
    elif dataset_name == "zinc15m":
        return DatasetConfig(
            util.PLINDER_COORDS_STD_DEV,
            complex_bucket_limit,
            max_atoms,
            "complex",
            train_lmdb_path,
            val_lmdb_path,
            train_key_path,
            val_key_path,
        )
    elif dataset_name == "crossdock":
        return DatasetConfig(
            util.PLINDER_COORDS_STD_DEV,
            complex_bucket_limit,
            max_atoms,
            "complex",
            train_lmdb_path,
            val_lmdb_path,
            train_key_path,
            val_key_path,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


def get_categorical_strategy(categorical_strategy, vocab):
    type_mask_index = None
    bond_mask_index = None

    if categorical_strategy == "mask":
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = util.BOND_MASK_INDEX
        categorical_interpolation = "unmask"
        categorical_noise = "mask"

    elif categorical_strategy == "uniform-sample":
        categorical_interpolation = "unmask"
        categorical_noise = "uniform-sample"

    elif categorical_strategy == "no-change":
        categorical_interpolation = "no-change"
        # doesn't matter what this is set to as categorical attributes are not interpolated
        categorical_noise = "uniform-sample"

    elif categorical_strategy == "auto-regressive":
        categorical_interpolation = "no-change"
        # doesn't matter what this is set to as categorical attributes are not interpolated
        categorical_noise = "uniform-sample"

    else:
        raise ValueError(
            f"Interpolation '{categorical_strategy}' is not supported.")

    n_bond_types = util.get_n_bond_types(categorical_strategy)
    return CategoricalStrategyConfig(
        type_mask_index,
        bond_mask_index,
        categorical_interpolation,
        categorical_noise,
        n_bond_types,
    )


def get_ot_config(optimal_transport):
    scale_ot = False
    batch_ot = False
    equivariant_ot = False

    if optimal_transport == "batch":
        batch_ot = True
    elif optimal_transport == "equivariant":
        equivariant_ot = True
    elif optimal_transport == "scale":
        scale_ot = True
        equivariant_ot = True
    elif optimal_transport not in ["None", "none", None]:
        raise ValueError(
            f"Unknown value for optimal_transport '{optimal_transport}'")
    return OptimalTransportConfig(scale_ot, batch_ot, equivariant_ot)


def get_transform_fn(data_config, vocab):
    if data_config.dataset_type == "molecule":
        transform_fn = util.mol_transform
    elif data_config.dataset_type == "pseudocomplex":
        # Randomly shift the ligand center of mass
        transform_fn = partial(util.mol_transform,
                               shift_com_std=util.PLINDER_LIG_CENTER_STD_DEV)
    elif data_config.dataset_type == "complex":
        transform_fn = partial(util.complex_transform,
                               radius=util.PLINDER_RADIUS)
    else:
        raise ValueError(f"Unknown dataset type {data_config.dataset_type}")

    transform_fn = partial(transform_fn,
                           vocab=vocab,
                           coord_std=data_config.coord_std)
    return transform_fn


def _get_dataset_cls(data_config):
    if data_config.dataset_type in ["molecule", "pseudocomplex"]:
        return LMDBGeometricDataset
    elif data_config.dataset_type == "complex":
        return LMDBPocketComplexDataset
    else:
        raise ValueError(f"Unknown dataset type {data_config.dataset_type}")


def get_datasets(transform_fn, data_config, mode="train"):
    DatasetClass = _get_dataset_cls(data_config)
    if mode == "train":
        key_path = data_config.train_key_path
        lmdb_path = data_config.train_lmdb_path
    else:
        key_path = data_config.val_key_path
        lmdb_path = data_config.val_lmdb_path
    return DatasetClass(
        key_path=key_path,
        lmdb_path=lmdb_path,
        transform=transform_fn,
        max_length=data_config.max_length,
    )


def get_dataset_from_batch(batch, transform_fn, data_config):
    DatasetClass = _get_dataset_cls(data_config)
    dataset = DatasetClass(batch, transform=transform_fn)
    return dataset


def get_non_autoregressive_interpolant(args, vocab, cat_config, ot_config,
                                       data_config):
    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        cat_config.n_bond_types,
        coord_noise="gaussian",
        type_noise=cat_config.categorical_noise,
        bond_noise=cat_config.categorical_noise,
        scale_ot=ot_config.scale_ot,
        zero_com=True,
        type_mask_index=cat_config.type_mask_index,
        bond_mask_index=cat_config.bond_mask_index,
    )

    common_args = {
        "prior_sampler": prior_sampler,
        "coord_interpolation": "linear",
        "type_interpolation": cat_config.categorical_interpolation,
        "bond_interpolation": cat_config.categorical_interpolation,
        "conf_coord_strategy": args.conf_coord_strategy,
    }

    train_interpolant_args = {
        **common_args,
        "coord_noise_std": args.coord_noise_std_dev,
        "type_dist_temp": args.type_dist_temp,
        "equivariant_ot": ot_config.equivariant_ot,
        "batch_ot": ot_config.batch_ot,
        "time_alpha": args.time_alpha,
        "time_beta": args.time_beta,
        "fixed_time": args.train_fixed_time,
    }

    eval_interpolant_args = {
        **common_args,
        "equivariant_ot": False,
        "batch_ot": False,
        "fixed_time": 0.9,
    }

    if data_config.dataset_type == "molecule":
        InterpolantClass = GeometricInterpolant
    elif data_config.dataset_type == "pseudocomplex":
        InterpolantClass = partial(PseudoGeometricComplexInterpolant,
                                   align_vector=args.align_vector)
    elif data_config.dataset_type == "complex":
        InterpolantClass = GeometricComplexInterpolant
    else:
        raise ValueError(f"Unknown dataset type {data_config.dataset_type}")

    train_interpolant = InterpolantClass(**train_interpolant_args)
    eval_interpolant = InterpolantClass(**eval_interpolant_args)

    return train_interpolant, eval_interpolant


def get_autoregressive_interpolant(args, vocab, data_config):
    ar_geometric_args = {
        "vocab": vocab,
        "decomposition_strategy": args.decomposition_strategy,
        "ordering_strategy": args.ordering_strategy,
        "coord_noise_std": args.coord_noise_std_dev,
        "t_per_ar_action": args.t_per_ar_action,
        "max_action_t": args.max_action_t,
        "max_interp_time": args.max_interp_time,
        "max_num_cuts": args.max_num_cuts,
        "min_group_size": args.min_group_size,
    }

    ar_geometric_train_args = {
        **ar_geometric_args,
        "time_alpha": args.time_alpha,
        "time_beta": args.time_beta,
        "fixed_time": args.train_fixed_time,
    }

    ar_geometric_eval_args = {
        **ar_geometric_args,
        "fixed_time": 0.9,
    }

    if data_config.dataset_type == "molecule":
        InterpolantClass = ARGeometricInterpolant
    elif data_config.dataset_type == "pseudocomplex":
        InterpolantClass = PseudoARGeometricComplexInterpolant
    elif data_config.dataset_type == "complex":
        InterpolantClass = ARGeometricComplexInterpolant
    else:
        raise ValueError(f"Unknown dataset type {data_config.dataset_type}")

    train_interpolant = InterpolantClass(**ar_geometric_train_args)
    eval_interpolant = InterpolantClass(**ar_geometric_eval_args)
    return train_interpolant, eval_interpolant


def get_interpolants(args, vocab, cat_config, ot_config, data_config):
    if args.categorical_strategy == "auto-regressive":
        train_interpolant, eval_interpolant = get_autoregressive_interpolant(
            args, vocab, data_config)
    else:
        train_interpolant, eval_interpolant = get_non_autoregressive_interpolant(
            args, vocab, cat_config, ot_config, data_config)
    return train_interpolant, eval_interpolant


def build_dm(args, vocab, batch=None, mode="train"):
    data_path = Path(args.data_path)

    data_config = get_dataset_config(args.dataset, data_path,
                                     args.is_pseudo_complex, args.max_atoms)
    cat_config = get_categorical_strategy(args.categorical_strategy, vocab)
    ot_config = get_ot_config(args.optimal_transport)

    transform_fn = get_transform_fn(data_config, vocab)

    # For training - which loads the dataset from disk
    if batch is None and mode == "train":
        train_dataset = get_datasets(transform_fn, data_config, mode="train")
        val_dataset = get_datasets(transform_fn, data_config, mode="val")

        if args.n_validation_mols < len(val_dataset):
            val_dataset = val_dataset.sample(args.n_validation_mols)
        if args.n_training_mols < len(train_dataset):
            train_dataset = train_dataset.sample(args.n_training_mols)

    # For prediction - which loads the dataset from batch
    elif batch is None and mode == "val":
        val_dataset = get_datasets(transform_fn, data_config, mode="val")
        train_dataset = val_dataset
    else:
        raise ValueError(f"Unknown mode {mode}")

    if batch is not None:
        val_dataset = get_dataset_from_batch(batch, transform_fn, data_config)
        train_dataset = val_dataset

    # args.train_fixed_time = 0.5 if args.distill else None
    args.train_fixed_time = None
    train_interpolant, eval_interpolant = get_interpolants(
        args, vocab, cat_config, ot_config, data_config)
    print(f"Using type {train_interpolant.__class__.__name__} for training")

    dm = GeometricInterpolantDM(
        train_dataset,
        val_dataset,
        None,
        args.batch_cost,
        train_interpolant=train_interpolant,
        val_interpolant=eval_interpolant,
        test_interpolant=None,
        bucket_limits=data_config.padded_sizes,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False,
        num_workers=args.num_workers,
    )
    return dm
