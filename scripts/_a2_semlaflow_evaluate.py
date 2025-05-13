""" Evalute trained pose prediciton model
Example usage:
python scripts/_a2_cgflow_evaluate.py --ckpt_path /home/to.shen/projects/CGFlow/wandb/equinv-plinder/icxk301o/checkpoints/last.ckpt --data_path /home/to.shen/projects/CGFlow/data/complex/plinder/smol --dataset plinder
"""

import argparse
from pathlib import Path
from functools import partial

import torch
import numpy as np
import lightning as L

import cgflow.scriptutil as util
import cgflow.util.rdkit as smolRD
from cgflow.models.pocket import PocketEncoder, LigandGenerator
from cgflow.models.fm import MolecularCFM, Integrator

from cgflow.data.datasets import GeometricDataset, PocketComplexDataset
from cgflow.data.datamodules import GeometricInterpolantDM
from cgflow.data.interpolate import GeometricInterpolant, GeometricNoiseSampler

# Default script arguments
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_N_MOLECULES = 10000
DEFAULT_N_REPLICATES = 3
DEFAULT_BATCH_COST = 8192
DEFAULT_BUCKET_COST_SCALE = "linear"
DEFAULT_INTEGRATION_STEPS = 100
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_ODE_SAMPLING_STRATEGY = "log"


def load_model(args, vocab):
    checkpoint = torch.load(args.ckpt_path)
    hparams = checkpoint["hyper_parameters"]

    hparams["compile_model"] = False
    hparams["integration-steps"] = args.integration_steps
    hparams["sampling_strategy"] = args.ode_sampling_strategy

    n_bond_types = util.get_n_bond_types(hparams["integration-type-strategy"])
    n_extra_atom_feats = 2 if args.is_autoregressive else 1

    # Set default arch to semla if nothing has been saved
    if hparams.get("architecture") is None:
        hparams["architecture"] = "semla"

    if hparams["architecture"] == "semla":
        pocket_enc = PocketEncoder(hparams["pocket-d_equi"],
                                   hparams["pocket-d_inv"],
                                   hparams["pocket-d_message"],
                                   hparams["pocket-n_layers"],
                                   hparams["pocket-n_attn_heads"],
                                   hparams["pocket-d_message_ff"],
                                   hparams["pocket-d_edge"],
                                   vocab.size,
                                   n_bond_types,
                                   len(smolRD.IDX_RESIDUE_MAP),
                                   fixed_equi=hparams["pocket-fixed_equi"])

        egnn_gen = LigandGenerator(hparams["d_equi"],
                                   hparams["d_inv"],
                                   hparams["d_message"],
                                   hparams["n_layers"],
                                   hparams["n_attn_heads"],
                                   hparams["d_message_ff"],
                                   hparams["d_edge"],
                                   vocab.size,
                                   n_bond_types,
                                   n_extra_atom_feats=n_extra_atom_feats,
                                   self_cond=hparams["self_cond"],
                                   pocket_enc=pocket_enc)

    else:
        raise ValueError("Unknown architecture hyperparameter.")

    type_mask_index = (vocab.indices_from_tokens([
        "<MASK>"
    ])[0] if hparams["integration-type-strategy"] == "mask" else None)
    bond_mask_index = None

    integrator = Integrator(
        args.integration_steps,
        type_strategy=hparams["integration-type-strategy"],
        bond_strategy=hparams["integration-bond-strategy"],
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        cat_noise_level=args.cat_sampling_noise_level,
    )
    fm_model = MolecularCFM.load_from_checkpoint(
        args.ckpt_path,
        gen=egnn_gen,
        vocab=vocab,
        integrator=integrator,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        **hparams,
    )
    return fm_model


def build_dm(args, hparams, vocab):
    is_complex = False

    if args.dataset == "qm9":
        coord_std = util.QM9_COORDS_STD_DEV
        bucket_limits = util.QM9_BUCKET_LIMITS

    elif args.dataset == "geom-drugs":
        coord_std = util.GEOM_COORDS_STD_DEV
        bucket_limits = util.GEOM_DRUGS_BUCKET_LIMITS

    elif args.dataset == "plinder" or args.dataset == "crossdock":
        coord_std = util.PLINDER_COORDS_STD_DEV
        bucket_limits = util.PLINDER_BUCKET_LIMITS
        is_complex = True

    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    n_bond_types = 5
    if 
    transform = partial(util.mol_transform, vocab=vocab, coord_std=coord_std)

    if args.dataset_split == "train":
        dataset_path = Path(args.data_path) / "train.smol"
    elif args.dataset_split == "val":
        dataset_path = Path(args.data_path) / "val.smol"
    elif args.dataset_split == "test":
        dataset_path = Path(args.data_path) / "test.smol"
    else:
        raise ValueError(f"Unknown dataset split {args.dataset}")

    if is_complex:
        dataset = PocketComplexDataset.load(dataset_path, transform=transform)
    else:
        dataset = GeometricDataset.load(dataset_path, transform=transform)
    dataset = dataset.sample(args.n_molecules, replacement=True)

    type_mask_index = None
    bond_mask_index = None
    if hparams['type_strategy'] == "mask":
        raise ValueError("Masking not supported for evaluation yet.")
    elif hparams['type_strategy'] == "no-change" or hparams[
            'type_strategy'] == "auto-regressive":
        categorical_interpolation = "no-change"
    elif hparams['type_strategy'] == "uniform-sample":
        categorical_interpolation = "unmask"
    else:
        raise ValueError(f"Unknown type strategy {hparams['type_strategy']}")

    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        coord_noise="gaussian",
        type_noise="uniform-sample",
        bond_noise="uniform-sample",
        scale_ot=False,
        zero_com=True,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
    )
    eval_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation="linear",
        type_interpolation=categorical_interpolation,
        bond_interpolation=categorical_interpolation,
        equivariant_ot=False,
        batch_ot=False,
    )
    dm = GeometricInterpolantDM(
        None,
        None,
        dataset,
        args.batch_cost,
        test_interpolant=eval_interpolant,
        bucket_limits=bucket_limits,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False,
    )
    return dm


def dm_from_ckpt(args, vocab):
    checkpoint = torch.load(args.ckpt_path)
    hparams = checkpoint["hyper_parameters"]
    dm = build_dm(args, hparams, vocab)
    return dm


def evaluate(args, model, dm, metrics, stab_metrics):
    results_list = []
    for replicate_index in range(args.n_replicates):
        print(
            f"Running replicate {replicate_index + 1} out of {args.n_replicates}"
        )
        molecules, _, stabilities = util.generate_molecules(
            model,
            dm,
            args.integration_steps,
            args.ode_sampling_strategy,
            stabilities=True,
        )

        print("Calculating metrics...")
        results = util.calc_metrics_(molecules,
                                     metrics,
                                     stab_metrics=stab_metrics,
                                     mol_stabs=stabilities)
        results_list.append(results)

    results_dict = {key: [] for key in results_list[0].keys()}
    for results in results_list:
        for metric, value in results.items():
            results_dict[metric].append(value.item())

    mean_results = {
        metric: np.mean(values)
        for metric, values in results_dict.items()
    }
    std_results = {
        metric: np.std(values)
        for metric, values in results_dict.items()
    }

    return mean_results, std_results, results_dict


def main(args):
    print(
        f"Running evaluation script for {args.n_replicates} replicates with {args.n_molecules} molecules each..."
    )
    print(f"Using model stored at {args.ckpt_path}")

    if args.n_replicates < 1:
        raise ValueError("n_replicates must be at least 1.")

    L.seed_everything(12345)
    util.disable_lib_stdout()
    util.configure_fs()

    print("Building model vocab...")
    vocab = util.build_vocab()
    print("Vocab complete.")

    print("Loading datamodule...")
    dm = dm_from_ckpt(args, vocab)
    print("Datamodule complete.")

    print("Loading model...")
    model = load_model(args, vocab)
    print("Model complete.")

    print("Initialising metrics...")
    metrics, stab_metrics, complex_metrics, conf_metrics = util.init_metrics(
        model, args.data_path, is_complex=args.is_complex)
    print("Metrics complete.")

    print("Running evaluation...")
    avg_results, std_results, list_results = evaluate(args, model, dm, metrics,
                                                      stab_metrics)
    print("Evaluation complete.")

    util.print_results(avg_results, std_results=std_results)

    print("All replicate results...")
    print(f"{'Metric':<22}Result")
    print("-" * 30)

    for metric, results_list in list_results.items():
        print(f"{metric:<22}{results_list}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--is_complex", action="store_true")
    parser.add_argument("--is_autoregressive", action="store_true")

    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--dataset_split",
                        type=str,
                        default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--n_molecules", type=int, default=DEFAULT_N_MOLECULES)
    parser.add_argument("--n_replicates",
                        type=int,
                        default=DEFAULT_N_REPLICATES)
    parser.add_argument("--integration_steps",
                        type=int,
                        default=DEFAULT_INTEGRATION_STEPS)
    parser.add_argument("--cat_sampling_noise_level",
                        type=int,
                        default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL)
    parser.add_argument("--ode_sampling_strategy",
                        type=str,
                        default=DEFAULT_ODE_SAMPLING_STRATEGY)

    parser.add_argument("--bucket_cost_scale",
                        type=str,
                        default=DEFAULT_BUCKET_COST_SCALE)

    args = parser.parse_args()
    main(args)
