"""
Script for generating molecules using a trained model and saving them.

Note that the script currently does not save the molecules in batches - all of the molecules are generated and then
all saved together in one Smol batch. If generating many molecules ensure you have enough memory to store them.
"""

import argparse
from pathlib import Path
from functools import partial

import torch
import lightning as L

import semlaflow.scriptutil as util
from semlaflow.util.molrepr import GeometricMolBatch
from semlaflow.models.fm import Integrator, MolecularCFM
from semlaflow.models.semla import EquiInvDynamics, SemlaGenerator

from semlaflow.data.datasets import GeometricDataset
from semlaflow.data.datamodules import GeometricInterpolantDM
from semlaflow.data.interpolate import GeometricInterpolant, GeometricNoiseSampler


# Default script arguments
DEFAULT_SAVE_FILE = "predictions.smol"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_N_MOLECULES = 5000
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

    dynamics = EquiInvDynamics(
        hparams["d_model"],
        hparams["d_message"],
        hparams["n_coord_sets"],
        hparams["n_layers"],
        n_neighbours_feats=hparams["n_neighbours_feats"],
        d_message_hidden=hparams["d_message_hidden"],
        d_edge=hparams["d_edge"],
        self_cond=hparams["self_cond"],
        coord_norm=hparams["coord_norm"],
    )
    egnn_gen = SemlaGenerator(
        hparams["d_model"],
        dynamics,
        vocab.size,
        hparams["n_atom_feats"],
        d_edge=hparams["d_edge"],
        n_edge_types=util.get_n_bond_types(hparams["integration-type-strategy"]),
        self_cond=hparams["self_cond"],
        size_emb=hparams["size_emb"],
        max_atoms=hparams["max_atoms"],
    )

    type_mask_index = (
        vocab.indices_from_tokens(["<MASK>"])[0] if hparams["train-type-interpolation"] == "mask" else None
    )
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
    if args.dataset == "qm9":
        coord_std = util.QM9_COORDS_STD_DEV
        bucket_limits = util.QM9_BUCKET_LIMITS

    elif args.dataset == "geom-drugs":
        coord_std = util.GEOM_COORDS_STD_DEV
        bucket_limits = util.GEOM_DRUGS_BUCKET_LIMITS

    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    n_bond_types = 5
    transform = partial(util.mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std)

    if args.dataset_split == "train":
        dataset_path = Path(args.data_path) / "train.smol"
    elif args.dataset_split == "val":
        dataset_path = Path(args.data_path) / "val.smol"
    elif args.dataset_split == "test":
        dataset_path = Path(args.data_path) / "test.smol"

    dataset = GeometricDataset.load(dataset_path, transform=transform)
    dataset = dataset.sample(args.n_molecules, replacement=True)

    type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0] if hparams["val-type-interpolation"] == "mask" else None
    bond_mask_index = None

    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        coord_noise="gaussian",
        type_noise=hparams["val-prior-type-noise"],
        bond_noise=hparams["val-prior-bond-noise"],
        scale_log_size=hparams["val-prior-noise-scale-log-size"],
        scale_factor=util.COORD_NOISE_SCALE,
        zero_com=True,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
    )
    eval_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation="linear",
        type_interpolation=hparams["val-type-interpolation"],
        bond_interpolation=hparams["val-bond-interpolation"],
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


def generate_smol_mols(output, model):
    coords = output["coords"]
    atom_dists = output["atomics"]
    bond_dists = output["bonds"]
    charge_dists = output["charges"]
    masks = output["mask"]

    mols = model.builder.smol_from_tensors(coords, atom_dists, masks, bond_dists=bond_dists, charge_dists=charge_dists)
    return mols


def save_predictions(args, raw_outputs, model):
    # Generate GeometricMols and then combine into one GeometricMolBatch
    mol_lists = [generate_smol_mols(output, model) for output in raw_outputs]
    mols = [mol for mol_list in mol_lists for mol in mol_list]
    batch = GeometricMolBatch.from_list(mols)

    save_path = Path(args.save_dir) / args.save_file
    batch_bytes = batch.to_bytes()
    save_path.write_bytes(batch_bytes)


def main(args):
    print(f"Running prediction script for {args.n_molecules} molecules...")
    print(f"Using model stored at {args.ckpt_path}")

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
    metrics, _ = util.init_metrics(args.data_path, model)
    print("Metrics complete.")

    print("Running generation...")
    molecules, raw_outputs = util.generate_molecules(model, dm, args.integration_steps, args.ode_sampling_strategy)
    print("Generation complete.")

    print(f"Saving predictions to {args.save_dir}/{args.save_file}")
    save_predictions(args, raw_outputs, model)
    print("Complete.")

    print("Calculating generative metrics...")
    results = util.calc_metrics_(molecules, metrics)
    util.print_results(results)
    print("Generation script complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--save_file", type=str, default=DEFAULT_SAVE_FILE)

    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--dataset_split", type=str, default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--n_molecules", type=int, default=DEFAULT_N_MOLECULES)
    parser.add_argument("--integration_steps", type=int, default=DEFAULT_INTEGRATION_STEPS)
    parser.add_argument("--cat_sampling_noise_level", type=int, default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL)
    parser.add_argument("--ode_sampling_strategy", type=str, default=DEFAULT_ODE_SAMPLING_STRATEGY)

    parser.add_argument("--bucket_cost_scale", type=str, default=DEFAULT_BUCKET_COST_SCALE)

    args = parser.parse_args()
    main(args)
