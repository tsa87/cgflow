"""Example usage
python scripts/_a2_cgflow_eval.py   --model_checkpoint /home/to.shen/projects/CGFlow/wandb/equinv-plinder/icxk301o/checkpoints/last.ckpt   --data_path /home/to.shen/projects/CGFlow/data/complex/plinder/smol   --dataset crossdock   --categorical_strategy auto-regressive   --ordering_strategy connected   --decomposition_strategy reaction   --pocket_n_layers 4   --d_message 64   --d_message_hidden 96   --time_alpha 1.0   --t_per_ar_action 0.3   --max_interp_time 0.4   --max_action_t 0.6   --max_num_cuts 2
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
from rdkit import RDLogger

import cgflow.scriptutil as util
from cgflow.buildutil import build_dm, build_model
from cgflow.util.profile import time_profile

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


DEFAULT_MODEL_CHECKPOINT = None
DEFAULT_OUTPUT_DIR = "evaluation_results"

DEFAULT_D_MODEL = 384
DEFAULT_N_LAYERS = 12
DEFAULT_D_MESSAGE = 128
DEFAULT_D_EDGE = 128
DEFAULT_N_COORD_SETS = 64
DEFAULT_N_ATTN_HEADS = 32
DEFAULT_D_MESSAGE_HIDDEN = 128
DEFAULT_COORD_NORM = "length"
DEFAULT_SIZE_EMB = 64
DEFAULT_MAX_ATOMS = 256

DEFAULT_POCKET_N_LAYERS = 4
DEFAULT_POCKET_D_INV = 256

DEFAULT_BATCH_COST = 600
DEFAULT_NUM_INFERENCE_STEPS = 100
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_COORD_NOISE_STD_DEV = 0.2
DEFAULT_TYPE_DIST_TEMP = 1.0
DEFAULT_TIME_ALPHA = 2.0
DEFAULT_TIME_BETA = 1.0
DEFAULT_OPTIMAL_TRANSPORT = "equivariant"
DEFAULT_CATEGORICAL_STRATEGY = "uniform-sample"
DEFAULT_CONF_COORD_STRATEGY = "gaussian"

# AR
DEFAULT_T_PER_AR_ACTION = 0.2
DEFAULT_MAX_INTERP_TIME = 1.0
DEFAULT_DECOMPOSITION_STRATEGY = "atom"
DEFAULT_ORDERING_STRATEGY = "connected"
DEFAULT_MAX_ACTION_T = 0.8
DEFAULT_MAX_NUM_CUTS = None
DEFAULT_MIN_GROUP_SIZE = 5

DEFAULT_N_VALIDATION_MOLS = 2000
DEFAULT_NUM_WORKERS = 0
DEFAULT_NUM_GPUS = 1
DEFAULT_SAMPLING_STRATEGY = "linear"


@time_profile(output_file="semla_eval.profile", lines_to_print=500)
def main(args):
    # Set torch properties for consistency with training
    torch.set_float32_matmul_precision("high")

    L.seed_everything(12345)
    util.disable_lib_stdout()
    util.configure_fs()

    # Load model checkpoint
    if args.model_checkpoint is None:
        raise ValueError("Model checkpoint must be provided for evaluation")

    checkpoint = torch.load(args.model_checkpoint, map_location="cpu")

    print("Building model vocab...")
    vocab = util.build_vocab()
    print("Vocab complete.")

    print("Loading validation datamodule...")
    dm = build_dm(args, vocab)
    print("Datamodule complete.")

    print("Building model from checkpoint...")
    model = build_model(args, dm, vocab)

    # Load model weights from checkpoint
    model.load_state_dict(checkpoint["state_dict"])
    print("Model loaded from checkpoint.")

    # Create a simple trainer for evaluation
    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        logger=None,
    )

    # Ensure the model's sampling parameters match command line args
    model.integrator.steps = args.num_inference_steps
    model.sampling_strategy = args.sampling_strategy

    # Run evaluation
    print(f"Evaluating model on {args.n_validation_mols} molecules...")
    results = trainer.validate(model, datamodule=dm)[0]

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"eval_results_{timestamp}.json"

    # Add evaluation parameters to results
    results["eval_parameters"] = {
        "model_checkpoint": args.model_checkpoint,
        "num_inference_steps": args.num_inference_steps,
        "sampling_strategy": args.sampling_strategy,
        "n_validation_mols": args.n_validation_mols,
        "dataset": args.dataset,
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation complete. Results saved to {results_file}")

    # Print key metrics
    print("\nKey Metrics:")
    for metric_name, value in results.items():
        if metric_name != "eval_parameters":
            print(f"{metric_name}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Evaluation setup args
    parser.add_argument(
        "--model_checkpoint", type=str, default=DEFAULT_MODEL_CHECKPOINT, help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default=DEFAULT_SAMPLING_STRATEGY,
        choices=["linear", "log"],
        help="Strategy for sampling time steps",
    )

    # Dataset args
    parser.add_argument("--data_path", type=str, help="Path to validation data")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--is_pseudo_complex", action="store_true")
    parser.add_argument("--n_validation_mols", type=int, default=DEFAULT_N_VALIDATION_MOLS)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--num_gpus", type=int, default=DEFAULT_NUM_GPUS)
    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)

    # Model args - needed to reconstruct the model architecture
    parser.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--d_message", type=int, default=DEFAULT_D_MESSAGE)
    parser.add_argument("--d_edge", type=int, default=DEFAULT_D_EDGE)
    parser.add_argument("--n_coord_sets", type=int, default=DEFAULT_N_COORD_SETS)
    parser.add_argument("--n_attn_heads", type=int, default=DEFAULT_N_ATTN_HEADS)
    parser.add_argument("--d_message_hidden", type=int, default=DEFAULT_D_MESSAGE_HIDDEN)
    parser.add_argument("--coord_norm", type=str, default=DEFAULT_COORD_NORM)
    parser.add_argument("--size_emb", type=int, default=DEFAULT_SIZE_EMB)
    parser.add_argument("--max_atoms", type=int, default=DEFAULT_MAX_ATOMS)
    parser.add_argument("--arch", type=str, default="semla")

    # Protein model args
    parser.add_argument("--pocket_n_layers", type=int, default=DEFAULT_POCKET_N_LAYERS)
    parser.add_argument("--pocket_d_inv", type=int, default=DEFAULT_POCKET_D_INV)
    parser.add_argument("--fixed_equi", action="store_true")

    # Training args - needed to reconstruct the model configuration
    parser.add_argument("--use_complex_metrics", action="store_true")
    parser.add_argument("--categorical_strategy", type=str, default=DEFAULT_CATEGORICAL_STRATEGY)
    parser.add_argument("--conf_coord_strategy", type=str, default=DEFAULT_CONF_COORD_STRATEGY)
    parser.add_argument("--optimal_transport", type=str, default=DEFAULT_OPTIMAL_TRANSPORT)
    parser.add_argument("--num_inference_steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)
    parser.add_argument("--cat_sampling_noise_level", type=int, default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL)
    parser.add_argument("--coord_noise_std_dev", type=float, default=DEFAULT_COORD_NOISE_STD_DEV)
    parser.add_argument("--type_dist_temp", type=float, default=DEFAULT_TYPE_DIST_TEMP)
    parser.add_argument("--time_alpha", type=float, default=DEFAULT_TIME_ALPHA)
    parser.add_argument("--time_beta", type=float, default=DEFAULT_TIME_BETA)

    # Autoregressive args
    parser.add_argument("--t_per_ar_action", type=float, default=DEFAULT_T_PER_AR_ACTION)
    parser.add_argument("--max_interp_time", type=float, default=DEFAULT_MAX_INTERP_TIME)
    parser.add_argument("--decomposition_strategy", type=str, default=DEFAULT_DECOMPOSITION_STRATEGY)
    parser.add_argument("--ordering_strategy", type=str, default=DEFAULT_ORDERING_STRATEGY)
    parser.add_argument("--max_action_t", type=float, default=DEFAULT_MAX_ACTION_T)
    parser.add_argument("--max_num_cuts", type=int, default=DEFAULT_MAX_NUM_CUTS)
    parser.add_argument("--min_group_size", type=int, default=DEFAULT_MIN_GROUP_SIZE)

    # Set defaults for model loading
    parser.set_defaults(
        trial_run=False,
        use_ema=True,
        self_condition=True,
        lr=0.0003,  # Not used for evaluation but needed for model building
        type_loss_weight=0.2,
        bond_loss_weight=1.0,
        charge_loss_weight=1.0,
        dist_loss_weight=0.0,
        lr_schedule="constant",
        warm_up_steps=10000,
        bucket_cost_scale="linear",
        epochs=1,  # Not used for evaluation but needed for model building
        acc_batches=1,
        val_check_epochs=1,
        gradient_clip_val=1.0,
        monitor="val-validity",
        monitor_mode="max",
        n_training_mols=1,  # Not used for evaluation but needed for model building
        use_complex_metrics=False,
    )

    args = parser.parse_args()
    main(args)
