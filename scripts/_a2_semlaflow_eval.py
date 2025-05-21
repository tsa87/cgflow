"""Example usage
python scripts/_a2_semlaflow_eval.py   \
    --model_checkpoint /projects/jlab/to.shen/cgflow-dev/weights/plinder_till_end.ckpt \
    --data_path /projects/jlab/to.shen/cgflow-dev/experiments/data/complex/plinder_15A \
    --use_complex_metrics False
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import torch
from rdkit import RDLogger

import cgflow.scriptutil as util
from cgflow.buildutil import build_dm, build_model
from cgflow.util.profile import time_profile

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

DEFAULT_MODEL_CHECKPOINT = "/projects/jlab/to.shen/cgflow-dev/weights/plinder_till_end.ckpt"
DEFAULT_DATA_PATH = "/projects/jlab/to.shen/cgflow-dev/experiments/data/complex/plinder_15A"
DEFAULT_OUTPUT_DIR = "evaluation_results"

DEFAULT_MAX_ATOMS = 2048
DEFAULT_NUM_INFERENCE_STEPS = 60
DEFAULT_USE_COMPLEX_METRICS = False
DEFAULT_N_VALIDATION_MOLS = 10
DEFAULT_N_TRAINING_MOLS = 10
DEFAULT_NUM_GPUS = 1
DEFAULT_BATCH_COST = 3000
DEFAULT_NUM_WORKERS = 0
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
    parser.add_argument("--model_checkpoint",
                        type=str,
                        default=DEFAULT_MODEL_CHECKPOINT,
                        help="Path to the checkpoint file")
    parser.add_argument("--data_path",
                        type=str,
                        default=DEFAULT_DATA_PATH,
                        help="Path to validation data")
    parser.add_argument("--output_dir",
                        type=str,
                        default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save evaluation results")
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default=DEFAULT_SAMPLING_STRATEGY,
        choices=["linear", "log"],
        help="Strategy for sampling time steps",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--n_validation_mols",
        type=int,
        default=DEFAULT_N_VALIDATION_MOLS,
        help="Number of validation molecules to evaluate",
    )
    parser.add_argument(
        "--batch_cost",
        type=int,
        default=DEFAULT_BATCH_COST,
        help="Batch cost",
    )
    parser.add_argument(
        "--max_atoms",
        type=int,
        default=DEFAULT_MAX_ATOMS,
        help="Maximum number of atoms",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of workers",
    )
    parser.add_argument(
        "--use_complex_metrics",
        type=bool,
        default=DEFAULT_USE_COMPLEX_METRICS,
        help="Whether to use complex metrics",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=DEFAULT_NUM_GPUS,
        help="Number of GPUs to use for evaluation",
    )

    args = parser.parse_args()
    checkpoint = torch.load(args.model_checkpoint, map_location='cpu')
    merged_args = {**checkpoint['hyper_parameters'], **vars(args)}
    args = SimpleNamespace(**merged_args)

    assert args.num_workers == 0, "num_workers must be 0 for evaluation for PoseCheck"

    main(args)
