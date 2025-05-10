import torch
import argparse
import numpy as np
import lightning as L
from semlaflow.util.profile import time_profile
import torch.distributed

import semlaflow.scriptutil as util
from semlaflow.buildutil import build_dm, build_model, build_trainer

# turn off rdkit logging
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

DEFAULT_MODEL_CHECKPOINT = None

DEFAULT_D_MODEL = 384
DEFAULT_N_LAYERS = 12
DEFAULT_N_PRO_LAYERS = 12
DEFAULT_D_MESSAGE = 128
DEFAULT_D_EDGE = 128
DEFAULT_N_COORD_SETS = 64
DEFAULT_N_ATTN_HEADS = 32
DEFAULT_D_MESSAGE_HIDDEN = 128
DEFAULT_COORD_NORM = "length"
DEFAULT_SIZE_EMB = 64

DEFAULT_MAX_ATOMS = 256

DEFAULT_EPOCHS = 10000
DEFAULT_VAL_CHECK_EPOCHS = 10
DEFAULT_LR = 0.0003
DEFAULT_BATCH_COST = 4096
DEFAULT_ACC_BATCHES = 1
DEFAULT_GRADIENT_CLIP_VAL = 1.0
DEFAULT_DIST_LOSS_WEIGHT = 0.0
DEFAULT_TYPE_LOSS_WEIGHT = 0.2
DEFAULT_BOND_LOSS_WEIGHT = 1.0
DEFAULT_CHARGE_LOSS_WEIGHT = 1.0
DEFAULT_CONF_COORD_STRATEGY = "gaussian"  # "gaussian" or "harmonic"
DEFAULT_CATEGORICAL_STRATEGY = "uniform-sample"
DEFAULT_LR_SCHEDULE = "constant"
DEFAULT_WARM_UP_STEPS = 10000
DEFAULT_BUCKET_COST_SCALE = "linear"

DEFAULT_N_VALIDATION_MOLS = 2000
DEFAULT_N_TRAINING_MOLS = np.inf
DEFAULT_NUM_INFERENCE_STEPS = 100
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_COORD_NOISE_STD_DEV = 0.2
DEFAULT_TYPE_DIST_TEMP = 1.0
DEFAULT_TIME_ALPHA = 2.0
DEFAULT_TIME_BETA = 1.0
DEFAULT_OPTIMAL_TRANSPORT = "equivariant"

# AR
DEFAULT_T_PER_AR_ACTION = 0.2
DEFAULT_MAX_INTERP_TIME = 1.0
DEFAULT_DECOMPOSITION_STRATEGY = "atom"
DEFAULT_ORDERING_STRATEGY = "connected"
DEFAULT_MAX_ACTION_T = 0.8
DEFAULT_MAX_NUM_CUTS = None
DEFAULT_MIN_GROUP_SIZE = 5

DEFAULT_POCKET_ENCODING = "c-alpha"  # 'calpha' or 'atom'

DEFAULT_MONITOR = "val-validity"
DEFAULT_MONITOR_MODE = "max"
DEFAULT_NUM_WORKERS = None
DEFAULT_NUM_GPUS = 1


@time_profile(output_file="semla.profile", lines_to_print=500)
def main(args, model_checkpoint=None):
    # Set some useful torch properties
    # Float32 precision should only affect computation on A100 and should in theory be a lot faster than the default setting
    # Increasing the cache size is required since the model will be compiled seperately for each bucket
    torch.set_float32_matmul_precision("high")
    # torch._dynamo.config.cache_size_limit = util.COMPILER_CACHE_SIZE

    # print(f"Set torch compiler cache size to {torch._dynamo.config.cache_size_limit}")

    L.seed_everything(12345)
    util.disable_lib_stdout()
    util.configure_fs()

    trainer = build_trainer(args)

    print("Arguments:")
    print(args)

    print("Building model vocab...")
    vocab = util.build_vocab()
    print("Vocab complete.")

    print("Loading datamodule...")
    dm = build_dm(args, vocab)
    print("Datamodule complete.")

    print("Building equinv model...")
    model = build_model(args, dm, vocab)
    print("Model complete.")

    if model_checkpoint is not None:
        print(f"Checkpoint found! Loading model checkpoint from {model_checkpoint}...")
        model.load_state_dict(model_checkpoint["state_dict"])
        print("Model checkpoint loaded.")

    print("Fitting datamodule to model...")
    trainer.fit(model, datamodule=dm)
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Setup args
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--is_pseudo_complex", action="store_true")
    parser.add_argument("--trial_run", action="store_true")
    parser.add_argument("--complex_debug", action="store_true")
    parser.add_argument("--model_checkpoint", type=str, default=DEFAULT_MODEL_CHECKPOINT)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--num_gpus", type=int, default=DEFAULT_NUM_GPUS)

    # Model args
    parser.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--n_pro_layers", type=int, default=DEFAULT_N_PRO_LAYERS)
    parser.add_argument("--d_message", type=int, default=DEFAULT_D_MESSAGE)
    parser.add_argument("--d_edge", type=int, default=DEFAULT_D_EDGE)
    parser.add_argument("--n_coord_sets", type=int, default=DEFAULT_N_COORD_SETS)
    parser.add_argument("--n_attn_heads", type=int, default=DEFAULT_N_ATTN_HEADS)
    parser.add_argument("--d_message_hidden", type=int, default=DEFAULT_D_MESSAGE_HIDDEN)
    parser.add_argument("--coord_norm", type=str, default=DEFAULT_COORD_NORM)
    parser.add_argument("--size_emb", type=int, default=DEFAULT_SIZE_EMB)
    parser.add_argument("--max_atoms", type=int, default=DEFAULT_MAX_ATOMS)
    parser.add_argument("--arch", type=str, default="semla")

    # Training args
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--val_check_epochs", type=int, default=DEFAULT_VAL_CHECK_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--acc_batches", type=int, default=DEFAULT_ACC_BATCHES)
    parser.add_argument("--gradient_clip_val", type=float, default=DEFAULT_GRADIENT_CLIP_VAL)
    parser.add_argument("--align_vector", action="store_true")
    parser.add_argument("--dist_loss_weight", type=float, default=DEFAULT_DIST_LOSS_WEIGHT)
    parser.add_argument("--type_loss_weight", type=float, default=DEFAULT_TYPE_LOSS_WEIGHT)
    parser.add_argument("--bond_loss_weight", type=float, default=DEFAULT_BOND_LOSS_WEIGHT)
    parser.add_argument("--charge_loss_weight", type=float, default=DEFAULT_CHARGE_LOSS_WEIGHT)
    parser.add_argument("--conf_coord_strategy", type=str, default=DEFAULT_CONF_COORD_STRATEGY)
    parser.add_argument("--categorical_strategy", type=str, default=DEFAULT_CATEGORICAL_STRATEGY)
    parser.add_argument("--lr_schedule", type=str, default=DEFAULT_LR_SCHEDULE)
    parser.add_argument("--warm_up_steps", type=int, default=DEFAULT_WARM_UP_STEPS)
    parser.add_argument("--bucket_cost_scale", type=str, default=DEFAULT_BUCKET_COST_SCALE)
    parser.add_argument("--no_ema", action="store_false", dest="use_ema")
    parser.add_argument("--no_self_condition", action="store_false", dest="self_condition")

    # Flow matching and sampling args
    parser.add_argument("--n_validation_mols", type=int, default=DEFAULT_N_VALIDATION_MOLS)
    parser.add_argument("--n_training_mols", type=int, default=DEFAULT_N_TRAINING_MOLS)
    parser.add_argument("--num_inference_steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)
    parser.add_argument("--cat_sampling_noise_level", type=int, default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL)
    parser.add_argument("--coord_noise_std_dev", type=float, default=DEFAULT_COORD_NOISE_STD_DEV)
    parser.add_argument("--type_dist_temp", type=float, default=DEFAULT_TYPE_DIST_TEMP)
    parser.add_argument("--time_alpha", type=float, default=DEFAULT_TIME_ALPHA)
    parser.add_argument("--time_beta", type=float, default=DEFAULT_TIME_BETA)
    parser.add_argument("--optimal_transport", type=str, default=DEFAULT_OPTIMAL_TRANSPORT)
    # Autoregressive args
    parser.add_argument("--t_per_ar_action", type=float, default=DEFAULT_T_PER_AR_ACTION)
    parser.add_argument("--max_interp_time", type=float, default=DEFAULT_MAX_INTERP_TIME)
    parser.add_argument("--decomposition_strategy", type=str, default=DEFAULT_DECOMPOSITION_STRATEGY)
    parser.add_argument("--ordering_strategy", type=str, default=DEFAULT_ORDERING_STRATEGY)
    parser.add_argument("--max_action_t", type=float, default=DEFAULT_MAX_ACTION_T)
    parser.add_argument("--max_num_cuts", type=int, default=DEFAULT_MAX_NUM_CUTS)
    parser.add_argument("--min_group_size", type=int, default=DEFAULT_MIN_GROUP_SIZE)
    # Complex args
    parser.add_argument("--pocket_encoding", type=str, default=DEFAULT_POCKET_ENCODING)

    # Logging args
    parser.add_argument("--monitor", type=str, default=DEFAULT_MONITOR)
    parser.add_argument("--monitor_mode", type=str, default=DEFAULT_MONITOR_MODE)
    parser.add_argument("--use_complex_metrics", action="store_true")

    parser.set_defaults(
        trial_run=False,
        use_ema=True,
        self_condition=True,
        # compile_model=False,
        # mixed_precision=False,
        # distill=False
    )

    args = parser.parse_args()

    if args.model_checkpoint is not None:
        model_checkpoint = torch.load(args.model_checkpoint)
    else:
        model_checkpoint = None

    main(args, model_checkpoint=model_checkpoint)
