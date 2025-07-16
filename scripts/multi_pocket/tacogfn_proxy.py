import argparse
import os

import wandb
from omegaconf import DictConfig, OmegaConf


def parse_args() -> DictConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file.",
        default="./configs/multi_pocket/tacogfn_zincdock.yaml",
    )

    # config override
    # save dir
    parser.add_argument("--name", type=str, help="Name of the experiment.")
    parser.add_argument("--result_dir", type=str, help="Directory to save results.")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases for logging.")
    parser.add_argument("--overwrite_existing_exp", action="store_true", help="Overwrite existing experiment.")

    # environment
    parser.add_argument("--env_dir", type=str, help="Directory containing environment data.")
    parser.add_argument("--max_atoms", type=int, help="Number of maximum atoms.")
    parser.add_argument(
        "--subsampling_ratio", type=float, help="Subsampling ratio for action space; Memory-variance trade-off."
    )

    # opt
    parser.add_argument("--num_steps", type=int, help="Number of training steps for optimization.")
    parser.add_argument("--num_sampling_per_step", type=int, help="Number of sampling for each train step.")
    parser.add_argument(
        "--temperature",
        type=int,
        nargs="+",
        help="Temperature (Exploration-Exploitation Trade-off). e.g., `--temp 32` (const); `--temp 1 64` (unif)",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")

    # pose prediction
    parser.add_argument("--pose_model", type=str, help="Path to the pose prediction model checkpoint.")
    parser.add_argument("--pose_steps", type=int, help="Number of steps for pose prediction.")

    args = parser.parse_args()

    # NOTE: override config
    param: DictConfig = OmegaConf.load(args.config)
    for key in vars(args):
        if key in ("config", "wandb", "overwrite_existing_exp"):
            continue
        assert key in param, f"Key {key} not found in config."
        value = getattr(args, key)
        if value is not None:
            param[key] = value
    param.wandb = args.wandb
    param.overwrite_existing_exp = args.overwrite_existing_exp
    return param


if __name__ == "__main__":
    from synthflow.config import Config, init_empty
    from synthflow.pocket_conditional.trainer_proxy import Proxy_MultiPocket_Trainer

    param = parse_args()

    config = init_empty(Config())

    config.desc = "Multi target training using CrossDock2020 pockets with the proxy model."
    config.print_every = 10
    config.checkpoint_every = 500
    config.store_all_checkpoints = True
    config.num_workers_retrosynthesis = 4
    config.overwrite_existing_exp = param.overwrite_existing_exp
    config.seed = param.seed

    # generative environment
    config.env_dir = param.env_dir
    config.algo.action_subsampling.sampling_ratio = param.subsampling_ratio
    config.algo.max_nodes = param.max_atoms
    config.log_dir = os.path.join(param.result_dir, param.name)

    # pocket conditional
    config.task.pocket_conditional.proxy = param.proxy
    config.task.pocket_conditional.protein_dir = param.protein_dir
    config.task.pocket_conditional.train_key = param.train_key_path

    # model training
    config.num_training_steps = param.num_steps
    config.algo.num_from_policy = 32

    # temperature
    if len(param.temperature) == 1:
        config.cond.temperature.sample_dist = "constant"
    elif len(param.temperature) == 2:
        config.cond.temperature.sample_dist = "uniform"
    else:
        raise ValueError("Temperature should be a single value for constant or two values for uniform.")
    config.cond.temperature.dist_params = param.temperature

    # pose prediction
    config.cgflow.ckpt_path = param.pose_model
    config.cgflow.num_inference_steps = param.pose_steps

    # extras
    config.algo.train_random_action_prob = param.random_action_prob

    trainer = Proxy_MultiPocket_Trainer(config)

    if param.wandb:
        wandb.init(
            name=param.name,
            project="cgflow-gen",
            group="tacogfn-proxy",
            config=OmegaConf.to_container(trainer.cfg, resolve=True),
        )

    trainer.run()
