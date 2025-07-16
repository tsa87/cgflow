import argparse
import datetime
import os

from omegaconf import DictConfig, OmegaConf

from synthflow.config import Config, init_empty


def parse_args() -> DictConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file.")

    # config override
    # save dir
    parser.add_argument("--result_dir", type=str, help="Directory to save results.")

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

    # docking
    parser.add_argument("--protein_path", type=str, help="Path to the protein structure file.")
    parser.add_argument("--center", type=float, nargs=3, help="Center coordinates (x y z) for search box.")
    parser.add_argument(
        "--ref_ligand_path", type=str, help="Path to the reference ligand file (required if center is not given)."
    )
    parser.add_argument("--size", type=float, nargs=3, help="Size (x y z) of the search box.")

    # pose prediction
    parser.add_argument("--pose_model", type=str, help="Path to the pose prediction model checkpoint.")
    parser.add_argument("--pose_steps", type=int, help="Number of steps for pose prediction.")

    args = parser.parse_args()

    # NOTE: override config
    param: DictConfig = OmegaConf.load(args.config)
    for key in vars(args):
        if key == "config":
            continue
        assert key in param, f"Key {key} not found in config."
        value = getattr(args, key)
        if value is not None:
            param[key] = value
    return param


if __name__ == "__main__":
    from tasks.unidock import UniDockVinaMOOTrainer

    param = parse_args()

    config = init_empty(Config())

    config.desc = "Multi objective optimization for UniDock Vina with QED"
    config.task.moo.objectives = ["vina", "qed"]
    config.print_every = 10
    config.num_workers_retrosynthesis = 4

    time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    config.log_dir = os.path.join(param.result_dir, time)

    # generative environment
    config.env_dir = param.env_dir
    config.algo.action_subsampling.sampling_ratio = param.subsampling_ratio
    config.algo.max_nodes = param.max_atoms

    # docking
    config.task.docking.protein_path = param.protein_path
    config.task.docking.ref_ligand_path = param.ref_ligand_path
    config.task.docking.center = param.center
    config.task.docking.size = param.size
    config.task.docking.ff_opt = "uff"

    # opt
    config.num_training_steps = param.num_steps
    config.algo.num_from_policy = param.num_sampling_per_step
    config.seed = param.seed

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
    config.algo.sampling_tau = param.sampling_tau  # EMA
    config.algo.train_random_action_prob = param.random_action_prob  # suggest to set positive value

    config.replay.use = True
    config.replay.warmup = config.algo.num_from_policy * param.replay_warmup_step
    config.replay.capacity = param.replay_capacity
    config.replay.num_from_replay = config.algo.num_from_policy  # buffer sampling size

    trainer = UniDockVinaMOOTrainer(config)
    trainer.run()
