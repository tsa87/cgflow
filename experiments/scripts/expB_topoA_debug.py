import argparse
import os

import wandb

from synthflow.config import Config, init_empty
from synthflow.tasks.gnina import Gnina_MOGFNTrainer


def main():
    parser = argparse.ArgumentParser(description="Gnina-QED optimization using 3D information")
    parser.add_argument("--prefix", type=str, default="topo2", help="Run prefix")
    parser.add_argument("--storage", type=str, default="logs/ex3_topo2/", help="Storage directory")
    parser.add_argument("--env_dir", type=str, default="data/envs/catalog", help="Environment directory")
    parser.add_argument("--pocket_dir", type=str, default="../topo2/", help="Pocket directory")
    parser.add_argument("--ckpt_path", type=str, default="weights/topo2_epoch209.ckpt", help="Checkpoint path")
    parser.add_argument(
        "--protein", type=str, default="4FM9_T633_prepared_ref_ligand_plinder.pdb", help="Target protein"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--redocking", action="store_true", help="Enable redocking")
    parser.add_argument("--num_inference_steps", type=int, default=80, help="Number of inference steps")

    args = parser.parse_args()

    prefix = args.prefix
    storage = args.storage
    env_dir = args.env_dir
    pocket_dir = args.pocket_dir
    ckpt_path = args.ckpt_path
    target = args.protein
    seed = args.seed
    redocking = args.redocking
    num_inference_steps = args.num_inference_steps

    # Initialize wandb if enabled
    wandb.init(group=prefix)
    wandb.config.update(
        {"protein": target, "seed": seed, "redocking": redocking, "num_inference_steps": num_inference_steps}
    )

    protein_path = os.path.join(pocket_dir, target)
    ref_ligand_path = os.path.join(pocket_dir, "ref_ligand.sdf")

    config = init_empty(Config())
    config.desc = "Gnina-QED optimization using 3D information"
    config.env_dir = env_dir

    config.num_training_steps = 10000
    config.num_validation_gen_steps = 0
    config.num_final_gen_steps = 0
    config.print_every = 1
    config.seed = seed

    config.task.docking.protein_path = protein_path
    config.task.docking.ref_ligand_path = ref_ligand_path
    config.task.docking.redocking = redocking

    config.cgflow.ckpt_path = ckpt_path
    config.cgflow.num_inference_steps = num_inference_steps

    config.log_dir = os.path.join(storage, prefix, target, f"seed-{seed}")

    # NOTE: Run
    trainer = Gnina_MOGFNTrainer(config)
    wandb.config.update({"prefix": prefix})
    trainer.run()
    trainer.terminate()


if __name__ == "__main__":
    main()
