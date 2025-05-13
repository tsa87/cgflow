import os
import sys

import wandb

from synthflow.config import Config, init_empty
from synthflow.tasks.gnina import Gnina_MOGFNTrainer


def main():
    prefix = sys.argv[1]
    storage = sys.argv[2]
    env_dir = sys.argv[3]
    pocket_dir = sys.argv[4]
    ckpt_path = sys.argv[5]

    wandb.init(group=prefix)
    target = wandb.config["protein"]
    seed = wandb.config["seed"]
    redocking = wandb.config["redocking"]
    num_inference_steps = wandb.config["num_inference_steps"]

    protein_path = os.path.join(pocket_dir, target + ".pdb")
    ref_ligand_path = os.path.join(pocket_dir, "ref_ligand.sdf")

    config = init_empty(Config())
    config.desc = "Gnina-QED optimization using 3D information"
    config.env_dir = env_dir

    config.num_training_steps = 5000
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
