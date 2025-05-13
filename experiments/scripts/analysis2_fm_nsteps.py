import os
import sys

import wandb

from synthflow.config import Config, init_empty
from synthflow.tasks.unidock_vina import UniDockVina_Trainer


def main():
    group = sys.argv[1]
    storage = sys.argv[2]
    env_dir = sys.argv[3]
    pocket_dir = sys.argv[4]
    ckpt_path = sys.argv[5]

    wandb.init(group=group)
    target = wandb.config["protein"]
    seed = wandb.config["seed"]
    num_inference_steps = wandb.config["num_inference_steps"]

    protein_path = os.path.join(pocket_dir, target, "protein.pdb")
    ref_ligand_path = os.path.join(pocket_dir, target, "ligand.mol2")

    config = init_empty(Config())
    config.desc = "Vina-QED optimization using 3D information"
    config.env_dir = env_dir

    # stock
    config.algo.action_subsampling.sampling_ratio = 0.05
    config.algo.train_random_action_prob = 0.1

    config.num_training_steps = 1000
    config.num_validation_gen_steps = 0
    config.num_final_gen_steps = 0
    config.print_every = 1
    config.seed = seed

    config.task.docking.protein_path = protein_path
    config.task.docking.ref_ligand_path = ref_ligand_path
    config.task.docking.redocking = True

    config.cgflow.ckpt_path = ckpt_path
    config.cgflow.num_inference_steps = num_inference_steps

    config.log_dir = os.path.join(storage, group, target, f"fm-{num_inference_steps}", f"seed-{seed}")

    # NOTE: Run
    trainer = UniDockVina_Trainer(config)
    trainer.run()
    trainer.terminate()


if __name__ == "__main__":
    main()
