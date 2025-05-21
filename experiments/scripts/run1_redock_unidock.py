import os
import sys

import wandb

from synthflow.config import Config, init_empty
from synthflow.tasks.unidock_vina import UniDockVina_MOGFNTrainer


def main():
    group = sys.argv[1]
    storage = sys.argv[2]
    env_dir = sys.argv[3]
    pocket_dir = sys.argv[4]
    ckpt_path = sys.argv[5]
    target  = sys.argv[6]
    seed = sys.argv[7]
    num_inference_steps = sys.argv[8]

    wandb.init(group=group)
    protein_path = os.path.join(pocket_dir, target, "protein.pdb")
    ref_ligand_path = os.path.join(pocket_dir, target, "ligand.mol2")

    config = init_empty(Config())
    config.desc = "Vina-QED optimization using 3D information"
    config.env_dir = env_dir

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

    config.log_dir = os.path.join(storage, group, target, f"seed-{seed}")

    # NOTE: Run
    trainer = UniDockVina_MOGFNTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
