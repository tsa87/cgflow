import os
import sys
import wandb
from omegaconf import OmegaConf

from cgflow.config import Config, init_empty
from cgflow.tasks.semla_vina import VinaMOOTrainer_semla


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
    config.task.docking.redocking = redocking

    config.semlaflow.ckpt_path = ckpt_path
    config.semlaflow.num_inference_steps = num_inference_steps

    config.log_dir = os.path.join(storage, prefix, target, f"seed-{seed}")

    # NOTE: Run
    prefix = f"{prefix}-{target}"
    trainer = VinaMOOTrainer_semla(config)
    wandb.config.update({"prefix": prefix, "config": OmegaConf.to_container(trainer.cfg)})
    trainer.run()
    wandb.finish()


if __name__ == "__main__":
    main()
