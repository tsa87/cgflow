import torch

from synthflow.config import Config, init_empty
from synthflow.tasks.autodock_vina import AutoDockVina_MOGFNTrainer

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.num_training_steps = 10
    config.print_every = 1
    config.log_dir = "./logs/debug-localopt/"
    config.env_dir = "./data/envs/stock"
    config.overwrite_existing_exp = True
    config.algo.action_subsampling.min_sampling = 100

    config.task.docking.protein_path = "./data/experiments/LIT-PCBA/ALDH1/protein.pdb"
    config.task.docking.ref_ligand_path = "./data/experiments/LIT-PCBA/ALDH1/ligand.mol2"

    config.cgflow.ckpt_path = "./weights/crossdocked2020_till_end.ckpt"
    config.cgflow.use_predicted_pose = True
    config.cgflow.num_inference_steps = 80

    trainer = AutoDockVina_MOGFNTrainer(config)
    trainer.run()
