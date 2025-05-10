import torch
from cgflow.config import init_empty, Config
from cgflow.tasks.semla_vina import VinaMOOTrainer_semla
from cgflow.tasks.unidock_vina import VinaMOOTrainer_unidock


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.num_training_steps = 10
    config.print_every = 1
    config.log_dir = "./logs/debug-semla/"
    config.env_dir = "./data/envs/catalog"
    config.overwrite_existing_exp = True
    config.algo.action_subsampling.min_sampling = 10

    config.task.docking.protein_path = "./data/experiments/LIT-PCBA/ADRB2/protein.pdb"
    config.task.docking.ref_ligand_path = "./data/experiments/LIT-PCBA/ADRB2/ligand.mol2"

    config.semlaflow.ckpt_path = "./weights/crossdocked_mini_epoch97.ckpt"
    config.semlaflow.use_predicted_pose = True
    config.semlaflow.num_inference_steps = 80

    tool = "semla"
    config.task.docking.redocking = True

    if tool == "semla":
        trainer = VinaMOOTrainer_semla(config)
    elif tool == "unidock":
        trainer = VinaMOOTrainer_unidock(config)
    else:
        raise ValueError(tool)
    trainer.run()
