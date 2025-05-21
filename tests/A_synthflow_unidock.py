from synthflow.config import Config, init_empty
from synthflow.tasks.unidock_vina import UniDockVina_MOGFNTrainer

if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.num_training_steps = 10
    config.print_every = 1
    config.log_dir = "./logs/debug-redock/"
    config.env_dir = "./experiments/data/envs/stock"
    config.overwrite_existing_exp = True

    config.task.docking.protein_path = "./experiments/data/LIT-PCBA/ALDH1/protein.pdb"
    config.task.docking.ref_ligand_path = "./experiments/data/LIT-PCBA/ALDH1/ligand.mol2"

    config.cgflow.ckpt_path = "./weights/plinder_till_end.ckpt"
    config.cgflow.use_predicted_pose = True
    config.cgflow.num_inference_steps = 50
    config.task.docking.redocking = True

    trainer = UniDockVina_MOGFNTrainer(config)
    trainer.run()
