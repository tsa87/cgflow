import wandb

from synthflow.config import Config, init_empty
from synthflow.pocket_conditional.trainer_autodock import AutoDock_MultiPocket_Trainer

if __name__ == "__main__":
    """Example of how this trainer can be run"""
    wandb.init(project="cgflow-update", group="sbdd-localopt")

    config = init_empty(Config())
    config.env_dir = "./data/stock"
    config.log_dir = "./logs/rebuttal-multipocket/sbdd_localopt"
    config.print_every = 10
    config.checkpoint_every = 500
    config.store_all_checkpoints = True

    config.algo.train_random_action_prob = 0.1
    config.algo.action_subsampling.sampling_ratio = 0.1  # stock

    config.cgflow.ckpt_path = "../weights/crossdocked2020_till_end.ckpt"
    config.cgflow.use_predicted_pose = True
    config.cgflow.num_inference_steps = 100

    config.task.pocket_conditional.pocket_dir = "./data/CrossDocked2020/"
    config.task.pocket_conditional.train_key = "./data/CrossDocked2020/train_keys.csv"

    trainer = AutoDock_MultiPocket_Trainer(config)
    trainer.run()
