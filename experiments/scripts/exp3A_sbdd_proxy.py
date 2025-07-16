import wandb

from synthflow.config import Config, init_empty
from synthflow.pocket_conditional.trainer_proxy import Proxy_MultiPocket_Trainer

if __name__ == "__main__":
    """Example of how this trainer can be run"""
    wandb.init(project="cgflow-cameraready", group="sbdd-proxy")

    config = init_empty(Config())
    config.env_dir = "./data/envs/stock-2504-druglike"
    config.log_dir = "./logs/camera-ready-multipocket/sbdd_proxy-bs32"
    config.print_every = 10
    config.checkpoint_every = 500
    config.store_all_checkpoints = True

    # model training
    config.algo.num_from_policy = 32

    config.algo.train_random_action_prob = 0.1
    config.algo.action_subsampling.sampling_ratio = 0.1  # stock

    config.cgflow.ckpt_path = "../weights/final/crossdock_epoch28.ckpt"
    config.cgflow.num_inference_steps = 20

    config.task.pocket_conditional.protein_dir = "/home/shwan/DATA/CrossDocked2020/protein/train/pdb/"
    config.task.pocket_conditional.train_key = "/home/shwan/DATA/CrossDocked2020/center_info/train.csv"

    trainer = Proxy_MultiPocket_Trainer(config)
    trainer.run()
