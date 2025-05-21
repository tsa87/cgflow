import wandb

from synthflow.config import Config, init_empty
from synthflow.pocket_conditional.trainer_proxy import Proxy_MultiPocket_Trainer

if __name__ == "__main__":
    """Example of how this trainer can be run"""
    wandb.init(project="cgflow", group="sbdd-proxy")

    config = init_empty(Config())
    config.env_dir = "./data/envs/stock-2504-druglike"
    config.log_dir = "./logs/service/sbdd-stock-zincdock"
    config.print_every = 10
    config.checkpoint_every = 500
    config.store_all_checkpoints = True

    config.algo.train_random_action_prob = 0.1
    config.algo.action_subsampling.sampling_ratio = 0.5

    config.cgflow.ckpt_path = "../weights/crossdocked2020_till_end.ckpt"
    config.cgflow.use_predicted_pose = True
    config.cgflow.num_inference_steps = 50

    config.task.pocket_conditional.proxy = ("TacoGFN_Reward", "QVina", "ZINCDock15M")
    config.task.pocket_conditional.pocket_dir = "./data/experiments/CrossDocked2020/crossdocked_pocket10/"
    config.task.pocket_conditional.train_key = "./data/experiments/CrossDocked2020/train_keys.csv"

    trainer = Proxy_MultiPocket_Trainer(config)
    trainer.run()
