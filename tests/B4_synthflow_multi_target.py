from synthflow.config import Config, init_empty
from synthflow.pocket_conditional.trainer_proxy import Proxy_MultiPocket_Trainer

if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.env_dir = "./experiments/data/envs/stock-2504-druglike"
    config.log_dir = "./logs/debug-multitarget"
    config.print_every = 1
    config.overwrite_existing_exp = True

    config.algo.train_random_action_prob = 0.1
    config.algo.action_subsampling.sampling_ratio = 0.1  # stock

    config.cgflow.ckpt_path = "./weights/final/crossdock_epoch27.ckpt"
    config.cgflow.num_inference_steps = 20

    config.task.pocket_conditional.protein_dir = "/home/shwan/DATA/CrossDocked2020/protein/train/pdb/"
    config.task.pocket_conditional.train_key = "/home/shwan/DATA/CrossDocked2020/center_info/train.csv"

    trainer = Proxy_MultiPocket_Trainer(config)
    trainer.run()
