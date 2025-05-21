import torch

from synthflow.config import Config, init_empty

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.num_training_steps = 100
    config.print_every = 1
    config.log_dir = "./logs/debug-sbdd/"
    config.env_dir = "./experiments/data/envs/stock-2504-druglike"
    config.overwrite_existing_exp = True

    config.cgflow.ckpt_path = "./weights/plinder_till_end.ckpt"
    config.cgflow.use_predicted_pose = True
    config.cgflow.num_inference_steps = 50
    config.algo.action_subsampling.sampling_ratio = 0.1

    config.replay.warmup = 0

    reward = "proxy"
    match reward:
        case "proxy":
            from synthflow.pocket_conditional.trainer_proxy import Proxy_MultiPocket_Trainer

            trainer = Proxy_MultiPocket_Trainer(config)

        case "unidock":
            from synthflow.pocket_conditional.trainer_unidock import UniDock_MultiPocket_Trainer

            trainer = UniDock_MultiPocket_Trainer(config)

        case "autodock":
            from synthflow.pocket_conditional.trainer_autodock import AutoDock_MultiPocket_Trainer

            config.task.docking.redocking = False
            trainer = AutoDock_MultiPocket_Trainer(config)

    trainer.run()
