import random
import time

import numpy as np
import torch
from tqdm import tqdm

from synthflow.config import Config, init_empty
from synthflow.pocket_specific.sampler import CGFlowSampler


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # NOTE: Create sampler
    config = init_empty(Config())
    config.algo.num_from_policy = 64  # batch size
    config.algo.action_subsampling.sampling_ratio = 0.1
    config.num_workers_retrosynthesis = 0
    device = "cuda"

    for step in [10, 20, 40, 60, 80, 100]:
        times = []
        for seed in tqdm([0, 1, 2, 3], leave=False):
            set_seed(0)
            ckpt_path = f"./logs-old/exp5-fm-abl/rebuttal/ALDH1/fm-{step}/seed-{seed}/model_state.pt"
            sampler = CGFlowSampler(config, ckpt_path, device)

            # NOTE: Run
            st = time.time()
            res = sampler.sample(128)
            end = time.time()
            times.append((end - st) / 128)
        print(step, np.mean(times), np.std(times))
