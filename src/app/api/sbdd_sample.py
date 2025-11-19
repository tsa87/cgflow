import random
import time
from pathlib import Path
from typing import List, Dict, Tuple, Union

import numpy as np
import torch

from rdkit import Chem
from synthflow.config import Config, init_empty
from synthflow.pocket_conditional.sampler import PocketConditionalSampler


def set_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def build_pocket_conditional_sampler(
    ckpt_path: Union[str, Path],
    temperature: Tuple[float, float],
    num_samples: int = 100,
    device: str = "cuda",
    seed: int = 1,
):
    ckpt_path = Path(ckpt_path)
    set_seed(seed)

    # Initialize sampler
    config = init_empty(Config())
    config.env_dir = 'experiments/data/envs/stock/'
    config.cgflow.ckpt_path = str(ckpt_path)
    config.cgflow.use_predicted_pose = True
    config.cgflow.num_inference_steps = 100
    config.algo.action_subsampling.sampling_ratio = 1.0
    config.algo.num_from_policy = num_samples
    sampler = PocketConditionalSampler(config, str(ckpt_path), device)
    sampler.update_temperature("uniform", temperature)
    
    return sampler

def sample_against_pockets(
    sampler: PocketConditionalSampler,
    pocket_paths: List[Path],
    num_samples: int = 100,
    seed: int = 1,
) -> Tuple[List[List[Dict]], float]:
    """
    Sample molecules against a list of pocket paths.

    Args:
        sampler: Initialized PocketConditionalSampler instance.
        pocket_paths: List of paths to pocket folders.
        num_samples: Number of molecules to sample per pocket.
        seed: Random seed for reproducibility.

    Returns:
        results: A list of list of samples per pocket.
        avg_runtime: Average sampling time per pocket.
    """
    set_seed(seed)

    results = []
    runtimes = []

    for pocket_path in pocket_paths:
        start_time = time.time()
        samples = sampler.sample_against_pocket(pocket_path, num_samples)
        runtimes.append(time.time() - start_time)
        results.append(samples)

    return results, float(np.mean(runtimes))