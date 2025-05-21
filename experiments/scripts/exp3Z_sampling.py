import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm

from synthflow.config import Config, init_empty
from synthflow.pocket_conditional.sampler import PocketConditionalSampler

POCKET_DIR = Path("./data/CrossDocked2020/")
TEST_KEY_PATH = Path("./data/experiments/CrossDocked2020/test_keys.csv")


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    temperature = [16.0, 64.0]
    # ckpt_path = "./logs/exp3-pocket_conditional/localopt-fm100-sr10/model_state_5000.pt"
    # save_path = Path("./result/localopt-fm100-sr10/5k-u32/")
    # ckpt_path = "./logs/exp3-pocket_conditional/proxy-fm100-sr10/model_state_12000.pt"
    ckpt_path = "./weights/sbdd-stock-zincdock/model_state_16000.pt"

    save_path = Path("./result/service/zincdock-proxy/16k-u16/")
    print(ckpt_path)
    print(save_path)

    with open(TEST_KEY_PATH) as f:
        test_keys = [line.strip().split(",") for line in f.readlines()]
    st, end = int(sys.argv[1]), int(sys.argv[2])
    test_keys = test_keys[st:end]

    # NOTE: Create sampler
    config = init_empty(Config())
    config.cgflow.ckpt_path = "../weights/crossdocked2020_till_end.ckpt"
    config.algo.action_subsampling.sampling_ratio = 0.1
    config.algo.num_from_policy = 100  # batch size
    device = "cuda"
    sampler = PocketConditionalSampler(config, ckpt_path, device)
    sampler.update_temperature("uniform", temperature)

    # NOTE: Run
    smiles_path = save_path / "smiles"
    pose_path = save_path / "pose"
    save_path.mkdir(exist_ok=True, parents=True)
    smiles_path.mkdir(exist_ok=True)
    pose_path.mkdir(exist_ok=True)
    runtime = []

    for name, key in tqdm(test_keys):
        set_seed(1)
        pocket_path = POCKET_DIR / name

        st = time.time()
        res = sampler.sample_against_pocket(pocket_path, 100)
        runtime.append(time.time() - st)

        with open(smiles_path / f"{key}.csv", "w") as w:
            w.write(",SMILES\n")
            for idx, sample in enumerate(res):
                smiles = sample["smiles"]
                w.write(f"{idx},{smiles}\n")

        out_path = pose_path / f"{key}.sdf"
        with Chem.SDWriter(str(out_path)) as w:
            for i, sample in enumerate(res):
                mol = sample["mol"]
                mol.SetIntProp("sample_idx", i)
                w.write(mol)

    print("avg time", np.mean(runtime))
