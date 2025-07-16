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

PROTEIN_DIR = Path("/home/shwan/DATA/CrossDocked2020/protein/test/pdb/")
TEST_KEY_PATH = Path("/home/shwan/DATA/CrossDocked2020/center_info/test.csv")


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    temperature = [16.0, 48.0]
    ckpt_path = "./logs/camera-ready-multipocket/sbdd_proxy-bs32/model_state_10000.pt"
    save_path = "./result/camera-ready/crossdock-proxy/bs32-10k-bs50-u16-32/"
    # save_path = "./result/camera-ready/crossdock-proxy/speed_check/"

    test_keys: list[tuple[str, tuple[float, float, float]]] = []
    with open(TEST_KEY_PATH) as f:
        for line in f.readlines():
            pdb_key, x, y, z = line.strip().split(",")
            test_keys.append((pdb_key, (float(x), float(y), float(z))))

    st, end = int(sys.argv[1]), int(sys.argv[2])
    test_keys = test_keys[st:end]

    # NOTE: Create sampler
    config = init_empty(Config())
    config.cgflow.ckpt_path = "../weights/final/crossdock_epoch28.ckpt"
    config.algo.action_subsampling.sampling_ratio = 0.1
    config.algo.num_from_policy = 50  # batch size
    device = "cuda"
    sampler = PocketConditionalSampler(config, ckpt_path, device)
    sampler.update_temperature("uniform", temperature)

    # NOTE: Run
    save_path = Path(save_path)
    smiles_path = save_path / "smiles"
    pose_path = save_path / "pose"
    save_path.mkdir(exist_ok=True, parents=True)
    smiles_path.mkdir(exist_ok=True)
    pose_path.mkdir(exist_ok=True)

    runtime: list[float] = []
    runtime_only: list[float] = []

    for key, center in tqdm(test_keys):
        set_seed(1)
        protein_path = PROTEIN_DIR / f"{key}.pdb"

        tick1 = time.time()
        sampler.set_pocket(protein_path, center)
        tick2 = time.time()
        res = sampler.sample(100)
        end = time.time()
        runtime.append(end - tick1)
        runtime_only.append(end - tick2)

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
    print("avg time(ony sampling)", np.mean(runtime_only))
