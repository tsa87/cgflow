import pickle
import shutil
import warnings
from multiprocessing import Pool
from pathlib import Path

import tqdm

from cgflow.util.data.pocket import ProteinPocket
from synthflow.utils.extract_pocket import extract_pocket_from_center

warnings.filterwarnings("ignore")

PROTEIN_KEYS = Path("/home/shwan/DATA/CrossDocked2020/center_info/train.csv")
LIGAND_ROOT_DIR = Path("/home/shwan/DATA/CrossDocked2020/crossdocked_pocket10/")
PROTEIN_ROOT_DIR = Path("/home/shwan/DATA/CrossDocked2020/protein/train/pdb/")
SAVE_DIR = Path("/home/shwan/Project/CGFlow/data/crossdock/pocket_15A/files/")


def runner(line):
    key, x, y, z = line.split(",")
    center = float(x), float(y), float(z)

    # pdb, bio_assembly, rec_chain_id, lig_chain_id = key.split("__")
    receptor_pdb = PROTEIN_ROOT_DIR / (key + ".pdb")
    out_pocket_path = SAVE_DIR / key / "pocket_15A.pdb"
    out_pocket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        extract_pocket_from_center(receptor_pdb, out_pocket_path, center, cutoff=15)
        assert out_pocket_path.exists()
        ProteinPocket.from_pdb(out_pocket_path, infer_res_bonds=True, sanitize=True)
    except KeyboardInterrupt as e:
        raise e
    except Exception:
        print(f"fail {key}")
        if out_pocket_path.exists():
            out_pocket_path.unlink()


if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    with open("/home/shwan/DATA/CrossDocked2020/center_info/train.csv") as f:
        lines = f.readlines()
        keys = set(ln.split(",")[0] for ln in lines)

    with tqdm.tqdm(total=len(lines)) as pbar:
        with Pool(4) as pool:
            res = pool.imap_unordered(runner, lines)
            for _ in res:
                pbar.update(1)

    with open("/home/shwan/DATA/CrossDocked2020/split_by_name.pkl", "rb") as f:
        split = pickle.load(f)

    for _, ligand_fn in tqdm.tqdm(split["train"]):
        ligand_file = LIGAND_ROOT_DIR / ligand_fn
        protein_key = ligand_file.stem[:6]
        if protein_key not in keys:
            continue
        save_file = SAVE_DIR / protein_key / ligand_file.name
        if not save_file.exists():
            shutil.copyfile(ligand_file, save_file)
