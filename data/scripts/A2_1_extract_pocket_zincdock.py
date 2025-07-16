from multiprocessing import Pool
from pathlib import Path

import tqdm

from cgflow.util.pocket import ProteinPocket
from synthflow.utils import extract_pocket

PROTEIN_ROOT_DIR = Path("/home/shwan/DATA/ZINCDock/protein/train/")
SAVE_DIR = Path("/home/shwan/Project/CGFlow/data/zincdock_data/pocket_15A/files/")


def runner(line):
    key, x, y, z = line.split(",")
    center = float(x), float(y), float(z)

    # pdb, bio_assembly, rec_chain_id, lig_chain_id = key.split("__")
    receptor_pdb = PROTEIN_ROOT_DIR / (key + ".pdb")
    out_pocket_path = SAVE_DIR / receptor_pdb.name
    try:
        extract_pocket.extract_pocket_from_center(receptor_pdb, out_pocket_path, center, cutoff=15)
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
    with open("/home/shwan/DATA/ZINCDock/center_info/train.csv") as f:
        lines = f.readlines()
    with tqdm.tqdm(total=len(lines)) as pbar:
        with Pool(4) as pool:
            res = pool.imap_unordered(runner, lines)
            for _ in res:
                pbar.update(1)
