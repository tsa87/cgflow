import multiprocessing
import os
import pickle
import random
from functools import partial
from pathlib import Path

import lmdb
from rdkit import Chem
import tqdm

from cgflow.util.molrepr import GeometricMol
from cgflow.util.pocket import ProteinPocket, PocketComplex


def run(key: str, tmp_dir: Path):
    save_path = tmp_dir / f"{key}.pkl"
    if save_path.exists():
        return
    pocket_path = POCKET_FILE_DIR / f"{key}.pdb"
    ligand_path = LIGAND_FILE_DIR / f"{key}.sdf"
    mols = list(Chem.SDMolSupplier(str(ligand_path)))
    random.shuffle(mols)
    mols = mols[:100]
    poc_obj = ProteinPocket.from_pdb(pocket_path, infer_res_bonds=True, sanitize=True)
    lig_objs = [GeometricMol.from_rdkit(mol) for mol in mols]
    poc_byte = poc_obj.to_bytes()
    lig_bytes = [obj.to_bytes() for obj in lig_objs]
    with open(save_path, "wb") as f:
        data = (poc_byte, lig_bytes)
        pickle.dump(data, f)


if __name__ == "__main__":
    NUM_WORKERS = len(os.sched_getaffinity(0))
    ROOT_DIR = Path("/home/shwan/Project/CGFlow/data/zincdock_data/pocket_15A/")
    POCKET_FILE_DIR = ROOT_DIR / "files"
    LIGAND_FILE_DIR = Path("/home/shwan/DATA/ZINCDock/data/docking/train/0_1000/")
    SAVE_DIR = ROOT_DIR / "lmdb"
    TMP_DIR = ROOT_DIR / "tmp_pkl"
    KEY_DIR = ROOT_DIR / "keys"
    SAVE_DIR.mkdir(exist_ok=True)
    KEY_DIR.mkdir(exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)

    random.seed(0)
    all_protein_keys = [file.stem for file in POCKET_FILE_DIR.iterdir()]
    all_ligand_keys = [file.stem for file in LIGAND_FILE_DIR.iterdir()]
    complex_keys = list(set(all_protein_keys) & set(all_ligand_keys))
    print(len(complex_keys))
    complex_keys.sort()
    random.shuffle(complex_keys)

    train_keys = complex_keys[:14000]
    val_keys = complex_keys[14000:]
    print("split:", len(train_keys), len(val_keys))

    with (KEY_DIR / "train.txt").open("w") as w:
        for key in train_keys:
            w.write(key + "\n")
    with (KEY_DIR / "val.txt").open("w") as w:
        for key in val_keys:
            w.write(key + "\n")

    if True:
        # val set
        keys = val_keys
        save_dir = str(SAVE_DIR / "val")

        with tqdm.trange(len(keys), unit="data", desc="Preprocessing") as pbar:
            with multiprocessing.Pool(NUM_WORKERS) as pool:
                for _ in pool.imap_unordered(partial(run, tmp_dir=TMP_DIR), keys):
                    pbar.update(1)

        print("save validation set")
        env = lmdb.Environment(save_dir, map_size=int(5e9))  # 5gb
        with env.begin(write=True) as txt:
            for key in keys:
                with open(TMP_DIR / f"{key}.pkl", "rb") as f:
                    complex_bytes = f.read()
                txt.put(key.encode(), complex_bytes)
        env.close()

        # test
        env = lmdb.Environment(save_dir, readonly=True, map_size=int(5e9))
        with env.begin() as txt:
            (poc_byte, lig_bytes) = pickle.loads(txt.get(keys[0].encode()))
            poc_obj = ProteinPocket.from_bytes(poc_byte)
            lig_obj = GeometricMol.from_bytes(lig_bytes[0])
            complex_obj = PocketComplex(poc_obj, lig_obj)
        env.close()

    if True:
        # train set
        keys = train_keys
        save_dir = str(SAVE_DIR / "train")

        with tqdm.trange(len(keys), unit="data", desc="Preprocessing") as pbar:
            with multiprocessing.Pool(NUM_WORKERS) as pool:
                for _ in pool.imap_unordered(partial(run, tmp_dir=TMP_DIR), keys):
                    pbar.update(1)

        print("save train set")
        env = lmdb.Environment(save_dir, map_size=int(1e11))  # 100gb
        with env.begin(write=True) as txt:
            for key in keys:
                with open(TMP_DIR / f"{key}.pkl", "rb") as f:
                    complex_bytes = f.read()
                txt.put(key.encode(), complex_bytes)
        env.close()
