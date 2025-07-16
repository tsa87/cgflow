import pickle
import random
from pathlib import Path

import lmdb
from tqdm import tqdm

from cgflow.util.data.molrepr import LigandMol
from cgflow.util.data.pocket import PocketComplex, ProteinPocket


def main():
    FILE_DIR = Path("/home/shwan/Project/CGFlow/data/crossdock/extract_15A/files/")
    ROOT_DIR = Path("/home/shwan/Project/CGFlow/data/crossdock/")
    # ROOT_DIR = Path("/home/shwan/Project/CGFlow/data/crossdock-small/")
    SAVE_DIR = ROOT_DIR / "extract_15A" / "lmdb"
    KEY_DIR = ROOT_DIR / "extract_15A" / "keys"
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    KEY_DIR.mkdir(exist_ok=True, parents=True)

    random.seed(12345)

    with open("/home/shwan/DATA/CrossDocked2020/split_by_name.pkl", "rb") as f:
        split = pickle.load(f)["train"]
    random.shuffle(split)
    train_split, valid_split = split[:99000], split[99000:]
    # train_split, valid_split = split[:1000], split[1000:1100]

    if True:
        # val set
        save_dir = SAVE_DIR / "val"
        key_path = KEY_DIR / "val.txt"
        env = lmdb.Environment(str(save_dir), map_size=int(1e11))
        key_writer = key_path.open("w")
        with env.begin(write=True) as txt:
            for _, ligand_fn in tqdm(valid_split):
                ligand_fn = ligand_fn.split("/")[-1]
                complex_key = ligand_fn.split(".")[0]
                protein_key = ligand_fn[:6]
                ligand_sdf_path = FILE_DIR / protein_key / ligand_fn
                pocket_pdb_path = FILE_DIR / protein_key / "pocket_15A.pdb"
                if ligand_sdf_path.is_file() is False or pocket_pdb_path.is_file() is False:
                    print(ligand_fn, "no file")
                    continue
                try:
                    lig_obj = LigandMol.from_sdf(ligand_sdf_path)
                    poc_obj = ProteinPocket.from_pdb(pocket_pdb_path, infer_res_bonds=True, sanitize=True)
                    complex_obj = PocketComplex(poc_obj, lig_obj)
                except KeyboardInterrupt as e:
                    raise e
                except Exception as e:
                    print(ligand_fn, "fail", e)
                    # raise e
                    continue
                else:
                    txt.put(complex_key.encode(), complex_obj.to_bytes())
                    key_writer.write(complex_key + "\n")
        env.close()
        key_writer.close()

    if True:
        # train set
        save_dir = SAVE_DIR / "train"
        key_path = KEY_DIR / "train.txt"
        env = lmdb.Environment(str(save_dir), map_size=int(1e11))
        key_writer = key_path.open("w")
        with env.begin(write=True) as txt:
            for _, ligand_fn in tqdm(train_split):
                ligand_fn = ligand_fn.split("/")[-1]
                complex_key = ligand_fn.split(".")[0]
                protein_key = ligand_fn[:6]
                ligand_sdf_path = FILE_DIR / protein_key / ligand_fn
                pocket_pdb_path = FILE_DIR / protein_key / "pocket_15A.pdb"
                if ligand_sdf_path.is_file() is False or pocket_pdb_path.is_file() is False:
                    continue
                try:
                    lig_obj = LigandMol.from_sdf(ligand_sdf_path)
                    poc_obj = ProteinPocket.from_pdb(pocket_pdb_path, infer_res_bonds=True, sanitize=True)
                    complex_obj = PocketComplex(poc_obj, lig_obj)
                except:
                    continue
                else:
                    txt.put(complex_key.encode(), complex_obj.to_bytes())
                    key_writer.write(complex_key + "\n")
        env.close()
        key_writer.close()


if __name__ == "__main__":
    main()
