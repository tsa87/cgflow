import gc
import pickle
from pathlib import Path

import lmdb
from tqdm import tqdm

from cgflow.util.molrepr import GeometricMol
from cgflow.util.pocket import ProteinPocket


def main():
    ROOT_DIR = Path("/home/shwan/Project/CGFlow/data/plinder_data/")
    FILE_DIR = ROOT_DIR / "files_15A"
    SAVE_DIR = ROOT_DIR / "extract_15A" / "lmdb"
    KEY_DIR = ROOT_DIR / "extract_15A" / "keys"
    SAVE_DIR.mkdir(exist_ok=True, parents=True)
    KEY_DIR.mkdir(exist_ok=True, parents=True)

    if True:
        # val set
        root_dir = FILE_DIR / "val"
        save_dir = SAVE_DIR / "val"
        key_path = KEY_DIR / "val.txt"
        env = lmdb.Environment(str(save_dir), map_size=int(1e11))
        key_writer = key_path.open("w")
        with env.begin(write=True) as txt:
            for system_dir in tqdm(sorted(root_dir.iterdir(), key=lambda k: k.name)):
                for ligand_dir in sorted(system_dir.iterdir(), key=lambda k: k.name):
                    complex_key = f"{system_dir.name}/{ligand_dir.name}"
                    ligand_sdf_path = ligand_dir / "ligand.sdf"
                    pocket_pdb_path = ligand_dir / "pocket_15A.pdb"
                    lig_obj = GeometricMol.from_sdf(ligand_dir / "ligand.sdf")
                    poc_obj = ProteinPocket.from_pdb(pocket_pdb_path, infer_res_bonds=True, sanitize=True)
                    data = (lig_obj.to_bytes(), poc_obj.to_bytes())
                    complex_bytes = pickle.dumps(data)
                    txt.put(complex_key.encode(), complex_bytes)
                    key_writer.write(complex_key + "\n")
        env.close()
        key_writer.close()

    if True:
        # train set
        root_dir = FILE_DIR / "train"
        save_dir = SAVE_DIR / "train"
        key_path = KEY_DIR / "train.txt"
        env = lmdb.Environment(str(save_dir), map_size=int(1e11))
        key_writer = key_path.open("w")
        txn = env.begin(write=True)
        counter = 0
        for system_dir in tqdm(sorted(root_dir.iterdir(), key=lambda k: k.name)):
            for ligand_dir in sorted(system_dir.iterdir(), key=lambda k: k.name):
                complex_key = f"{system_dir.name}/{ligand_dir.name}"
                ligand_sdf_path = ligand_dir / "ligand.sdf"
                pocket_pdb_path = ligand_dir / "pocket_15A.pdb"
                lig_obj = GeometricMol.from_sdf(ligand_sdf_path)
                poc_obj = ProteinPocket.from_pdb(pocket_pdb_path, infer_res_bonds=True, sanitize=True)
                complex_bytes = pickle.dumps((lig_obj.to_bytes(), poc_obj.to_bytes()))
                txn.put(complex_key.encode(), complex_bytes)
                key_writer.write(complex_key + "\n")
            counter += 1
            if counter == 10000:
                counter = 0
                txn.commit()
                gc.collect()
                txn = env.begin(write=True)
        txn.commit()
        env.sync()
        env.close()
        key_writer.close()


if __name__ == "__main__":
    main()
