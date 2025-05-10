"""Preprocessing script only for Geom Drugs, QM9 is done in the QM9 notebook"""

import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from semlaflow.util.molrepr import GeometricMol
from semlaflow.util.pocket import PocketComplex, ProteinPocket

CROSSDOCK_FOLDER = "semlaflow/saved/data/crossdock/data/crossdocked_pocket10"
CROSSDOCK_SPLIT_FILE = "semlaflow/saved/data/crossdock/data/split_by_name.pt"
SAVE_FOLDER = "semlaflow/saved/data/crossdock/smol"
DEFAULT_NUM_CHUNKS = 10


def chunk_generator(data, chunk_size):
    """Yield chunks of data lazily."""
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


# @time_profile(output_file='semla.profile', lines_to_print=500)
def main():
    # if crossdock_pocket_folder is not empty
    save_path = Path(SAVE_FOLDER)
    data_path = Path(CROSSDOCK_FOLDER)

    complex_path_pairs = torch.load(CROSSDOCK_SPLIT_FILE)
    train_pairs = complex_path_pairs["train"]
    test_pairs = complex_path_pairs["test"]

    train_list = []
    for pdb_file, sdf_file in tqdm(train_pairs):
        pdb_path = data_path / pdb_file
        sdf_path = data_path / sdf_file
        try:
            train_list.append(
                PocketComplex(
                    ProteinPocket.from_pdb(pdb_path, infer_res_bonds=True, sanitize=True),
                    GeometricMol.from_sdf(sdf_path),
                ).to_bytes()
            )
        except Exception as e:
            print(f"Failed to process {pdb_file} and {sdf_file} due to {e}")

    test_list = []
    for pdb_file, sdf_file in tqdm(test_pairs):
        pdb_path = data_path / pdb_file
        sdf_path = data_path / sdf_file
        try:
            test_list.append(
                PocketComplex(
                    ProteinPocket.from_pdb(pdb_path, infer_res_bonds=True, sanitize=True),
                    GeometricMol.from_sdf(sdf_path),
                ).to_bytes()
            )
        except Exception as e:
            print(f"Failed to process {pdb_file} and {sdf_file} due to {e}")

    train_save_path = save_path / "train.smol"
    with open(train_save_path, "wb") as f:
        pickle.dump(train_list, f)

    val_save_path = save_path / "val.smol"
    with open(val_save_path, "wb") as f:
        pickle.dump(test_list, f)


if __name__ == "__main__":
    main()
