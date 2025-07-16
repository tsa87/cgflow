"""Preprocessing script only for Geom Drugs, QM9 is done in the QM9 notebook"""

import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from cgflow.util.molrepr import GeometricMol
from cgflow.util.pocket import PocketComplex, ProteinPocket

CROSSDOCK_FOLDER = "/scratch/to.shen/crossdock/crossdocked_pocket10/"
CROSSDOCK_SPLIT_FILE = "/scratch/to.shen/crossdock/split_by_name.pt"
SAVE_FOLDER = "/projects/jlab/to.shen/cgflow-dev/data/complex/crossdock-mmseq2"

# LIT-PCBA PDB ids
with open("../mmseq2/mmseq2_30_pct_remove_list.txt", "r") as f:
    pdb_ids = f.readlines()
pdb_ids = [x.strip() for x in pdb_ids]


# @time_profile(output_file='semla.profile', lines_to_print=500)
def main():
    # if crossdock_pocket_folder is not empty
    save_path = Path(SAVE_FOLDER)
    data_path = Path(CROSSDOCK_FOLDER)

    complex_path_pairs = torch.load(CROSSDOCK_SPLIT_FILE)
    train_pairs = complex_path_pairs["train"]
    test_pairs = complex_path_pairs["test"]

    # Combine train and test pairs
    train_pairs += test_pairs

    train_list = []
    for pdb_file, sdf_file in tqdm(train_pairs):
        # Skip the pdb files if they are in the pdb_templates
        if any([pdb_id in pdb_file.split("_rec")[0] for pdb_id in pdb_ids]):
            print(f"Skipping {pdb_file} as it is in the pdb_templates")
            continue

        pdb_path = data_path / pdb_file
        sdf_path = data_path / sdf_file
        try:
            train_list.append(
                PocketComplex(
                    ProteinPocket.from_pdb(pdb_path,
                                           infer_res_bonds=True,
                                           sanitize=True),
                    GeometricMol.from_sdf(sdf_path),
                ).to_bytes())
        except Exception as e:
            print(f"Failed to process {pdb_file} and {sdf_file} due to {e}")

    test_list = []
    for pdb_file, sdf_file in tqdm(test_pairs):
        pdb_path = data_path / pdb_file
        sdf_path = data_path / sdf_file
        try:
            test_list.append(
                PocketComplex(
                    ProteinPocket.from_pdb(pdb_path,
                                           infer_res_bonds=True,
                                           sanitize=True),
                    GeometricMol.from_sdf(sdf_path),
                ).to_bytes())
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
