"""Preprocessing script only for Geom Drugs, QM9 is done in the QM9 notebook"""

import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from semlaflow.util.molrepr import GeometricMol
from semlaflow.util.pocket import PocketComplex, ProteinPocket

CROSSDOCK_FOLDER = "semlaflow/saved/data/crossdock/data/crossdocked_pocket10"
CROSSDOCK_SPLIT_FILE = "semlaflow/saved/data/crossdock/data/split_by_name.pt"
SAVE_FOLDER = "semlaflow/saved/data/crossdock-no-litpcba/smol"
DEFAULT_NUM_CHUNKS = 10


# LIT-PCBA PDB ids
pdb_ids = [
    "3p0g",
    "3pds",
    "3sn6",
    "4lde",
    "4ldl",
    "4ldo",
    "4qkx",
    "6mxt",
    "4wp7",
    "4wpn",
    "4x4l",
    "5ac2",
    "5l2m",
    "5l2n",
    "5l2o",
    "5tei",
    "1l2i",
    "2b1v",
    "2b1z",
    "2p15",
    "2q70",
    "2qr9",
    "2qse",
    "2qzo",
    "4ivw",
    "4pps",
    "5drj",
    "5du5",
    "5due",
    "5dzi",
    "5e1c",
    "1xp1",
    "1xqc",
    "2ayr",
    "2iog",
    "2iok",
    "2ouz",
    "2pog",
    "2r6w",
    "3dt3",
    "5aau",
    "5fqv",
    "5t92",
    "5ufx",
    "6b0f",
    "6chw",
    "5fv7",
    "2v3d",
    "2v3e",
    "2xwd",
    "2xwe",
    "3rik",
    "3ril",
    "4i3k",
    "4i3l",
    "4umx",
    "4xrx",
    "4xs3",
    "5de1",
    "5l57",
    "5l58",
    "5lge",
    "5sun",
    "5svf",
    "5tqh",
    "6adg",
    "6b0z",
    "5h84",
    "5h86",
    "5mlj",
    "1pme",
    "2ojg",
    "3sa0",
    "3w55",
    "4qp3",
    "4qp4",
    "4qp9",
    "4qta",
    "4qte",
    "4xj0",
    "4zzn",
    "5ax3",
    "5buj",
    "5v62",
    "6g9h",
    "1fap",
    "1nsg",
    "2fap",
    "3fap",
    "4drh",
    "4dri",
    "4drj",
    "4fap",
    "4jsx",
    "4jt5",
    "5gpg",
    "6b73",
    "3gqy",
    "3gr4",
    "3h6o",
    "3me3",
    "3u2z",
    "4g1n",
    "4jpg",
    "5x1v",
    "5x1w",
    "1zgy",
    "2i4j",
    "2p4y",
    "2q5s",
    "2yfe",
    "3b1m",
    "3hod",
    "3r8a",
    "4ci5",
    "4fgy",
    "4prg",
    "5tto",
    "5two",
    "5y2t",
    "5z5s",
    "2vuk",
    "3zme",
    "4ago",
    "4agq",
    "5g4o",
    "5o1i",
    "3a2i",
    "3a2j",
]


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
        # Skip the pdb files if they are in the pdb_templates
        if any([pdb_id in pdb_file.split("_rec")[0] for pdb_id in pdb_ids]):
            print(f"Skipping {pdb_file} as it is in the pdb_templates")
            continue

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
