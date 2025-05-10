"""Preprocessing script only for Geom Drugs, QM9 is done in the QM9 notebook"""

import argparse
import os
import pickle
import random
import shutil
from pathlib import Path

from tqdm import tqdm

from semlaflow.util.molrepr import GeometricMol
from semlaflow.util.pocket import PocketComplex, ProteinPocket

CROSSDOCK_FOLDER = "semlaflow/saved/data/crossdock/data/crossdocked_pocket10"
CROSSDOCK_POCKET_FOLDER = "semlaflow/saved/data/crossdock/pocket"
ZINC_FOLDER = "semlaflow/saved/data/zinc15m/docking/train/0_1000"
SAVE_FOLDER = "semlaflow/saved/data/zinc15m/smol"
POCKET_LIGAND_PAIR_PATH = "semlaflow/saved/data/zinc15m/pocket_ligand_pairs.pkl"


DEFAULT_NUM_CHUNKS = 10


def extract_pocket_files(crossdock_folder, pocket_folder):
    # find all pdb files in the crossdock dataset
    saved_ids = set()
    for root, dirs, files in tqdm(os.walk(crossdock_folder)):  # noqa
        for file in files:
            if file.endswith(".pdb"):
                pdb_id = file.split("_rec")[0]
                if pdb_id not in saved_ids:
                    saved_ids.add(pdb_id)
                    shutil.copyfile(
                        os.path.join(root, file),
                        os.path.join(pocket_folder, f"{pdb_id}.pdb"),
                    )


def build_pocket_ligand_pairs(zinc_folder, pocket_folder):
    """Build a list of tuples containing the path to a pocket pdb file and a ligand sdf file"""
    complex_path_pairs = []

    for pdb_id in tqdm(os.listdir(zinc_folder)):
        for sdf_id in list(os.listdir(os.path.join(zinc_folder, pdb_id))):
            pdb_path = os.path.join(pocket_folder, pdb_id + ".pdb")
            sdf_path = os.path.join(zinc_folder, pdb_id, sdf_id)
            complex_path_pairs.append((pdb_path, sdf_path))

    return complex_path_pairs


def chunk_generator(data, chunk_size):
    """Yield chunks of data lazily."""
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


# @time_profile(output_file='semla.profile', lines_to_print=500)
def main(args):
    # if crossdock_pocket_folder is not empty
    save_path = Path(SAVE_FOLDER)
    assert args.num_chunks > 1

    if os.path.exists(CROSSDOCK_POCKET_FOLDER) and os.listdir(CROSSDOCK_POCKET_FOLDER):
        print(f"Folder {CROSSDOCK_POCKET_FOLDER} is not empty, skipping extraction")
    else:
        extract_pocket_files(CROSSDOCK_FOLDER, CROSSDOCK_POCKET_FOLDER)

    print("Building complex path pairs")
    if os.path.exists(POCKET_LIGAND_PAIR_PATH):
        with open(POCKET_LIGAND_PAIR_PATH, "rb") as f:
            complex_path_pairs = pickle.load(f)
    else:
        complex_path_pairs = build_pocket_ligand_pairs(ZINC_FOLDER, CROSSDOCK_POCKET_FOLDER)
        with open(POCKET_LIGAND_PAIR_PATH, "wb") as f:
            pickle.dump(complex_path_pairs, f)
    print("Finished building complex path pairs")

    random.shuffle(complex_path_pairs)

    chunk_size = len(complex_path_pairs) // args.num_chunks
    for i, chunk in enumerate(chunk_generator(complex_path_pairs, chunk_size)):
        complete_list = []
        for pdb_path, sdf_path in tqdm(chunk):
            complete_list.append(
                PocketComplex(
                    ProteinPocket.from_pdb(pdb_path, infer_res_bonds=True, sanitize=True),
                    GeometricMol.from_sdf(sdf_path),
                ).to_bytes()
            )

        chunk_save_path = save_path / f"train_{str(i)}.smol"
        with open(chunk_save_path, "wb") as f:
            pickle.dump(complete_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=DEFAULT_NUM_CHUNKS,
        help="Number of chunks to split the data into",
    )
    args = parser.parse_args()

    main(args)
