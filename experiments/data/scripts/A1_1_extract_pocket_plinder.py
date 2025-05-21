import gc
import shutil
from multiprocessing import Pool
from pathlib import Path

import tqdm
from plinder.core import PlinderSystem
from plinder.core.loader import PlinderDataset
from rdkit import Chem

from cgflow.util.molrepr import GeometricMol
from cgflow.util.pocket import ProteinPocket
from synthflow.utils import extract_pocket

CORE_ATOMS = ["H", "C", "N", "O", "F", "P", "S", "Cl"]
OTHER_ATOMS = ["B", "Si", "Br", "As", "I"]
AVAIL_ATOMS = CORE_ATOMS + OTHER_ATOMS


def runner(key):
    savedir = out_dir / key

    try:
        system = PlinderSystem(system_id=key)
        system._archive = Path("/home/shwan/.local/share/plinder/2024-06/v2/systems/") / key
        receptor_pdb = system.receptor_pdb
        ligand_sdfs = system.ligand_sdfs.items()
    except:
        return

    # pdb, bio_assembly, rec_chain_id, lig_chain_id = key.split("__")
    for lig_key, lig_sdf in ligand_sdfs:
        try:
            system_dir = savedir / lig_key
            # save_protein_path = system_dir / "receptor.pdb"
            save_pocket_path = system_dir / "pocket_15A.pdb"
            save_ligand_path = system_dir / "ligand.sdf"
            if save_ligand_path.exists():
                continue

            smi = system.smiles[lig_key]
            mol = Chem.MolFromSmiles(smi)
            if any(atom.GetSymbol() not in AVAIL_ATOMS for atom in mol.GetAtoms()):
                print(f"pass {key}/{lig_key}: {smi}")
                continue

            system_dir.mkdir(exist_ok=True, parents=True)

            try:
                extract_pocket.extract_pocket_from_center(
                    receptor_pdb, save_pocket_path, cutoff=15, ref_ligand_path=lig_sdf
                )
                assert save_pocket_path.exists()
                GeometricMol.from_sdf(lig_sdf)
                ProteinPocket.from_pdb(save_pocket_path, infer_res_bonds=True, sanitize=True)
            except KeyboardInterrupt as e:
                raise e
            except Exception:
                print(f"fail {key}/{lig_key}")
                if save_pocket_path.exists():
                    save_pocket_path.unlink()
                system_dir.rmdir()
            else:
                shutil.copy(lig_sdf, save_ligand_path)
        except:
            return
    del system


if __name__ == "__main__":
    ROOT_DIR = Path("/home/shwan/Project/CGFlow/data/plinder_data/files_15A/")
    ROOT_DIR.mkdir(exist_ok=True, parents=True)
    if False:
        dataset = PlinderDataset(split="val", use_alternate_structures=False)
        keys = dataset._system_ids
        del dataset
        gc.collect()

        out_dir = ROOT_DIR / "val"
        with tqdm.tqdm(total=len(keys)) as pbar:
            with Pool(4) as pool:
                res = pool.imap_unordered(runner, keys)
                for _ in res:
                    pbar.update(1)

    if True:
        dataset = PlinderDataset(split="train", use_alternate_structures=False)
        keys = sorted(dataset._system_ids)
        del dataset
        gc.collect()

        out_dir = ROOT_DIR / "train"
        with tqdm.tqdm(total=len(keys)) as pbar:
            with Pool(4) as pool:
                res = pool.imap_unordered(runner, keys)
                for _ in res:
                    pbar.update(1)
