import argparse
import multiprocessing
import os
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import BondType
from tqdm import tqdm

ATOMS = ["B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
BONDS = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]


def get_clean_smiles(smiles: str) -> str | None:
    if "[2H]" in smiles or "[13C]" in smiles:
        return None

    # smi -> mol
    mol = Chem.MolFromSmiles(smiles, replacements={"[C]": "C", "[CH]": "C", "[CH2]": "C", "[N]": "N"})
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    # refine smi
    smi = Chem.MolToSmiles(mol)
    if smi is None:
        return None

    fail = False
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        atom: Chem.Atom
        if atom.GetSymbol() not in ATOMS:
            fail = True
            break
        elif atom.GetIsotope() != 0:
            fail = True
            break
        if atom.GetFormalCharge() not in [-1, 0, 1]:
            fail = True
            break
        if atom.GetNumExplicitHs() not in [0, 1]:
            fail = True
            break
    if fail:
        return None

    for bond in mol.GetBonds():
        if bond.GetBondType() not in BONDS:
            fail = True
            break
    if fail:
        return None

    # return the largest fragment
    smis = smi.split(".")
    smi = max(smis, key=len)

    return smi


def main(
    block_path: str,
    save_block_path: str,
    num_cpus: int,
):
    block_file = Path(block_path)
    assert block_file.suffix == ".smi"

    print("Read SMI file")
    with block_file.open() as f:
        lines = f.readlines()[1:]
    smiles_list: list[str] = [ln.strip().split()[0] for ln in lines]
    ids: list[str] = [ln.strip().split()[1] for ln in lines]
    print("Including mols:", len(smiles_list))

    print("Refine building blocks...")
    clean_smiles_list: list[str | None] = []
    for idx in tqdm(range(0, len(smiles_list), 10000)):
        chunk = smiles_list[idx : idx + 10000]
        with multiprocessing.Pool(num_cpus) as pool:
            results = pool.map(get_clean_smiles, chunk)
        clean_smiles_list.extend(results)
    clean_ids = [id for i, id in enumerate(ids) if clean_smiles_list[i] is not None]
    clean_smiles = [smi for smi in clean_smiles_list if smi is not None and len(smi) > 0]

    with open(save_block_path, "w") as w:
        for id, smiles in zip(clean_ids, clean_smiles, strict=True):
            assert smiles is not None, "Clean SMILES should not be None"
            assert len(smiles) > 0, "Clean SMILES should not be empty"
            w.write(f"{smiles}\t{id}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get clean building blocks")
    # refine
    parser.add_argument(
        "-b", "--building_block_path", type=str, help="Path to input enamine building block file (.smi)", required=True
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        help="Path to output smiles file",
        default="./building_blocks/enamine_blocks.smi",
    )
    parser.add_argument("--cpu", type=int, help="Num Workers", default=len(os.sched_getaffinity(0)))
    args = parser.parse_args()

    main(args.building_block_path, args.out_path, args.cpu)
