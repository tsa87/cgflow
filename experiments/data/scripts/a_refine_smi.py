import argparse
import multiprocessing
from pathlib import Path

from _a_refine import get_clean_smiles
from tqdm import tqdm
import torch


def filter_by_druglikeness(smiles_list, ids, model_path, device, threshold=0.5):
    """Filter SMILES strings based on druglikeness score from RNNLM model."""
    from DeepDL.src.models import RNNLM
    
    print(f"Loading druglikeness model from {model_path}")
    model = RNNLM.load_model(model_path, device)
    
    filtered_smiles = []
    filtered_ids = []
    
    print("Filtering molecules based on druglikeness")
    for smiles, id in tqdm(zip(smiles_list, ids), total=len(smiles_list)):
        if smiles is None:
            continue
        
        try:
            # Score is normalized to [0, 1] range
            score = model.test(smiles) 
            if score >= threshold:
                filtered_smiles.append(smiles)
                filtered_ids.append(id)
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
            continue
    
    
    print(f"After druglikeness filtering: {len(filtered_smiles)} molecules remaining")
    
    return filtered_smiles, filtered_ids


def main(block_path: str, save_block_path: str, num_cpus: int, 
         filter_druglike: bool = False, model_path: str = None, 
         device: str = "cpu", threshold: float = 60):
    block_file = Path(block_path)
    assert block_file.suffix == ".smi"

    print("Read SMI Files")
    with block_file.open() as f:
        lines = f.readlines()[1:]
    smiles_list = [ln.strip().split()[0] for ln in lines]
    ids = [ln.strip().split()[1] for ln in lines]
    print("Including Mols:", len(smiles_list))
    
    print("Run Building Blocks...")
    clean_smiles_list = []
    for idx in tqdm(range(0, len(smiles_list), 10000)):
        chunk = smiles_list[idx : idx + 10000]
        with multiprocessing.Pool(num_cpus) as pool:
            results = pool.map(get_clean_smiles, chunk)
        clean_smiles_list.extend(results)

    filtered_smiles = clean_smiles_list
    filtered_ids = ids
    
    # Filter by druglikeness if requested
    if filter_druglike and model_path:
        filtered_smiles, filtered_ids = filter_by_druglikeness(
            clean_smiles_list, ids, model_path, device, threshold
        )
        print(f"After druglikeness filtering: {len(filtered_smiles)} molecules remaining")

    with open(save_block_path, "w") as w:
        for smiles, id in zip(filtered_smiles, filtered_ids, strict=True):
            if smiles is not None:
                w.write(f"{smiles}\t{id}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get clean building blocks")
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
    parser.add_argument("--cpu", type=int, help="Num Workers", default=1)
    parser.add_argument(
        "--filter_druglike", 
        action="store_true", 
        help="Filter molecules based on druglikeness"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        help="Path to the druglikeness model", 
        default="scripts/DeepDL/test/result/rnn_worlddrug"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        help="Device to run the model (cuda:0, cpu, etc.)", 
        default="cuda"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        help="Druglikeness score threshold (0-100)", 
        default=60
    )
    args = parser.parse_args()

    main(
        args.building_block_path, 
        args.out_path, 
        args.cpu, 
        args.filter_druglike, 
        args.model_path, 
        args.device,
        args.threshold
    )
