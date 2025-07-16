import argparse
from pathlib import Path

from druglikeness.deepdl import DeepDL


def main(
    block_path: str,
    save_block_path: str,
    device: str = "cpu",
    threshold: float = 60.0,
):
    block_file = Path(block_path)
    assert block_file.suffix == ".smi"

    print("Read SMI file")
    with block_file.open() as f:
        lines = f.readlines()[1:]
    smiles_list: list[str] = [ln.strip().split()[0] for ln in lines]
    ids: list[str] = [ln.strip().split()[1] for ln in lines]
    print("Including mols:", len(smiles_list))

    print("Initializing DeepDL model")
    model = DeepDL.from_pretrained("extended", device)
    batch_size = 256 if device == "cuda" else 64

    print("Filtering molecules based on druglikeness")
    scores: list[float] = []
    for i in range(0, len(smiles_list), 100000):
        _chunk = smiles_list[i : i + 100000]
        print(f"Screening {i + len(_chunk)} / {len(smiles_list)} ...")
        scores += model.screening(_chunk, naive=True, batch_size=batch_size, verbose=True)
    num_pass = sum([v > threshold for v in scores])
    print(f"After druglikeness filtering: {num_pass} molecules remaining")

    with open(save_block_path, "w") as w:
        for id, smiles, score in zip(ids, smiles_list, scores, strict=True):
            if score < threshold:
                continue
            assert smiles is not None, "Clean SMILES should not be None"
            assert len(smiles) > 0, "Clean SMILES should not be empty"
            w.write(f"{smiles}\t{id}\t{score:.2f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get clean building blocks")
    # refine
    parser.add_argument(
        "-b",
        "--building_block_path",
        type=str,
        help="Path to input enamine building block file (.smi)",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        help="Path to output smiles file",
        default="./building_blocks/druglike_blocks.smi",
    )
    parser.add_argument("--threshold", type=float, help="Druglikeness score threshold (0-100)", default=60)
    parser.add_argument("--cuda", action="store_true", help="Use cuda for druglikeness scoring")
    args = parser.parse_args()

    main(
        args.building_block_path,
        args.out_path,
        "cuda" if args.cuda else "cpu",
        args.threshold,
    )
