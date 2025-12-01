import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Pocket
    parser.add_argument("--protein_path", type=Path, required=True, help="Directory containing environment data.")
    parser.add_argument(
        "--ref_ligand_path", type=Path, help="Path to the reference ligand file to determine pocket center."
    )
    parser.add_argument("--center", type=float, nargs=3, help="Pocket center coordinates.")

    # Environment
    parser.add_argument("--env_dir", type=Path, required=True, help="Directory for environment.")

    # Generation
    parser.add_argument("--save_dir", type=Path, required=True, help="Directory to save sampled results.")
    parser.add_argument("--num_samples", type=int, help="Number of samples per pocket.", default=100)
    parser.add_argument(
        "--temperature",
        type=int,
        nargs=2,
        help="Temperature (Exploration-Exploitation Trade-off). example: `--temp 1 64` (unif)",
        default=[16, 48],
    )
    parser.add_argument(
        "--subsampling_ratio",
        type=float,
        default=0.1,
        help="Subsampling ratio for action space; Memory-variance trade-off.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for training.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation.")

    # model
    parser.add_argument(
        "--flow_model",
        type=Path,
        help="Path to the flow model checkpoint.",
        default="./weights/cgflow_crossdock.ckpt",
    )
    parser.add_argument(
        "--gfn_model",
        type=Path,
        help="Path to the GFN model checkpoint.",
        default="./weights/3dsynthflow_tacogfn.ckpt",
    )
    args = parser.parse_args()

    if args.center is None and args.ref_ligand_path is None:
        parser.error("Either --center or --ref_ligand_path must be provided to determine the pocket center.")

    return args


def set_seed(seed: int):
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    from rdkit import Chem

    from synthflow.config import Config, init_empty
    from synthflow.pocket_conditional.sampler import PocketConditionalSampler

    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Create sampler
    config = init_empty(Config())
    config.env_dir = args.env_dir
    config.cgflow.ckpt_path = args.flow_model
    config.algo.action_subsampling.sampling_ratio = args.subsampling_ratio
    config.algo.max_nodes = 40
    config.algo.num_from_policy = args.batch_size  # batch size
    sampler = PocketConditionalSampler(config, args.gfn_model, args.device)
    sampler.update_temperature("uniform", args.temperature)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    smiles_path = save_dir / "smiles.csv"
    pose_path = save_dir / "pose.sdf"

    protein_path = args.protein_path
    if args.ref_ligand_path is not None:
        sampler.set_pocket(protein_path, ref_ligand_path=args.ref_ligand_path)
    else:
        assert args.center is not None
        sampler.set_pocket(protein_path, center=args.center)

    res = sampler.sample(100)

    with open(smiles_path, "w") as w:
        w.write(",SMILES\n")
        for idx, sample in enumerate(res):
            smiles = sample["smiles"]
            w.write(f"{idx},{smiles}\n")

    with Chem.SDWriter(str(pose_path)) as w:
        for i, sample in enumerate(res):
            mol = sample["mol"]
            mol.SetIntProp("sample_idx", i)
            w.write(mol)
