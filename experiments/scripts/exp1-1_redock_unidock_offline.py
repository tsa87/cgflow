import os
import sys

from synthflow.config import Config, init_empty
from synthflow.tasks.unidock_vina import UniDockVina_MOGFNTrainer

TARGETS = [
    "ADRB2",
    "ALDH1",
    "ESR_ago",
    "ESR_antago",
    "FEN1",
    "GBA",
    "IDH1",
    "KAT2A",
    "MAPK1",
    "MTORC1",
    "OPRK1",
    "PKM2",
    "PPARG",
    "TP53",
    "VDR",
]


def main():
    idx = int(sys.argv[1])

    target_idx = idx % 15
    seed = idx // 15
    target = TARGETS[target_idx]

    storage = "./logs/exp1-redocking/"
    env_dir = "./data/envs/catalog/"
    pocket_dir = "./data/test/LIT-PCBA/"
    ckpt_path = "../weights/final/crossdock_epoch28.ckpt"

    num_inference_steps = 20

    protein_path = os.path.join(pocket_dir, target, "protein.pdb")
    ref_ligand_path = os.path.join(pocket_dir, target, "ligand.mol2")

    config = init_empty(Config())
    config.desc = "Vina-QED optimization using 3D information"
    config.env_dir = env_dir

    config.num_training_steps = 1000
    config.num_validation_gen_steps = 0
    config.num_final_gen_steps = 0
    config.print_every = 1
    config.seed = seed

    config.algo.action_subsampling.sampling_ratio = 0.01

    config.task.docking.protein_path = protein_path
    config.task.docking.ref_ligand_path = ref_ligand_path
    config.task.docking.redocking = True

    config.cgflow.ckpt_path = ckpt_path
    config.cgflow.num_inference_steps = num_inference_steps

    config.log_dir = os.path.join(storage, "redock-catalog", target, f"seed-{seed}")

    # NOTE: Run
    trainer = UniDockVina_MOGFNTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
