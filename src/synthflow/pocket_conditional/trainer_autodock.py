import random
from pathlib import Path

import torch
from rdkit import Chem
from torch import Tensor

from rxnflow.utils.misc import get_worker_env
from synthflow.config import Config
from synthflow.pocket_conditional.affinity import autodock_vina
from synthflow.pocket_conditional.env import SynthesisEnvContext3D_pocket_conditional
from synthflow.pocket_conditional.trainer import PocketConditionalTask, PocketConditionalTrainer


class AutoDock_MultiPocket_Task(PocketConditionalTask):
    pocket_path: Path

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.root_pocket_dir = Path(cfg.task.pocket_conditional.pocket_dir)
        self.pocket_pdb_to_files: dict[str, list[str]] = {}
        with open(cfg.task.pocket_conditional.train_key) as f:
            for ln in f.readlines():
                filename, pocket_pdb_key = ln.strip().split(",")
                self.pocket_pdb_to_files.setdefault(pocket_pdb_key, []).append(filename)
        self.pocket_pdb_keys: list[str] = list(self.pocket_pdb_to_files.keys())
        random.shuffle(self.pocket_pdb_keys)
        self.index_path = Path(cfg.log_dir) / "index.csv"

        self.redocking = cfg.task.docking.redocking

    def sample_conditional_information(self, n: int, train_it: int) -> dict[str, Tensor]:
        ctx: SynthesisEnvContext3D_pocket_conditional = get_worker_env("ctx")
        # set next pocket
        pocket_pdb = self.pocket_pdb_keys[train_it % len(self.pocket_pdb_keys)]
        pocket_filename = random.choice(self.pocket_pdb_to_files[pocket_pdb])
        self.pocket_path = self.root_pocket_dir / pocket_filename
        ctx.set_pocket(self.pocket_path)

        # log what pocket is selected for each training iterations
        with open(self.index_path, "a") as w:
            w.write(f"{self.oracle_idx},{pocket_filename}\n")
        return super().sample_conditional_information(n, train_it)

    def calculate_affinity(self, mols: list[Chem.Mol]) -> Tensor:
        vina_module = autodock_vina.VinaDocking(self.pocket_path)
        scores = []
        output_result_path = self.save_dir / f"oracle{self.oracle_idx}_opt.sdf"

        if self.redocking:
            res = [vina_module.docking(mol) for mol in mols]
        else:
            res = vina_module.local_opt_batch(mols)
        del vina_module

        with Chem.SDWriter(str(output_result_path)) as w:
            for mol, (score, docked_mol) in zip(mols, res, strict=True):
                scores.append(min(score, 0))
                docked_mol = Chem.Mol() if docked_mol is None else docked_mol
                docked_mol.SetIntProp("sample_idx", mol.GetIntProp("sample_idx"))
                w.write(docked_mol)
        scores = [min(score, 0.0) for score, _ in res]
        return torch.tensor(scores, dtype=torch.float32)


class AutoDock_MultiPocket_Trainer(PocketConditionalTrainer):
    def setup_task(self):
        self.task = AutoDock_MultiPocket_Task(cfg=self.cfg)
