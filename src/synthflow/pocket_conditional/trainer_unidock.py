import random
from pathlib import Path

import torch
from rdkit import Chem
from torch import Tensor

from rxnflow.utils.misc import get_worker_env

from synthflow.base.env_ctx_cgflow import SynthesisEnvContext3D_cgflow
from synthflow.config import Config
from synthflow.pocket_conditional.affinity import unidock_vina
from synthflow.pocket_conditional.trainer import PocketConditionalTask, PocketConditionalTrainer
from synthflow.utils.extract_pocket import get_mol_center


class UniDock_MultiPocket_Task(PocketConditionalTask):
    pocket_path: Path

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.root_protein_dir = Path(cfg.task.pocket_conditional.protein_dir)

        # TODO: add protein db structure
        self.pdb_to_pocket: dict[str, tuple[float, float, float]] = {}
        with open(cfg.task.pocket_conditional.train_key) as f:
            for ln in f.readlines():
                pdb, x, y, z = ln.strip().split(",")
                self.pdb_to_pocket[pdb] = (float(x), float(y), float(z))

        self.pdb_keys: list[str] = sorted(list(self.pdb_to_pocket.keys()))
        random.shuffle(self.pdb_keys)

        self.index_path = Path(cfg.log_dir) / "index.csv"
        self.search_mode: str = "fast"

    def sample_conditional_information(self, n: int, train_it: int) -> dict[str, Tensor]:
        ctx: SynthesisEnvContext3D_cgflow = get_worker_env("ctx")
        # set next pocket
        pdb_code = self.pdb_keys[train_it % len(self.pdb_keys)]
        protein_path = self.root_protein_dir / f"{pdb_code}.pdb"
        center = self.pdb_to_pocket[pdb_code]
        ctx.set_pocket(protein_path, center)
        self.pocket_key = pdb_code
        self.pocket_filename = protein_path
        # log what pocket is selected for each training iterations
        with open(self.index_path, "a") as w:
            w.write(f"{self.oracle_idx},{pdb_code}\n")
        return super().sample_conditional_information(n, train_it)

    def calculate_affinity(self, mols: list[Chem.Mol]) -> Tensor:
        center = get_mol_center(self.pocket_path)
        res = unidock_vina.docking(mols, self.pocket_path, center, search_mode=self.search_mode)
        output_result_path = self.save_dir / f"oracle{self.oracle_idx}_redock.sdf"
        with Chem.SDWriter(str(output_result_path)) as w:
            for mol, (docked_mol, _) in zip(mols, res, strict=True):
                docked_mol = Chem.Mol() if docked_mol is None else docked_mol
                docked_mol.SetIntProp("sample_idx", mol.GetIntProp("sample_idx"))
                w.write(docked_mol)
        scores = [min(v, 0.0) for mol, v in res]
        return torch.tensor(scores, dtype=torch.float32)


class UniDock_MultiPocket_Trainer(PocketConditionalTrainer):
    def setup_task(self):
        self.task = UniDock_MultiPocket_Task(cfg=self.cfg)
