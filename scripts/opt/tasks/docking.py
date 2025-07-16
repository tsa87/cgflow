from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from rxnflow.base import BaseTask

from synthflow.config import Config
from synthflow.utils.extract_pocket import get_mol_center


class BaseDockingTask(BaseTask, ABC):
    cfg: Config

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.protein_path: Path = Path(cfg.task.docking.protein_path)
        self.size: tuple[float, float, float] = cfg.task.docking.size
        # set center
        if cfg.task.docking.center is not None:
            center = cfg.task.docking.center
        else:
            assert cfg.task.docking.ref_ligand_path is not None, (
                "Neither center coordinates nor a reference ligand path is provided."
            )
            cx, cy, cz = get_mol_center(cfg.task.docking.ref_ligand_path)
            center = round(cx, 3), round(cy, 3), round(cz, 3)
        self.center: tuple[float, float, float] = center

        # binding affinity estimation
        self.redocking: bool = cfg.task.docking.redocking
        self.ff_opt: str = cfg.task.docking.ff_opt
        assert self.ff_opt in ["none", "uff", "mmff"], (
            f"ff_opt must be one of ['none', 'uff', 'mmff']. Got {self.ff_opt}."
        )

        self.save_dir: Path = Path(cfg.log_dir) / "pose"
        self.save_dir.mkdir()

        self.topn_affinity: OrderedDict[str, float] = OrderedDict()
        self.batch_affinity: list[float] = []

    @abstractmethod
    def calc_affinities(self, mols: list[RDMol]) -> list[float]:
        raise NotImplementedError

    def compute_rewards(self, mols: list[RDMol]) -> Tensor:
        self.save_pose(mols)
        fr = self.calc_affinity_reward(mols)
        return fr.reshape(-1, 1)

    def calc_affinity_reward(self, mols: list[RDMol]) -> Tensor:
        affinities = self.calc_affinities(mols)
        self.batch_affinity = affinities
        self.update_storage(mols, affinities)
        fr = torch.tensor(affinities, dtype=torch.float32) * -0.1
        return fr.clip(min=1e-5)

    def save_pose(self, mols: list[RDMol]):
        out_path = self.save_dir / f"oracle{self.oracle_idx}.sdf"
        with Chem.SDWriter(str(out_path)) as w:
            for mol in mols:
                assert mol.HasProp("sample_idx")
                w.write(mol)

    def update_storage(self, mols: list[RDMol], scores: list[float]):
        smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
        self.topn_affinity.update(zip(smiles_list, scores, strict=True))
        topn = sorted(list(self.topn_affinity.items()), key=lambda v: v[1])[:1000]
        self.topn_affinity = OrderedDict(topn)
