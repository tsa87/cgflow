import torch
import tempfile
import warnings
from pathlib import Path

from rdkit import Chem
from semlaflow.util.pocket import PocketComplex
from semlaflow.util.metrics import GenerativeMetric


def load_pc(mol, holo):
    from posecheck import PoseCheck

    pc = PoseCheck()
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        pocket_pdb_path = tmpdir / "pocket.pdb"

        holo.write_pdb(pocket_pdb_path, include_bonds=True)
        with warnings.catch_warnings():
            pc.load_protein_from_pdb(pocket_pdb_path)
        pc.load_ligands_from_mols([mol])
    return pc


class Clash(GenerativeMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("n_clashes", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_valid_clashes", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.Mol], holo_pocks: list[PocketComplex]) -> None:
        assert len(mols) == len(holo_pocks), "Number of molecules and holos must match"

        for mol, holo in zip(mols, holo_pocks, strict=False):
            if mol is None or holo is None:
                continue
            try:
                pc = load_pc(mol, holo)
                # Check for clashes
                clashes = pc.calculate_clashes()
                self.n_clashes += clashes[0]
                self.n_valid_clashes += 1
            except Exception as e:
                print("Error in calculating clashes", e)
                continue

    def compute(self) -> torch.Tensor:
        return self.n_clashes.float() / self.n_valid_clashes.float()


class Interactions(GenerativeMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 'HBAcceptor', 'HBDonor', 'Hydrophobic', 'VdWContact'
        self.add_state("n_hydrophobic", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_vdw", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_hbacceptor", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_hbdonor", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_valid", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mols: list[Chem.Mol], holo_pocks: list[PocketComplex]) -> None:
        assert len(mols) == len(holo_pocks), "Number of molecules and holos must match"
        for mol, holo in zip(mols, holo_pocks, strict=False):
            if mol is None or holo is None:
                continue
            try:
                pc = load_pc(mol, holo)
                # Check for interactions
                interactions = pc.calculate_interactions()

                if len(interactions.sum()) > 0:
                    interactions = interactions.sum().groupby(level=2).sum().to_dict()
                else:
                    interactions = {}

                for key, value in interactions.items():
                    if key == "Hydrophobic":
                        self.n_hydrophobic += value
                    elif key == "VdWContact":
                        self.n_vdw += value
                    elif key == "HBAcceptor":
                        self.n_hbacceptor += value
                    elif key == "HBDonor":
                        self.n_hbdonor += value

                self.n_valid += 1
            except Exception as e:
                print("Error in calculating interactions", e)
                continue

    def compute(self):
        return {
            "hydrophobic": self.n_hydrophobic.float() / self.n_valid.float(),
            "vdw": self.n_vdw.float() / self.n_valid.float(),
            "hbacceptor": self.n_hbacceptor.float() / self.n_valid.float(),
            "hbdonor": self.n_hbdonor.float() / self.n_valid.float(),
        }
