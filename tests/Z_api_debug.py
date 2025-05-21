import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdChemReactions import ReactionFromSmarts

from synthflow.api.api import CGFlowAPI


class Prediction:

    def __init__(
        self,
        cgflow_ckpt_path: str | Path,
        protein_path: str | Path,
        ref_ligand_path: str | Path,
        device: str | torch.device = "cuda",
        num_inference_steps: int = 100,
    ):
        # NOTE: flow-matching module
        self.cgflow_api = CGFlowAPI(cgflow_ckpt_path,
                                    num_inference_steps,
                                    device,
                                    fp16=True)
        self.cgflow_api.set_protein(protein_path, ref_ligand_path)
        self.mol_ongoing: Chem.Mol
        self.poses_ongoing: np.ndarray

    def get_trajectory(
        self,
        mol: Chem.Mol,
        traj_idx: int,
        is_last_step: bool,
    ) -> tuple[Chem.Mol, np.ndarray, np.ndarray]:
        if traj_idx > 0:
            # transfer poses information from previous state to current state if state is updated
            if mol.GetNumAtoms() != self.mol_ongoing.GetNumAtoms():
                self.poses_ongoing = self.update_coords(
                    mol, self.poses_ongoing)
            # set the coordinates to flow-matching ongoing state (\\hat{x}_1 -> x_{t-\\delta t})
            mol.GetConformer().SetPositions(self.poses_ongoing)

        # run cgflow binding pose prediction (x_{t-\\delta t} -> x_t}
        upd_mols, traj_xt, traj_x1, _ = self.cgflow_api.run([mol],
                                                            traj_idx,
                                                            is_last_step,
                                                            return_traj=True)
        self.mol_ongoing = Chem.Mol(upd_mols[0])
        self.poses_ongoing = traj_xt[0][-1]
        return upd_mols[0], traj_xt[0], traj_x1[0]

    def update_coords(self, obj: Chem.Mol,
                      prev_coords: np.ndarray) -> np.ndarray:
        out_coords = np.zeros((obj.GetNumAtoms(), 3))
        for atom in obj.GetAtoms():
            if atom.HasProp("react_atom_idx"):
                new_aidx = atom.GetIdx()
                prev_aidx = atom.GetIntProp("react_atom_idx")
                out_coords[new_aidx] = prev_coords[prev_aidx]
        return out_coords


def get_refined_obj(obj: Chem.Mol) -> Chem.Mol:
    """get refined molecule while retaining atomic coordinates and states"""
    org_obj = obj
    new_obj = Chem.MolFromSmiles(Chem.MolToSmiles(obj))

    org_conf = org_obj.GetConformer()
    new_conf = Chem.Conformer(new_obj.GetNumAtoms())

    is_added = (org_conf.GetPositions() == 0.0).all(-1).tolist()
    atom_order = list(
        map(int,
            org_obj.GetProp("_smilesAtomOutputOrder").strip()[1:-1].split(
                ",")))
    atom_mapping = [(org_aidx, new_aidx)
                    for new_aidx, org_aidx in enumerate(atom_order)
                    if not is_added[org_aidx]]

    # transfer atomic information (coords, indexing)
    for org_aidx, new_aidx in atom_mapping:
        org_atom = org_obj.GetAtomWithIdx(org_aidx)
        new_atom = new_obj.GetAtomWithIdx(new_aidx)
        org_atom_info = org_atom.GetPropsAsDict()
        # print(org_atom.GetIsAromatic(), new_atom.GetIsAromatic())
        for k in ["gen_order", "react_atom_idx"]:
            if k in org_atom_info:
                new_atom.SetIntProp(k, org_atom_info[k])
        new_conf.SetAtomPosition(new_aidx, org_conf.GetAtomPosition(org_aidx))
    new_obj.AddConformer(new_conf)
    return new_obj


def remove_star(mol: Chem.Mol) -> tuple[Chem.RWMol, list[int]]:
    non_star_idcs = [
        i for i, atom in enumerate(mol.GetAtoms()) if atom.GetSymbol() != "*"
    ]
    non_star_mol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            non_star_mol.RemoveAtom(atom.GetIdx())
    non_star_mol.UpdatePropertyCache()
    return non_star_mol, non_star_idcs


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    target = "ALDH1"
    module = Prediction(
        "./weights/plinder_till_end.ckpt",
        f"./experiments/data/LIT-PCBA/{target}/protein.pdb",
        f"./experiments/data/LIT-PCBA/{target}/ligand.mol2",
        "cuda",
        num_inference_steps=100,
    )

    template = "[*:1]-[1*].[*:2]-[2*]>>[*:1]-[*:2]"
    rxn = ReactionFromSmarts(template)

    path = [
        "[1*]c1cc(C(=O)O)c2c(c1)C(=O)CC2",
        "[2*]NC(=O)C[1*]",
        "[2*]c1cn(C)nn1",
    ]

    history: list[tuple[Chem.Mol, np.ndarray, np.ndarray]] = []
    mol = Chem.Mol()
    for step in range(3):
        tick_st = time.time()
        if step < len(path):
            block = Chem.MolFromSmiles(path[step])
            if step == 0:
                mol = block
                conf = Chem.Conformer(mol.GetNumAtoms())
                mol.AddConformer(conf)
            else:
                mol = rxn.RunReactants((mol, block))[0][0]
                mol = get_refined_obj(mol)
        else:
            mol = Chem.Mol(mol)
        tick_end = time.time()
        mol, traj_xt, traj_x1 = module.get_trajectory(mol, step, (step == 2))
        print(Chem.MolToSmiles(mol), tick_end - tick_st)
        history.append((Chem.Mol(mol), traj_xt, traj_x1))

    print("Generation complete.")
    w1 = Chem.SDWriter("./example_ongoing.sdf")
    w2 = Chem.SDWriter("./example_predicted.sdf")
    print(len(history))
    for mol, xt, x1 in history:
        num_traj = xt.shape[0]
        print(num_traj)
        for t in range(num_traj):
            mol.RemoveAllConformers()
            conf_t = Chem.Conformer(mol.GetNumAtoms())
            conf_t.SetPositions(xt[t])
            conf_1 = Chem.Conformer(mol.GetNumAtoms())
            conf_1.SetPositions(x1[t])
            idx1 = mol.AddConformer(conf_t, True)
            idx2 = mol.AddConformer(conf_1, True)
            _mol, _ = remove_star(mol)
            w1.write(_mol, idx1)
            w2.write(_mol, idx2)
    w1.close()
    w2.close()
