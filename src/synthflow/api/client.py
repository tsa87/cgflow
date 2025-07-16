from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from rdkit import Chem

from rxnflow.envs.reaction import Reaction

from .api import CGFlowAPI


class CGFlowForSyntheticPathway:
    """The wrapper class to get flow-matching trajectory of synthetic pathway"""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cuda",
        protocol_path: str | Path = Path(__file__).parent / "protocol.yaml",
    ):
        # NOTE: flow-matching module
        self.api = CGFlowAPI(checkpoint_path, device)

        # load reactions
        self.reactions: dict[str, Reaction] = {}
        protocol_config = OmegaConf.load(protocol_path)
        for name, cfg in protocol_config.items():
            self.reactions[str(name)] = Reaction(cfg.forward)

    def set_protein(self, protein_path: str | Path, ref_ligand_path: str | Path):
        self.api.set_pocket(protein_path, ref_ligand_path=ref_ligand_path)

    def generate(
        self, pathway: list[tuple[str, str | None]], num_inference_steps: int = 100
    ) -> list[tuple[Chem.Mol, np.ndarray, np.ndarray]]:
        """generate flow matching trajectory from a given trajectory.

        Parameters
        ----------
        pathway : list[tuple[str, str | None]]
            (building block smiles, protocol)

        Returns
        -------
        flow matching trajectory : list[tuple[Chem.Mol, np.ndarray, np.ndarray]]
        (molecule, xt, x1)

        """

        mol = Chem.Mol()
        history: list[tuple[Chem.Mol, np.ndarray, np.ndarray]] = []
        state = {}

        for step_idx in range(self.api.num_ar_steps):
            # run auto-regressive step
            if step_idx < len(pathway):
                block_smi, protocol_name = pathway[step_idx]
                block = Chem.MolFromSmiles(block_smi)
                if step_idx == 0:
                    assert protocol_name is None
                    mol = block
                    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
                else:
                    assert protocol_name is not None
                    rxn_name, _, block_order = protocol_name.split("_")
                    assert rxn_name in self.reactions, f"Unknown reaction: {rxn_name}"
                    reaction = self.reactions[rxn_name]
                    assert block_order in ["b0", "b1"], f"Unknown block order: {block_order}"
                    if block_order == "b0":
                        mol = reaction.forward(block, mol, strict=True)[0][0]
                    elif block_order == "b1":
                        mol = reaction.forward(mol, block, strict=True)[0][0]
                    mol = self.get_refined_obj(mol)
            # run flow-matching step
            mol, traj_xt, traj_x1 = self.run_fm(mol, step_idx, state, num_inference_steps)
            history.append((Chem.Mol(mol), traj_xt, traj_x1))

        return history

    def run_fm(
        self,
        mol: Chem.Mol,
        curr_step: int,
        state: dict[str, Any],
        num_inference_steps: int = 50,
    ) -> tuple[Chem.Mol, np.ndarray, np.ndarray]:
        if curr_step > 0:
            state_mol, state_pose = state["mol"], state["pose"]
            # transfer poses information from previous state to current state if state is updated
            if mol.GetNumAtoms() != state_mol.GetNumAtoms():
                state_pose = self.update_coords(mol, state_pose)
            # set the coordinates to flow-matching ongoing state (\\hat{x}_1 -> x_{t-\\delta t})
            mol.GetConformer().SetPositions(state_pose)

        # run cgflow binding pose prediction (x_{t-\\delta t} -> x_t}
        out = self.api.run([mol], curr_step, num_inference_steps, return_traj=True)[0]
        upd_mol = out.mol
        traj_xt = out.traj_xt
        traj_x1 = out.traj_x1_hat

        # move to numpy
        traj_xt_arr = traj_xt.double().numpy()
        traj_x1_arr = traj_x1.double().numpy()

        state_mol = Chem.Mol(upd_mol)
        state_pose = traj_xt_arr[-1]
        state.update({"mol": state_mol, "pose": state_pose})
        return upd_mol, traj_xt_arr, traj_x1_arr

    def update_coords(self, obj: Chem.Mol, prev_coords: np.ndarray) -> np.ndarray:
        out_coords = np.zeros((obj.GetNumAtoms(), 3))
        for atom in obj.GetAtoms():
            if atom.HasProp("react_atom_idx"):
                new_aidx = atom.GetIdx()
                prev_aidx = atom.GetIntProp("react_atom_idx")
                out_coords[new_aidx] = prev_coords[prev_aidx]
        return out_coords

    @staticmethod
    def get_refined_obj(obj: Chem.Mol) -> Chem.Mol:
        """get refined molecule while retaining atomic coordinates and states"""
        org_obj = obj
        new_obj = Chem.MolFromSmiles(Chem.MolToSmiles(obj))

        org_conf = org_obj.GetConformer()
        new_conf = Chem.Conformer(new_obj.GetNumAtoms())

        is_added = (org_conf.GetPositions() == 0.0).all(-1).tolist()
        atom_order = list(map(int, org_obj.GetProp("_smilesAtomOutputOrder").strip()[1:-1].split(",")))
        atom_mapping = [(org_aidx, new_aidx) for new_aidx, org_aidx in enumerate(atom_order) if not is_added[org_aidx]]

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
