import numpy as np
from pathlib import Path
from rdkit import Chem

import torch
from torch import Tensor

from gflownet.utils.misc import get_worker_device
from rxnflow.envs.env import MolGraph
from cgflow.api import SemlaFlowAPI
from .env import SynthesisEnv3D, SynthesisEnvContext3D


class SynthesisEnvContext3D_semla(SynthesisEnvContext3D):
    def __init__(
        self,
        env: SynthesisEnv3D,
        num_cond_dim: int,
        semlaflow_ckpt_path: str | Path,
        protein_path: str | Path,
        ref_ligand_path: str | Path,
        use_predicted_pose: bool = True,
        num_inference_steps: int = 100,
    ):
        super().__init__(env, num_cond_dim, protein_path, ref_ligand_path)
        device = get_worker_device()

        # NOTE: flow-matching module
        self.semla_api = SemlaFlowAPI.from_protein(
            semlaflow_ckpt_path,
            protein_path,
            ref_ligand_path,
            device=device,
            num_inference_steps=num_inference_steps,
        )
        self.use_predicted_pose = use_predicted_pose

        # temporary cache
        # create when each batch sampling starts
        # removed when each batch sampling is finished
        self.state_mols: dict[int, Chem.Mol]  # x_t
        self.state_coords: dict[int, np.ndarray]  # x_t

    def set_binding_pose_batch(self, graphs: list[MolGraph], traj_idx: int, is_last_step: bool) -> None:
        """run semlaflow binding pose prediction module (x_{t-\\delta t} -> x_t)"""
        # PERF: current implementation use inplace operations during this function to reduce overhead. be careful.
        input_objs = []
        for g in graphs:
            idx = g.graph["sample_idx"]
            obj = g.mol
            if traj_idx > 0:
                # transfer poses information from previous state to current state if state is updated
                if obj.GetNumAtoms() != self.state_mols[idx].GetNumAtoms():
                    self.state_coords[idx] = self.update_coords(obj, self.state_coords[idx])
                # set the coordinates to flow-matching ongoing state (\\hat{x}_1 -> x_{t-\\delta t})
                obj.GetConformer().SetPositions(self.state_coords[idx].copy())  # use copy (sometime error occurs)
            else:
                # initialize empty state
                self.state_mols = {}
                self.state_coords = {}
                pass
            input_objs.append(obj)

        # run semlaflow binding pose prediction (x_{t-\\delta t} -> x_t}
        upd_objs, xt_list, x1_list = self.semla_api.step(input_objs, traj_idx, is_last_step, inplace=True)

        # update the molecule state (gen_order, coordinates)
        for local_idx, g in enumerate(graphs):
            # set the coordinates to flow-matching predicted state (x_t -> \\hat{x}_1)
            # if current step is last step, use x_1 instead of \\hat{x}_1
            idx = g.graph["sample_idx"]

            xt = xt_list[local_idx][-1]
            x1_hat = x1_list[local_idx][-1]

            if self.use_predicted_pose and (not is_last_step):
                g.mol.GetConformer().SetPositions(x1_hat)
            g.graph["updated"] = False

            self.state_mols[idx] = Chem.Mol(g.mol)
            self.state_coords[idx] = xt

        # save or remove the temporary cache
        if is_last_step:
            del self.state_mols, self.state_coords

    def update_coords(self, obj: Chem.Mol, prev_coords: np.ndarray) -> np.ndarray:
        """update previous state's coords to current state's coords

        Parameters
        ----------
        obj : Chem.Mol
            Current state molecule
        prev_coords : np.ndarray
            Coordinates of the previous state

        Returns
        -------
        np.ndarray
            Coordinates of the current state
        """
        out_coords = np.zeros((obj.GetNumAtoms(), 3))
        for atom in obj.GetAtoms():
            if atom.HasProp("react_atom_idx"):
                new_aidx = atom.GetIdx()
                prev_aidx = atom.GetIntProp("react_atom_idx")
                out_coords[new_aidx] = prev_coords[prev_aidx]
        return out_coords

    # NOTE: action masking
    def _graph_to_data_dict(self, g: MolGraph) -> dict[str, Tensor]:
        """Convert a networkx Graph to a torch tensors"""
        data = super()._graph_to_data_dict(g)
        level = torch.tensor([g.mol.GetNumHeavyAtoms() / 50]).view(1, 1)
        data["level"] = level
        return data

    def get_block_data(
        self,
        block_type: str,
        block_indices: Tensor | int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get the block features for the given type and indices

        Parameters
        ----------
        block_type : str
            Block type
        block_indices : torch.Tensor | int
            Block indices for the given block type

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            typ: index tensor for given block type
            descs: molecular feature of blocks
            fp: molecular fingerprints of blocks
            level: block level for action masking (Num Heavy Atom <= 50)
        """
        typ, desc, fp, _ = super().get_block_data(block_type, block_indices)
        if "-" in block_type:  # HACK: change the block type name
            level = (desc[:, 1] / 5 + 0.2).view(-1, 1)  # NumHeavyAtom + 10 <= 50
        else:
            level = (desc[:, 1] / 5).view(-1, 1)  # NumHeavyAtom <= 50
        return typ, desc, fp, level
