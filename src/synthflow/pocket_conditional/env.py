from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch import Tensor

from gflownet.utils.misc import get_worker_device
from rxnflow.envs.env import MolGraph
from synthflow.api import CGFlowAPI
from synthflow.base.env import SynthesisEnv3D, SynthesisEnvContext3D

DISTANCE_DIM = 32


class SynthesisEnvContext3D_pocket_conditional(SynthesisEnvContext3D):
    def __init__(
        self,
        env: SynthesisEnv3D,
        num_cond_dim: int,
        cgflow_ckpt_path: str | Path,
        use_predicted_pose: bool = True,
        num_inference_steps: int = 100,
    ):
        super().__init__(env, num_cond_dim)

        # NOTE: flow-matching module
        device = get_worker_device()
        self.cgflow_api = CGFlowAPI(cgflow_ckpt_path, num_inference_steps, device=device)
        self.use_predicted_pose = use_predicted_pose

        # change dimension
        self.ligand_atom_feat = self.num_node_dim
        self.ligand_fm_dim = self.cgflow_api.cfg.n_coord_sets * 3
        self.pocket_fm_dim = self.cgflow_api.cfg.pocket_d_inv
        self.num_node_dim += self.ligand_fm_dim + self.pocket_fm_dim  # ( lig_node_info, lig_fm_feat, poc_fm_feat )
        self.distance_dim = DISTANCE_DIM
        self.num_edge_dim += self.distance_dim

        self.num_cond_dim += self.pocket_fm_dim

        # temporary cache
        # create when each batch sampling starts
        # removed when each batch sampling is finished
        self.state_coords: dict[int, np.ndarray]  # x_t

    def set_pocket(self, pocket_path: str | Path):
        self.cgflow_api.set_pocket(pocket_path)

        pos = self.cgflow_api._tmp_pocket.coords
        center = torch.from_numpy(self.cgflow_api._tmp_center)

        calpha_index = [idx for idx, atom in enumerate(self.cgflow_api._tmp_pocket.atoms) if atom.atom_name == "CA"]
        calpha_pos = pos[calpha_index]

        calpha_x = torch.zeros(len(calpha_index), self.num_node_dim)
        calpha_x_fm_inv = self.cgflow_api._tmp_pocket_data["invs"][0].cpu()[calpha_index]  # [natom, F]
        calpha_x[:, -self.pocket_fm_dim :] = _layernorm(calpha_x_fm_inv)

        self._tmp_pocket_path: Path = Path(pocket_path)
        self._tmp_pocket_data: dict[str, torch.Tensor] = {"x": calpha_x, "pos": calpha_pos}
        self._tmp_pocket_cond: torch.Tensor = _layernorm(self.cgflow_api._tmp_pocket_data["invs"][0].sum(0)).cpu()
        self._tmp_pocket_center: torch.Tensor = center

    def trunc_pocket(self):
        del self._tmp_pocket_path
        del self._tmp_pocket_data
        del self._tmp_pocket_cond
        del self._tmp_pocket_center
        self.cgflow_api.trunc_pocket()

    def set_binding_pose_batch(self, graphs: list[MolGraph], traj_idx: int, is_last_step: bool) -> None:
        """run cgflow binding pose prediction module (x_{t-\\delta t} -> x_t)"""
        # PERF: current implementation use inplace operations during this function to reduce overhead. be careful.
        input_objs = []
        for g in graphs:
            idx = g.graph["sample_idx"]
            obj = g.mol
            if traj_idx == 0:
                # initialize empty state
                self.state_coords = {}
            else:
                # load binding pose from previous state if state is updated
                if g.graph["updated"]:
                    self.state_coords[idx] = self.update_coords(obj, self.state_coords[idx])
                # set the coordinates to flow-matching ongoing state (x_t)
                obj.GetConformer().SetPositions(self.state_coords[idx].copy())  # use copy (sometime error occurs)
            input_objs.append(obj)

        # run cgflow binding pose prediction (x_{i\lambda} -> x_{(i+1)\lambda})
        upd_objs, xt_list, x1_list, hidden_list = self.cgflow_api.run(input_objs, traj_idx, is_last_step, inplace=True)

        # update the molecule state
        for local_idx, g in enumerate(graphs):
            idx = g.graph["sample_idx"]
            g._mol = upd_objs[local_idx]

            xt = xt_list[local_idx][-1]
            x1_hat = x1_list[local_idx][-1]
            hidden_emb = hidden_list[local_idx]
            if self.use_predicted_pose and (not is_last_step):
                # set the coordinates to predicted pose (\\hat{x}_1) instead of state x_t
                # if it is the last step, use x_{t=1} instead of \\hat{x}_1
                g.mol.GetConformer().SetPositions(x1_hat)
            g.graph["updated"] = False
            g.graph["hidden_emb"] = hidden_emb
            self.state_coords[idx] = xt

        # save or remove the temporary cache
        if is_last_step:
            del self.state_coords

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

    def _graph_to_data_dict(
        self,
        g: MolGraph,
    ) -> dict[str, Tensor]:
        """Use CGFlow embeddings"""
        assert isinstance(g, MolGraph)
        self.setup_graph(g)

        if len(g.nodes) == 0:
            lig_x = torch.zeros((1, self.num_node_dim))
            lig_x[0, 0] = 1
            lig_pos = self._tmp_pocket_center.reshape(1, 3)
            l2l_edge_attr = torch.zeros((0, self.num_edge_dim))
            l2l_edge_index = torch.zeros((2, 0), dtype=torch.long)
            lig_graph_attr = torch.zeros((self.num_graph_dim,))
        else:
            # NOTE: node feature
            lig_x = torch.zeros((len(g.nodes), self.num_node_dim))
            lig_pos = self._tmp_pocket_center.reshape(1, 3).repeat(len(g.nodes), 1)
            # atom labeling
            for i, n in enumerate(g.nodes):
                ad = g.nodes[n]
                for k, sl in zip(self.atom_attrs, self.atom_attr_slice, strict=False):
                    idx = self.atom_attr_values[k].index(ad[k]) if k in ad else 0
                    lig_x[i, sl + idx] = 1  # One-hot encode the attribute value
                if ad["v"] != "*":
                    lig_pos[i] = torch.from_numpy(ad["pos"])  # atom coordinates
            # normalized flow-matching feature
            lig_x_fm = torch.from_numpy(g.graph["hidden_emb"])  # [Natom, F]
            lig_x[:, self.ligand_atom_feat : self.ligand_atom_feat + self.ligand_fm_dim] = _layernorm(lig_x_fm)

            # NOTE: edge feature
            l2l_edge_attr = torch.zeros((len(g.edges) * 2, self.num_edge_dim))
            l2l_edge_index = (
                torch.tensor([e for i, j in g.edges for e in [(i, j), (j, i)]], dtype=torch.long).view(-1, 2).T
            )
            for i, e in enumerate(g.edges):
                ad = g.edges[e]
                for k, sl in zip(self.bond_attrs, self.bond_attr_slice, strict=False):
                    if ad[k] in self.bond_attr_values[k]:
                        idx = self.bond_attr_values[k].index(ad[k])
                    else:
                        idx = 0
                    l2l_edge_attr[i * 2, sl + idx] = 1
                    l2l_edge_attr[i * 2 + 1, sl + idx] = 1

            # NOTE: graph feature
            # Add molecular properties (multi-modality)
            mol = self.graph_to_obj(g)
            lig_graph_attr = self.get_obj_features(mol)

        # pocket info
        poc_x = self._tmp_pocket_data["x"]
        poc_pos = self._tmp_pocket_data["pos"]

        # create protein-ligand message passing
        n_lig = lig_pos.size(0)
        n_poc = poc_pos.size(0)
        u = torch.arange(n_lig).repeat_interleave(n_poc)  # ligand indices
        v = torch.arange(n_lig, n_lig + n_poc).repeat(n_lig)  # pocket indices
        p2l_edge_index = torch.stack([v, u])  # Pocket to ligand
        l2p_edge_index = torch.stack([u, v])  # Ligand to pocket
        p2l_edge_attr = l2p_edge_attr = torch.zeros((u.shape[0], self.num_edge_dim))

        # complex
        x = torch.cat([lig_x, poc_x], dim=0)
        pos = torch.cat([lig_pos, poc_pos], dim=0)
        edge_index = torch.cat([l2l_edge_index, p2l_edge_index, l2p_edge_index], dim=1)
        edge_attr = torch.cat([l2l_edge_attr, p2l_edge_attr, l2p_edge_attr], dim=0)
        graph_attr = lig_graph_attr

        # add distance info
        u, v = edge_index
        distance = torch.norm(pos[v] - pos[u], dim=-1)
        edge_attr[:, -self.distance_dim :] = _rbf(distance, D_count=self.distance_dim)

        complex_data = dict(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            graph_attr=graph_attr.reshape(1, -1),
            protocol_mask=self.create_masks(g).reshape(1, -1),
        )
        return complex_data


def _rbf(D, D_min=0.0, D_max=20.0, D_count=32, device="cpu"):
    """From https://github.com/jingraham/neurips19-graph-protein-design"""
    D_mu = torch.linspace(D_min, D_max, D_count, device=device).view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    return torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))


def _layernorm(x: Tensor) -> Tensor:
    return F.layer_norm(x, (x.shape[-1],))
