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

DISTANCE_DIM = 16


class SynthesisEnvContext3D_cgflow(SynthesisEnvContext3D):
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
        self.cgflow_api = CGFlowAPI(cgflow_ckpt_path, device, num_inference_steps)
        self.use_predicted_pose = use_predicted_pose

        # change dimension
        self.ligand_atom_feat = self.num_node_dim

        self.ligand_fm_dim = self.cgflow_api.d_inv
        self.pocket_fm_dim = self.cgflow_api.d_inv_pocket
        self.num_node_dim += self.ligand_fm_dim + self.pocket_fm_dim
        self.distance_dim = DISTANCE_DIM
        self.num_edge_dim += self.distance_dim

        # temporary cache to store xt
        # create when each batch sampling starts
        # removed when each batch sampling is finished
        self.state_coords: dict[int, np.ndarray] = {}

    def set_pocket(
        self,
        protein_path: str | Path,
        center: tuple[float, float, float] | None = None,
        ref_ligand_path: str | Path | None = None,
        extract: bool = True,
    ):
        """set pocket and center for pose prediction

        Parameters
        ----------
        protein_path : str | Path
            Protein (or Pocket) structure file path (PDB format)
        center : tuple[float, float, float] | None
            Binding site center
        ref_ligand_path : str | Path | None
            Reference ligand structure file path (PDB format) to extract the binding site center
        extract : bool
            If True, the pocket will be extracted from the protein structure using the center or reference ligand.
            Else, the protein file will be considered as the pocket directly.
        """
        self.cgflow_api.set_pocket(protein_path, center, ref_ligand_path, extract)

        # load cgflow embeddings
        pocket_embedding = self.cgflow_api._tmp_pocket_embedding
        center_t = self.cgflow_api._tmp_center.float()

        x = _layernorm(pocket_embedding.inv[0]).cpu().float()
        pos = pocket_embedding.coords[0].cpu().float()
        pad_x = torch.zeros(x.shape[0], self.num_node_dim)
        pad_x[:, -self.pocket_fm_dim :] = x

        self._tmp_pocket_data: dict[str, torch.Tensor] = {"x": pad_x, "pos": pos}
        self._tmp_center: torch.Tensor = center_t

    def trunc_pocket(self):
        del self._tmp_pocket_data
        del self._tmp_center
        self.cgflow_api.trunc_pocket()

    def initialize(self):
        self.state_coords = {}

    def set_binding_pose_batch(
        self,
        graphs: list[MolGraph],
        traj_idx: int,
        is_last_step: bool = False,
        **kwargs,
    ) -> None:
        """run cgflow binding pose prediction module (x_{t-\\delta t} -> x_t)"""
        # PERF: current implementation use inplace operations during this function to reduce overhead. be careful.
        if len(graphs) == 0:
            return
        input_objs = []
        for g in graphs:
            idx = g.graph["sample_idx"]
            obj = g.mol
            if traj_idx == 0:
                # check the cache is initialized
                assert len(self.state_coords) == 0, (
                    'Environmental context is not initialized. Call "initialize" method first.'
                )
            else:
                # mapping the atom index of the previous state coordinates
                # if the molecule structure is updated
                if g.graph["updated"]:
                    self.state_coords[idx] = self.update_coords(obj, self.state_coords[idx])
                # set the coordinates to flow-matching state (x_t)
                obj.GetConformer().SetPositions(self.state_coords[idx])
            input_objs.append(obj)

        # run cgflow binding pose prediction (x_{i\lambda} -> x_{(i+1)\lambda})
        cgflow_outs = self.cgflow_api.run(input_objs, traj_idx, return_traj=True, inplace=True)

        # update the molecule state
        for g, out in zip(graphs, cgflow_outs, strict=True):
            idx = g.graph["sample_idx"]
            # update graph
            g._mol = out.mol
            if self.use_predicted_pose:
                # set the coordinates to predicted pose (\\hat{x}_1) instead of state x_t
                # if it is the last step, use x_{t=1} instead of \\hat{x}_1
                g.mol.GetConformer().SetPositions(out.x1_hat.double().numpy())
            g.graph["x_equi"] = out.x_equi.numpy()
            g.graph["x_inv"] = out.x_inv.numpy()
            g.graph["updated"] = False
            # save state coordinates cache
            self.state_coords[idx] = out.xt.double().numpy()

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

    def _graph_to_data_dict(self, g: MolGraph) -> dict[str, Tensor]:
        """Use CGFlow embeddings"""
        assert isinstance(g, MolGraph)
        self.setup_graph(g)

        # load ligand info
        if hasattr(g, "_Data_cache") and g._Data_cache is not None:
            lig_graph = g._Data_cache

        elif len(g.nodes) == 0:
            x = torch.zeros((1, self.num_node_dim))
            x[0, 0] = 1
            lig_graph = {
                "x": x,
                "pos": torch.zeros((1, 3)),
                "edge_index": torch.zeros((2, 0), dtype=torch.long),
                "edge_attr": torch.zeros((0, self.num_edge_dim)),
                "graph_attr": torch.zeros((self.num_graph_dim,)),
            }

        else:
            # NOTE: node feature
            x = torch.zeros((len(g.nodes), self.num_node_dim))
            pos = torch.zeros((len(g.nodes), 3))
            # atom labeling
            for i, n in enumerate(g.nodes):
                ad = g.nodes[n]
                for k, sl in zip(self.atom_attrs, self.atom_attr_slice, strict=False):
                    idx = self.atom_attr_values[k].index(ad[k]) if k in ad else 0
                    x[i, sl + idx] = 1  # One-hot encode the attribute value
                if ad["v"] != "*":
                    pos[i] = torch.from_numpy(ad["pos"]) - self._tmp_center.view(1, 3)  # atom coordinates
            # normalized flow-matching feature
            x_fm = _layernorm(torch.from_numpy(g.graph["x_inv"]))  # [Natom, F]
            x[:, self.ligand_atom_feat : self.ligand_atom_feat + self.ligand_fm_dim] = x_fm

            # NOTE: edge feature
            edge_index = torch.tensor([e for i, j in g.edges for e in [(i, j), (j, i)]], dtype=torch.long).view(-1, 2).T
            edge_attr = torch.zeros((len(g.edges) * 2, self.num_edge_dim))
            for i, e in enumerate(g.edges):
                ad = g.edges[e]
                for k, sl in zip(self.bond_attrs, self.bond_attr_slice, strict=False):
                    if ad[k] in self.bond_attr_values[k]:
                        idx = self.bond_attr_values[k].index(ad[k])
                    else:
                        idx = 0
                    edge_attr[i * 2, sl + idx] = 1
                    edge_attr[i * 2 + 1, sl + idx] = 1

            # NOTE: graph feature
            # Add molecular properties (multi-modality)
            mol = self.graph_to_obj(g)
            graph_attr = self.get_obj_features(mol)

            lig_graph = {
                "x": x,
                "pos": pos,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "graph_attr": graph_attr,
            }
        # add cache
        g._Data_cache = lig_graph

        # get concat graph
        lig_x, lig_pos, l2l_edge_index, l2l_edge_attr, lig_graph_attr = (
            lig_graph["x"],
            lig_graph["pos"],
            lig_graph["edge_index"],
            lig_graph["edge_attr"],
            lig_graph["graph_attr"],
        )

        # pocket info
        poc_x = self._tmp_pocket_data["x"]
        poc_pos = self._tmp_pocket_data["pos"]

        # create protein-ligand message passing (fully-connected)
        n_lig = lig_pos.size(0)
        n_poc = poc_pos.size(0)
        u = torch.arange(n_lig).repeat_interleave(n_poc)  # ligand indices
        v = torch.arange(n_lig, n_lig + n_poc).repeat(n_lig)  # pocket indices
        p2l_edge_index = torch.stack([v, u])  # Pocket to ligand
        p2l_edge_attr = torch.zeros((u.shape[0], self.num_edge_dim))

        x = torch.cat([lig_x, poc_x], dim=0)
        pos = torch.cat([lig_pos, poc_pos], dim=0)
        edge_index = torch.cat([l2l_edge_index, p2l_edge_index], dim=1)
        edge_attr = torch.cat([l2l_edge_attr, p2l_edge_attr], dim=0)
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


def _rbf(D, D_min=0.0, D_max=10.0, D_count=16, device="cpu"):
    """From https://github.com/jingraham/neurips19-graph-protein-design"""
    D_mu = torch.linspace(D_min, D_max, D_count, device=device).view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    return torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))


@torch.no_grad()
def _layernorm(x: Tensor) -> Tensor:
    return F.layer_norm(x, (x.shape[-1],))
