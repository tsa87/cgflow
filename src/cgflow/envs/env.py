from pathlib import Path
import numpy as np
import torch
from torch import Tensor
import torch_cluster
from rdkit import Chem

from rxnflow.envs.action import RxnAction, RxnActionType
from rxnflow.envs.env import SynthesisEnv, MolGraph
from rxnflow.envs.env_context import SynthesisEnvContext

from cgflow.utils import extract_pocket
from cgflow.utils.featurize_pocket import POCKET_NODE_DIM, POCKET_EDGE_DIM, generate_pocket_data, _rbf


class SynthesisEnv3D(SynthesisEnv):
    def step(self, g: MolGraph, action: RxnAction) -> MolGraph:
        """Applies the action to the current state and returns the next state retaining the coordinate inform.

        Args:
            g (MolGraph): Current state as an RDKit mol.
            action tuple[int, Optional[int], Optional[int]]: Action indices to apply to the current state.
                (ActionType, reaction_template_idx, reactant_idx)

        Returns:
            (Chem.Mol): Next state as an RDKit mol.
        """
        state_info = g.graph
        state_info["updated"] = True
        if action.action is RxnActionType.FirstBlock:
            # initialize state
            g = MolGraph(action.block, **state_info)
            g.mol.AddConformer(Chem.Conformer(g.mol.GetNumAtoms()))

        elif action.action is RxnActionType.BiRxn:
            protocol = self.protocol_dict[action.protocol]
            ps = protocol.rxn_forward.forward(g.mol, Chem.MolFromSmiles(action.block), strict=True)
            assert len(ps) == 1, "reaction is Fail"
            refined_obj = self.get_refined_obj(ps[0][0])
            g = MolGraph(refined_obj, **state_info)
        else:
            # In our setup, Stop and UniRxn is invalid.
            raise ValueError(action.action)
        return g

    def get_refined_obj(self, obj: Chem.Mol) -> Chem.Mol:
        """get refined molecule while retaining atomic coordinates and states"""
        org_obj = obj
        new_obj = Chem.MolFromSmiles(Chem.MolToSmiles(obj))

        org_conf = org_obj.GetConformer()
        new_conf = Chem.Conformer(new_obj.GetNumAtoms())

        # get atom mapping between org_obj and new_obj
        # mask the newly added atoms after reaction.
        is_added = (org_conf.GetPositions() == 0.0).all(-1).tolist()
        atom_order = list(map(int, org_obj.GetProp("_smilesAtomOutputOrder").strip()[1:-1].split(",")))
        atom_mapping = [(org_aidx, new_aidx) for new_aidx, org_aidx in enumerate(atom_order) if not is_added[org_aidx]]

        # transfer atomic information (coords, indexing)
        for org_aidx, new_aidx in atom_mapping:
            org_atom = org_obj.GetAtomWithIdx(org_aidx)
            new_atom = new_obj.GetAtomWithIdx(new_aidx)
            org_atom_info = org_atom.GetPropsAsDict()
            for k in ["gen_order", "react_atom_idx"]:
                if k in org_atom_info:
                    new_atom.SetIntProp(k, org_atom_info[k])
            new_conf.SetAtomPosition(new_aidx, org_conf.GetAtomPosition(org_aidx))
        new_obj.AddConformer(new_conf)
        return new_obj


class SynthesisEnvContext3D(SynthesisEnvContext):
    """This context specifies how to create molecules by applying reaction templates."""

    def __init__(
        self,
        env: SynthesisEnv3D,
        num_cond_dim: int,
        protein_path: str | Path,
        ref_ligand_path: str | Path,
    ):
        super().__init__(env, num_cond_dim)
        self.lig_num_node_dim = self.num_node_dim
        self.lig_num_edge_dim = self.num_edge_dim
        self.num_node_dim += POCKET_NODE_DIM
        self.num_edge_dim += POCKET_EDGE_DIM

        self.protein_path = protein_path = Path(protein_path)
        self.ref_ligand_path = ref_ligand_path = Path(ref_ligand_path)
        pocket_path = extract_pocket.extract_pocket_crossdocked_v1(protein_path, ref_ligand_path)
        self.pocket_center: tuple[float, float, float] = extract_pocket.get_mol_center(pocket_path)
        self.pocket_center_t: Tensor = torch.tensor(self.pocket_center, dtype=torch.float32)
        self.pocket_data: dict[str, Tensor] = self.construct_pocket_data(pocket_path)

    def set_binding_pose_batch(self, gs: list[MolGraph], traj_idx: int, **kwargs) -> None:
        raise NotImplementedError

    def construct_pocket_data(self, pocket_path: Path) -> dict[str, Tensor]:
        pocket_data: dict[str, Tensor] = generate_pocket_data(pocket_path, top_k=10)
        seq = pocket_data["seq"]
        pos = pocket_data["pos"]
        edge_attr = pocket_data["edge_attr"]
        edge_index = pocket_data["edge_index"]

        # match dimension
        x = torch.zeros((seq.shape[0], self.num_node_dim), dtype=torch.float32)
        for i, aa_idx in enumerate(seq):
            x[i, -aa_idx] = 1
        placeholder = torch.zeros((edge_attr.shape[0], self.lig_num_edge_dim), dtype=torch.float32)
        edge_attr = torch.cat([placeholder, edge_attr], dim=-1)

        return dict(x=x, pos=pos, edge_attr=edge_attr, edge_index=edge_index)

    def _graph_to_data_dict(self, g: MolGraph) -> dict[str, Tensor]:
        """Convert a networkx Graph to a torch tensors"""

        lig_data = super()._graph_to_data_dict(g)
        lig_x = lig_data["x"]
        lig_edge_attr = lig_data["edge_attr"]
        lig_edge_index = lig_data["edge_index"]
        lig_graph_attr = lig_data["graph_attr"]
        protocol_mask = lig_data["protocol_mask"]
        level = lig_data["level"]

        # add pos
        if len(g.nodes) == 0:
            lig_pos = self.pocket_center_t.reshape(1, 3)
        else:
            lig_pos = torch.empty((len(g.nodes), 3), dtype=torch.float32)
            for i, n in enumerate(g.nodes):
                ad = g.nodes[n]
                if ad["v"] == "*":
                    lig_pos[i] = self.pocket_center_t
                else:
                    lig_pos[i] = torch.from_numpy(ad["pos"])
            u, v = lig_edge_index
            distance = torch.norm(lig_pos[v] - lig_pos[u], dim=-1)
            lig_edge_attr[:, -16:] = _rbf(distance, D_count=16)

        poc_x = self.pocket_data["x"]
        poc_pos = self.pocket_data["pos"]
        poc_edge_attr = self.pocket_data["edge_attr"]
        poc_edge_index = self.pocket_data["edge_index"] + lig_x.shape[0]

        # create protein-ligand message passing
        u, v = torch_cluster.knn(poc_pos, lig_pos, 10)

        distance = torch.norm(poc_pos[v] - lig_pos[u], dim=-1)
        l2p_edge_attr = torch.zeros((u.shape[0], self.num_edge_dim))
        l2p_edge_attr[:, -16:] = _rbf(distance, D_count=16)
        p2l_edge_attr = l2p_edge_attr

        v = v + lig_x.shape[0]  # shifting protein node index
        l2p_edge_index = torch.stack([u, v])
        p2l_edge_index = torch.stack([v, u])

        return dict(
            x=torch.cat([lig_x, poc_x], 0),
            pos=torch.cat([lig_pos, poc_pos], 0),
            edge_index=torch.cat([lig_edge_index, poc_edge_index, l2p_edge_index, p2l_edge_index], 1),
            edge_attr=torch.cat([lig_edge_attr, poc_edge_attr, l2p_edge_attr, p2l_edge_attr], 0),
            graph_attr=lig_graph_attr,
            protocol_mask=protocol_mask,
            level=level,
        )

    def setup_graph(self, g: MolGraph):
        if not g.is_setup:
            obj = g.mol
            if g.mol.GetNumAtoms() > 0:
                docked_pos = np.array(obj.GetConformer().GetPositions(), dtype=np.float32)
            else:
                docked_pos = np.empty((0, 3), dtype=np.float32)
            for a in obj.GetAtoms():
                attrs = {
                    "atomic_number": a.GetAtomicNum(),
                    "chi": a.GetChiralTag(),
                    "charge": a.GetFormalCharge(),
                    "aromatic": a.GetIsAromatic(),
                    "expl_H": a.GetNumExplicitHs(),
                }
                aid = a.GetIdx()
                g.add_node(
                    aid,
                    v=a.GetSymbol(),
                    pos=docked_pos[aid],
                    **{attr: val for attr, val in attrs.items()},
                )
            for b in obj.GetBonds():
                attrs = {"type": b.GetBondType()}
                g.add_edge(
                    b.GetBeginAtomIdx(),
                    b.GetEndAtomIdx(),
                    **{attr: val for attr, val in attrs.items()},
                )
            g.is_setup = True
