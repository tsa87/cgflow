import numpy as np
from rdkit import Chem

from rxnflow.envs.action import RxnAction, RxnActionType
from rxnflow.envs.env import MolGraph, SynthesisEnv
from rxnflow.envs.env_context import SynthesisEnvContext


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

    env: SynthesisEnv3D

    def set_binding_pose_batch(self, gs: list[MolGraph], traj_idx: int, is_last_step: bool, **kwargs) -> None:
        raise NotImplementedError

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
