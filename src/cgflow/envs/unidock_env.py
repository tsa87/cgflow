from rxnflow.envs.env import MolGraph

from cgflow.utils import unidock
from .env import SynthesisEnvContext3D


class SynthesisEnvContext3D_unidock(SynthesisEnvContext3D):
    def set_binding_pose_batch(self, gs: list[MolGraph], traj_idx: int, **kwargs) -> None:
        objs = [g.mol for g in gs]
        docking_results = unidock.docking(objs, self.protein_path, self.pocket_center)
        for obj, (docked_obj, docking_score) in zip(objs, docking_results, strict=True):
            if docked_obj is None:
                continue
            if obj.GetNumAtoms() != docked_obj.GetNumAtoms():
                continue
            flag = True
            for a1, a2 in zip(obj.GetAtoms(), docked_obj.GetAtoms(), strict=True):
                if a1.GetSymbol() == "*":
                    if a2.GetSymbol() != "C":
                        flag = False
                        break
                elif a1.GetSymbol() != a2.GetSymbol():
                    flag = False
                    break
            if not flag:
                continue
            conf = docked_obj.GetConformer()
            obj.AddConformer(conf)
            obj.SetDoubleProp("docking_score", docking_score)
