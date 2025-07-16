import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem

from synthflow.api.client import CGFlowForSyntheticPathway


def _remove_star(mol: Chem.Mol) -> Chem.Mol:
    non_star_mol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            non_star_mol.RemoveAtom(atom.GetIdx())
    non_star_mol.UpdatePropertyCache()
    return non_star_mol.GetMol()


def to_sdf(
    history: list[tuple[Chem.Mol, np.ndarray, np.ndarray]],
    xt_sdf_path: str | Path,
    x1_sdf_path: str | Path,
):
    w_xt = Chem.SDWriter(str(xt_sdf_path))
    w_x1 = Chem.SDWriter(str(x1_sdf_path))
    for mol, xt, x1 in history:
        mol = Chem.Mol(mol)
        num_steps = xt.shape[0]

        mol.RemoveAllConformers()
        indices = []
        for t in range(num_steps):
            conf_xt = Chem.Conformer(mol.GetNumAtoms())
            conf_xt.SetPositions(xt[t])
            idx = mol.AddConformer(conf_xt, True)
            indices.append(idx)
        _mol = _remove_star(mol)
        for idx in indices:
            w_xt.write(_mol, idx)

        mol.RemoveAllConformers()
        indices = []
        for t in range(num_steps):
            conf_x1 = Chem.Conformer(mol.GetNumAtoms())
            conf_x1.SetPositions(x1[t])
            idx = mol.AddConformer(conf_x1, True)
            indices.append(idx)
        _mol = _remove_star(mol)
        for idx in indices:
            w_x1.write(_mol, idx)
    w_xt.close()
    w_x1.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein_path", type=Path, help="protein pdb path")
    parser.add_argument("--ref_ligand_path", type=Path, help="reference ligand path to infer pocket center")
    parser.add_argument("--trajectory", type=str, help="json string of trajectory")
    parser.add_argument("--num_inference_steps", type=int, help="Number of inference steps", default=60)
    parser.add_argument("--prefix", type=str, help="prefix of out sdf file", default="output")

    parser.add_argument("--model_path", type=Path, default="./weights/plinder_till_end.ckpt")
    args = parser.parse_args()

    print("input trajectory")
    trajectory = json.loads(args.trajectory)
    print(trajectory)

    """Example of how this trainer can be run"""
    print("load client")
    client = CGFlowForSyntheticPathway(args.model_path, device="cuda")

    print("set protein")
    client.set_protein(args.protein_path, args.ref_ligand_path)

    print("start flow matching inference")
    history = client.generate(trajectory, args.num_inference_steps)

    print("save to sdf files")
    xt_sdf_path = args.prefix + "_xt.sdf"
    x1_sdf_path = args.prefix + "_x1.sdf"
    to_sdf(history, xt_sdf_path, x1_sdf_path)
