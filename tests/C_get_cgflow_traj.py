import json

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


def to_sdf(history: list[tuple[Chem.Mol, np.ndarray, np.ndarray]], xt_sdf_path: str, x1_sdf_path: str):
    w_xt = Chem.SDWriter(xt_sdf_path)
    w_x1 = Chem.SDWriter(x1_sdf_path)
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
    """Example of how this trainer can be run"""
    print("load client")
    client = CGFlowForSyntheticPathway(
        # "./result/cgflow_0604/cgflow/i4dl6xzg/checkpoints/epoch24-step26950-rmsd2.52.ckpt", device="cuda"
        "./weights/final/crossdock_small_epoch28.ckpt",
        device="cuda",
    )

    print("set protein")
    target = "ALDH1"
    client.set_protein(
        f"./experiments/data/test/LIT-PCBA/{target}/protein.pdb",
        f"./experiments/data/test/LIT-PCBA/{target}/ligand.mol2",
    )

    print("start flow matching inference")
    pathway_str = '[["[17*]C(CBr)c1ccc(F)cc1", null], ["[12*]C([15*])c1ccc(F)cc1", "rxn18_linker_b0"], ["[19*]C(Cc1ccccc1)C(C)=O", "rxn36_brick_b0"]]'
    pathway = json.loads(pathway_str)
    history = client.generate(pathway, num_inference_steps=100)

    print("save to sdf files")
    to_sdf(history, "example_ongoing.sdf", "example_predicted.sdf")
