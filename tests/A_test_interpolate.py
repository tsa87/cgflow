from rdkit import Chem

from cgflow.buildutil.build_dm import build_dm
from cgflow.data.interpolate import ARGeometricInterpT
from cgflow.scriptutil import load_config

if __name__ == "__main__":
    config = load_config("configs/cgflow/debug.yaml")
    dm = build_dm(config)
    dataset = dm.train_dataset
    assert dataset is not None

    times = [i / 100 for i in range(100)]
    interps: list[ARGeometricInterpT] = dataset.get_trajectory(7, times)
    with Chem.SDWriter("interpolant.sdf") as w:
        for interp in interps:
            new_ligand = interp.masked_ligand_mol.copy_with(interp.xt)
            mol = new_ligand.to_rdkit(sanitise=True)
            w.write(mol)

        data = interps[0].ligand_mol.to_rdkit(sanitise=True)
        w.write(data)
