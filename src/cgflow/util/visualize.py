import tempfile
from pathlib import Path

import py3Dmol
from rdkit import Chem
from rdkit.Chem import Mol as RDMol

from cgflow.util.pocket import ProteinPocket


def mol_to_3dview(mol, size=(300, 300), style="stick", surface=False, opacity=0.5, scale=1.0):
    """Draw molecule in 3D

    Args:
    ----
        mol: rdMol, molecule to show
        size: tuple(int, int), canvas size
        style: str, type of drawing molecule
               style can be 'line', 'stick', 'sphere', 'carton'
        surface, bool, display SAS
        opacity, float, opacity of surface, range 0.0-1.0
    Return:
    ----
        viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """
    assert style in ("line", "stick", "sphere", "carton")
    mblock = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(mblock, "mol")
    viewer.setStyle({style: {"scale": scale}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {"opacity": opacity})
    viewer.zoomTo()
    return viewer


def complex_to_3dview(
    ligand: RDMol,
    pocket: ProteinPocket = None,
    reference_ligand: RDMol = None,
):
    view = py3Dmol.view(width=800, height=800)
    model_idx = 0

    with tempfile.TemporaryDirectory() as tmpdirname:
        if pocket is not None:
            tmpdir = Path(tmpdirname)
            pocket_pdb_path = tmpdir / "pocket.pdb"
            pocket.write_pdb(pocket_pdb_path)

            with open(pocket_pdb_path) as ifile:
                pocket_system = "".join([x for x in ifile])
                view.addModelsAsFrames(pocket_system)
                model_idx += 1

        ligand_blk = Chem.MolToMolBlock(ligand)
        ligand_system = "".join([x for x in ligand_blk])
        view.addModelsAsFrames(ligand_system)
        view.setStyle(
            {"model": model_idx},
            {"stick": {"colorscheme": "blackCarbon"}, "sphere": {"scale": 0.25}},
        )
        model_idx += 1

        if reference_ligand is not None:
            ref_ligand_blk = Chem.MolToMolBlock(reference_ligand)
            ref_ligand_system = "".join([x for x in ref_ligand_blk])
            view.addModelsAsFrames(ref_ligand_system)
            view.setStyle(
                {"model": model_idx},
                {"stick": {"colorscheme": "cyanCarbon"}, "sphere": {"scale": 0.25}},
            )

    return view
