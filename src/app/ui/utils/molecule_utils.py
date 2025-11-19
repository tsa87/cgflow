"""Molecule handling utilities for the drug discovery platform."""

from typing import List, Dict, Any, Optional, Tuple
import io
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def get_ligands_for_protein(
    protein_file: str, 
    binding_site: Optional[str] = None, 
    n_results: int = 10
) -> List[Chem.Mol]:
    """
    Retrieve potential ligands for a protein.
    
    In production, this would connect to a real database or service.
    Currently returns mock data for demonstration purposes.
    
    Args:
        protein_file: Path to the protein PDB file
        binding_site: Optional binding site specification (residues or coordinates)
        n_results: Maximum number of results to return
        
    Returns:
        List of RDKit molecule objects with properties set
    """
    try:
        # Example molecules for demonstration
        example_smiles = [
            "CCO", "CC(=O)O", "CNC", "c1ccccc1", "CC(C)CC(=O)O", 
            "COc1ccc(CC(=O)O)cc1", "CC1=C(C(=O)O)C(C)=CC=C1", 
            "Cc1ccc(S(=O)(=O)NC(=O)NN2CC=C(C)CC2)cc1", 
            "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1",
            "CCC1(CC)C(=O)NC(=O)NC1=O"
        ]
        
        # Create random scores for demonstration
        scores = np.random.uniform(low=-12.5, high=-5.0, size=len(example_smiles))
        scores.sort()
        
        # Create molecules and add properties
        mols = []
        for i, (smiles, score) in enumerate(zip(example_smiles, scores)):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol.SetProp("_Name", f"Compound_{i+1}")
                mol.SetProp("Binding_Affinity", f"{score:.2f}")
                mol.SetProp("MW", f"{Descriptors.MolWt(mol):.2f}")
                mol.SetProp("LogP", f"{Descriptors.MolLogP(mol):.2f}")
                mol.SetProp("SMILES", smiles)
                mols.append(mol)
        
        return mols[:n_results]
    except Exception as e:
        raise RuntimeError(f"Error retrieving ligands: {str(e)}")


def perform_docking(
    protein_file: str, 
    ligand_smiles: str, 
    binding_site: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform a molecular docking simulation.
    
    In production, this would call actual docking software like AutoDock Vina.
    Currently returns mock data for demonstration purposes.
    
    Args:
        protein_file: Path to the protein PDB file
        ligand_smiles: SMILES string of the ligand
        binding_site: Optional binding site specification
        
    Returns:
        Dictionary with docking results
    """
    try:
        # Generate a 3D conformer for visualization
        mol = Chem.MolFromSmiles(ligand_smiles)
        if not mol:
            raise ValueError(f"Invalid SMILES string: {ligand_smiles}")
            
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Mock docking scores and binding poses
        docking_score = np.random.uniform(low=-12.0, high=-6.0)
        
        return {
            "ligand_mol": mol,
            "docking_score": docking_score,
            "binding_mode": "Hydrophobic interactions with residues PHE123, VAL45; H-bond with THR98"
        }
    except Exception as e:
        raise RuntimeError(f"Error performing docking: {str(e)}")


def download_ligands_as_sdf(ligands: List[Chem.Mol]) -> str:
    """
    Convert a list of molecules to SDF format for download.
    
    Args:
        ligands: List of RDKit molecule objects
        
    Returns:
        String in SDF format containing all molecules
    """
    try:
        sdf_file = io.StringIO()
        writer = Chem.SDWriter(sdf_file)
        for mol in ligands:
            writer.write(mol)
        writer.close()
        return sdf_file.getvalue()
    except Exception as e:
        raise RuntimeError(f"Error converting ligands to SDF: {str(e)}")


def calculate_drug_likeness(mol: Chem.Mol) -> Tuple[Dict[str, float], int]:
    """
    Calculate Lipinski's Rule of Five parameters for a molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Tuple of (properties_dict, violation_count)
    """
    try:
        properties = {
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "H_Donors": Descriptors.NumHDonors(mol),
            "H_Acceptors": Descriptors.NumHAcceptors(mol)
        }
        
        violations = 0
        if properties["MW"] > 500: violations += 1
        if properties["LogP"] > 5: violations += 1
        if properties["H_Donors"] > 5: violations += 1
        if properties["H_Acceptors"] > 10: violations += 1
        
        return properties, violations
    except Exception as e:
        raise ValueError(f"Error calculating drug-likeness: {str(e)}") 