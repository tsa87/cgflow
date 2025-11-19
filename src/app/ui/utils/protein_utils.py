"""Protein-related utility functions for the protein-ligand app."""

import os
import tempfile
import py3Dmol
from stmol import showmol
import streamlit as st
import requests

def load_protein_from_file(file_path):
    """Load a protein structure from a PDB file."""
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"
    
    try:
        with open(file_path, 'r') as f:
            pdb_data = f.read()
        return pdb_data, None
    except Exception as e:
        return None, str(e)

def load_protein_from_pdb_id(pdb_id):
    """Download and load a protein structure from the PDB database."""
    try:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url)
        
        if response.status_code != 200:
            return None, f"Failed to download PDB {pdb_id}: {response.status_code}"
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
        temp_file.write(response.content)
        temp_file.close()
        
        return temp_file.name, None
    except Exception as e:
        return None, str(e)

def save_uploaded_protein(uploaded_file):
    """Save an uploaded protein file to disk."""
    try:
        # Save uploaded file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()
        return temp_file.name, None
    except Exception as e:
        return None, str(e)

def render_protein_structure(pdb_data=None, pdb_file=None, width=700, height=500, style="cartoon", color_scheme="spectrum", spin=True):
    """
    Render a protein structure with py3Dmol.
    
    Args:
        pdb_data: String containing PDB data
        pdb_file: Path to PDB file (alternative to pdb_data)
        width: Viewer width
        height: Viewer height
        style: Visualization style ('cartoon', 'line', 'stick', 'sphere', etc.)
        color_scheme: Color scheme ('spectrum', 'chain', 'residue', etc.)
        spin: Whether to enable spinning
        
    Returns:
        py3Dmol view object
    """
    if pdb_data is None and pdb_file is not None:
        with open(pdb_file, 'r') as f:
            pdb_data = f.read()
    
    if pdb_data is None:
        return None
    
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_data, 'pdb')
    
    # Set style based on parameters
    if style == "cartoon":
        view.setStyle({'cartoon': {'color': color_scheme}})
    elif style == "line":
        view.setStyle({'line': {'color': color_scheme}})
    elif style == "stick":
        view.setStyle({'stick': {'color': color_scheme}})
    elif style == "sphere":
        view.setStyle({'sphere': {'color': color_scheme}})
    else:
        # Default to cartoon
        view.setStyle({'cartoon': {'color': color_scheme}})
    
    view.zoomTo()
    if spin:
        view.spin(True)
    
    return view

def display_protein_structure(pdb_data=None, pdb_file=None, width=700, height=500, **kwargs):
    """Display a protein structure in Streamlit using stmol."""
    view = render_protein_structure(pdb_data, pdb_file, width, height, **kwargs)
    if view:
        return showmol(view, height=height, width=width)
    return None

def render_protein_ligand_complex(protein_file, ligand_mol, width=700, height=500):
    """
    Render a protein-ligand complex with py3Dmol.
    
    Args:
        protein_file: Path to protein PDB file
        ligand_mol: RDKit molecule with 3D coordinates
        width: Viewer width
        height: Viewer height
        
    Returns:
        py3Dmol view object
    """
    from rdkit import Chem
    
    # Load protein
    with open(protein_file, 'r') as f:
        protein_data = f.read()
    
    # Convert ligand to PDB
    ligand_pdb = Chem.MolToPDBBlock(ligand_mol)
    
    # Create viewer
    view = py3Dmol.view(width=width, height=height)
    
    # Add protein
    view.addModel(protein_data, 'pdb')
    view.setStyle({'cartoon': {'color': 'gray'}})
    
    # Add ligand
    view.addModel(ligand_pdb, 'pdb')
    view.setStyle({'model': -1}, {'stick': {'color': 'spectrum', 'radius': 0.2}})
    
    view.zoomTo()
    view.spin(True)
    
    return view

def extract_protein_info(pdb_file):
    """Extract basic information from a PDB file."""
    if not os.path.exists(pdb_file):
        return {}
    
    info = {
        "Chains": set(),
        "Residues": 0,
        "Atoms": 0,
        "Hetero Atoms": 0
    }
    
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    chain = line[21:22].strip()
                    if chain:
                        info["Chains"].add(chain)
                    info["Atoms"] += 1
                elif line.startswith("HETATM"):
                    info["Hetero Atoms"] += 1
    except Exception:
        return {}
    
    # Convert chain set to string
    info["Chains"] = ", ".join(sorted(info["Chains"]))
    info["Total Atoms"] = info["Atoms"] + info["Hetero Atoms"]
    
    return info 