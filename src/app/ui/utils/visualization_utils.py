"""Visualization utilities for rendering proteins and molecules."""

from typing import Dict, Any, Optional
import py3Dmol
from stmol import showmol
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import tempfile
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def render_protein(
    pdb_file: str, 
    width: int = 700, 
    height: int = 500, 
    spin: bool = True
) -> py3Dmol.view:
    """
    Render a protein structure from a PDB file using py3Dmol.
    
    Args:
        pdb_file: Path to the PDB file
        width: Viewer width in pixels
        height: Viewer height in pixels
        spin: Whether to enable rotation animation
        
    Returns:
        A py3Dmol view object of the rendered protein
    """
    try:
        with open(pdb_file, 'r') as f:
            pdb_data = f.read()
        
        view = py3Dmol.view(width=width, height=height)
        view.addModel(pdb_data, 'pdb')
        view.setStyle({'cartoon': {'color': 'spectrum'}})
        view.zoomTo()
        view.spin(spin)
        return view
    except Exception as e:
        raise ValueError(f"Error rendering protein: {str(e)}")

def display_mol(
    mol: Chem.Mol, 
    width: int = 300, 
    height: int = 200
) -> str:
    """
    Generate an SVG representation of a molecule.
    
    Args:
        mol: RDKit molecule object
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        SVG string representation of the molecule
    """
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg
    except Exception as e:
        raise ValueError(f"Error displaying molecule: {str(e)}")

def mol_to_3dmol(mol, width=400, height=300, style="stick", surface=False, spin=False):
    """
    Convert an RDKit molecule to a py3Dmol view.
    
    Args:
        mol: RDKit molecule
        width: Viewer width
        height: Viewer height
        style: Visualization style ('stick', 'line', 'sphere', 'cartoon')
        surface: Whether to show molecular surface
        spin: Whether to enable spinning
        
    Returns:
        py3Dmol view object
    """
    if mol is None:
        return None
    
    # Ensure 3D coordinates
    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
    
    # Convert to PDB block
    pdb_block = Chem.MolToPDBBlock(mol)
    
    # Create viewer
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_block, 'pdb')
    
    # Set style
    if style == "stick":
        view.setStyle({'stick': {'radius': 0.2, 'color': 'spectrum'}})
    elif style == "line":
        view.setStyle({'line': {'color': 'spectrum'}})
    elif style == "sphere":
        view.setStyle({'sphere': {'radius': 0.5, 'color': 'spectrum'}})
    elif style == "cartoon":
        view.setStyle({'cartoon': {'color': 'spectrum'}})
    else:
        # Default to stick
        view.setStyle({'stick': {'radius': 0.2, 'color': 'spectrum'}})
    
    # Add surface if requested
    if surface:
        view.addSurface(py3Dmol.VDW, {'opacity': 0.5, 'color': 'white'})
    
    # Center and zoom
    view.zoomTo()
    
    # Enable spinning if requested
    if spin:
        view.spin(True)
    
    return view

def display_mol_3d(mol, width=400, height=300, **kwargs):
    """Display a molecule in 3D in Streamlit."""
    view = mol_to_3dmol(mol, width, height, **kwargs)
    if view:
        return showmol(view, height=height, width=width)
    return None

def ligand_interaction_map(interactions, width=600, height=400):
    """
    Generate a 2D interaction map for protein-ligand interactions.
    
    Args:
        interactions: Dictionary of interaction types and their details
        width: Plot width
        height: Plot height
        
    Returns:
        Matplotlib figure
    """
    # Extract residues involved in interactions
    residues = set()
    interaction_types = {}
    
    # Process hydrogen bonds
    for hbond in interactions.get("hydrogen_bonds", []):
        acceptor = hbond["acceptor"]
        if acceptor.startswith("LIG"):
            continue
        res_name = acceptor.split(":")[0]
        residues.add(res_name)
        interaction_types[res_name] = interaction_types.get(res_name, []) + ["H-bond"]
    
    # Process hydrophobic interactions
    for hydro in interactions.get("hydrophobic", []):
        prot_atom = hydro["protein_atom"]
        res_name = prot_atom.split(":")[0]
        residues.add(res_name)
        interaction_types[res_name] = interaction_types.get(res_name, []) + ["Hydrophobic"]
    
    # Process pi-stacking
    for pi in interactions.get("pi_stacking", []):
        prot_ring = pi["protein_ring"]
        res_name = prot_ring.split(":")[0]
        residues.add(res_name)
        interaction_types[res_name] = interaction_types.get(res_name, []) + ["π-stacking"]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    # Prepare data
    residues = sorted(list(residues))
    data = []
    
    for res in residues:
        types = interaction_types.get(res, [])
        unique_types = set(types)
        
        for itype in unique_types:
            count = types.count(itype)
            data.append((res, itype, count))
    
    if not data:
        # No data, return empty plot
        return fig
    
    # Convert to dataframe
    import pandas as pd
    df = pd.DataFrame(data, columns=["Residue", "Interaction", "Count"])
    
    # Pivot for heatmap
    pivot = df.pivot(index="Residue", columns="Interaction", values="Count").fillna(0)
    
    # Create heatmap
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=.5, ax=ax)
    
    plt.title("Protein-Ligand Interactions")
    plt.tight_layout()
    
    return fig

def plot_energy_components(energy_components, width=600, height=400):
    """
    Plot energy components from binding energy calculation.
    
    Args:
        energy_components: Dictionary of energy components
        width: Plot width
        height: Plot height
        
    Returns:
        Matplotlib figure
    """
    # Create a copy without the total component for the bar chart
    components = {k: v for k, v in energy_components.items() if k != "total"}
    
    # Create plot
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    # Sort by absolute value
    sorted_components = sorted(components.items(), key=lambda x: abs(x[1]), reverse=True)
    labels = [item[0].replace("_", " ").title() for item in sorted_components]
    values = [item[1] for item in sorted_components]
    
    # Color mapping - negative (favorable) in blue, positive (unfavorable) in red
    colors = ['#1f77b4' if v < 0 else '#d62728' for v in values]
    
    # Create bar chart
    bars = ax.bar(labels, values, color=colors)
    
    # Add total energy as text
    total = energy_components.get("total", sum(values))
    ax.text(0.5, 0.9, f"Total Energy: {total:.2f} kcal/mol", 
            horizontalalignment='center', transform=ax.transAxes,
            fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Format plot
    ax.set_ylabel("Energy (kcal/mol)")
    ax.set_title("Binding Energy Components")
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height < 0 else -0.1),
                f"{height:.1f}", ha='center', va='bottom' if height < 0 else 'top',
                color='black', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_interaction_distance_plot(interactions, width=600, height=400):
    """
    Create a plot showing interaction distances.
    
    Args:
        interactions: Dictionary of interaction types and their details
        width: Plot width
        height: Plot height
        
    Returns:
        Matplotlib figure
    """
    # Collect all distances by interaction type
    distances = {}
    
    # Process hydrogen bonds
    if "hydrogen_bonds" in interactions:
        distances["H-bond"] = [h["distance"] for h in interactions["hydrogen_bonds"]]
    
    # Process hydrophobic interactions
    if "hydrophobic" in interactions:
        distances["Hydrophobic"] = [h["distance"] for h in interactions["hydrophobic"]]
    
    # Process pi-stacking
    if "pi_stacking" in interactions:
        distances["π-stacking"] = [p["distance"] for p in interactions["pi_stacking"]]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    # Create boxplot if we have data
    if distances:
        data = []
        labels = []
        
        for itype, dist_list in distances.items():
            if dist_list:
                data.append(dist_list)
                labels.append(itype)
        
        if data:
            ax.boxplot(data, labels=labels)
            ax.set_ylabel("Distance (Å)")
            ax.set_title("Interaction Distances")
    else:
        ax.text(0.5, 0.5, "No interaction data available", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
    
    plt.tight_layout()
    return fig 