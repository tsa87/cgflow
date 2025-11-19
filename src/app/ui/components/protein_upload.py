"""Protein upload component for the drug discovery platform."""

import streamlit as st
import tempfile
import requests
from typing import Optional, Tuple
from stmol import showmol

from src.app.utils.visualization_utils import render_protein


def protein_upload_tab() -> None:
    """Display the protein upload tab with file upload and visualization."""
    st.header("Upload Protein Structure")
    st.write("Upload a protein structure file (PDB format) to start the ligand discovery process.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a PDB file", type=["pdb"])
    
    use_example = st.checkbox("Use example protein (HIV-1 Protease)")
    
    protein_file = None
    
    if use_example:
        protein_file = _load_example_protein()
    elif uploaded_file is not None:
        protein_file = _save_uploaded_protein(uploaded_file)
    
    if protein_file:
        _display_protein_visualization(protein_file)
        _display_protein_info(use_example)
        
        # Store the protein file path in session state for use in other tabs
        st.session_state.protein_file = protein_file


def _load_example_protein() -> str:
    """
    Load an example protein (HIV-1 Protease).
    
    Returns:
        Path to the saved PDB file
    """
    try:
        # Download example PDB file (HIV-1 Protease)
        example_url = "https://files.rcsb.org/download/1HSG.pdb"
        response = requests.get(example_url)
        if response.status_code != 200:
            st.error(f"Failed to download example protein: HTTP {response.status_code}")
            return None
            
        protein_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
        protein_file.write(response.content)
        protein_file.close()
        st.success("Example protein loaded: HIV-1 Protease (PDB ID: 1HSG)")
        return protein_file.name
    except Exception as e:
        st.error(f"Error loading example protein: {str(e)}")
        return None


def _save_uploaded_protein(uploaded_file) -> str:
    """
    Save an uploaded protein file.
    
    Args:
        uploaded_file: The uploaded file from Streamlit
        
    Returns:
        Path to the saved PDB file
    """
    try:
        protein_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
        protein_file.write(uploaded_file.getvalue())
        protein_file.close()
        st.success(f"Uploaded: {uploaded_file.name}")
        return protein_file.name
    except Exception as e:
        st.error(f"Error saving uploaded protein: {str(e)}")
        return None


def _display_protein_visualization(protein_file: str) -> None:
    """
    Display the protein structure visualization.
    
    Args:
        protein_file: Path to the protein PDB file
    """
    st.header("Protein Structure Visualization")
    
    # Add rotation control
    rotation_enabled = st.checkbox("Enable protein rotation", value=True)
    
    # Display protein structure
    try:
        view = render_protein(protein_file, spin=rotation_enabled)
        showmol(view, height=500, width=700)
        
        # Add button to toggle rotation after display
        if st.button("Toggle Rotation"):
            if rotation_enabled:
                st.session_state.rotation_enabled = False
                view = render_protein(protein_file, spin=False)
            else:
                st.session_state.rotation_enabled = True
                view = render_protein(protein_file, spin=True)
    except Exception as e:
        st.error(f"Error displaying protein structure: {str(e)}")


def _display_protein_info(is_example: bool) -> None:
    """
    Display information about the protein.
    
    Args:
        is_example: Whether this is the example protein
    """
    st.header("Protein Information")
    
    # For a real app, extract protein info from the PDB file
    # Here we'll just show mock data for demonstration
    if is_example:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Protein Name:** HIV-1 Protease")
            st.write("**PDB ID:** 1HSG")
            st.write("**Chain Count:** 2")
        with col2:
            st.write("**Resolution:** 2.0 Å")
            st.write("**Source Organism:** Human Immunodeficiency Virus 1")
            st.write("**Publication:** Nature (1990)")
    else:
        st.info("Protein information would be extracted from the PDB file") 