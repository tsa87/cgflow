"""Ligand discovery component for the drug discovery platform."""

import streamlit as st
import pandas as pd
import time
from typing import List, Dict, Any

from rdkit import Chem
from src.app.utils.visualization_utils import display_mol
from src.app.utils.molecule_utils import get_ligands_for_protein, download_ligands_as_sdf


def ligand_discovery_tab() -> None:
    """Display the ligand discovery tab with search functionality."""
    st.header("Discover Potential Ligands")
    
    if 'protein_file' not in st.session_state:
        st.warning("Please upload a protein structure in the 'Protein Upload' tab first.")
        return
    
    # Configuration options
    _setup_sidebar_options()
    
    # Get parameters from session state or sidebar
    search_method = st.session_state.get('search_method', 'Structure-based Virtual Screening')
    max_results = st.session_state.get('max_results', 10)
    
    # Allow binding site selection
    binding_site = st.text_input("Binding Site (optional - specify residues or coordinates)")
    
    if st.button("Find Potential Ligands"):
        _perform_ligand_search(
            st.session_state.protein_file,
            binding_site=binding_site,
            max_results=max_results
        )


def _setup_sidebar_options() -> None:
    """Setup sidebar options for ligand discovery."""
    st.sidebar.subheader("Ligand Search Parameters")
    
    # Configure search method
    st.session_state.search_method = st.sidebar.selectbox(
        "Search Method",
        ["Structure-based Virtual Screening", "Similarity-based Search", "Fragment-based Design"],
        key="search_method_select"
    )
    
    # Configure result limit
    st.session_state.max_results = st.sidebar.slider(
        "Maximum Results", 
        5, 20, 10, 
        key="max_results_slider"
    )


def _perform_ligand_search(protein_file: str, binding_site: str = None, max_results: int = 10) -> None:
    """
    Perform ligand search and display results.
    
    Args:
        protein_file: Path to the protein PDB file
        binding_site: Optional binding site specification
        max_results: Maximum number of results to return
    """
    try:
        with st.spinner("Searching for ligands..."):
            # In a real app, this would take time as it performs actual calculations
            # For demo purposes, we'll add a small delay
            time.sleep(2)
            
            # Get ligands
            ligands = get_ligands_for_protein(
                protein_file, 
                binding_site=binding_site,
                n_results=max_results
            )
            
            # Store ligands in session state
            st.session_state.ligands = ligands
            
            # Display results
            _display_ligand_results(ligands)
    except Exception as e:
        st.error(f"Error searching for ligands: {str(e)}")


def _display_ligand_results(ligands: List[Chem.Mol]) -> None:
    """
    Display ligand search results.
    
    Args:
        ligands: List of RDKit molecule objects
    """
    st.header(f"Found {len(ligands)} Potential Ligands")
    
    # Create a DataFrame for easier display
    ligand_data = []
    for mol in ligands:
        try:
            ligand_data.append({
                "Name": mol.GetProp("_Name"),
                "Structure": mol,
                "SMILES": mol.GetProp("SMILES"),
                "Binding Affinity (kcal/mol)": float(mol.GetProp("Binding_Affinity")),
                "MW": float(mol.GetProp("MW")),
                "LogP": float(mol.GetProp("LogP"))
            })
        except Exception as e:
            st.warning(f"Error processing ligand: {str(e)}")
    
    df = pd.DataFrame(ligand_data)
    
    # Display structures with data
    for i, row in df.iterrows():
        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            try:
                svg = display_mol(row["Structure"])
                st.markdown(f"<div>{svg}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Error displaying structure: {str(e)}")
        with col2:
            st.write(f"**{row['Name']}**")
            st.write(f"SMILES: `{row['SMILES']}`")
            st.write(f"Binding Affinity: {row['Binding Affinity (kcal/mol)']:.2f} kcal/mol")
            st.write(f"MW: {row['MW']:.2f} | LogP: {row['LogP']:.2f}")
            
            # Button to select this ligand for docking
            if st.button(f"Dock {row['Name']}", key=f"dock_{i}"):
                st.session_state.selected_ligand = {
                    "name": row['Name'],
                    "smiles": row['SMILES'],
                    "mol": row['Structure']
                }
                st.info(f"Selected {row['Name']} for docking. Go to the 'Docking Analysis' tab.")
    
    # Add download button for all ligands
    if ligands:
        st.divider()
        try:
            sdf_data = download_ligands_as_sdf(ligands)
            st.download_button(
                label="Download All Ligands (SDF)",
                data=sdf_data,
                file_name="potential_ligands.sdf",
                mime="chemical/x-mdl-sdfile"
            )
        except Exception as e:
            st.error(f"Error creating download: {str(e)}") 