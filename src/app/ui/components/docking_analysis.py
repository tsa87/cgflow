"""Docking analysis component for the drug discovery platform."""

import streamlit as st
import time
from typing import Dict, Any, Optional
from stmol import showmol
import py3Dmol

from rdkit import Chem
from src.app.utils.visualization_utils import display_mol
from src.app.utils.molecule_utils import perform_docking, calculate_drug_likeness


def docking_analysis_tab() -> None:
    """Display the docking analysis tab with docking functionality."""
    st.header("Molecular Docking Analysis")
    
    if 'protein_file' not in st.session_state:
        st.warning("Please upload a protein structure in the 'Protein Upload' tab first.")
        return
    elif 'selected_ligand' not in st.session_state:
        _handle_no_ligand_selected()
        return
    
    # Configure docking parameters in sidebar
    _setup_sidebar_options()
    
    # Get the selected ligand from session state
    ligand = st.session_state.selected_ligand
    
    # Display ligand information and docking controls
    _display_ligand_info(ligand)
    
    # Display docking results if available
    if 'docking_results' in st.session_state:
        _display_docking_results(st.session_state.docking_results)


def _handle_no_ligand_selected() -> None:
    """Handle the case where no ligand is selected."""
    st.info("Please select a ligand in the 'Ligand Discovery' tab for docking analysis.")
    
    # Option to enter custom SMILES
    st.header("Or Enter Custom Ligand")
    custom_smiles = st.text_input("Enter SMILES for a custom ligand:")
    
    if custom_smiles and st.button("Dock Custom Ligand"):
        try:
            mol = Chem.MolFromSmiles(custom_smiles)
            if mol:
                st.session_state.selected_ligand = {
                    "name": "Custom Ligand",
                    "smiles": custom_smiles,
                    "mol": mol
                }
            else:
                st.error("Invalid SMILES string. Please check and try again.")
        except Exception as e:
            st.error(f"Error processing SMILES: {str(e)}")


def _setup_sidebar_options() -> None:
    """Setup sidebar options for docking analysis."""
    st.sidebar.subheader("Docking Parameters")
    
    # Configure docking parameters
    st.session_state.exhaustiveness = st.sidebar.slider(
        "Search Exhaustiveness", 
        1, 10, 8, 
        key="exhaustiveness_slider",
        help="Higher values provide more thorough search but take longer"
    )
    
    st.session_state.energy_range = st.sidebar.slider(
        "Energy Range (kcal/mol)", 
        1, 10, 3, 
        key="energy_range_slider",
        help="Maximum energy difference between best and worst binding mode"
    )


def _display_ligand_info(ligand: Dict[str, Any]) -> None:
    """
    Display information about the selected ligand and docking controls.
    
    Args:
        ligand: Dictionary containing ligand information
    """
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"**Selected Ligand**: {ligand['name']}")
        try:
            svg = display_mol(ligand['mol'], width=300, height=200)
            st.markdown(svg, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying molecule: {str(e)}")
    
    with col2:
        st.write(f"SMILES: `{ligand['smiles']}`")
        
        # Start docking button
        if st.button("Perform Docking Analysis"):
            _run_docking_simulation(
                st.session_state.protein_file,
                ligand['smiles']
            )


def _run_docking_simulation(protein_file: str, ligand_smiles: str) -> None:
    """
    Run a docking simulation and store the results.
    
    Args:
        protein_file: Path to the protein PDB file
        ligand_smiles: SMILES string of the ligand
    """
    try:
        with st.spinner("Running molecular docking simulation..."):
            # In a real app, this would call an actual docking software
            # For demo purposes, we'll add a delay and return mock results
            time.sleep(3)
            
            docking_results = perform_docking(
                protein_file,
                ligand_smiles
            )
            
            # Store results
            st.session_state.docking_results = docking_results
    except Exception as e:
        st.error(f"Error in docking simulation: {str(e)}")


def _display_docking_results(results: Dict[str, Any]) -> None:
    """
    Display docking simulation results.
    
    Args:
        results: Dictionary containing docking results
    """
    st.divider()
    st.header("Docking Results")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write(f"**Docking Score**: {results['docking_score']:.2f} kcal/mol")
        st.write(f"**Binding Mode**: {results['binding_mode']}")
        
        # Lipinski's Rule of Five evaluation
        try:
            mol = results['ligand_mol']
            properties, violations = calculate_drug_likeness(mol)
            
            st.header("Drug-likeness Analysis")
            st.write("Lipinski's Rule of Five:")
            
            st.write(f"- Molecular Weight: {properties['MW']:.2f} (≤500: {'✓' if properties['MW'] <= 500 else '✗'})")
            st.write(f"- LogP: {properties['LogP']:.2f} (≤5: {'✓' if properties['LogP'] <= 5 else '✗'})")
            st.write(f"- H-Bond Donors: {properties['H_Donors']} (≤5: {'✓' if properties['H_Donors'] <= 5 else '✗'})")
            st.write(f"- H-Bond Acceptors: {properties['H_Acceptors']} (≤10: {'✓' if properties['H_Acceptors'] <= 10 else '✗'})")
            st.write(f"- Violations: {violations} (Good: {'✓' if violations <= 1 else '✗'})")
        except Exception as e:
            st.error(f"Error in drug-likeness calculation: {str(e)}")
        
    with col2:
        # In a real app, this would display the actual docked pose
        # For demo purposes, we'll just show a 3D visualization of the ligand
        try:
            mol = results['ligand_mol']
            viewer = py3Dmol.view(width=400, height=400)
            
            # Convert molecule to PDB format for py3Dmol
            mb = Chem.MolToMolBlock(mol)
            viewer.addModel(mb, "mol")
            viewer.setStyle({'stick': {}})
            viewer.zoomTo()
            viewer.spin(True)
            showmol(viewer, height=400, width=400)
        except Exception as e:
            st.error(f"Error displaying 3D molecule: {str(e)}")
    
    # Download options
    st.header("Download Results")
    
    try:
        # In a real app, we would generate proper files
        st.download_button(
            label="Download Docking Results Report (PDF)",
            data=b"Placeholder for PDF report",  # This would be a real PDF in a production app
            file_name=f"docking_results_{results['ligand_mol'].GetProp('_Name') if results['ligand_mol'].HasProp('_Name') else 'ligand'}.pdf",
            mime="application/pdf"
        )
        
        st.download_button(
            label="Download Docked Pose (PDB)",
            data=Chem.MolToPDBBlock(results['ligand_mol']),
            file_name=f"docked_pose_{results['ligand_mol'].GetProp('_Name') if results['ligand_mol'].HasProp('_Name') else 'ligand'}.pdb",
            mime="chemical/x-pdb"
        )
    except Exception as e:
        st.error(f"Error creating downloads: {str(e)}") 