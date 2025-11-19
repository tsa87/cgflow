"""File handling utilities for the protein-ligand app."""

import os
import tempfile
import io
import zipfile
import base64
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import PandasTools

def save_uploadedfile(uploaded_file, suffix=None):
    """Save an uploaded file to disk and return the file path."""
    if suffix is None:
        suffix = os.path.splitext(uploaded_file.name)[1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def create_download_link(data, filename, text=None):
    """Create a download link for data."""
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text or filename}</a>'
    return href

def download_button(data, file_name, button_text, mime_type):
    """Create a Streamlit download button for data."""
    if isinstance(data, str):
        data = data.encode()
    
    st.download_button(
        label=button_text,
        data=data,
        file_name=file_name,
        mime=mime_type
    )

def save_smiles_to_csv(smiles_list, names=None, scores=None, properties=None):
    """Save a list of SMILES to a CSV file."""
    if names is None:
        names = [f"Compound_{i+1}" for i in range(len(smiles_list))]
    
    data = {"Name": names, "SMILES": smiles_list}
    
    if scores is not None:
        data["Score"] = scores
    
    if properties is not None:
        for key, values in properties.items():
            data[key] = values
    
    df = pd.DataFrame(data)
    csv_data = df.to_csv(index=False)
    return csv_data

def save_mols_to_sdf(mols):
    """Save a list of RDKit molecules to an SDF string."""
    sio = io.StringIO()
    writer = Chem.SDWriter(sio)
    
    for mol in mols:
        writer.write(mol)
    
    writer.close()
    return sio.getvalue()

def create_molecules_zip(mols, formats=None):
    """
    Create a zip file containing molecules in various formats.
    
    Args:
        mols: List of RDKit molecules
        formats: List of formats to include. Default: ['sdf', 'pdb', 'smiles', 'csv']
    
    Returns:
        Bytes containing the zip file
    """
    if formats is None:
        formats = ['sdf', 'pdb', 'smiles', 'csv']
    
    # Create a memory file for the zip
    memory_file = io.BytesIO()
    
    with zipfile.ZipFile(memory_file, 'w') as zf:
        # Add molecules in SDF format
        if 'sdf' in formats:
            sdf_content = save_mols_to_sdf(mols)
            zf.writestr('molecules.sdf', sdf_content)
        
        # Add molecules in SMILES format
        if 'smiles' in formats:
            smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
            names = [mol.GetProp('_Name') if mol.HasProp('_Name') else f"Compound_{i+1}" 
                    for i, mol in enumerate(mols)]
            
            smiles_content = save_smiles_to_csv(smiles_list, names)
            zf.writestr('molecules.csv', smiles_content)
        
        # Add individual PDB files for each molecule
        if 'pdb' in formats:
            for i, mol in enumerate(mols):
                name = mol.GetProp('_Name') if mol.HasProp('_Name') else f"compound_{i+1}"
                
                # Add hydrogens and generate 3D coordinates if needed
                if mol.GetNumConformers() == 0:
                    mol = Chem.AddHs(mol)
                    Chem.AllChem.EmbedMolecule(mol)
                    Chem.AllChem.MMFFOptimizeMolecule(mol)
                
                # Get PDB as string
                pdb_content = Chem.MolToPDBBlock(mol)
                
                # Add to zip
                zf.writestr(f'pdb/{name}.pdb', pdb_content)
    
    # Reset pointer to start of memory file
    memory_file.seek(0)
    return memory_file.getvalue()

def dataframe_to_excel(df):
    """Convert a DataFrame to Excel bytes."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
    
    output.seek(0)
    return output.getvalue()

def molecule_dataframe_to_csv(df, include_structures=False):
    """Convert a DataFrame with molecules to CSV."""
    if include_structures:
        # Create a copy to avoid modifying the original
        df_copy = df.copy()
        
        # If there's a Structure column with RDKit molecules, convert to SMILES
        if 'Structure' in df_copy.columns:
            if hasattr(df_copy.Structure.iloc[0], 'GetSubstructMatch'):
                df_copy['Structure'] = df_copy.Structure.apply(lambda x: Chem.MolToSmiles(x) if x is not None else '')
    else:
        df_copy = df
    
    csv_data = df_copy.to_csv(index=False)
    return csv_data

def validate_smiles(smiles):
    """Validate a SMILES string by trying to create an RDKit molecule."""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None 