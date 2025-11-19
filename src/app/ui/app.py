"""
Protein-Ligand Discovery Platform

A Streamlit application for drug discovery through protein-ligand interactions.
For research purposes only - not for clinical use.
"""

import streamlit as st
import os
import logging
import sys

# Add the src directory to the path so we can use absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.app.components import protein_upload_tab, ligand_discovery_tab, docking_analysis_tab


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_app():
    """Configure the Streamlit application."""
    st.set_page_config(
        page_title="Drug Discovery Platform", 
        layout="wide", 
        initial_sidebar_state="expanded",
        menu_items={
            'About': "Drug Discovery Platform | For research purposes only"
        }
    )


def main():
    """Run the main application."""
    try:
        # Configure the application
        initialize_app()
        
        # Header
        st.title("Protein-Ligand Discovery Platform")
        st.subheader("Upload a protein structure and discover potential binding ligands")
        
        # Sidebar configuration
        st.sidebar.header("Configuration")
        
        # Create tabs for the main app areas
        tab1, tab2, tab3 = st.tabs(["Protein Upload", "Ligand Discovery", "Docking Analysis"])
        
        # Tab 1: Protein Upload
        with tab1:
            protein_upload_tab()
        
        # Tab 2: Ligand Discovery
        with tab2:
            ligand_discovery_tab()
        
        # Tab 3: Docking Analysis
        with tab3:
            docking_analysis_tab()
        
        # Footer
        st.divider()
        st.caption("Drug Discovery Platform | For research purposes only | Not for clinical use")
    
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()