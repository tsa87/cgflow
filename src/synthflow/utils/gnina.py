import subprocess
import tempfile
from pathlib import Path

from openbabel import pybel

ob_log_handler = pybel.ob.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)

EMPTY_MOL = """
     RDKit          3D

  0  0  0  0  0  0  0  0  0  0999 V2000
M  END
>  <docking_score>
0.0

$$$$
"""


def run_gnina_local_opt(ligand_path: str, protein_pdbqt_path: str, out_pdbqt_path: str = None):
    """
    Run GNINA local optimization with a simple command.

    Args:
        ligand_path: Path to ligand file (SDF or PDBQT)
        protein_pdbqt_path: Path to protein file in PDBQT format
        out_pdbqt_path: Path to save the output PDBQT file (optional)

    Returns:
        GNINA stdout output
    """
    cmd = ["gnina", "-r", protein_pdbqt_path, "-l", ligand_path, "--local_only"]

    if out_pdbqt_path:
        cmd.extend(["--out", out_pdbqt_path])

    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return res.stdout


def local_opt(
    ligand_path: str | Path, protein_pdbqt_path: str | Path, save_path: str | Path, num_workers: int = 1
) -> list[float]:
    """
    Perform local optimization using gnina.

    Args:
        ligand_path: Path to ligand file in SDF format
        protein_pdbqt_path: Path to protein file in PDBQT format
        save_path: Path to save the output SDF file
        num_workers: Number of parallel workers (default: 1)

    Returns:
        List of docking scores
    """
    ligand_path = str(ligand_path)
    protein_pdbqt_path = str(protein_pdbqt_path)
    save_path = str(save_path)

    try:
        # Run gnina directly with the SDF file
        stdout = run_gnina_local_opt(ligand_path, protein_pdbqt_path)

        # Parse the scores and create output SDF
        scores = []
        with open(save_path, "w") as w:
            # Parse all Affinity lines from the stdout
            affinity_lines = [line for line in stdout.split("\n") if line.startswith("Affinity:")]

            # Read the original SDF molecules to preserve structure
            mols = list(pybel.readfile("sdf", ligand_path))

            # Process each molecule with its corresponding score
            for i, (pbmol, affinity_line) in enumerate(zip(mols, affinity_lines, strict=False)):
                try:
                    # Parse score
                    parts = affinity_line.split()
                    energy = float(parts[1])
                    scores.append(energy)

                    # Update molecule data with only the docking score
                    pbmol.data.clear()
                    pbmol.data.update(
                        {
                            "docking_score": energy,
                        }
                    )

                    # Write the molecule to SDF
                    w.write(pbmol.write("sdf"))
                except Exception:
                    w.write(EMPTY_MOL)

            # If we have fewer affinity lines than molecules, fill with empty mols
            for _ in range(len(mols) - len(affinity_lines)):
                scores.append(0.0)
                w.write(EMPTY_MOL)
    except Exception:
        with open(save_path, "w") as w:
            w.write(EMPTY_MOL)
        scores = [0.0]

    return scores


def run_gnina_docking(
    ligand_path: str,
    protein_pdbqt_path: str,
    out_path: str,
    center: tuple[float, float, float],
    size: float | tuple[float, float, float] = 20.0,
    exhaustiveness: int = 8,
    cnn_scoring: str = "rescore",
    cnn_model: str = None,
    num_modes: int = 9,
):
    """
    Run gnina docking with specified parameters.

    Args:
        ligand_path: Path to ligand file
        protein_pdbqt_path: Path to protein PDBQT file
        out_path: Path to output PDBQT file
        center: Tuple of x, y, z coordinates of the center of the search box
        size: Size of the search box (can be a single float for a cube or a tuple for x,y,z dimensions)
        exhaustiveness: Exhaustiveness of the search
        cnn_scoring: CNN scoring mode ('none', 'rescore', 'refinement', 'metrorescore', 'metrorefine', 'all')
        cnn_model: CNN model to use. If None, use default crossdock model.
        num_modes: Maximum number of binding modes to generate
    """
    # If size is a single value, expand it to a tuple
    if isinstance(size, (int, float)):
        size = (float(size), float(size), float(size))

    cmd = [
        "gnina",
        "-r",
        protein_pdbqt_path,
        "-l",
        ligand_path,
        "--out",
        out_path,
        "--center_x",
        str(center[0]),
        "--center_y",
        str(center[1]),
        "--center_z",
        str(center[2]),
        "--size_x",
        str(size[0]),
        "--size_y",
        str(size[1]),
        "--size_z",
        str(size[2]),
        "--exhaustiveness",
        str(exhaustiveness),
        "--cnn_scoring",
        cnn_scoring,
        "--num_modes",
        str(num_modes),
    ]

    # Add CNN model if specified
    if cnn_model:
        cmd.extend(["--cnn", cnn_model])

    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return res.stdout


def docking(
    ligand_path: str | Path,
    protein_pdbqt_path: str | Path,
    save_path: str | Path,
    center: tuple[float, float, float],
    size: float | tuple[float, float, float] = 20.0,
    exhaustiveness: int = 8,
    cnn_scoring: str = "rescore",
    cnn_model: str = None,
    num_modes: int = 9,
    num_workers: int = 1,
) -> list[float]:
    """
    Perform docking using gnina.

    Args:
        ligand_path: Path to ligand file in SDF format
        protein_pdbqt_path: Path to protein file in PDBQT format
        save_path: Path to save the output SDF file
        center: Tuple of x, y, z coordinates of the center of the search box
        size: Size of the search box (can be a single float for a cube or a tuple for x,y,z dimensions)
        exhaustiveness: Exhaustiveness of the search
        cnn_scoring: CNN scoring mode ('none', 'rescore', 'refinement', 'metrorescore', 'metrorefine', 'all')
        cnn_model: CNN model to use. If None, use default crossdock model.
        num_modes: Maximum number of binding modes to generate
        num_workers: Number of parallel workers (default: 1)

    Returns:
        List of best docking scores for each molecule in the input SDF
    """
    ligand_path = str(ligand_path)
    protein_pdbqt_path = str(protein_pdbqt_path)
    save_path = str(save_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dir = Path(tmp_dir)
        out_pdbqt_path = str(dir / "out.pdbqt")

        # Run GNINA docking
        stdout = run_gnina_docking(
            ligand_path,
            protein_pdbqt_path,
            out_pdbqt_path,
            center,
            size,
            exhaustiveness,
            cnn_scoring,
            cnn_model,
            num_modes,
        )

        # Parse output to extract scores for each molecule
        best_scores = []
        all_scores = []  # Store all scores for writing to SDF later

        try:
            # Split output by molecule results
            mol_sections = stdout.split("Using random seed:")

            # Process each molecule section (skip first empty section)
            for section in mol_sections[1:]:
                mol_scores = []
                if "mode |  affinity" in section:
                    result_table = section.split("mode |  affinity")[1].strip().split("\n")
                    # Skip the header separator line
                    if result_table and "-----+-----" in result_table[0]:
                        result_table = result_table[1:]

                    # Process each mode result
                    for line in result_table:
                        if line.strip():
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    score = float(parts[1])  # affinity score
                                    mol_scores.append(score)
                                except (ValueError, IndexError):
                                    mol_scores.append(0.0)

                # Add best score for this molecule
                if mol_scores:
                    best_scores.append(min(mol_scores))  # Best score is lowest affinity
                    all_scores.append(mol_scores)
                else:
                    best_scores.append(0.0)
                    all_scores.append([0.0])

            # If no molecules were processed, add a default score
            if not best_scores:
                best_scores = [0.0]
                all_scores = [[0.0]]

            # Write SDF output
            with open(save_path, "w") as w:
                if Path(out_pdbqt_path).exists():
                    try:
                        mols = list(pybel.readfile("pdbqt", out_pdbqt_path))
                        mol_idx = 0
                        pose_idx = 0

                        for pbmol in mols:
                            pbmol.data.clear()

                            # Get scores for current molecule
                            if mol_idx < len(all_scores):
                                mol_scores = all_scores[mol_idx]
                                if pose_idx < len(mol_scores):
                                    affinity = mol_scores[pose_idx]
                                else:
                                    affinity = 0.0

                                # Add properties to the molecule
                                pbmol.data.update(
                                    {"docking_score": affinity, "pose_num": pose_idx + 1, "mol_num": mol_idx + 1}
                                )

                                # Try to extract additional data from the results table
                                section = mol_sections[mol_idx + 1] if mol_idx + 1 < len(mol_sections) else ""
                                if "mode |  affinity" in section:
                                    result_lines = section.split("mode |  affinity")[1].strip().split("\n")
                                    if len(result_lines) > 1 and pose_idx < len(result_lines) - 1:
                                        line = result_lines[pose_idx + 1]
                                        parts = line.split()
                                        if len(parts) >= 6:
                                            try:
                                                pbmol.data.update(
                                                    {
                                                        "intramol_energy": float(parts[3]),
                                                        "cnn_score": float(parts[5]),
                                                        "cnn_affinity": float(parts[7]) if len(parts) >= 8 else 0.0,
                                                    }
                                                )
                                            except (ValueError, IndexError):
                                                pass

                                # Write the molecule to SDF
                                w.write(pbmol.write("sdf"))

                                pose_idx += 1
                                if pose_idx >= num_modes:
                                    pose_idx = 0
                                    mol_idx += 1

                    except Exception as e:
                        print(f"Error writing molecules to SDF: {e}")
                        w.write(EMPTY_MOL)
                else:
                    w.write(EMPTY_MOL)

        except Exception as e:
            print(f"Error parsing GNINA output: {e}")
            with open(save_path, "w") as w:
                w.write(EMPTY_MOL)
            best_scores = [0.0]

        return best_scores
