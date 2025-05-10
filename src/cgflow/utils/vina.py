from functools import partial
from pathlib import Path
import tempfile
import subprocess
import multiprocessing
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


def run_vina_local_opt(args: tuple[str, str], protein_pdbqt_path: str):
    in_pdbqt_path, out_pdbqt_path = args
    res = subprocess.run(
        [
            "vina1.2.5",
            "--receptor",
            protein_pdbqt_path,
            "--ligand",
            in_pdbqt_path,
            "--out",
            out_pdbqt_path,
            "--autobox",
            "--local_only",
            "--seed",
            "1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return res.stdout


def run_vina_docking(
    args: tuple[str, str],
    protein_pdbqt_path: str,
    center: tuple[float, float, float],
    size: float,
    exhaustiveness: int,
):
    in_pdbqt_path, out_pdbqt_path = args
    res = subprocess.run(
        [
            "vina1.2.5",
            "--receptor",
            protein_pdbqt_path,
            "--ligand",
            in_pdbqt_path,
            "--out",
            out_pdbqt_path,
            "--center_x",
            str(center[0]),
            "--center_y",
            str(center[1]),
            "--center_z",
            str(center[2]),
            "--size_x",
            str(size),
            "--size_y",
            str(size),
            "--size_z",
            str(size),
            "--exhaustiveness",
            str(exhaustiveness),
            "--seed",
            "1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return res.stdout


def local_opt(
    ligand_path: str | Path, protein_pdbqt_path: str | Path, save_path: str | Path, num_workers: int = 8
) -> list[float]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        dir = Path(tmp_dir)
        args = []
        for i, pbmol in enumerate(pybel.readfile("sdf", str(ligand_path))):
            in_pdbqt_path = str(dir / f"{i}_in.pdbqt")
            out_pdbqt_path = str(dir / f"{i}_out.pdbqt")
            pbmol.write("pdbqt", in_pdbqt_path)
            args.append((in_pdbqt_path, out_pdbqt_path))

        func = partial(run_vina_local_opt, protein_pdbqt_path=str(protein_pdbqt_path))
        with multiprocessing.Pool(num_workers) as pool:
            result = pool.map(func, args)

        scores = []
        with open(save_path, "w") as w:
            for i, stdout in enumerate(result):
                try:
                    energy = float(stdout.split("Estimated Free Energy of Binding   :")[1].split()[0])
                    internel_energy = float(stdout.split("Final Total Internal Energy    :")[1].split()[0])
                    torsional_energy = float(stdout.split("Torsional Free Energy          :")[1].split()[0])
                    unbounded_energy = float(stdout.split("Unbound System's Energy        :")[1].split()[0])
                    pbmol = next(pybel.readfile("pdbqt", str(dir / f"{i}_out.pdbqt")))
                    pbmol.data.clear()
                    pbmol.data.update(
                        {
                            "docking_score": energy,
                            "internel_energy": internel_energy,
                            "torsional_energy": torsional_energy,
                            "unbounded_energy": unbounded_energy,
                        }
                    )
                    scores.append(energy)
                    w.write(pbmol.write("sdf"))
                except Exception:
                    scores.append(0.0)
                    w.write(EMPTY_MOL)
        return scores


def docking(
    ligand_path: str | Path,
    protein_pdbqt_path: str | Path,
    save_path: str | Path,
    center: tuple[float, float, float],
    size: float = 20.0,
    exhaustiveness: int = 8,
    num_workers: int = 8,
) -> list[float]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        dir = Path(tmp_dir)
        args = []
        for i, pbmol in enumerate(pybel.readfile("sdf", str(ligand_path))):
            in_pdbqt_path = str(dir / f"{i}_in.pdbqt")
            out_pdbqt_path = str(dir / f"{i}_out.pdbqt")
            pbmol.write("pdbqt", in_pdbqt_path)
            args.append((in_pdbqt_path, out_pdbqt_path))

        func = partial(
            run_vina_docking,
            protein_pdbqt_path=str(protein_pdbqt_path),
            center=center,
            size=size,
            exhaustiveness=exhaustiveness,
        )
        with multiprocessing.Pool(num_workers) as pool:
            result = pool.map(func, args)

        scores = []
        with open(save_path, "w") as w:
            for i, stdout in enumerate(result):
                try:
                    energy = float(stdout.split("Estimated Free Energy of Binding   :")[1].split()[0])
                    internel_energy = float(stdout.split("Final Total Internal Energy    :")[1].split()[0])
                    torsional_energy = float(stdout.split("Torsional Free Energy          :")[1].split()[0])
                    unbounded_energy = float(stdout.split("Unbound System's Energy        :")[1].split()[0])
                    pbmol = next(pybel.readfile("pdbqt", str(dir / f"{i}_out.pdbqt")))
                    pbmol.data.clear()
                    pbmol.data.update(
                        {
                            "docking_score": energy,
                            "internel_energy": internel_energy,
                            "torsional_energy": torsional_energy,
                            "unbounded_energy": unbounded_energy,
                        }
                    )
                    scores.append(energy)
                    w.write(pbmol.write("sdf"))
                except Exception:
                    scores.append(0.0)
                    w.write(EMPTY_MOL)
        return scores
