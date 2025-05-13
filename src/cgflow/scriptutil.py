"""Util file for Equinv scripts"""

import math
import resource
from pathlib import Path

import numpy as np
import torch
from openbabel import pybel
from rdkit import RDLogger
from torchmetrics import MetricCollection
from tqdm import tqdm

import cgflow.util.complex_metrics as ComplexMetrics
import cgflow.util.functional as smolF
import cgflow.util.metrics as Metrics
import cgflow.util.rdkit as smolRD
from cgflow.data.datasets import GeometricDataset, PocketComplexDataset
from cgflow.util.tokeniser import Vocabulary

# Declarations to be used in scripts
QM9_COORDS_STD_DEV = 1.723299503326416
GEOM_COORDS_STD_DEV = 2.407038688659668
# Use the same GEOM_COORDS_STD_DEV for ease of finetuning
PLINDER_COORDS_STD_DEV = GEOM_COORDS_STD_DEV  # 2.2693647416252976
PLINDER_LIG_CENTER_STD_DEV = 1.866057902527167
PLINDER_RADIUS = 10.0

QM9_BUCKET_LIMITS = [12, 16, 18, 20, 22, 24, 30]
GEOM_DRUGS_BUCKET_LIMITS = [
    24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 96, 192, 384
]
PLINDER_BUCKET_LIMITS = [128, 170, 256, 512, 1024, 2048]

PROJECT_PREFIX = "equinv"
BOND_MASK_INDEX = 5
COMPILER_CACHE_SIZE = 128


def disable_lib_stdout():
    pybel.ob.obErrorLog.StopLogging()
    RDLogger.DisableLog("rdApp.*")


# Need to ensure the limits are large enough when using OT since lots of preprocessing needs to be done on the batches
# OT seems to cause a problem when there are not enough allowed open FDs
def configure_fs(limit=4096):
    """
    Try to increase the limit on open file descriptors
    If not possible use a different strategy for sharing files in torch
    """

    n_file_resource = resource.RLIMIT_NOFILE
    soft_limit, hard_limit = resource.getrlimit(n_file_resource)

    print(f"Current limits (soft, hard): {(soft_limit, hard_limit)}")

    if limit > soft_limit:
        try:
            print(f"Attempting to increase open file limit to {limit}...")
            resource.setrlimit(n_file_resource, (limit, hard_limit))
            print("Limit changed successfully!")

        except:
            print(
                "Limit change unsuccessful. Using torch file_system file sharing strategy instead."
            )

            import torch.multiprocessing

            torch.multiprocessing.set_sharing_strategy("file_system")

    else:
        print("Open file limit already sufficiently large.")


# Applies transformations to a molecule:
# 1. Normalizes coordinate values by dividing by coord_std.
# 2. Applies a random 3D rotation to the molecule.
# 3. Removes the molecule's center of mass.
# 4. Optionally shifts the molecule's center of mass by a random Gaussian offset (if shift_com_std > 0).
# 5. Converts atomic numbers to categorical indices using a predefined vocabulary.
# 6. Encodes atomic charges into a non-negative index representation.
def mol_transform(molecule, vocab, coord_std, shift_com_std=0.0):
    rotation = tuple(np.random.rand(3) * np.pi * 2)
    molecule = molecule.scale(1.0 / coord_std).rotate(rotation).zero_com()

    if shift_com_std > 0.0:
        molecule = molecule.shift(np.random.normal(0, shift_com_std, 3))

    return molecule._copy_with(
        atomics=smolF.atomics_to_index(molecule.atomics, vocab),
        charges=smolF.charge_to_index(molecule.charges),
    )


# Applies transformations to a protein-ligand complex:
# 1. Normalizes coordinate values by dividing by coord_std.
# 2. If fix_pos is False, applies a random 3D rotation to the entire complex.
# 3. Removes the center of mass of the holo pocket from the complex.
# 4. Converts the ligand's atomic numbers to categorical indices.
# 5. Encodes ligand atomic charges as non-negative indices.
# 6. Converts the holo pocket into a geometric molecular representation.
# 7. Converts the holo's atomic numbers and charges to categorical indices.
# 8. Restores the holo pocket's original scale (for evaluating PoseCheck metrics).
def complex_transform(complex, vocab, coord_std, radius=np.inf, fix_pos=False):
    # Perform the transformation on the ligand
    ligand_centroid = smolRD.get_mol_centroid(complex.ligand.to_rdkit(vocab))
    holo_subset = complex.holo.select_residues_by_distance(
        ligand_centroid, radius)
    assert len(holo_subset) > 0, "No holo subset found"
    complex = complex.copy_with(holo=holo_subset)

    if not fix_pos:
        rotation = tuple(np.random.rand(3) * np.pi * 2)
        complex = complex.scale(1.0 /
                                coord_std).rotate(rotation).zero_holo_com()

    transformed_ligand = complex.ligand._copy_with(
        atomics=smolF.atomics_to_index(complex.ligand.atomics, vocab),
        charges=smolF.charge_to_index(complex.ligand.charges),
    )

    # Convert the holo pocket into a geometric molecule and transform atomic properties
    holo_mol = complex.holo.to_geometric_mol()
    transformed_holo_mol = holo_mol._copy_with(
        atomics=smolF.atomics_to_index(holo_mol.atomics, vocab),
        charges=smolF.charge_to_index(holo_mol.charges),
    )

    # Restore the holo pocket to its original scale
    unscaled_holo = complex.holo.copy().scale(coord_std)

    return complex.copy_with(ligand=transformed_ligand,
                             holo_mol=transformed_holo_mol,
                             holo=unscaled_holo)


# When training a distilled model atom types and bonds are already distributions over categoricals
def distill_transform(molecule, coord_std):
    rotation = tuple(np.random.rand(3) * np.pi * 2)
    molecule = molecule.scale(1.0 / coord_std).rotate(rotation).zero_com()

    charge_idxs = [
        smolRD.CHARGE_IDX_MAP[charge] for charge in molecule.charges.tolist()
    ]
    charge_idxs = torch.tensor(charge_idxs)

    transformed = molecule._copy_with(charges=charge_idxs)
    return transformed


def get_n_bond_types(cat_strategy):
    n_bond_types = len(smolRD.BOND_IDX_MAP.keys()) + 1
    n_bond_types = n_bond_types + 1 if cat_strategy == "mask" else n_bond_types
    return n_bond_types


def build_vocab():
    # Need to make sure PAD has index 0
    special_tokens = ["<PAD>", "<MASK>"]
    core_atoms = ["H", "C", "N", "O", "F", "P", "S", "Cl"]
    other_atoms = ["Br", "B", "Al", "Si", "As", "I", "Hg", "Bi"]
    tokens = special_tokens + core_atoms + other_atoms
    return Vocabulary(tokens)


# TODO support multi gpus
def calc_train_steps(dm, epochs, acc_batches):
    dm.setup("train")
    steps_per_epoch = math.ceil(len(dm.train_dataloader()) / acc_batches)
    return steps_per_epoch * epochs


def init_metrics(model=None, data_path=None, is_complex=False):
    if data_path is not None:
        print("Loading training data for novelty metric...")
        # Load the train data separately from the DM, just to access the list of train SMILES
        train_path = Path(data_path) / "train.smol"
        if is_complex:
            train_dataset = PocketComplexDataset.load(train_path)
        else:
            train_dataset = GeometricDataset.load(train_path)
        train_smiles = [mol.str_id for mol in train_dataset]

        print("Creating RDKit mols from training SMILES...")
        train_mols = model.builder.mols_from_smiles(train_smiles,
                                                    explicit_hs=True)
        train_mols = [mol for mol in train_mols if mol is not None]
        metrics = {"novelty": Metrics.Novelty(train_mols)}
    else:
        metrics = {}
        print("No training data provided. Skipping novelty metric.")

    metrics = {
        **metrics,
        "validity": Metrics.Validity(),
        "connected-validity": Metrics.Validity(connected=True),
        "uniqueness": Metrics.Uniqueness(),
        "energy-validity": Metrics.EnergyValidity(),
        "opt-energy-validity": Metrics.EnergyValidity(optimise=True),
        "energy": Metrics.AverageEnergy(),
        "energy-per-atom": Metrics.AverageEnergy(per_atom=True),
        "strain": Metrics.AverageStrainEnergy(),
        "strain-per-atom": Metrics.AverageStrainEnergy(per_atom=True),
        "opt-rmsd": Metrics.AverageOptRmsd(),
    }
    stability_metrics = {
        "atom-stability": Metrics.AtomStability(),
        "molecule-stability": Metrics.MoleculeStability(),
    }
    complex_metrics = {
        "clash": ComplexMetrics.Clash(),
        "interactions": ComplexMetrics.Interactions(),
    }
    conf_metrics = {
        "conformer-rmsd": Metrics.MolecularPairRMSD(),
        "conformer-no-align-rmsd": Metrics.MolecularPairRMSD(align=False),
        "conformer-centroid-rmsd": Metrics.CentroidRMSD(),
    }

    metrics = MetricCollection(metrics, compute_groups=False)
    stability_metrics = MetricCollection(stability_metrics,
                                         compute_groups=False)
    complex_metrics = MetricCollection(complex_metrics, compute_groups=False)
    conf_metrics = MetricCollection(conf_metrics, compute_groups=False)

    return metrics, stability_metrics, complex_metrics, conf_metrics


def generate_molecules(model, dm, steps, strategy, stabilities=False):
    test_dl = dm.test_dataloader()
    model.eval()
    cuda_model = model.to("cuda")

    outputs = []
    for batch in tqdm(test_dl):
        batch = {k: v.cuda() for k, v in batch[0].items()}
        output = cuda_model._generate(batch, steps, strategy)
        outputs.append(output)

    molecules = [cuda_model._generate_mols(output) for output in outputs]
    molecules = [mol for mol_list in molecules for mol in mol_list]

    if not stabilities:
        return molecules, outputs

    stabilities = [
        cuda_model._generate_stabilities(output) for output in outputs
    ]
    stabilities = [
        mol_stab for mol_stabs in stabilities for mol_stab in mol_stabs
    ]
    return molecules, outputs, stabilities


def calc_metrics_(
    rdkit_mols,
    metrics,
    stab_metrics=None,
    mol_stabs=None,
    complex_metrics=None,
    holo_pocks=None,
    conf_metrics=None,
    data_mols=None,
):
    metrics.reset()
    metrics.update(rdkit_mols)
    results = metrics.compute()

    if stab_metrics is not None:
        stab_metrics.reset()
        stab_metrics.update(mol_stabs)
        stab_results = stab_metrics.compute()
        results = {**results, **stab_results}

    if complex_metrics is not None:
        complex_metrics.reset()
        complex_metrics.update(rdkit_mols, holo_pocks)
        complex_results = complex_metrics.compute()
        results = {**results, **complex_results}

    if conf_metrics is not None:
        conf_metrics.reset()
        conf_metrics.update(rdkit_mols, data_mols)
        conf_results = conf_metrics.compute()
        results = {**results, **conf_results}

    return results


def print_results(results, std_results=None):
    print()
    print(f"{'Metric':<22}Result")
    print("-" * 30)

    for metric, value in results.items():
        result_str = f"{metric:<22}{value:.5f}"
        if std_results is not None:
            std = std_results[metric]
            result_str = f"{result_str} +- {std:.7f}"

        print(result_str)
    print()
