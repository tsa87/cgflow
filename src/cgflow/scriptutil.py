"""Util file for Equinv scripts"""

import math
import resource
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from rdkit import RDLogger
from torchmetrics import MetricCollection
from tqdm import tqdm

import cgflow.util.metrics as Metrics


def load_config(path: str | Path) -> DictConfig:
    """
    Load a configuration file from the given path.

    Args:
        path (str): The path to the configuration file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    path = Path(path)
    config: DictConfig = OmegaConf.load(path)
    container: dict = OmegaConf.to_container(config)

    # if there is _yaml_, load it and override the config.
    new_container: dict[str, Any] = {}
    for key in container.keys():
        sub_config = container[key]
        if isinstance(sub_config, dict) and "_yaml_" in sub_config:
            yaml_path = path.parent / sub_config.pop("_yaml_")
            base_config: DictConfig = load_config(yaml_path)
            OmegaConf.set_struct(base_config, True)  # avoid invalid override
            sub_config = OmegaConf.merge(base_config, sub_config)
        new_container[key] = sub_config
    return OmegaConf.create(new_container)


def disable_lib_stdout():
    RDLogger.DisableLog("rdApp.*")


# Need to ensure the limits are large enough when using OT since lots of preprocessing needs to be done on the batches
# OT seems to cause a problem when there are not enough allowed open FDs
def configure_fs(limit: int = 4096, verbose: bool = False):
    """
    Try to increase the limit on open file descriptors
    If not possible use a different strategy for sharing files in torch
    """

    n_file_resource = resource.RLIMIT_NOFILE
    soft_limit, hard_limit = resource.getrlimit(n_file_resource)

    if verbose:
        print(f"Current limits (soft, hard): {(soft_limit, hard_limit)}")

    if limit > soft_limit:
        try:
            if verbose:
                print(f"Attempting to increase open file limit to {limit}...")
            resource.setrlimit(n_file_resource, (limit, hard_limit))
            if verbose:
                print("Limit changed successfully!")

        except:
            if verbose:
                print("Limit change unsuccessful. Using torch file_system file sharing strategy instead.")

            import torch.multiprocessing

            torch.multiprocessing.set_sharing_strategy("file_system")

    else:
        if verbose:
            print("Open file limit already sufficiently large.")


# TODO support multi gpus
def calc_train_steps(dm, epochs, acc_batches):
    dm.setup("train")
    steps_per_epoch = math.ceil(len(dm.train_dataloader()) / acc_batches)
    return steps_per_epoch * epochs


def init_metrics():
    metrics = {
        "energy-validity": Metrics.EnergyValidity(),
        "opt-energy-validity": Metrics.EnergyValidity(optimise=True),
        "energy": Metrics.AverageEnergy(),
        "energy-per-atom": Metrics.AverageEnergy(per_atom=True),
        "strain": Metrics.AverageStrainEnergy(),
        "strain-per-atom": Metrics.AverageStrainEnergy(per_atom=True),
        "opt-rmsd": Metrics.AverageOptRmsd(),
    }
    complex_metrics = {
        "clash": Metrics.Clash(),
        "interactions": Metrics.Interactions(),
    }
    conf_metrics = {
        "conformer-rmsd": Metrics.MolecularPairRMSD(),
        "conformer-no-align-rmsd": Metrics.MolecularPairRMSD(align=False),
        "conformer-centroid-rmsd": Metrics.CentroidRMSD(),
    }
    metrics = MetricCollection(metrics, compute_groups=False)
    complex_metrics = MetricCollection(complex_metrics, compute_groups=False)
    conf_metrics = MetricCollection(conf_metrics, compute_groups=False)
    return metrics, complex_metrics, conf_metrics


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

    stabilities = [cuda_model._generate_stabilities(output) for output in outputs]
    stabilities = [mol_stab for mol_stabs in stabilities for mol_stab in mol_stabs]
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
