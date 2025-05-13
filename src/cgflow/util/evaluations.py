
import numpy as np
import pandas as pd

import semlaflow.util.metrics as Metrics
import semlaflow.util.complex_metrics as ComplexMetrics


def calculate_metrics(generated_mols, reference_mols=None, pocket_raw=None, is_complex=False):
    """Calculate quality metrics for generated molecules"""
    all_metrics = {}

    # Initialize all metrics with empty lists to gather per-sample results
    metric_functions = {
        "validity": Metrics.Validity(),
        "fc_validity": Metrics.Validity(connected=True),
        "uniqueness": Metrics.Uniqueness(),
        "energy_validity": Metrics.EnergyValidity(),
        "average_energy": Metrics.AverageEnergy(),
        "average_energy_per_atom": Metrics.AverageEnergy(per_atom=True),
        "average_strain": Metrics.AverageStrainEnergy(),
        "average_strain_per_atom": Metrics.AverageStrainEnergy(per_atom=True),
        "average_opt_rmsd": Metrics.AverageOptRmsd()
    }

    if reference_mols:
        metric_functions.update({
            "molecular_accuracy": Metrics.MolecularAccuracy(),
            "pair_rmsd": Metrics.MolecularPairRMSD(),
            "pair_no_align_rmsd": Metrics.MolecularPairRMSD(align=False)
        })

    if pocket_raw and is_complex:
        metric_functions.update({
            "clash_score": ComplexMetrics.Clash(),
            "interactions": ComplexMetrics.Interactions()
        })

    # Collect individual metric values
    for key, metric in metric_functions.items():
        results = []
        for idx in range(len(generated_mols)):
            mol = generated_mols[idx:idx+1]
            ref = reference_mols[idx:idx+1] if reference_mols else None
            pocket = pocket_raw[idx:idx+1] if pocket_raw else None

            if key.startswith("interactions"):
                interaction_values = metric(mol, pocket or pocket_raw)
                for int_key, val in interaction_values.items():
                    all_metrics.setdefault(f"interactions_{int_key}", []).append(val)
            else:
                if "pair" in key or "accuracy" in key:
                    val = metric(mol, ref)
                elif "clash" in key or "interactions" in key:
                    val = metric(mol, pocket or pocket_raw)
                else:
                    val = metric(mol)
                all_metrics.setdefault(key, []).append(val)

    # Compute mean, median, std for each metric
    summary = {"Metric": [], "Mean": [], "Median": [], "Std": []}
    for key, values in all_metrics.items():
        values = np.array(values, dtype=np.float32)
        # remove nan values
        values = values[~np.isnan(values)]
        
        summary["Metric"].append(key)
        summary["Mean"].append(values.mean())
        summary["Median"].append(np.median(values))
        summary["Std"].append(values.std())

    return pd.DataFrame(summary), all_metrics