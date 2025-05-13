import random
from pathlib import Path

from omegaconf import OmegaConf
from rdkit.Chem import BRICS
from rdkit.Chem.rdchem import Mol as RDMol

from .rule_cgflow import CGFlowRule

data_folder = Path(__file__).parent / "data"
data = OmegaConf.load(data_folder / "reaction_cgflow.yaml")
RULES = [CGFlowRule(info) for info in data]


def find_brics_bonds(mol: RDMol) -> list[tuple[tuple[int, int], tuple[str, str]]]:
    return list(BRICS.FindBRICSBonds(mol))


def find_rxn_bonds(mol: RDMol) -> list[tuple[tuple[int, int], tuple[str, str], int]]:
    res: list[tuple[tuple[int, int], tuple[str, str], int]] = []
    for idx, rule in enumerate(RULES):
        pattern_matches: list[tuple[int, ...]] = mol.GetSubstructMatches(rule.product_pattern)
        for matches in pattern_matches:
            i1, i2 = random.choice(rule.order)
            res.append(((matches[i1], matches[i2]), rule.label, idx))
    random.shuffle(res)
    storage: set[tuple[int, int]] = set()
    final_res: list[tuple[tuple[int, int], tuple[str, str], int]] = []
    for v in res:
        if v[0] in storage:
            continue
        storage.add(v[0])
        final_res.append(v)
    return final_res
