import random
from collections.abc import Sequence
from pathlib import Path

from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Mol as RDMol

from .rule_cgflow import CGFlowRule

data_folder = Path(__file__).parent / "data"
data = OmegaConf.load(data_folder / "reaction_cgflow.yaml")
RULES = [CGFlowRule(info) for info in data]


def break_bonds(mol: RDMol, bonds: Sequence[tuple[int, int]], sanitize=True) -> tuple[RDMol, ...]:
    """breaks the bonds in a molecule and returns the results"""
    eMol = Chem.EditableMol(mol)
    nAts = mol.GetNumAtoms()

    dummyPositions = []
    for ia, ib in bonds:
        obond = mol.GetBondBetweenAtoms(ia, ib)
        bondType = obond.GetBondType()
        eMol.RemoveBond(ia, ib)

        atoma = Chem.Atom(0)
        atoma.SetNoImplicit(True)
        idxa = nAts
        nAts += 1
        eMol.AddAtom(atoma)
        eMol.AddBond(ia, idxa, bondType)

        atomb = Chem.Atom(0)
        atomb.SetNoImplicit(True)
        idxb = nAts
        nAts += 1
        eMol.AddAtom(atomb)
        eMol.AddBond(ib, idxb, bondType)
        if mol.GetNumConformers():
            dummyPositions.append((idxa, ib))
            dummyPositions.append((idxb, ia))
    res = eMol.GetMol()
    if sanitize:
        Chem.SanitizeMol(res)
    if mol.GetNumConformers():
        for conf in mol.GetConformers():
            resConf = res.GetConformer(conf.GetId())
            for ia, pa in dummyPositions:
                resConf.SetAtomPosition(ia, conf.GetAtomPosition(pa))
    return Chem.GetMolFrags(res, asMols=True)


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
