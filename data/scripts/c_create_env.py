import argparse
import functools
import multiprocessing
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts
from tqdm import tqdm

from rxnflow.utils.featurization import get_block_features


class Conversion:
    def __init__(self, info: DictConfig):
        self.key = info.key
        self.template = info.original + ">>" + info.convert
        self.rxn: ChemicalReaction = ReactionFromSmarts(self.template)
        self.rxn.Initialize()

    def run(self, mol: Chem.Mol) -> list[Chem.Mol]:
        res = self.rxn.RunReactants((mol,), 10)
        return list([v[0] for v in res])


def _run_reaction(smi: str, rxn: Conversion) -> list[str]:
    mol = Chem.MolFromSmiles(smi)
    prod_mols = rxn.run(mol)
    return list(set(Chem.MolToSmiles(mol) for mol in prod_mols))


def get_block(env_dir: Path, block_file: Path, protocol_dir: Path, num_cpus: int):
    # load block
    with block_file.open() as f:
        lines = f.readlines()[1:]
    enamine_block_list: list[str] = [ln.split()[0] for ln in lines]
    enamine_id_list: list[str] = [ln.strip().split()[1] for ln in lines]

    # run conversion
    block_dir = env_dir / "blocks/"
    block_dir.mkdir(parents=True, exist_ok=True)
    conversion_config = OmegaConf.load(protocol_dir / "reactant.yaml")
    for i in tqdm(range(len(conversion_config))):
        info_i = conversion_config[i]
        rxn = Conversion(info_i)
        _func = functools.partial(_run_reaction, rxn=rxn)
        with multiprocessing.Pool(num_cpus) as pool:
            res = pool.map(_func, enamine_block_list)
        del rxn, _func

        brick_to_id: dict[str, list[str]] = {}
        for id, bricks in zip(enamine_id_list, res, strict=True):
            for smi in bricks:
                brick_to_id.setdefault(smi, []).append(id)
        if len(brick_to_id) == 0:
            continue
        with open(block_dir / f"{info_i.key}.smi", "w") as w:
            for smi, id_list in brick_to_id.items():
                w.write(f"{smi}\t{';'.join(sorted(id_list))}\n")

        brick_list = list(brick_to_id.keys())
        for j in tqdm(range(i + 1, len(conversion_config)), leave=False):
            info_j = conversion_config[j]
            rxn = Conversion(info_j)
            _func = functools.partial(_run_reaction, rxn=rxn)
            with multiprocessing.Pool(num_cpus) as pool:
                res = pool.map(_func, brick_list)
            del rxn, _func

            linker_to_id: dict[str, list[str]] = {}
            for brick, linkers in zip(brick_list, res, strict=True):
                for smi in linkers:
                    linker_to_id.setdefault(smi, []).extend(brick_to_id[brick])
            if len(linker_to_id) == 0:
                continue
            with open(block_dir / f"{info_i.key}-{info_j.key}.smi", "w") as w:
                for smi, ids in linker_to_id.items():
                    w.write(f"{smi}\t{';'.join(sorted(ids))}\n")


def get_block_data(env_dir: Path, num_cpus: int):
    block_smi_dir = env_dir / "blocks/"
    save_block_data_path = env_dir / "bb_feature.pt"

    data: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for smi_file in tqdm(list(block_smi_dir.iterdir())):
        with smi_file.open() as f:
            lines = f.readlines()
        if len(lines) == 0:
            continue
        smi_list = [ln.split()[0] for ln in lines]
        fp_list = []
        desc_list = []
        for idx in tqdm(range(0, len(smi_list), 10000), leave=False):
            chunk = smi_list[idx : idx + 10000]
            with multiprocessing.Pool(num_cpus) as pool:
                results = pool.map(get_block_features, chunk)
            for fp, desc in results:
                fp_list.append(fp)
                desc_list.append(desc)
        block_descs = torch.from_numpy(np.stack(desc_list, 0))
        block_fps = torch.from_numpy(np.stack(fp_list, 0))
        data[smi_file.stem] = (block_descs, block_fps)
    torch.save(data, save_block_data_path)


def get_workflow(env_dir: Path, protocol_dir: Path):
    protocol_config = OmegaConf.load(protocol_dir / "protocol.yaml")
    save_workflow_path = env_dir / "workflow.yaml"

    firstblock_protocols: dict[str, dict] = {}
    unirxn_protocols: dict[str, dict] = {}
    birxn_protocols: dict[str, dict] = {}
    workflow_config = {"FirstBlock": firstblock_protocols, "UniRxn": unirxn_protocols, "BiRxn": birxn_protocols}

    # firstblock
    pattern_to_types: dict[int, list[str]] = {}
    for block_file in Path(env_dir / "blocks/").iterdir():
        block_type = block_file.stem
        protocol_name = "block" + block_type
        with block_file.open() as f:
            if len(f.readline()) == 0:
                continue
        for pattern in map(int, block_type.split("-")):
            pattern_to_types.setdefault(pattern, []).append(block_type)
        # TODO: Remove here, currently, only brick for firstblock
        if "-" in block_type:
            continue
        firstblock_protocols[protocol_name] = {"block_types": [block_type]}

    # remove redundant items
    pattern_to_types = {k: sorted(list(set(v))) for k, v in pattern_to_types.items()}
    pattern_dict = {
        pattern: {
            "brick": [t for t in block_types if ("-" not in t)],
            "linker": [t for t in block_types if ("-" in t)],
        }
        for pattern, block_types in pattern_to_types.items()
    }

    # birxn (no unirxn)
    for rxn_name, cfg in protocol_config.items():
        rxn_name = str(rxn_name)
        if cfg.ordered:
            block_orders = [0, 1]
        else:
            assert cfg.block_type[0] == cfg.block_type[1]
            block_orders = [0]

        for order in block_orders:
            is_block_first = order == 0
            state_pattern = cfg.block_type[1 - order]
            block_pattern = cfg.block_type[order]
            for t in ["brick", "linker"]:
                protocol_name = rxn_name + f"_{t}_" + ("b0" if is_block_first else "b1")
                block_keys = pattern_dict[block_pattern][t]
                if len(block_keys) > 0:
                    birxn_protocols[protocol_name] = {
                        "forward": cfg.forward,
                        "reverse": cfg.reverse,
                        "is_block_first": order == 0,
                        "state_pattern": state_pattern,
                        "block_types": block_keys,
                    }
    OmegaConf.save(workflow_config, save_workflow_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create environment")
    parser.add_argument(
        "-b",
        "--building_block_path",
        type=Path,
        help="Path of input building block smiles file",
        default="./building_blocks/enamine_stock.smi",
    )
    parser.add_argument(
        "-p",
        "--protocol_dir",
        type=Path,
        help="Path of input synthesis protocol directory",
        default="./template/real/",
    )
    parser.add_argument(
        "-o",
        "--env_dir",
        type=Path,
        help="Path of output environment directory",
        default="./envs/stock/",
    )
    parser.add_argument("--cpu", type=int, help="Num Workers", default=len(os.sched_getaffinity(0)))
    args = parser.parse_args()

    env_dir: Path = args.env_dir
    protocol_dir: Path = args.protocol_dir
    block_file: Path = args.building_block_path
    num_cpus: int = args.cpu

    assert not env_dir.exists()

    print("convert building blocks to ready-to-compose fragments")
    get_block(env_dir, block_file, protocol_dir, num_cpus)
    print("pre-calculate building block features")
    get_block_data(env_dir, num_cpus)
    print("create workflow")
    get_workflow(env_dir, protocol_dir)
