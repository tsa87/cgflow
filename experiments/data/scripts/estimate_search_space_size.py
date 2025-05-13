from pathlib import Path

from omegaconf import OmegaConf

nblocks: dict[str, int] = {}
for path in Path("./envs/stock/blocks").iterdir():
    with path.open() as f:
        nblocks[path.stem] = len(f.readlines())

workflow = OmegaConf.load("./envs/stock/workflow.yaml")
birxns = workflow["BiRxn"]

# nstep 2
count_2 = []
reactant_2 = {}
for bt in range(1, 33):
    n = nblocks[str(bt)]
    n_reactants = 0
    for k, v in birxns.items():
        if bt == v["state_pattern"] and "brick" in k:
            assert len(v["block_types"]) == 1
            brick_pattern = v["block_types"][0]
            n_reactants += nblocks[brick_pattern]
    reactant_2[bt] = n_reactants
    count_2.append(n * n_reactants)
print(sum(count_2) // 2)

# nstep 3
count_3 = []
reactant_3 = {}
for bt in range(1, 33):
    n = nblocks[str(bt)]
    n_reactants = 0
    for k, v in birxns.items():
        if not (bt == v["state_pattern"] and "linker" in k):
            continue
        _bt_rxn = birxns[k.replace("linker", "brick")]["block_types"][0]
        for linker_pattern in v["block_types"]:
            _n_linkers = nblocks[linker_pattern]
            _bts = linker_pattern.split("-")
            assert _bt_rxn in _bts
            _bt_remain = _bts[1] if _bts[0] == _bt_rxn else _bts[0]
            _n_reactants = reactant_2[bt]
            n_reactants += _n_linkers * _n_reactants
    reactant_3[bt] = n_reactants
    count_3.append(n * n_reactants)
print(sum(count_3) // 6)

# nstep 4
count_4 = []
reactant_4 = {}
for bt in range(1, 33):
    n = nblocks[str(bt)]
    n_reactants = 0
    for k, v in birxns.items():
        if not (bt == v["state_pattern"] and "linker" in k):
            continue
        _bt_rxn = birxns[k.replace("linker", "brick")]["block_types"][0]
        for linker_pattern in v["block_types"]:
            _n_linkers = nblocks[linker_pattern]
            _bts = linker_pattern.split("-")
            assert _bt_rxn in _bts
            _bt_remain = _bts[1] if _bts[0] == _bt_rxn else _bts[0]
            _n_reactants = reactant_3[bt]
            n_reactants += _n_linkers * _n_reactants
    reactant_4[bt] = n_reactants
    count_4.append(n * n_reactants)
print(sum(count_4) // 24)
