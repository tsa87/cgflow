from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class MOOTaskConfig:
    """Common Config for the MOOTasks

    Attributes
    ----------
    objectives : list[str]
        The objectives to use for the multi-objective optimization.
    n_valid : int
        The number of valid cond_info tensors to sample.
    n_valid_repeats : int
        The number of times to repeat the valid cond_info tensors.
    online_pareto_front : bool
        Whether to calculate the pareto front online.
    """

    objectives: list[str] = field(default_factory=lambda: [])
    n_valid: int = 15
    n_valid_repeats: int = 128
    log_topk: bool = False
    online_pareto_front: bool = True


@dataclass
class DockingTaskConfig:
    """Config for DockingTask

    Attributes
    ----------
    protein_path: str (path)
        Protein path
    center: tuple[float, float, float]
        Pocket center
    ref_ligand_path: str (path)
        Reference ligand path
    size: tuple[float, float, float]
        Search box size
    redocking: bool
        if True, run docking; otherwise, run local opt
    ff_opt: str
        When redocking is False, run FF optimization before local opt
    """

    protein_path: str = MISSING
    center: tuple[float, float, float] | None = None
    size: tuple[float, float, float] = (30, 30, 30)
    ref_ligand_path: str | None = None
    redocking: bool = True  # TODO: change to docking mode (score-only, local-opt, redock)
    ff_opt: str = "mmff"  # 'none', 'uff', 'mmff'
    exhaustiveness: int = 8  # Exhaustiveness for Vina docking


@dataclass
class ConstraintConfig:
    """Config for Filtering

    Attributes
    ----------
    rule: str (path)
        DrugFilter Rule
            - None
            - lipinski
            - veber
    """

    rule: str | None = None


@dataclass
class PocketConditionalConfig:
    """Config for PocketConditional Training

    Attributes
    ----------
    proxy: tuple[str, str, str] (proxy_name, docking_program, train_dataset)
        Proxy Key from PharmacoNet
    """

    proxy: tuple[str, str, str] = ("TacoGFN_Reward", "QVina", "ZINCDock15M")
    protein_dir: str = "./data/experiments/CrossDocked2020/"
    train_key: str = "./data/experiments/CrossDocked2020/train_keys.csv"


@dataclass
class TasksConfig:
    moo: MOOTaskConfig = field(default_factory=MOOTaskConfig)
    constraint: ConstraintConfig = field(default_factory=ConstraintConfig)
    docking: DockingTaskConfig = field(default_factory=DockingTaskConfig)
    pocket_conditional: PocketConditionalConfig = field(default_factory=PocketConditionalConfig)
