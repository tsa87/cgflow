from dataclasses import dataclass
from typing import Any
from omegaconf import MISSING


@dataclass
class SemlaFlowConfig:
    dataset: str = MISSING
    data_path: str | None = None
    is_pseudo_complex: bool = False
    trial_run: bool = False
    val_check_epochs: int = 1
    monitor: str = "val-validity"
    monitor_mode: str = "max"

    # Model args
    arch: str = "semla"
    d_model: int = 384
    n_layers: int = 12
    n_pro_layers: int = 6
    d_message: int = 128
    d_edge: int = 128
    n_coord_sets: int = 64
    n_attn_heads: int = 32
    d_message_hidden: int = 128
    coord_norm: str = "length"
    size_emb: int = 64
    max_atoms: int = 256
    integration_steps: int = 100

    # Training args
    epochs: int = 200
    lr: float = 0.0003
    batch_cost: int = 2048  # 4096,
    acc_batches: int = 1
    gradient_clip_val: float = 1.0
    type_loss_weight: float = 0.0
    bond_loss_weight: float = 0.0
    charge_loss_weight: float = 0.0
    coord_align: bool = False
    t_per_ar_action: float = 0.25
    max_interp_time: float = 0.5
    ordering_strategy: str = "connected"
    decomposition_strategy: str = "reaction"
    max_action_t: float = 0.75
    max_num_cuts: int = 3
    categorical_strategy: str = "auto-regressive"
    lr_schedule: str = "constant"
    warm_up_steps: int = 10000
    bucket_cost_scale: str = "linear"
    use_ema: bool = True  # Defaults to True since --no_ema negates it
    self_condition: bool = True
    distill: bool = False

    # Flow matching and sampling args
    conf_coord_strategy: str = "gaussian"
    complex_debug: bool = False
    dist_loss_weight: float = 0.0
    ode_sampling_strategy: str = "linear"
    n_training_mols: int | float = float("inf")
    n_validation_mols: int | float = float("inf")
    num_inference_steps: int = 100
    cat_sampling_noise_level: int = 1
    coord_noise_std_dev: float = 0.2
    type_dist_temp: float = 1.0
    time_alpha: float = 1.0
    time_beta: float = 1.0
    time_discretization: None = None  # FIXME: modify this - what type?
    optimal_transport: str = "None"
    pocket_encoding: str = "gvp"
    num_workers: int | None = None


def get_cfg_plinder(**kwargs) -> SemlaFlowConfig:
    params: dict[str, Any] = dict(
        dataset="plinder",
        n_pro_layers=6,
        t_per_ar_action=0.25,
        max_interp_time=0.5,
        max_action_t=0.75,
        max_num_cuts=3,
        pocket_encoding="c-alpha",
        time_alpha=2.0,
    )
    params.update(kwargs)
    return SemlaFlowConfig(**params)


def get_cfg_crossdocked(**kwargs) -> SemlaFlowConfig:
    params: dict[str, Any] = dict(
        dataset="crossdocked",
        n_pro_layers=6,
        t_per_ar_action=0.3,
        max_interp_time=0.4,
        max_action_t=0.6,
        max_num_cuts=2,
        pocket_encoding="gvp",
        time_alpha=1.0,
    )
    params.update(kwargs)
    return SemlaFlowConfig(**params)
