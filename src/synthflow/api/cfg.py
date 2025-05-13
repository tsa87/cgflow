from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class CGFlowConfig:
    # Model args
    arch: str = "semla"
    dataset: str = MISSING
    self_condition: bool = True
    d_model: int = 384
    n_layers: int = 12
    d_message: int = 64
    d_edge: int = 128
    n_coord_sets: int = 64
    n_attn_heads: int = 32
    d_message_hidden: int = 96
    coord_norm: str = "length"

    # Protein model args
    pocket_n_layers: int = 4
    pocket_d_inv: int = 256
    fixed_equi: bool = False

    # Flow matching and sampling args
    num_inference_steps: int = 100
    ode_sampling_strategy: str = "linear"
    optimal_transport: str = "None"

    # Auto-regressive setting
    decomposition_strategy: str = "reaction"
    ordering_strategy: str = "connected"
    t_per_ar_action: float = 0.3
    max_interp_time: float = 0.4
    max_action_t: float = 0.6
    max_num_cuts: int = 2
