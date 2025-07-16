from dataclasses import dataclass, field

from gflownet.algo.config import TBConfig


@dataclass(slots=True)
class SubsamplingConfig:
    """Configuration for action space subsampling

    Attributes
    ----------
    sampling_ratio : float
        global subsampling ratio, [0, 1]
    min_sampling : int
        the minimum number of actions for each block types
    importance_temp : float
        [Experimental] the temperature of importance weighting during sampling
    """

    sampling_ratio: float = 1.0
    min_sampling: int = 100
    importance_temp: float = 1.0


@dataclass(slots=True)
class AlgoConfig:
    """Generic configuration for algorithms

    Attributes
    ----------
    method : str
        The name of the algorithm to use (e.g. "TB")
    num_from_policy : int
        The number of on-policy samples for a training batch.
        If using a replay buffer, see `replay.num_from_replay` for the number of samples from the replay buffer, and
        `replay.num_new_samples` for the number of new samples to add to the replay buffer (e.g. `num_from_policy=0`,
        and `num_new_samples=N` inserts `N` new samples in the replay buffer at each step, but does not make that data
        part of the training batch).
    max_len : int
        The maximum length of a trajectory. In this project, it is fixed value
    max_nodes : int
        The maximum number of nodes in a generated graph
    train_random_action_prob : float
        The probability of taking a random action during training
    sampling_tau : float
        The EMA factor for the sampling model (theta_sampler = tau * theta_sampler + (1-tau) * theta)
    """

    method: str = "TB"
    num_from_policy: int = 64
    max_nodes: int = 40
    max_len: int = 4
    illegal_action_logreward: float = -100
    train_random_action_prob: float = 0.05
    sampling_tau: float = 0.0
    tb: TBConfig = field(default_factory=TBConfig)
    action_subsampling: SubsamplingConfig = field(default_factory=SubsamplingConfig)
    # unused parameters
    num_from_dataset: int = 0
    valid_num_from_policy: int = 0
    valid_num_from_dataset: int = 0
    train_det_after: int | None = None
    valid_random_action_prob: float = 0.0
