import random
from pathlib import Path

from pmnet_appl import BaseProxy, get_docking_proxy
from rdkit import Chem
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from rxnflow.base.task import BaseTask
from rxnflow.utils.misc import get_worker_env

from synthflow.base.env_ctx_cgflow import SynthesisEnvContext3D_cgflow
from synthflow.base.trainer import RxnFlow3DTrainer
from synthflow.config import Config
from synthflow.pocket_conditional.affinity import pmnet_proxy

"""
Summary
- ProxyTask: Base Class
- ProxyTask_MultiPocket & ProxyTrainer_MultiPocket: Train Pocket-Conditioned RxnFlow.
"""


class ProxyTask(BaseTask):
    pocket_key: str

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.objectives = cfg.task.moo.objectives
        self.proxy: BaseProxy = self._load_task_models(cfg.task.pocket_conditional.proxy)
        self.reward_function = pmnet_proxy.get_reward_function(self.proxy, self.objectives)
        self.last_reward: dict[str, Tensor] = {}  # For Logging
        self.index_path = Path(cfg.log_dir) / "index.csv"
        self.save_dir = Path(cfg.log_dir) / "pose/"
        self.save_dir.mkdir(exist_ok=True)

    def compute_rewards(self, mols: list[RDMol]) -> Tensor:
        self.save_pose(mols)
        r, info = self.reward_function(mols, self.pocket_key)
        self.sample_reward_info = info
        flat_rewards = r.view(-1, 1)
        assert flat_rewards.shape[0] == len(mols)
        return flat_rewards

    def _load_task_models(self, proxy_arg: tuple[str, str, str]) -> BaseProxy:
        proxy_model, proxy_type, proxy_dataset = proxy_arg
        proxy = get_docking_proxy(proxy_model, proxy_type, proxy_dataset, None, self.cfg.device)
        return proxy

    def save_pose(self, mols: list[RDMol]):
        out_path = self.save_dir / f"oracle{self.oracle_idx}.sdf"
        with Chem.SDWriter(str(out_path)) as w:
            for i, mol in enumerate(mols):
                mol.SetIntProp("sample_idx", i)
                w.write(mol)


class Proxy_MultiPocket_Task(ProxyTask):
    """For multi-pocket environments (Pre-training)"""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.root_protein_dir = Path(cfg.task.pocket_conditional.protein_dir)

        # TODO: add protein db structure
        self.pdb_to_pocket: dict[str, tuple[float, float, float]] = {}
        with open(cfg.task.pocket_conditional.train_key) as f:
            for ln in f.readlines():
                pdb, x, y, z = ln.strip().split(",")
                self.pdb_to_pocket[pdb] = (float(x), float(y), float(z))

        self.pdb_keys: list[str] = sorted(list(self.pdb_to_pocket.keys()))
        self.pdb_keys = [key for key in self.pdb_keys if key in self.proxy._cache.keys()]
        random.shuffle(self.pdb_keys)

    def compute_rewards(self, mols: list[RDMol]) -> Tensor:
        # print(f"compute reward for {self.pocket_key}")
        rewards = super().compute_rewards(mols)
        return rewards

    def _load_task_models(self, proxy_arg: tuple[str, str, str]) -> BaseProxy:
        proxy_model, proxy_type, proxy_dataset = proxy_arg
        proxy = get_docking_proxy(proxy_model, proxy_type, proxy_dataset, "train", self.cfg.device)
        return proxy

    def sample_conditional_information(self, n: int, train_it: int) -> dict[str, Tensor]:
        ctx: SynthesisEnvContext3D_cgflow = get_worker_env("ctx")

        # set pocket (up to 100 trials)
        for index in range(train_it, train_it + 100):
            pdb_code = self.pdb_keys[index % len(self.pdb_keys)]
            protein_path = self.root_protein_dir / f"{pdb_code}.pdb"
            center = self.pdb_to_pocket[pdb_code]
            try:
                ctx.set_pocket(protein_path, center)
            except:
                print(f"Failed to set pocket for {pdb_code} at {protein_path}. Try next pocket.")
                continue
            else:
                break
        else:
            raise ValueError("Fail to set pocket")
        self.pocket_key = pdb_code
        self.pocket_filename = protein_path
        # log what pocket is selected for each training iterations
        with open(self.index_path, "a") as w:
            w.write(f"{self.oracle_idx},{pdb_code}\n")
        return super().sample_conditional_information(n, train_it)


class Proxy_MultiPocket_Trainer(RxnFlow3DTrainer):
    task: Proxy_MultiPocket_Task

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.desc = "Proxy-QED-SA optimization for multiple targets"
        base.task.moo.objectives = ["vina", "qed"]
        base.validate_every = 0
        base.num_training_steps = 200_000
        base.checkpoint_every = 1_000

        base.model.num_emb = 128

        # GFN parameters
        base.cond.temperature.sample_dist = "uniform"
        base.cond.temperature.dist_params = [1, 64]

        # model training
        base.algo.sampling_tau = 0.0
        base.algo.train_random_action_prob = 0.2
        base.algo.num_from_policy = 32

        # it is mandatory to set this to False for multi-pocket training
        base.replay.use = False

        # training learning rate
        base.opt.learning_rate = 1e-4
        base.opt.lr_decay = 10_000
        base.algo.tb.Z_learning_rate = 1e-2
        base.algo.tb.Z_lr_decay = 20_000

    def setup_env_context(self):
        ckpt_path = self.cfg.cgflow.ckpt_path
        use_predicted_pose = self.cfg.cgflow.use_predicted_pose
        num_inference_steps = self.cfg.cgflow.num_inference_steps
        self.ctx = SynthesisEnvContext3D_cgflow(
            self.env,
            self.task.num_cond_dim,
            ckpt_path,
            use_predicted_pose,
            num_inference_steps,
        )

    def setup_task(self):
        self.task = Proxy_MultiPocket_Task(cfg=self.cfg)

    def log(self, info, index, key):
        for obj in self.task.objectives:
            info[f"sampled_{obj}_avg"] = self.task.sample_reward_info[obj].mean().item()
        super().log(info, index, key)
