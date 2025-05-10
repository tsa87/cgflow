from rxnflow.base.trainer import RxnFlowTrainer
from cgflow.config import Config
from cgflow.envs.env import SynthesisEnvContext3D, SynthesisEnv3D


class RxnFlow3DTrainer(RxnFlowTrainer):
    cfg: Config
    env: SynthesisEnv3D

    def set_default_hps(self, base: Config):
        """rxnflow.config.Config -> cgflow.config.Config"""
        super().set_default_hps(base)

    def get_default_cfg(self):
        """rxnflow.config.Config -> cgflow.config.Config"""
        return Config()

    def setup_env(self):
        self.env = SynthesisEnv3D(self.cfg.env_dir)

    def setup_env_context(self):
        protein_path = self.cfg.task.docking.protein_path
        ref_ligand_path = self.cfg.task.docking.ref_ligand_path
        self.ctx = SynthesisEnvContext3D(self.env, self.task.num_cond_dim, protein_path, ref_ligand_path)


class RxnFlow3DTrainer_unidock(RxnFlow3DTrainer):
    def setup_env_context(self):
        from .envs.unidock_env import SynthesisEnvContext3D_unidock

        protein_path = self.cfg.task.docking.protein_path
        ref_ligand_path = self.cfg.task.docking.ref_ligand_path
        self.ctx = SynthesisEnvContext3D_unidock(self.env, self.task.num_cond_dim, protein_path, ref_ligand_path)

    def setup_algo(self):
        from .algo.unidock_algo import SynthesisTB3D_unidock

        assert self.cfg.algo.method == "TB"
        self.algo = SynthesisTB3D_unidock(self.env, self.ctx, self.cfg)


class RxnFlow3DTrainer_semla(RxnFlow3DTrainer):
    def setup_env_context(self):
        from .envs.semla_env import SynthesisEnvContext3D_semla

        protein_path = self.cfg.task.docking.protein_path
        ref_ligand_path = self.cfg.task.docking.ref_ligand_path
        ckpt_path = self.cfg.semlaflow.ckpt_path
        use_predicted_pose = self.cfg.semlaflow.use_predicted_pose
        num_inference_steps = self.cfg.semlaflow.num_inference_steps
        self.ctx = SynthesisEnvContext3D_semla(
            self.env,
            self.task.num_cond_dim,
            ckpt_path,
            protein_path,
            ref_ligand_path,
            use_predicted_pose,
            num_inference_steps,
        )

    def setup_algo(self):
        from .algo.semla_algo import SynthesisTB3D_semla

        assert self.cfg.algo.method == "TB"
        self.algo = SynthesisTB3D_semla(self.env, self.ctx, self.cfg)
