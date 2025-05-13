from synthflow.base.trainer import RxnFlow3DTrainer
from synthflow.pocket_specific.env import SynthesisEnvContext3D_single


class RxnFlow3DTrainer_single(RxnFlow3DTrainer):
    def setup_env_context(self):
        protein_path = self.cfg.task.docking.protein_path
        ref_ligand_path = self.cfg.task.docking.ref_ligand_path
        ckpt_path = self.cfg.cgflow.ckpt_path
        use_predicted_pose = self.cfg.cgflow.use_predicted_pose
        num_inference_steps = self.cfg.cgflow.num_inference_steps
        self.ctx = SynthesisEnvContext3D_single(
            self.env,
            self.task.num_cond_dim,
            ckpt_path,
            protein_path,
            ref_ligand_path,
            use_predicted_pose,
            num_inference_steps,
        )
