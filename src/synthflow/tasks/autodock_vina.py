import tempfile

import numpy as np
from rdkit import Chem
from vina import Vina

from synthflow.config import Config, init_empty
from synthflow.pocket_specific.trainer import RxnFlow3DTrainer_single
from synthflow.tasks.docking import BaseDockingMOGFNTask, BaseDockingMOOTask, BaseDockingTask
from synthflow.utils import autodock


def setup_vina_module(self: BaseDockingTask) -> Vina:
    return autodock.create_vina_from_protein(self.protein_path, center=self.center, size=(30, 30, 30))


def _run_localopt(self, mols: list[Chem.Mol]) -> list[float]:
    assert hasattr(self, "vina_module")
    vina_module: Vina = self.vina_module

    scores = []
    out_result_path = self.save_dir / f"oracle{self.oracle_idx}_localopt.sdf"
    writer = Chem.SDWriter(str(out_result_path))
    with tempfile.NamedTemporaryFile() as tmp:
        for mol in mols:
            try:
                mol = Chem.AddHs(mol, addCoords=True)
                mol_pdbqt_string = autodock.ligand_rdmol_to_pdbqt_string(mol)
                vina_module.set_ligand_from_string(mol_pdbqt_string)
                score = vina_module.optimize()[0]
                # save pose
                with open(tmp.name, "w") as f:
                    vina_module.write_pose(tmp.name, overwrite=True)
                with open(tmp.name) as f:
                    pose = f.read()
                docked_mol = autodock.ligand_pdbqt_string_to_rdmol(pose)
            except Exception:
                score = 0.0
                docked_mol = Chem.Mol()
            docked_mol.SetIntProp("sample_idx", mol.GetIntProp("sample_idx"))
            docked_mol.SetDoubleProp("docking_score", score)
            writer.write(docked_mol)
            scores.append(score)
    writer.close()
    return [min(v, 0.0) for v in scores]


def _calc_vina_score_batch(self, mols: list[Chem.Mol]) -> list[float]:
    if self.redocking:
        raise NotImplementedError("Redocking is not implemented for AutoDockVinaTask")
        # return _run_redocking(self, objs)
    else:
        return _run_localopt(self, mols)


class AutoDockVina_Task(BaseDockingTask):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.vina_module = setup_vina_module(self)

    _calc_affinity_batch = _calc_vina_score_batch


class AutoDockVina_MOOTask(BaseDockingMOOTask):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.vina_module = setup_vina_module(self)

    _calc_affinity_batch = _calc_vina_score_batch


class AutoDockVina_MOGFNTask(BaseDockingMOGFNTask):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.vina_module = setup_vina_module(self)

    _calc_affinity_batch = _calc_vina_score_batch


class AutoDockVina_Trainer(RxnFlow3DTrainer_single):
    task: AutoDockVina_Task

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.num_training_steps = 1000
        base.task.constraint.rule = None
        base.task.docking.redocking = False

        # hparams
        base.algo.action_subsampling.sampling_ratio = 0.01
        base.algo.train_random_action_prob = 0.05
        base.algo.sampling_tau = 0.9
        base.cond.temperature.sample_dist = "uniform"
        base.cond.temperature.dist_params = [0, 64]

    def setup_task(self):
        self.task = AutoDockVina_Task(cfg=self.cfg)

    def log(self, info, index, key):
        self.add_extra_info(info)
        super().log(info, index, key)

    def add_extra_info(self, info):
        if len(self.task.batch_affinity) > 0:
            info["sample_vina_avg"] = np.mean(self.task.batch_affinity)
        best_vinas = list(self.task.topn_affinity.values())
        for topn in [10, 100, 1000]:
            if len(best_vinas) > topn:
                info[f"top{topn}_vina"] = np.mean(best_vinas[:topn])


class AutoDockVina_MOOTrainer(AutoDockVina_Trainer):
    task: AutoDockVina_MOOTask

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.task.moo.objectives = ["vina", "qed"]
        base.task.docking.redocking = False

    def setup_task(self):
        self.task = AutoDockVina_MOOTask(cfg=self.cfg)

    def add_extra_info(self, info):
        for prop, fr in self.task.avg_reward_info.items():
            info[f"sample_r_{prop}_avg"] = fr
        super().add_extra_info(info)


class AutoDockVina_MOGFNTrainer(AutoDockVina_Trainer):
    task: AutoDockVina_MOGFNTask

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.task.moo.objectives = ["vina", "qed"]
        base.task.docking.redocking = False

    def setup_task(self):
        self.task = AutoDockVina_MOGFNTask(cfg=self.cfg)

    def add_extra_info(self, info):
        for prop, fr in self.task.avg_reward_info.items():
            info[f"sample_r_{prop}_avg"] = fr
        super().add_extra_info(info)


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = "./logs/debug-autodock/"
    config.env_dir = "./data/envs/stock"
    config.overwrite_existing_exp = True

    config.cgflow.ckpt_path = "./weights/crossdocked_till_end.ckpt"
    config.task.docking.protein_path = "./data/experiments/examples/6oim_protein.pdb"
    config.task.docking.ref_ligand_path = "./data/experiments/examples/6oim_ligand.pdb"

    trial = AutoDockVina_MOOTrainer(config)
    trial.run()
