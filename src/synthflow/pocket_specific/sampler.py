from synthflow.base.sampler import SynthFlowSampler


class PocketSpecificSampler(SynthFlowSampler):
    def setup_env_context(self):
        super().setup_env_context()

        # set protein binding site
        protein_path = self.cfg.task.docking.protein_path
        center = self.cfg.task.docking.center
        ref_ligand_path = self.cfg.task.docking.ref_ligand_path

        if center is None:
            assert ref_ligand_path is not None, (
                "Either `center` or `ref_ligand_path` must be provided to identify the binding site."
            )
        self.ctx.set_pocket(protein_path, ref_ligand_path=ref_ligand_path)
