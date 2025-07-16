import os
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from rdkit import Chem
from torch import Tensor

from cgflow.api import CGFlowInference
from cgflow.models.model import PocketEmbedding
from cgflow.util.data.molrepr import LigandMol
from cgflow.util.data.pocket import ProteinPocket
from cgflow.util.dataclasses import LigandBatch, LigandTensor, PocketBatch
from synthflow.utils import extract_pocket

from .utils import extend_tensor, pad_tensors, remove_dummy

BATCH_COST = 8000
BUCKET_SIZES: list[int] = [16, 32, 48, 64]
MAX_NUM_BATCH: int = 32


@dataclass
class CGFlowOutput:
    mol: Chem.Mol
    xt: Tensor  # [N, 3]
    x1_hat: Tensor  # [N, 3]
    x_equi: Tensor  # [N, 3, d_equi]
    x_inv: Tensor  # [N, d_inv]
    traj_xt: Tensor | None = None  # [T, N, 3]
    traj_x1_hat: Tensor | None = None  # [T, N, 3]


class CGFlowAPI:
    """CGFlow Interface for Synth3DFlow"""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        num_inference_steps: int = 50,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device: torch.device = torch.device(device)
        self.module: CGFlowInference = CGFlowInference.from_pretrained(checkpoint_path, device=self.device)
        self.num_ar_steps: int = self.module.max_ar_steps
        self.num_inference_steps: int = num_inference_steps

        self.d_equi: int = self.module.d_equi
        self.d_inv: int = self.module.d_inv
        self.d_equi_pocket: int = self.module.d_equi_pocket
        self.d_inv_pocket: int = self.module.d_inv_pocket

    def set_pocket(
        self,
        protein_path: str | Path,
        center: tuple[float, float, float] | None = None,
        ref_ligand_path: str | Path | None = None,
        extract: bool = True,
    ):
        """set pocket and center for pose prediction

        Parameters
        ----------
        protein_path : str | Path
            Protein (or Pocket) structure file path (PDB format)
        center : tuple[float, float, float] | None
            Binding site center
        ref_ligand_path : str | Path | None
            Reference ligand structure file path (PDB format) to extract the binding site center
        extract : bool
            If True, the pocket will be extracted from the protein structure using the center or reference ligand.
            Else, the protein file will be considered as the pocket directly.
        """
        pocket, center = self.load_pocket(protein_path, center, ref_ligand_path, extract)
        center_arr = np.array(center, dtype=np.float32)
        pocket = pocket.shift(-center_arr)

        # encode pocket with cgflow
        pocket_tensor = pocket.to_geometric_mol().to_tensor().to(self.device)
        pocket_batch = PocketBatch.from_tensors([pocket_tensor])
        pocket_embedding = self.module.encode_pocket(pocket_batch)

        self._tmp_center: Tensor = torch.from_numpy(center_arr)
        self._tmp_pocket_embedding: PocketEmbedding = pocket_embedding

    def trunc_pocket(self):
        del self._tmp_center
        del self._tmp_pocket_embedding

    def run(
        self,
        mols: list[Chem.Mol],
        curr_step: int,
        num_inference_steps: int | None = None,
        return_traj: bool = False,
        inplace: bool = False,
    ) -> list[CGFlowOutput]:
        """Predict Binding Pose in an autoregressive manner

        Parameters
        ----------
        mols : list[Chem.Mol]
            molecules, the newly added atoms' coordinates are (0, 0, 0)
        curr_step : int
            current generation step
        num_inference_steps : bool
            Number of flow matching inference steps (default: self.num_inference_steps)
        return_traj : bool
            if True, return trajectory of xt and x1-hat
        inplace : bool
            if True, update the coordinates of the input molecules in place

        Returns
        -------
        - molecule with updated coords
        - trajectory of xt      [num_atoms, 3] (if return_traj=True, [num_traj, num_atoms, 3])
        - trajectory of x1-hat  [num_atoms, 3] (if return_traj=True, [num_traj, num_atoms, 3])
        - hidden embedding
            - x_equi            [num_atoms, 3, d_equi]
            - x_inv             [num_atoms, d_inv]
        """

        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps

        # copy molecule
        if not inplace:
            mols = [Chem.Mol(mol) for mol in mols]

        # set gen order when it is unlabeled
        for mol in mols:
            self.update_mol_cache(mol, curr_step)

        # run cgflow and return
        return self._run(mols, curr_step, num_inference_steps, return_traj)

    def _run(
        self,
        mols: Sequence[Chem.Mol],
        curr_step: int,
        num_inference_steps: int,
        return_traj: bool = False,
    ) -> list[CGFlowOutput]:
        """Predict Binding Pose in an autoregressive manner

        Parameters
        ----------
        mols : list[Chem.Mol]
            molecule w/o dummy atoms
        curr_step : int
            current generation step
        num_inference_steps : int
            flow matching inference steps
        return_traj : bool
            if True, return trajectory

        Returns
        -------
        list[tuple[Tensor, Tensor, tuple[Tensor, Tensor]]]
            - xt            [num_atoms, 3] ([num_traj, num_atoms, 3] if return_traj=True)
            - x1-hat        [num_atoms, 3] ([num_traj, num_atoms, 3] if return_traj=True)
            - hidden_emb
                - x_equi    [num_atoms, 3, d_equi]
                - x_inv     [num_atoms, d_inv]
        """
        # mask dummy atoms
        masked_mols, dummy_masks = zip(*[remove_dummy(mol) for mol in mols], strict=True)

        # collate ligand info
        ligand_datas = [self.get_ligand_tensors(mol) for mol in masked_mols]
        ligand_tensors: list[LigandTensor] = [data[0] for data in ligand_datas]
        gen_order_list: list[Tensor] = [data[1] for data in ligand_datas]

        # initialize the ligand coordinates (set prior to a new fragment, zero-com)
        ligand_tensors = [self.initialize_ligands(t) for t in ligand_tensors]

        # create loader (sort data for efficient batching)
        loader = self.iterator(ligand_tensors, gen_order_list)

        result: dict[int, CGFlowOutput] = {}
        for ligand_batch, gen_steps, sample_indices in loader:
            """
            ligand_batch: LigandBatch
                Batched ligand tensor
            gen_steps: Tensor
                Padded tensor of generation steps for each atom in the batch
            sample_indices: list[int]
                The index of each sample in the original list
            """
            # Move all the data to device
            ligand_batch = ligand_batch.to(self.device)
            gen_steps = gen_steps.to(self.device)

            # Expand pocket embedding to match the batch size
            pocket_embedding = self._expand_pocket_embedding(self._tmp_pocket_embedding, ligand_batch.batch_size)

            # flow matching inference for binding pose prediction
            fm_trajs, (x_equis, x_invs) = self.module.run(
                ligand_batch, pocket_embedding, gen_steps, curr_step, num_inference_steps
            )
            # add conformer for each ligand
            for i, sample_idx in enumerate(sample_indices):
                masked_mol = masked_mols[sample_idx]
                mol = mols[sample_idx]
                is_valid_atom = dummy_masks[sample_idx]
                n_valid_atoms = sum(is_valid_atom)
                assert masked_mol.GetNumAtoms() == n_valid_atoms, (
                    "Number of atoms in the molecule does not match the number of valid atoms"
                )
                assert mol.GetNumAtoms() == len(is_valid_atom), (
                    "Number of atoms in the molecule does not match the number of valid atoms"
                )
                if not return_traj:
                    fm_trajs = [fm_trajs[-1]]

                # collect coordinates of valid atoms
                xt_traj = torch.stack([xt.coords[i, :n_valid_atoms].to("cpu") for xt, _ in fm_trajs])
                x1_traj = torch.stack([x1.coords[i, :n_valid_atoms].to("cpu") for _, x1 in fm_trajs])
                # shift to center
                xt_traj = xt_traj + self._tmp_center.view(1, 1, 3)  # [T, N, 3]
                x1_traj = x1_traj + self._tmp_center.view(1, 1, 3)  # [T, N, 3]
                # pad dummy atom (-[*])
                xt_traj = extend_tensor(xt_traj, is_valid_atom, is_batched=True)
                x1_traj = extend_tensor(x1_traj, is_valid_atom, is_batched=True)
                xt, x1 = xt_traj[-1], x1_traj[-1]  # last step

                # collect embeddings of valid atoms
                _x_equi = x_equis[i, :n_valid_atoms].to("cpu").float()
                _x_inv = x_invs[i, :n_valid_atoms].to("cpu").float()
                # pad dummy atom (-[*])
                _x_equi = extend_tensor(_x_equi, is_valid_atom)
                _x_inv = extend_tensor(_x_inv, is_valid_atom)

                # update the pose
                mol.GetConformer().SetPositions(xt.double().numpy())

                if return_traj:
                    result[sample_idx] = CGFlowOutput(mol, xt, x1, _x_equi, _x_inv, xt_traj, x1_traj)
                else:
                    result[sample_idx] = CGFlowOutput(mol, xt, x1, _x_equi, _x_inv, None, None)
        # to list
        num_mols = len(mols)
        return [result[i] for i in range(num_mols)]

    def update_mol_cache(self, mol: Chem.Mol, step: int):
        """save generative information to each atom in the molecule"""
        assert mol.GetNumConformers() <= 1, "Molecule should have 0 or 1 conformer"
        if mol.GetNumConformers() == 0:
            mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
        coords = mol.GetConformer().GetPositions()
        # set gen-order info
        for aidx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(aidx)
            pos = coords[aidx]

            # set gen order
            if (pos == 0.0).all():
                order = step
                atom.SetIntProp("gen_order", order)
            else:
                assert atom.GetSymbol() != "*", "Dummy atom should not have coordinates"
                if atom.HasProp("gen_order"):
                    order = atom.GetIntProp("gen_order")
                else:
                    # e.g., C-[*] -> the property of atom C is lost during the rxn
                    order = step - 1
                    atom.SetIntProp("gen_order", order)

        # set attachment info
        for aidx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(aidx)
            gen_order = atom.GetIntProp("gen_order")
            is_dummy = atom.GetSymbol() == "*"

            neighbors: tuple[Chem.Atom, ...] = atom.GetNeighbors()
            # there is only one neighbor for dummy atom
            if is_dummy:
                assert len(neighbors) == 1, "dummy atom should have only one neighbor"

            for neigh_atom in neighbors:
                # dummy atom's neighbor is attachment atom
                if is_dummy:
                    neigh_atom.SetBoolProp("is_attachment", True)
                if neigh_atom.GetSymbol() == "*":
                    atom.SetBoolProp("is_attachment", True)
                # attachment atom has a neighbor atom with different gen order
                if neigh_atom.GetIntProp("gen_order") != gen_order:
                    atom.SetBoolProp("is_attachment", True)
                    neigh_atom.SetBoolProp("is_attachment", True)

    def get_ligand_tensors(self, mol: Chem.Mol) -> tuple[LigandTensor, Tensor]:
        m = LigandMol.from_rdkit(mol, remove_hs=False)
        m.attachments = torch.tensor([atom.HasProp("is_attachment") for atom in mol.GetAtoms()], dtype=torch.bool)
        gen_orders = torch.tensor([atom.GetIntProp("gen_order") for atom in mol.GetAtoms()], dtype=torch.int32)
        return m.to_tensor(), gen_orders

    def initialize_ligands(self, ligand: LigandTensor) -> LigandTensor:
        """initialize the ligand coordinates"""
        # first check the atom is belong to new fragment or not
        coords = ligand.coords  # [L, 3]
        set_to_prior = (coords == 0).all(dim=-1, keepdim=True)  # [L,]

        # then, move to center
        coords = coords - self._tmp_center.view(1, 3)  # [L, 3]

        # set prior for new fragment atoms
        prior = self.module.prior_like(coords)
        coords = torch.where(set_to_prior, prior, coords)  # [L, 3]

        return ligand.copy_with(coords=coords)

    def load_pocket(
        self,
        protein_path: str | Path,
        center: tuple[float, float, float] | None = None,
        ref_ligand_path: str | Path | None = None,
        extract: bool = True,
        force_pocket_extract: bool = False,
    ) -> tuple[ProteinPocket, tuple[float, float, float]]:
        if center is None:
            assert ref_ligand_path is not None, "Reference ligand path must be provided if center is not given"
            center = extract_pocket.get_mol_center(ref_ligand_path)

        if extract:
            with TemporaryDirectory() as dir:
                pocket_path = os.path.join(dir, "pocket.pdb")
                extract_pocket.extract_pocket_from_center(
                    protein_path,
                    pocket_path,
                    center=center,
                    cutoff=15.0,
                    force_pocket_extract=force_pocket_extract,
                )
                pocket = ProteinPocket.from_pdb(pocket_path, infer_res_bonds=True)
        else:
            pocket = ProteinPocket.from_pdb(protein_path, infer_res_bonds=True)
        return pocket, center

    def iterator(
        self,
        mols: list[LigandTensor],
        gen_orders: list[Tensor],
    ) -> Generator[tuple[LigandBatch, Tensor, list[int]]]:
        lengths: list[int] = [mol.length for mol in mols]
        sample_idcs: list[list[int]] = [[] for _ in BUCKET_SIZES]

        # add bucket
        for idx, size in enumerate(lengths):
            for k, threshold in enumerate(BUCKET_SIZES):
                # add sample to the first bucket that can hold it
                if size <= threshold:
                    sample_idcs[k].append(idx)
                    break
            else:
                # if there is no bucket for this size, put it in the last bucket
                sample_idcs[-1].append(idx)

        # sort a ligand size for each bucket
        for bucket in sample_idcs:
            bucket.sort(key=lambda i: lengths[i])

        # collate batch
        def __get_batch(batch_idxs: list[int]) -> tuple[LigandBatch, Tensor, list[int]]:
            collated_ligands = [mols[i] for i in batch_idxs]
            collated_gen_orders = [gen_orders[i] for i in batch_idxs]
            ligand_batch = LigandBatch.from_tensors(collated_ligands)
            gen_orders_batch = pad_tensors(collated_gen_orders)
            return ligand_batch, gen_orders_batch, batch_idxs

        for bucket in sample_idcs:
            batch, curr_cost = [], 0
            for idx in bucket:
                if (curr_cost > BATCH_COST) or (len(batch) == MAX_NUM_BATCH):
                    yield __get_batch(batch)
                    batch, curr_cost = [], 0
                batch.append(idx)
                curr_cost += lengths[idx]
            if len(batch) > 0:
                yield __get_batch(batch)

    def _expand_pocket_embedding(self, embedding: PocketEmbedding, batch_size: int) -> PocketEmbedding:
        """Expand the pocket embedding to match the batch size"""
        return PocketEmbedding(
            coords=embedding.coords.expand(batch_size, -1, -1),  # [B, L, 3]
            equi=embedding.equi.expand(batch_size, -1, -1, -1),  # [B, L, 3, d_equi]
            inv=embedding.inv.expand(batch_size, -1, -1),  # [B, L, d_inv]
            mask=embedding.mask.expand(batch_size, -1),  # [B, L]
        )
