from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from torch import Tensor

from cgflow.models.fm import Integrator
from cgflow.models.pocket import PocketEncoder
from cgflow.util.molrepr import GeometricMol, GeometricMolBatch
from cgflow.util.pocket import ProteinPocket
from cgflow.util.tokeniser import Vocabulary
from synthflow.utils import extract_pocket

from .cfg import CGFlowConfig
from .datamodule import MAX_NUM_BATCH, CGFlow_DM
from .inference import CGFlowInference
from .utils import build_vocab


class CGFlowAPI:
    cfg: CGFlowConfig
    vocab: Vocabulary
    dm: CGFlow_DM
    model: CGFlowInference
    pocket_encoder: PocketEncoder | None
    integrator: Integrator

    def __init__(
        self,
        checkpoint: str | Path | dict[str, Any],
        num_inference_steps: int,
        device: str | torch.device = "cpu",
        fp16: bool = True,
    ):
        # cal environment
        self.device: torch.device = torch.device(device)
        self.fp16: bool = fp16

        # load checkpoint and setup modules
        if isinstance(checkpoint, str | Path):
            ckpt = torch.load(checkpoint,
                              map_location="cpu",
                              weights_only=False)
        else:
            ckpt = checkpoint
        self.setup_config(ckpt, num_inference_steps)
        self.setup_vocab()
        self.setup_datamodule()
        self.setup_model(ckpt["state_dict"])
        del ckpt

    def setup_config(self, ckpt: dict[str, Any], num_inference_steps: int):
        hparam = ckpt["hyper_parameters"]
        config = CGFlowConfig(
            dataset=hparam["dataset"],
            self_condition=hparam["self_condition"],
            d_model=hparam["d_model"],
            n_layers=hparam["n_layers"],
            d_message=hparam["d_message"],
            d_edge=hparam["d_edge"],
            n_coord_sets=hparam["n_coord_sets"],
            n_attn_heads=hparam["n_attn_heads"],
            d_message_hidden=hparam["d_message_hidden"],
            pocket_d_inv=hparam["pocket_d_inv"],
            pocket_n_layers=hparam["pocket_n_layers"],
            fixed_equi=hparam["fixed_equi"],
            t_per_ar_action=hparam["t_per_ar_action"],
            max_interp_time=hparam["max_interp_time"],
            max_action_t=hparam["max_action_t"],
            max_num_cuts=hparam["max_num_cuts"],
            num_inference_steps=num_inference_steps,
        )
        self.cfg = config
        # print(self.cfg)

    def setup_vocab(self):
        self.vocab = build_vocab()

    def setup_datamodule(self):
        self.dm = CGFlow_DM(self.cfg, self.vocab, self.device)

    def setup_model(self, model_state_dict: dict[str, Any]):
        self.model = CGFlowInference(self.cfg,
                                     self.dm.interpolant,
                                     model_state_dict,
                                     self.device,
                                     compile=False)

    # set pocket info
    def set_pocket(self, pocket_path: str | Path):
        pocket: ProteinPocket = self.load_pocket(pocket_path)
        center = pocket.coords.numpy().mean(0)
        self.dm.set_center(center)
        self._tmp_pocket = pocket
        self._tmp_pocket_data = self.__encode_pocket(pocket)
        self._tmp_pocket_batch = {
            k: v.repeat((MAX_NUM_BATCH, ) + (1, ) * (v.dim() - 1))
            for k, v in self._tmp_pocket_data.items()
        }
        self._tmp_center = center

    def set_protein(self, protein_path: str | Path,
                    ref_ligand_path: str | Path):
        pocket_path = self.extract_pocket(protein_path, ref_ligand_path)
        pocket: ProteinPocket = self.load_pocket(pocket_path)
        center = pocket.coords.numpy().mean(0)
        self.dm.set_center(center)
        self._tmp_pocket = pocket
        self._tmp_pocket_data = self.__encode_pocket(pocket)
        self._tmp_pocket_batch = {
            k: v.repeat((MAX_NUM_BATCH, ) + (1, ) * (v.dim() - 1))
            for k, v in self._tmp_pocket_data.items()
        }
        self._tmp_center = center

    def trunc_pocket(self):
        del self._tmp_pocket
        del self._tmp_pocket_data
        del self._tmp_pocket_batch
        del self._tmp_center
        del self.dm._tmp_coord_center

    def __encode_pocket(self, pocket: ProteinPocket) -> dict[str, Tensor]:
        """Pre-calculate pocket encoding"""
        pocket = self.dm.transform_pocket(pocket)
        pocket_data = self.dm._batch_to_dict(
            GeometricMolBatch.from_list([pocket]))
        pocket_data = {k: v.to(self.device) for k, v in pocket_data.items()}
        pocket_data = self.model.encode_pocket(pocket_data)
        return pocket_data

    # run pose prediction
    @torch.no_grad()
    def run(
        self,
        mols: list[Chem.Mol],
        curr_step: int,
        is_last: bool = False,
        inplace: bool = False,
        return_traj: bool = False,
    ) -> tuple[list[Chem.Mol], list[np.ndarray], list[np.ndarray],
               list[np.ndarray]]:
        """Predict Binding Pose in an autoregressive manner

        Parameters
        ----------
        mols : list[Chem.Mol]
            molecules, the newly added atoms' coordinates are (0, 0, 0)
        curr_step : int
            current generation step
        is_last : bool
            whether the generation is finished
        inplace : bool
            if True, input molecule informations are updated
        return_traj : bool
            whether return all flow matching trajectory
            if False, only the last output is return (num_trajs = 1)

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            - trajectory of xt      [num_trajs, num_atoms, 3]
            - trajectory of \\hatx1 [num_trajs, num_atoms, 3]
        """
        if not inplace:
            mols = [Chem.Mol(mol) for mol in mols]

        # set gen order when it is unlabeled
        for mol in mols:
            self.set_gen_order(mol, curr_step)

        # mask dummy atoms
        masked_mols, idcs_list = zip(*[remove_dummy(mol) for mol in mols],
                                     strict=True)

        # run cgflow
        dev = "cpu" if self.device == torch.device("cpu") else "cuda"
        with torch.autocast(dev, enabled=self.fp16):
            __traj_xt_list, __traj_x1_list, __hidden_emb_list = self._run(
                masked_mols, curr_step, is_last, return_traj)

        # Add dummy atoms & pose stateate
        upd_mols: list[Chem.Mol] = []
        traj_xt_list: list[np.ndarray] = []
        traj_x1_list: list[np.ndarray] = []
        hidden_emb_list: list[np.ndarray] = []
        for i, mol in enumerate(mols):
            num_atoms = mol.GetNumAtoms()
            atom_indices = idcs_list[i]
            traj_xt = __traj_xt_list[i]
            traj_x1 = __traj_x1_list[i]
            emb = __hidden_emb_list[i]

            # add dummy atom
            traj_xt = expand_trajs(traj_xt, atom_indices, num_atoms)
            traj_x1 = expand_trajs(traj_x1, atom_indices, num_atoms)
            emb = expand_hidden_emb(emb, atom_indices, num_atoms)

            # set pose
            mol.GetConformer().SetPositions(traj_xt[-1])

            # add to list
            upd_mols.append(mol)
            traj_xt_list.append(traj_xt)
            traj_x1_list.append(traj_x1)
            hidden_emb_list.append(emb)
        return upd_mols, traj_xt_list, traj_x1_list, hidden_emb_list

    @torch.no_grad()
    def _run(
        self,
        mols: Sequence[Chem.Mol],
        curr_step: int,
        is_last: bool = False,
        return_traj: bool = False,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Predict Binding Pose in an autoregressive manner

        Parameters
        ----------
        mols : list[Chem.Mol]
            molecule w/o dummy atoms
        curr_step : int
            current generation step
        is_last : bool
            whether the generation is finished
        return_traj : bool
            whether return all flow matching trajectory

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            - trajectory of xt          [num_trajs, num_atoms, 3]
            - trajectory of \\hat{x}1   [num_trajs, num_atoms, 3]
            - hidden_emb                [num_atoms, Fh]
        """
        # helper function
        rescale = lambda pos: self.dm.rescale_coords(pos.cpu().numpy())  # noqa
        process = lambda v: (rescale(v[0]), rescale(v[1]))  # noqa

        ligand_datas = [self.get_ligand_data(mol) for mol in mols]
        ligand_list = [data[0] for data in ligand_datas]
        gen_order_list = [data[1] for data in ligand_datas]

        # sort with data size to minimize padding
        lengths = [len(orders) for orders in gen_order_list]
        sorted_indices = sorted(range(len(ligand_datas)),
                                key=lambda i: lengths[i])
        inverse_indices = sorted(range(len(sorted_indices)),
                                 key=lambda i: sorted_indices[i])

        # create loader (sort data for efficient batching)
        sorted_ligand_list = [ligand_list[i] for i in sorted_indices]
        sorted_gen_order_list = [gen_order_list[i] for i in sorted_indices]
        loader = self.dm.iterator(sorted_ligand_list, sorted_gen_order_list)

        sorted_xt_traj_list: list[np.ndarray] = []
        sorted_x1_traj_list: list[np.ndarray] = []
        sorted_hidden_emb_list: list[np.ndarray] = []
        for (curr, prior_coords), gen_steps in loader:
            N = gen_steps.shape[0]
            masks = curr["mask"].bool().numpy()  # atom mask
            pocket: dict[str, Tensor] = {
                k: v[:N]
                for k, v in self._tmp_pocket_batch.items()
            }  # match the batch size

            # set coordinates of newly added atoms to prior
            newly_added = gen_steps == curr_step
            curr["coords"][newly_added, :] = prior_coords[newly_added, :]

            # Move all the data to device
            curr = {k: v.to(self.device) for k, v in curr.items()}
            gen_steps = gen_steps.to(self.device)

            # flow matching inference for binding pose prediction
            fm_trajs, (h_equi, h_inv) = self.model.run(curr, pocket, gen_steps,
                                                       curr_step, is_last,
                                                       return_traj)
            rescaled_trajs = list(map(process, fm_trajs))

            # NOTE: we cannot use current h_inv since is is propagated from untrained layer.
            hidden_emb = h_equi.flatten(
                2
            )  # [batch, num_atoms, 3, n_hidden] -> [batch, num_atoms, 3*n_hidden]

            # add conformer for each ligand
            for i in range(N):
                mask = masks[i]
                xt_traj = [xt[i][mask] for xt, _ in rescaled_trajs]
                x1_traj = [x1[i][mask] for _, x1 in rescaled_trajs]
                sorted_xt_traj_list.append(
                    np.stack(xt_traj, axis=0, dtype=np.float_))
                sorted_x1_traj_list.append(
                    np.stack(x1_traj, axis=0, dtype=np.float_))
                sorted_hidden_emb_list.append(
                    hidden_emb[i].cpu().numpy()[mask])

        # reordering
        xt_traj_list = [sorted_xt_traj_list[i] for i in inverse_indices]
        x1_traj_list = [sorted_x1_traj_list[i] for i in inverse_indices]
        hidden_emb_list = [sorted_hidden_emb_list[i] for i in inverse_indices]
        return xt_traj_list, x1_traj_list, hidden_emb_list

    def set_gen_order(self, mol: Chem.Mol, step: int) -> list[int]:
        gen_orders: list[int] = []
        coords = mol.GetConformer().GetPositions()
        for aidx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(aidx)
            pos = coords[aidx]
            if (pos == 0.0).all():
                order = step
                atom.SetIntProp("gen_order", order)
            else:
                if atom.HasProp("gen_order"):
                    order = atom.GetIntProp("gen_order")
                else:
                    order = step - 1  # e.g., C-[*] -> the information of C is removed during the rxn
                    atom.SetIntProp("gen_order", order)
            gen_orders.append(order)
        return gen_orders

    def get_ligand_data(self, mol: Chem.Mol) -> tuple[GeometricMol, list[int]]:
        g = GeometricMol.from_rdkit(mol)
        gen_orders: list[int] = [
            atom.GetIntProp("gen_order") for atom in mol.GetAtoms()
        ]
        return g, gen_orders

    def extract_pocket(
        self,
        protein_path: str | Path,
        ref_ligand_path: str | Path,
        force_pocket_extract: bool = False,
    ) -> Path:
        return extract_pocket.extract_pocket_from_center(
            protein_path,
            ref_ligand_path=ref_ligand_path,
            force_pocket_extract=force_pocket_extract,
            cutoff=15)

    def load_pocket(self, pocket_path: str | Path) -> ProteinPocket:
        return ProteinPocket.from_pdb(pocket_path, infer_res_bonds=True)


def remove_dummy(mol: Chem.Mol) -> tuple[Chem.RWMol, list[int]]:
    non_star_idcs = [
        i for i, atom in enumerate(mol.GetAtoms()) if atom.GetSymbol() != "*"
    ]
    non_star_mol = Chem.RWMol(mol)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            non_star_mol.RemoveAtom(atom.GetIdx())
    return non_star_mol, non_star_idcs


def expand_trajs(coords: np.ndarray, atom_indices: list[int],
                 num_atoms: int) -> np.ndarray:
    """Expands [T, V', 3] coordinates to [T, V, 3], filling unspecified indices with zero."""
    if coords.shape[1] == num_atoms:
        return coords
    else:
        num_trajs = coords.shape[0]
        expanded_coords = np.zeros((num_trajs, num_atoms, 3))
        expanded_coords[:, atom_indices] = coords
        return expanded_coords.copy()


def expand_hidden_emb(emb: np.ndarray, atom_indices: list[int],
                      num_atoms: int) -> np.ndarray:
    """Expands [T, V', 3] coordinates to [T, V, 3], filling unspecified indices with zero."""
    if emb.shape[1] == num_atoms:
        return emb
    else:
        expanded_emb = np.zeros((num_atoms, emb.shape[-1]), dtype=np.float32)
        expanded_emb[atom_indices] = emb
        return expanded_emb
