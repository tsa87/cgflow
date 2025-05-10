import numpy as np
import torch
from rdkit import Chem

from .api import SemlaFlowAPI, remove_dummy


def expand_coordinates(coords: np.ndarray, node_indices: list[int], num_nodes: int) -> np.ndarray:
    """Expands [V', 3] coordinates to [V, 3], filling unspecified indices with zero."""
    if coords.shape[0] == num_nodes:
        return coords.astype(np.float_)
    else:
        expanded_coords = np.zeros((num_nodes, 3), dtype=np.float_)
        expanded_coords[node_indices] = coords
        return expanded_coords


class SemlaFlowVisualier(SemlaFlowAPI):
    """API for Visualization (FlowMatching)"""

    @torch.no_grad()
    def get_trajectory(
        self,
        mol: Chem.Mol,
        curr_step: int,
        is_last: bool = False,
        inplace: bool = False,
    ) -> tuple[Chem.Mol, list[tuple[np.ndarray, np.ndarray]]]:
        """Return Binding Pose Prediction History

        Parameters
        ----------
        mol : list[Chem.Mol]
            molecules, the newly added atoms' coordinates are (0, 0, 0)
        curr_step : int
            current generation step
        is_last : bool
            whether the generation is finished
        inplace : bool
            if True, input molecule informations are updated

        Returns
        -------
        tuple[Mol, list[tuple[np.ndarray, np.ndarray]]]
            - Molecule with updated `gen_orders`
            - flow matching trajectory
                - updated_coordinates
                - predicted_coordinates
        """
        if not inplace:
            mol = Chem.Mol(mol)

        # Mask dummy atoms
        masked_mol, atom_indices = remove_dummy(mol)

        # run semlaflow
        traj, gen_orders = self._get_trajectory(masked_mol, curr_step, is_last)

        # update generation order
        for aidx, order in zip(atom_indices, gen_orders, strict=True):
            mol.GetAtomWithIdx(aidx).SetIntProp("gen_order", order)

        # Add dummy atoms & pose update
        reindexed_traj = self.reindexing(atom_indices, traj, mol.GetNumAtoms())

        # update pose
        mol.GetConformer().SetPositions(reindexed_traj[-1][0])  # update to x_t

        return mol, reindexed_traj

    @torch.no_grad()
    def _get_trajectory(
        self,
        mol: Chem.Mol,
        curr_step: int,
        is_last: bool = False,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[int]]:
        """Predict Binding Pose in an autoregressive manner

        Parameters
        ----------
        mols : list[Chem.Mol]
            molecules, the newly added atoms' coordinates are (0, 0, 0)
        curr_step : int
            current generation step
        is_last : bool
            whether the generation is finished

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]], list[int]
            - trajectory
                - x_t: updated_coordinates
                - \\hat x_1: predicted_coordinates
            - gen_order
        """
        data, gen_orders = self.get_ligand_data(mol, curr_step)

        (curr, prior_coords), gen_steps = next(self.dm.iterator([data], [gen_orders]))

        # set coordinates of newly added atoms to prior
        newly_added = gen_steps == curr_step
        curr["coords"][newly_added, :] = prior_coords[newly_added, :]

        # Move all the data to device
        curr = {k: v.to(self.device) for k, v in curr.items()}
        gen_steps = gen_steps.to(self.device)

        # Match the batch size
        num_data = gen_steps.shape[0]
        holo = {k: v[:num_data] for k, v in self.holo_data.items()}

        # flow matching inference for binding pose prediction (single data batch)
        ## Compute the start and end times for each interpolation interval
        curr_time = curr_step * self.cfg.t_per_ar_action
        gen_times = torch.clamp(gen_steps * self.cfg.t_per_ar_action, max=self.cfg.max_action_t)

        ## If we are at the last step, we need to make sure that the end time is 1.0
        if is_last:
            end_time = 1.0
        else:
            end_time = (curr_step + 1) * self.cfg.t_per_ar_action

        ## flow matching inference
        num_fm_steps = max(1, int((end_time - curr_time) / (1.0 / self.cfg.num_inference_steps)))
        history = self.model._step_interval_inference(
            curr, gen_times, num_fm_steps, curr_time, end_time, holo=holo, return_traj=True
        )

        ## rescaling
        out_history = []
        for updated, predict in history:
            # rescaling
            upd_coords = self.dm.rescale_coords(updated.cpu().numpy())
            pred_coords = self.dm.rescale_coords(predict.cpu().numpy())

            # batch -> single datapoint
            upd_coords = upd_coords[0]  # [V, 3]
            pred_coords = pred_coords[0]  # [V, 3]
            out_history.append((upd_coords, pred_coords))
        return out_history, gen_orders

    def reindexing(self, atom_indices: list[int], traj: list[tuple[np.ndarray, np.ndarray]], num_atoms: int):
        reindexed_traj: list[tuple[np.ndarray, np.ndarray]] = []
        for upd_coords, pred_coords in traj:
            # add dummy atom
            upd_coords = expand_coordinates(upd_coords, atom_indices, num_atoms)
            pred_coords = expand_coordinates(pred_coords, atom_indices, num_atoms)
            reindexed_traj.append((upd_coords, pred_coords))
        return reindexed_traj
