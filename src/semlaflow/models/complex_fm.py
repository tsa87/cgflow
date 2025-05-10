import torch

from torchmetrics import MetricCollection
from semlaflow.models.fm import BaseMolecularCFM

from semlaflow.models.pocket import PocketEncoder
import semlaflow.util.complex_metrics as ComplexMetrics


def _compute_complex_model_args(
    batch,
    t,
    cond_batch=None,
    holo=None,
    holo_pocks=None,
    pocket_encoder=None,
):
    """Predict molecular coordinates and atom types

    Args:
        batch (dict[str, Tensor]): Batched pointcloud data
        t (torch.Tensor): Interpolation times between 0 and 1, shape [batch_size]
        training (bool): Whether to run forward in training mode
        cond_batch (dict[str, Tensor]): Predictions from previous step, if we are using self conditioning
        holo (torch.Tensor): Batched pointcloud data for the holo structure
        holo_pocks (list[ProteinPocket]): List of ProteinPocket objects
        pocket_encoder (GVP_embedding): Pocket encoder model
    Returns:
        (predicted coordinates, atom type logits (unnormalised probabilities))
        Both torch.Tensor, shapes [batch_size, num_atoms, 3] and [batch_size, num atoms, vocab_size]
    """
    if pocket_encoder is not None:
        assert holo is not None
        holo["atomics"] = pocket_encoder(holo_pocks, holo)

    lig_coords = batch["coords"]
    lig_atom_types = batch["atomics"]
    lig_bonds = batch["bonds"]
    lig_mask = batch["mask"]

    # Prepare invariant atom features
    lig_times = t.view(-1, 1, 1).expand(-1, lig_coords.size(1), -1)
    lig_features = torch.cat((lig_times, lig_atom_types), dim=2)

    if holo is not None:
        pro_coords = holo["coords"]
        pro_atom_types = holo["atomics"]
        pro_mask = holo["mask"]

        pro_times = t.view(-1, 1, 1).expand(-1, pro_coords.size(1), -1)
        pro_inv_feats = torch.cat((pro_times, pro_atom_types), dim=2)
    else:
        pro_coords = None
        pro_atom_types = None
        pro_mask = None
        pro_inv_feats = None

    model_args = {
        "lig_coords": lig_coords,
        "lig_inv_feats": lig_features,
        "lig_edge_feats": lig_bonds,
        "lig_atom_mask": lig_mask,
        "pro_coords": pro_coords,
        "pro_inv_feats": pro_inv_feats,
        "pro_atom_mask": pro_mask,
    }

    if cond_batch is not None:
        model_args = {
            **model_args,
            "lig_cond_coords": cond_batch["coords"],
            "lig_cond_atomics": cond_batch["atomics"],
            "lig_cond_bonds": cond_batch["bonds"],
        }
    return model_args


class ComplexMolecularCFM(BaseMolecularCFM):
    def __init__(
        self,
        pocket_encoder: PocketEncoder,
        use_gvp: bool,
        use_complex_metrics: bool,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert not self.distill

        if use_gvp:
            self.pocket_encoder = pocket_encoder
        else:
            self.pocket_encoder = None

        if use_complex_metrics:
            complex_metrics = {
                "clash": ComplexMetrics.Clash(),
                "interactions": ComplexMetrics.Interactions(),
            }
            self.complex_metrics = MetricCollection(complex_metrics, compute_groups=False)
        else:
            self.complex_metrics = None

    def _extract_from_batch(self, batch):
        prior, data, interpolated, holo_mols, holo_pocks, times = batch
        return {
            "prior": prior,
            "data": data,
            "interpolated": interpolated,
            "holo_mols": holo_mols,
            "holo_pocks": holo_pocks,
            "times": times,
        }

    def _compute_forward_args(
        self,
        batch_dict,
        training=True,
        cond_batch=None,
    ):
        return {
            "batch": batch_dict["interpolated"],
            "t": batch_dict["times"],
            "training": training,
            "cond_batch": cond_batch,
            "holo_mols": batch_dict["holo_mols"],
            "holo_pocks": batch_dict["holo_pocks"],
        }

    def _compute_model_args(self, batch, t, cond_batch=None, holo_mols=None, holo_pocks=None):
        """Predict molecular coordinates and atom types

        Args:
            batch (dict[str, Tensor]): Batched pointcloud data
            t (torch.Tensor): Interpolation times between 0 and 1, shape [batch_size]
            training (bool): Whether to run forward in training mode
            cond_batch (dict[str, Tensor]): Predictions from previous step, if we are using self conditioning
            holo (torch.Tensor): Batched pointcloud data for the holo structure

        Returns:
            (predicted coordinates, atom type logits (unnormalised probabilities))
            Both torch.Tensor, shapes [batch_size, num_atoms, 3] and [batch_size, num atoms, vocab_size]
        """
        return _compute_complex_model_args(
            batch,
            t,
            cond_batch=cond_batch,
            holo=holo_mols,
            holo_pocks=holo_pocks,
            pocket_encoder=self.pocket_encoder,
        )

    def forward(self, batch, t, training=False, cond_batch=None, holo_mols=None, holo_pocks=None):
        return super().forward(
            batch=batch,
            t=t,
            training=training,
            cond_batch=cond_batch,
            holo_mols=holo_mols,
            holo_pocks=holo_pocks,
        )

    def _generate(self, prior, steps, strategy="linear", holo_mols=None, holo_pocks=None):
        return super()._generate(
            prior=prior,
            steps=steps,
            strategy=strategy,
            holo=holo_mols,
            holo_pocks=holo_pocks,
        )

    def _compute_gen_batch(self, batch):
        prior, data, interpolated, holo_mols, holo_pocks, times = batch
        gen_batch = self._generate(
            prior,
            self.integrator.steps,
            self.sampling_strategy,
            holo_mols=holo_mols,
            holo_pocks=holo_pocks,
        )
        return gen_batch

    def _compute_data_mols(self, batch):
        prior, data, interpolated, holo_mols, holo_pocks, times = batch
        return self._generate_mols(data, rescale=True)

    def _compute_interp_mols(self, batch):
        prior, data, interpolated, holo_mols, holo_pocks, times = batch
        gen_interp_steps = max(1, int((1 - times[0].item()) * self.integrator.steps))
        gen_interp_batch = self._generate(interpolated, gen_interp_steps, holo_mols=holo_mols, holo_pocks=holo_pocks)
        gen_interp_mols = self._generate_mols(gen_interp_batch)
        return gen_interp_mols

    def _compute_complex_metrics(self, batch):
        # Compute the complex metrics additionally
        prior, data, interpolated, holo_mols, holo_pocks, times = batch

        if holo_pocks is not None and self.complex_metrics is not None:
            gen_batch = self._compute_gen_batch(batch)
            gen_mols = self._generate_mols(gen_batch)
            # Check if it's not a psuedo complex batch
            self.complex_metrics.update(gen_mols, holo_pocks)

    def validation_step(self, batch, b_idx):
        super().validation_step(batch, b_idx)
        self._compute_complex_metrics(batch)

    def _compute_and_log_complex_metrics(self):
        if self.complex_metrics is not None:
            complex_metrics_results = self.complex_metrics.compute()
            self.complex_metrics.reset()
            for metric, value in complex_metrics_results.items():
                progbar = True if metric == "validity" else False
                self.log(f"val-{metric}", float(value), on_epoch=True, logger=True, prog_bar=progbar, sync_dist=True)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        self._compute_and_log_complex_metrics()

    def predict_step(self, batch, batch_idx):
        prior, data, interpolated, holo_mols, holo_pocks, times = batch
        gen_batch = self._generate(
            prior,
            self.integrator.steps,
            self.sampling_strategy,
            holo_mols=holo_mols,
            holo_pocks=holo_pocks,
        )
        gen_mols = self._generate_mols(gen_batch)
        return gen_mols
