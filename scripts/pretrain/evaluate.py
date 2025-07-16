import argparse
import math
from pathlib import Path

import lightning as L
import torch
from omegaconf import OmegaConf
from rdkit import Chem
from tqdm import tqdm

import cgflow.scriptutil as util
from cgflow.buildutil import build_cfm, build_test_dm
from cgflow.models.utils import mol_from_tensor, mols_from_batch
from cgflow.util.dataclasses import LigandBatch


def sync_config(config, model_cfg):
    # unfreeze structure
    OmegaConf.set_struct(config, False)
    OmegaConf.set_struct(model_cfg, False)

    # sync config
    config.transform = model_cfg.transform
    config.prior_dist = model_cfg.prior_dist
    config.interpolant = model_cfg.interpolant

    # freeze structure
    OmegaConf.set_struct(config, True)
    OmegaConf.set_struct(model_cfg, True)


def main(
    checkpoint_path: str,
    config,
    save_dir: str | Path,
    sampling_steps: int = 50,
    sampling_strategy: str = "linear",
):
    torch.set_float32_matmul_precision("high")

    L.seed_everything(12345)
    util.disable_lib_stdout()
    util.configure_fs()

    # create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = OmegaConf.create(checkpoint["hyper_parameters"])
    sync_config(config, model_cfg)

    # load datamodule
    print("Loading datamodule...")
    dm = build_test_dm(config)
    assert dm.test_dataset is not None, "Train dataset is None. Check your datamodule configuration."
    dataloader = dm.test_dataloader()
    print("Datamodule complete.")

    # build model
    print("Building cgflow model...")
    # here we can set metrics
    model_cfg.cfm.use_energy_metric = True
    model_cfg.cfm.use_complex_metric = False

    # build model
    model = build_cfm(model_cfg, dm.test_dataset.interpolant)
    model = model.half()
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to("cuda")
    model.eval()
    print("Model complete.")

    print("Evaluationg model on test dataset...")
    num_batches = math.ceil(len(dataloader.dataset) / dataloader.batch_size)
    with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        for batch_idx, batch in tqdm(enumerate(dataloader), total=num_batches, desc="Evaluating"):
            # TODO: is there any better way to save result?

            # flow matching sampling
            trajectory = model.run_reconstruct(batch, sampling_steps, sampling_strategy)
            traj_mols = [(mol_from_tensor(curr[0]), mol_from_tensor(pred[0])) for curr, pred in trajectory]
            true_mols = mols_from_batch(LigandBatch(**batch["ligand_mol"]))
            recon_mols = mols_from_batch(trajectory[-1][0])

            # caclulate metrics
            model.conf_metrics.update(recon_mols, true_mols)
            if model.energy_metrics:
                model.energy_metrics.update(recon_mols)
            if model.complex_metrics:
                model.complex_metrics.update(recon_mols, batch["pocket_raw"])

            if False:
                # save trajectory for the first molecule of the batch
                w1 = Chem.SDWriter(str(save_dir / f"sample{batch_idx}_true.sdf"))
                w2 = Chem.SDWriter(str(save_dir / f"sample{batch_idx}_state.sdf"))
                w3 = Chem.SDWriter(str(save_dir / f"sample{batch_idx}_pred.sdf"))
                w1.write(true_mols[0])
                for curr_mols, pred_mols in traj_mols:
                    w2.write(curr_mols)
                    w3.write(pred_mols)
                w1.close()
                w2.close()
                w3.close()

                # early stop
                if batch_idx == 20:
                    break
    print("Evaluation complete.")

    print()
    print("Conformer Metrics")
    print("-----------------")
    conf_metrics_results = model.conf_metrics.compute()
    for key, value in conf_metrics_results.items():
        print(f"{key}: {value:.4f}")

    if model.energy_metrics:
        print()
        print("Energy Metrics")
        print("--------------")
        energy_metrics_results = model.energy_metrics.compute()
        for key, value in energy_metrics_results.items():
            print(f"{key}: {value:.4f}")

    if model.complex_metrics:
        print()
        print("Complex Metrics")
        print("--------------")
        complex_metrics_results = model.complex_metrics.compute()
        for key, value in complex_metrics_results.items():
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Setup args
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--config", type=str, default="./configs/cgflow/test.yaml")
    parser.add_argument("--save_dir", type=str, default="./result/eval/")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--strategy", type=str, default="linear")
    # TODO: add parameters

    args = parser.parse_args()
    config = util.load_config(args.config)
    main(args.checkpoint, config, args.save_dir, args.steps, args.strategy)
