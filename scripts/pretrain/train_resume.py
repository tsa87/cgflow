import sys
from pathlib import Path

import lightning as L
import torch
from omegaconf import OmegaConf

import cgflow.scriptutil as util
from cgflow.buildutil import build_cfm, build_dm, build_trainer


def print_rank0(*args, rank: int = 0):
    if rank == 0:
        print(*args)


def main(checkpoint_path: str | Path):
    torch.set_float32_matmul_precision("high")
    L.seed_everything(12345)
    util.disable_lib_stdout()
    util.configure_fs()

    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["hyper_parameters"])

    # construct trainer
    trainer = build_trainer(config.trainer)
    rank = trainer.global_rank

    print_rank0("Arguments:", rank=rank)
    print_rank0(OmegaConf.to_yaml(config), rank=rank)

    print_rank0("Loading datamodule...", rank=rank)
    dm = build_dm(config)
    assert dm.train_dataset is not None, "Train dataset is None. Check your datamodule configuration."
    print_rank0("Datamodule complete.", rank=rank)

    print_rank0("Building cgflow model...", rank=rank)
    model = build_cfm(config, dm.train_dataset.interpolant)
    print_rank0(f"Using CFM class {model.__class__.__name__}", rank=rank)
    print_rank0("Model complete.", rank=rank)

    print_rank0("Resumming model training...", rank=rank)
    trainer.fit(model, datamodule=dm, ckpt_path=checkpoint_path)
    print_rank0("Training complete.", rank=rank)


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python train_resume.py <checkpoint_path>"
    ckpt_path = sys.argv[1]
    main(ckpt_path)
