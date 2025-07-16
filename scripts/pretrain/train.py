import argparse
import os

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

import cgflow.scriptutil as util


def parse_args() -> DictConfig:
    parser = argparse.ArgumentParser()

    # Setup args
    # TODO: add parameters
    parser.add_argument("--config", type=str, default="./configs/cgflow/train.yaml")
    # save path
    parser.add_argument("--name", type=str)
    parser.add_argument("--save_dir", type=str)
    # model training
    parser.add_argument("--num_gpus", type=int)
    parser.add_argument("--batch_cost", type=int)
    parser.add_argument("--num_workers", type=int)
    args = parser.parse_args()

    config = util.load_config(args.config)

    # Only override if args are provided (not None)
    if args.name is not None:
        config.trainer.wandb_name = args.name

    if args.save_dir is not None:
        save_name = args.name if args.name is not None else "default"
        config.trainer.save_dir = os.path.join(args.save_dir, save_name)

    if args.num_gpus is not None:
        config.trainer.num_gpus = args.num_gpus

    if args.batch_cost is not None:
        config.datamodule.batch_cost = args.batch_cost

    if args.num_workers is not None:
        config.datamodule.num_workers = args.num_workers

    # NOTE: sync configs
    config.ligand_decoder.d_pocket_inv = config.pocket_encoder.d_inv
    config.ligand_decoder.d_pocket_equi = config.pocket_encoder.d_equi

    # TODO: check the config is valid
    pass

    return config


def print_rank0(*args, rank: int = 0):
    if rank == 0:
        print(*args)


def main(config):
    from cgflow.buildutil import build_cfm, build_dm, build_trainer

    torch.set_float32_matmul_precision("high")

    L.seed_everything(12345)
    util.disable_lib_stdout()
    util.configure_fs()

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
    print_rank0(f"Model complete. CFM class {model.__class__.__name__}", rank=rank)

    print_rank0("Fitting datamodule to model...", rank=rank)
    trainer.fit(model, datamodule=dm)
    print_rank0("Training complete.", rank=rank)


if __name__ == "__main__":
    config = parse_args()
    main(config)
