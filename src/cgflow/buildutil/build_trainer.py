from dataclasses import dataclass
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy


@dataclass
class TrainerConfig:
    save_dir: str = "./result/"
    wandb_project: str | None = "cgflow"
    wandb_group: str | None = None
    wandb_name: str | None = None
    epoch: int = 10000
    log_every_n_steps = 50
    check_val_every_n_epoch: int = 10
    val_check_interval: float | None = None
    checkpoint_epochs: int = 10
    num_gpus: int = 1
    monitor: str = "val/rmsd"
    monitor_mode: str = "min"
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    precision: str | int = "16-mixed"


def build_trainer(config: TrainerConfig):
    epochs: int = config.epoch
    if config.wandb_project is None:
        logger = None
    else:
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        logger = WandbLogger(
            name=config.wandb_name,
            project=config.wandb_project,
            group=config.wandb_group,
            log_model=True,
            save_dir=config.save_dir,
        )
        # if config.num_gpus == 1:
        #     logger.experiment.config.update(config)  # error on multi-gpu

    # callback
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpointing = ModelCheckpoint(
        filename="epoch{epoch}-step{step}-rmsd{val/rmsd:.2f}",
        every_n_epochs=config.checkpoint_epochs,
        monitor=config.monitor,
        mode=config.monitor_mode,
        save_last=True,
        save_top_k=3,
        auto_insert_metric_name=False,
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=config.num_gpus,
        strategy="auto" if config.num_gpus == 1 else DDPStrategy(find_unused_parameters=True),
        min_epochs=epochs,
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=config.log_every_n_steps,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        val_check_interval=config.val_check_interval,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[lr_monitor, checkpointing],
        precision=config.precision,
        use_distributed_sampler=False,
    )
    return trainer
