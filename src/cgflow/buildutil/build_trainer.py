import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

import cgflow.scriptutil as util


# bfloat16 training produced significantly worse models than full so use default 16-bit instead
def get_precision(args):
    return "16-mixed"


def build_trainer(args):
    assert not args.trial_run
    epochs = 1 if args.trial_run else args.epochs
    log_steps = 1
    # HACK log_steps = 1 if args.trial_run else 50

    assert args.dataset in [
        "qm9",
        "geom-drugs",
        "plinder",
        "plinder-ligand",
        "zinc15m",
        "crossdock",
    ], f"Unknown dataset {args.dataset}"
    val_check_epochs = args.val_check_epochs

    project_name = f"{util.PROJECT_PREFIX}-{args.dataset}"
    precision = get_precision(args)

    print(f"Using precision '{precision}'")

    logger = WandbLogger(project=project_name,
                         save_dir="wandb",
                         log_model=True)
    if args.num_gpus == 1:
        logger.experiment.config.update(vars(args))  # error on multi-gpu

    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpointing = ModelCheckpoint(
        every_n_epochs=val_check_epochs,
        monitor=args.monitor,
        mode=args.monitor_mode,
        save_last=True,
    )

    # Overwrite if doing a trial run
    val_check_epochs = 1 if args.trial_run else val_check_epochs
    logger = None if args.trial_run else logger

    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        strategy='auto' if args.num_gpus == 1 else DDPStrategy(
            find_unused_parameters=True),
        min_epochs=epochs,
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=log_steps,
        accumulate_grad_batches=args.acc_batches,
        gradient_clip_val=args.gradient_clip_val,
        check_val_every_n_epoch=val_check_epochs,
        callbacks=[lr_monitor, checkpointing],
        precision=precision,
        use_distributed_sampler=False,
    )
    return trainer
