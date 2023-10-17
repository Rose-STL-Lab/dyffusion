from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pytorch_lightning as pl
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from src.utilities.utils import get_logger


log = get_logger(__name__)


class WatchModel(Callback):
    """
    Make wandb watch model at the beginning of the run.
    This will log the gradients of the model (as a histogram for each or some weights updates).
    """

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log_type = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger: WandbLogger = get_wandb_logger(trainer=trainer)
        try:
            logger.watch(model=trainer.model, log=self.log_type, log_freq=self.log_freq, log_graph=True)
        except TypeError:
            log.info(
                f" Pytorch-lightning/Wandb version seems to be too old to support 'log_graph' arg in wandb.watch(.)"
                f" Wandb version={wandb.__version__}"
            )
            wandb.watch(models=trainer.model, log=self.log_type, log_freq=self.log_freq)  # , log_graph=True)


class SummarizeBestValMetric(Callback):
    """Make wandb log in run.summary the best achieved monitored val_metric as opposed to the last"""

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger: WandbLogger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment
        # When using DDP multi-gpu training, one usually needs to get the actual model by .module, and
        # trainer.model.module.module will be the same as pl_module

        model = pl_module  # .module if isinstance(trainer.model, DistributedDataParallel) else pl_module
        experiment.define_metric(model.monitor, summary=model.hparams.mode)
        experiment.define_metric(f"{model.monitor}_epoch", summary=model.hparams.mode)
        # Store the maximum epoch at all times
        # The following leads to a weird error in wandb file:
        #   /opt/conda/lib/python3.8/site-packages/wandb/sdk/internal/handler.py
        # where, this is an example print out before the problematic line 62:
        # print(target, v, key_list) --> 0 0 ('epoch', 'max')
        # experiment.define_metric('epoch', summary='max')


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = True):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        logger.experiment.log_artifact(ckpts)


# def save_arrays_as_line_plot(
#         lightning_module: pl.LightningModule,
#         arrays: Dict[str, np.ndarray],
#         wandb_metric_key: str,
# ):
#     # we want to zip the arrays together into the columns of a table
#     data = [row for row in zip(*arrays.values())]
#     table = wandb.Table(data=data, columns=list(arrays.keys()))


def save_arrays_as_line_plot(
    lightning_module: pl.LightningModule,
    x_array: np.ndarray,
    key_to_array: Dict[str, np.ndarray],
    wandb_key_stem: str,
    x_label: str = "x",
    log_as_step: bool = True,
    log_as_table: bool = False,
    update_summary: bool = True,
):
    # we want to zip the arrays together into the columns of a table
    step = lightning_module.trainer.global_step
    key_to_array_actual = {f"{wandb_key_stem}/{key}".replace("//", "/"): y for key, y in key_to_array.items()}
    if log_as_table:
        if not hasattr(lightning_module, "logger") and hasattr(lightning_module.logger, "experiment"):
            return
        log_dict = dict()
        for wandb_key, y_array in key_to_array_actual.items():
            y_label = wandb_key.split("/")[-1]
            data = [[x, y] for x, y in zip(x_array, y_array)]
            table = wandb.Table(data=data, columns=[x_label + "_x", y_label])

            log_dict["plot_" + wandb_key] = wandb.plot.line(table, x_label + "_x", y_label, title=wandb_key)
        lightning_module.logger.experiment.log(log_dict, step=step)

    if log_as_step:
        # define our custom x axis metric
        wandb.define_metric(x_label)

        # define which metrics will be plotted against it
        for wandb_key, y_array in key_to_array_actual.items():
            wandb.define_metric(wandb_key, step_metric=x_label)

        # now zip the arrays together, and log each step together
        for i, (x, *y) in enumerate(zip(x_array, *key_to_array.values()), start=1):
            lightning_module.logger.experiment.log(
                {x_label: x, **{key: y for key, y in zip(key_to_array_actual.keys(), y)}}
            )
            if update_summary:
                lightning_module.logger.experiment.summary.update(
                    {
                        f"{wandb_key_stem}/t{i}/{key_name}".replace("//", "/"): y_value
                        for key_name, y_value in zip(key_to_array.keys(), y)
                    }
                )

        # y_label2 = wandb_metric_stem if y_label in wandb_metric_stem else f'{wandb_metric_stem}/{y_label}'
        # for x, y in zip(x_array, y_array):
        #     lightning_module.logger.experiment.log({y_label2: y, x_label: x})


class MyWandbLogger(pl.loggers.WandbLogger):
    """Same as pl.WandbLogger, but also saves the last checkpoint as 'last.ckpt' and uploads it to wandb."""

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        super().after_save_checkpoint(checkpoint_callback)
        self.save_last(checkpoint_callback)
        self.save_best(checkpoint_callback)

    @rank_zero_only
    def save_last(self, ckpt_cbk):
        if isinstance(ckpt_cbk, Sequence):
            ckpt_cbk = [c for c in ckpt_cbk if c.last_model_path]
            if len(ckpt_cbk) == 0:
                raise Exception("No checkpoint callback has a last_model_path attribute. Ckpt callback is: {ckpt_cbk}")
            ckpt_cbk = ckpt_cbk[0]

        last_ckpt = ckpt_cbk.last_model_path
        if self.save_last and last_ckpt:
            self.experiment.save(last_ckpt)  # , base_path=".")
            # print(f'saved last ckpt: {last_ckpt}')

    @rank_zero_only
    def save_best(self, ckpt_cbk):
        # Save best model
        if not isinstance(ckpt_cbk, Sequence):
            ckpt_cbk = [ckpt_cbk]

        for ckpt_cbk in ckpt_cbk:
            best_ckpt = ckpt_cbk.best_model_path
            if best_ckpt and os.path.isfile(best_ckpt):
                # copy best ckpt to a file called 'best.ckpt' and upload it to wandb
                monitor = ckpt_cbk.monitor.replace("/", "_")
                fname_wandb = f"best-{monitor}.ckpt"
                shutil.copyfile(best_ckpt, fname_wandb)
                self.experiment.save(fname_wandb)  # , base_path=".")
                # log.info(f"Wandb: Saved best ckpt '{best_ckpt}' as '{fname_wandb}'.")
                # log.info(f"Saved best ckpt to the wandb cloud as '{fname_wandb}'.")


def get_wandb_logger(trainer: Trainer) -> WandbLogger | MyWandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, (WandbLogger, MyWandbLogger)):
        return trainer.logger

    if isinstance(trainer.loggers, list):
        for logger in trainer.loggers:
            if isinstance(logger, (WandbLogger, MyWandbLogger)):
                return logger

    raise Exception("You are using wandb related callback, but WandbLogger was not found for some reason...")
