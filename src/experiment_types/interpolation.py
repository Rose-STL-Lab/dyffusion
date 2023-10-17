from typing import Any, List

import numpy as np
import torch
import torchmetrics
from einops import rearrange
from torch import Tensor

from src.experiment_types._base_experiment import BaseExperiment


class InterpolationExperiment(BaseExperiment):
    r"""Base class for all interpolation experiments."""

    def __init__(self, stack_window_to_channel_dim: bool = True, **kwargs):
        super().__init__(**kwargs)
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.hidden_dims
        self.save_hyperparameters(ignore=["model"])
        assert self.horizon >= 2, "horizon must be >=2 for interpolation experiments"

    @property
    def horizon_range(self) -> List[int]:
        # h = horizon
        # We use timesteps  w-l+1, ..., w-1, w+h to predict timesteps w, ..., w+h-1
        # interpolate between step t=0 and t=horizon
        return list(np.arange(1, self.horizon))

    @property
    def true_horizon(self) -> int:
        return self.horizon

    @property
    def horizon_name(self) -> str:
        s = f"{self.true_horizon}h"
        return s

    @property
    def short_description(self) -> str:
        name = super().short_description
        name += f" (h={self.horizon_name})"
        return name

    @property
    def WANDB_LAST_SEP(self) -> str:
        return "/ipol/"

    def actual_num_input_channels(self, num_input_channels: int) -> int:
        if self.hparams.stack_window_to_channel_dim:
            return num_input_channels * self.window + num_input_channels
        return 2 * num_input_channels  # inputs and targets are concatenated

    # --------------------------------- Metrics
    def get_metrics(self, split: str, split_name: str, **kwargs) -> torch.nn.ModuleDict:
        metrics = {
            f"{split_name}/{self.horizon_name}_avg{self.WANDB_LAST_SEP}mse": torchmetrics.MeanSquaredError(
                squared=True
            )
        }
        for h in self.horizon_range:
            metrics[f"{split_name}/t{h}{self.WANDB_LAST_SEP}mse"] = torchmetrics.MeanSquaredError(squared=True)
        return torch.nn.ModuleDict(metrics)

    @property
    def default_monitor_metric(self) -> str:
        return f"val/{self.horizon_name}_avg{self.WANDB_LAST_SEP}mse"

    @torch.no_grad()
    def _evaluation_step(
        self,
        batch: Any,
        batch_idx: int,
        split: str,
        dataloader_idx: int = None,
        return_only_preds_and_targets: bool = False,
    ):
        log_dict = dict()
        compute_metrics = split != "predict"
        split_metrics = getattr(self, f"{split}_metrics") if compute_metrics else None
        dynamics = batch["dynamics"]  # dynamics is a (b, t, c, h, w) tensor

        return_dict = dict()
        avg_mse_key = f"{split}/{self.horizon_name}_avg{self.WANDB_LAST_SEP}mse"
        avg_mse_tracker = split_metrics[avg_mse_key] if compute_metrics else None

        inputs = self.get_evaluation_inputs(dynamics, split=split)
        extra_kwargs = {}
        for k, v in batch.items():
            if k != "dynamics":
                extra_kwargs[k] = self.get_ensemble_inputs(v, split=split, add_noise=False)

        for t_step in self.horizon_range:
            # dynamics[, self.window] is already the first target frame (t_step=1)
            targets = dynamics[:, self.window + t_step - 1, ...]  # (b, c, h, w)
            time = torch.full((inputs.shape[0],), t_step, device=self.device, dtype=torch.long)

            results = self.predict(inputs, time=time, **extra_kwargs)
            results["targets"] = targets
            preds = results["preds"]
            results = {f"t{t_step}_{k}": v for k, v in results.items()}

            if return_only_preds_and_targets:
                return_dict[f"t{t_step}_preds"] = preds
                return_dict[f"t{t_step}_targets"] = targets
            else:
                return_dict = {**return_dict, **results}
            if not compute_metrics:
                continue
            if self.use_ensemble_predictions(split):
                preds = preds.mean(dim=0)  # average over ensemble
            # Compute mse
            metric_name = f"{split}/t{t_step}{self.WANDB_LAST_SEP}mse"
            metric = split_metrics[metric_name]
            metric(preds, targets)  # compute metrics (need to be in separate line to the following line!)
            log_dict[metric_name] = metric

            # Add contribution to the average mse from this time step's MSE
            avg_mse_tracker(preds, targets)

        if compute_metrics:
            log_kwargs = dict()
            log_kwargs["sync_dist"] = True  # for DDP training
            # Log the average MSE
            log_dict[avg_mse_key] = avg_mse_tracker
            self.log_dict(log_dict, on_step=False, on_epoch=True, **log_kwargs)  # log metric objects

        return return_dict

    def get_inputs_from_dynamics(self, dynamics: Tensor, split: str, **kwargs) -> Tensor:
        """Get the inputs from the dynamics tensor.
        Since we are doing interpolation, this consists of the first window frames plus the last frame.
        """
        assert dynamics.shape[1] == self.window + self.horizon, "dynamics must have shape (b, t, c, h, w)"
        past_steps = dynamics[:, : self.window, ...]  # (b, window, c, lat, lon) at time 0
        last_step = dynamics[:, -1, ...]  # (b, c, lat, lon) at time t=window+horizon
        if self.hparams.stack_window_to_channel_dim:
            past_steps = rearrange(past_steps, "b window c lat lon -> b (window c) lat lon")
        else:
            last_step = last_step.unsqueeze(1)  # (b, 1, c, lat, lon)
        inputs = torch.cat([past_steps, last_step], dim=1)  # (b, window*c + c, lat, lon)
        return inputs

    def get_evaluation_inputs(self, dynamics: Tensor, split: str, **kwargs) -> Tensor:
        inputs = self.get_inputs_from_dynamics(dynamics, split)
        inputs = self.get_ensemble_inputs(inputs, split)
        return inputs

    # --------------------------------- Training
    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the loss for the given batch."""
        dynamics = batch["dynamics"]  # dynamics is a (b, t, c, h, w) tensor
        split = "train" if self.training else "val"
        inputs = self.get_inputs_from_dynamics(dynamics, split=split)  # (b, c, h, w) at time 0
        b = dynamics.shape[0]

        possible_times = torch.tensor(self.horizon_range, device=self.device, dtype=torch.long)  # (h,)
        # take random choice of time
        t = possible_times[torch.randint(len(possible_times), (b,), device=self.device, dtype=torch.long)]  # (b,)
        # t = torch.randint(start_t, max_t, (b,), device=self.device, dtype=torch.long)  # (b,)
        targets = dynamics[torch.arange(b), self.window + t - 1, ...]  # (b, c, h, w)
        # We use timesteps  w-l+1, ..., w-1, w+h to predict timesteps w, ..., w+h-1
        # so t=0 corresponds to interpolating w, t=1 to w+1, ..., t=h-1 to w+h-1

        loss = self.model.get_loss(
            inputs=inputs, targets=targets, time=t, **{k: v for k, v in batch.items() if k != "dynamics"}
        )  # function of BaseModel or BaseDiffusion classes
        return loss
