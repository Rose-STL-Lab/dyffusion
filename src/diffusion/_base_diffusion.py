from __future__ import annotations

import inspect
from abc import abstractmethod
from typing import Any

import torch
from torch import Tensor

from src.models._base_model import BaseModel


class BaseDiffusion(BaseModel):
    def __init__(
        self, model: BaseModel, timesteps: int, sampling_timesteps: int = None, sampling_schedule=None, **kwargs
    ):
        signature = inspect.signature(BaseModel.__init__).parameters
        base_kwargs = {k: model.hparams.get(k) for k in signature if k in model.hparams}
        base_kwargs.update(kwargs)  # override base_kwargs with kwargs
        super().__init__(**kwargs)
        if model is None:
            raise ValueError(
                "Arg ``model`` is missing..." " Please provide a backbone model for the diffusion model (e.g. a Unet)"
            )
        self.save_hyperparameters(ignore=["model"])
        # self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.model = model

        self.spatial_shape = model.spatial_shape
        self.num_input_channels = model.num_input_channels
        self.num_output_channels = model.num_output_channels
        self.num_conditional_channels = model.num_conditional_channels
        # self.num_timesteps = int(timesteps)

        # if hasattr(model, 'example_input_array'):
        #     self.example_input_array = model.example_input_array

    @property
    def short_description(self) -> str:
        name = super().short_description
        name += f" (timesteps={self.num_timesteps})"
        return name

    def sample(self, condition=None, num_samples=1, **kwargs):
        # sample from the model
        raise NotImplementedError()

    def predict_forward(self, inputs, condition=None, metadata: Any = None, **kwargs):
        channel_dim = 1
        p_losses_args = inspect.signature(self.p_losses).parameters.keys()
        if inputs is not None and condition is not None:
            if "static_condition" in p_losses_args:
                kwargs["static_condition"] = condition
                inital_condition = inputs
            else:
                # Concatenate the "inputs" and the condition along the channel dimension as conditioning
                try:
                    inital_condition = torch.cat([inputs, condition], dim=channel_dim)
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Could not concatenate the inputs (shape={inputs.shape}) and the condition "
                        f"(shape={condition.shape}) along the channel dimension (dim={channel_dim})"
                        f" due to the following error:\n{e}"
                    )
        else:  # if inputs is not None:
            inital_condition = inputs

        return self.sample(inital_condition, **kwargs)

    @abstractmethod
    def p_losses(self, targets: Tensor, condition: Tensor = None, t: Tensor = None, **kwargs):
        """Compute the loss for the given targets and condition.

        Args:
            targets (Tensor): Target data tensor of shape :math:`(B, C_{out}, *)`
            condition (Tensor): Condition data tensor of shape :math:`(B, C_{in}, *)`
            t (Tensor): Timestep of shape :math:`(B,)`
        """
        raise NotImplementedError()

    def forward(
        self,
        inputs: Tensor,
        targets: Tensor = None,
        condition: Tensor = None,
        time: Tensor = None,
    ):
        b, c, h, w = targets.shape if targets is not None else inputs.shape
        if time is not None:
            t = time
        else:
            t = torch.randint(0, self.num_timesteps, (b,), device=self.device, dtype=torch.long)

        p_losses_args = inspect.signature(self.p_losses).parameters.keys()
        kwargs = {}
        if "static_condition" in p_losses_args:
            # method handles it internally
            kwargs["static_condition"] = condition
            kwargs["condition"] = inputs
        elif condition is None:
            kwargs["condition"] = inputs
        else:
            channel_dim = 1
            kwargs["condition"] = torch.cat([inputs, condition], dim=channel_dim)

        return self.p_losses(targets, t=t, **kwargs)

    def get_loss(self, inputs: Tensor, targets: Tensor, metadata: Any = None, **kwargs):
        """Get the loss for the given inputs and targets.

        Args:
            inputs (Tensor): Input data tensor of shape :math:`(B, C_{in}, *)`
            targets (Tensor): Target data tensor of shape :math:`(B, C_{out}, *)`
            metadata (Any): Optional metadata
        """
        results = self(inputs, targets, **kwargs)
        return results
