from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Optional, Sequence, Tuple, Union

from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor

from src.utilities.utils import (
    disable_inference_dropout,
    enable_inference_dropout,
    get_logger,
    get_loss,
)


class BaseModel(LightningModule):
    r"""This is a template base class, that should be inherited by any stand-alone ML model.
    Methods that need to be implemented by your concrete ML model (just as if you would define a :class:`torch.nn.Module`):
        - :func:`__init__`
        - :func:`forward`

    The other methods may be overridden as needed.
    It is recommended to define the attribute
        >>> self.example_input_array = torch.randn(<YourModelInputShape>)  # batch dimension can be anything, e.g. 7


    .. note::
        Please use the function :func:`predict` at inference time for a given input tensor, as it postprocesses the
        raw predictions from the function :func:`raw_predict` (or model.forward or model())!

    Args:
        name (str): optional string with a name for the model
        verbose (bool): Whether to print/log or not

    Read the docs regarding LightningModule for more information:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        num_input_channels: int = None,
        num_output_channels: int = None,
        num_conditional_channels: int = 0,
        spatial_shape: Union[Sequence[int], int] = None,
        loss_function: str = "mean_squared_error",
        datamodule_config: Optional[DictConfig] = None,
        name: str = "",
        verbose: bool = True,
    ):
        super().__init__()
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.monitor
        self.save_hyperparameters(ignore=["verbose", "model"])
        # Get a logger
        self.log_text = get_logger(name=self.__class__.__name__ if name == "" else name)
        self.name = name
        self.verbose = verbose
        if not self.verbose:  # turn off info level logging
            self.log_text.setLevel(logging.WARN)

        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.num_conditional_channels = num_conditional_channels
        self.spatial_shape = spatial_shape
        self.datamodule_config = datamodule_config

        # Get the loss function
        self.criterion = get_loss(loss_function)
        self._channel_dim = None
        self.ema_scope = None  # EMA scope for the model. May be set by the BaseExperiment instance
        # self._parent_module = None    # BaseExperiment instance (only needed for edge cases)

    @property
    def short_description(self) -> str:
        return self.name if self.name else self.__class__.__name__

    def get_parameters(self) -> list:
        """Return the parameters for the optimizer."""
        return list(self.parameters())

    @property
    def num_params(self):
        """Returns the number of parameters in the model"""
        return sum(p.numel() for p in self.get_parameters() if p.requires_grad)

    @property
    def channel_dim(self):
        if self._channel_dim is None:
            self._channel_dim = 1
        return self._channel_dim

    def forward(self, X: Tensor, condition: Tensor = None, **kwargs):
        r"""Standard ML model forward pass (to be implemented by the specific ML model).

        Args:
            X (Tensor): Input data tensor of shape :math:`(B, *, C_{in})`
        Shapes:
            - Input: :math:`(B, *, C_{in})`,

            where :math:`B` is the batch size, :math:`*` is the spatial dimension(s) of the data,
            and :math:`C_{in}` is the number of input features/channels.
        """
        raise NotImplementedError("Base model is an abstract class!")

    def get_loss(
        self,
        inputs: Tensor,
        targets: Tensor,
        condition: Tensor = None,
        metadata: Any = None,
        predictions_mask: Optional[Tensor] = None,
        return_predictions: bool = False,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Get the loss for the given inputs and targets.

        Args:
            inputs (Tensor): Input data tensor of shape :math:`(B, *, C_{in})`
            targets (Tensor): Target data tensor of shape :math:`(B, *, C_{out})`
            condition (Tensor): Conditional data tensor of shape :math:`(B, *, C_{cond})`
            metadata (Any): Optional metadata
            predictions_mask (Tensor): Mask for the predictions, before computing the loss. Default: None (no mask)
            return_predictions (bool): Whether to return the predictions or not. Default: False.
                                    Note: this will return all the predictions, not just the masked ones (if any).
        """
        # Predict
        predictions = self(inputs, condition=condition, **kwargs)
        # Compute loss
        if predictions_mask is not None:
            loss = self.criterion(predictions[predictions_mask], targets)
        else:
            loss = self.criterion(predictions, targets)
        if return_predictions:
            return loss, predictions
        return loss

    def predict_forward(self, inputs: Tensor, metadata: Any = None, **kwargs):
        """Forward pass for prediction. Usually the same as the forward pass,
        but can be different for some models (e.g. sampling in probabilistic models).
        """
        y = self(inputs, **kwargs)
        return y

    # Auxiliary methods
    @contextmanager
    def inference_dropout_scope(self, condition: bool, context=None):
        assert isinstance(condition, bool), f"Condition must be a boolean, got {condition}"
        if condition:
            enable_inference_dropout(self)
            if context is not None:
                print(f"{context}: Switched to enabled inference dropout")
        try:
            yield None
        finally:
            if condition:
                disable_inference_dropout(self)
                if context is not None:
                    print(f"{context}: Switched to disabled inference dropout")

    def enable_inference_dropout(self):
        """Set all dropout layers to training mode"""
        enable_inference_dropout(self)

    def disable_inference_dropout(self):
        """Set all dropout layers to eval mode"""
        disable_inference_dropout(self)

    def register_buffer_dummy(self, name, tensor, **kwargs):
        try:
            self.register_buffer(name, tensor, **kwargs)
        except TypeError:  # old pytorch versions do not have the arg 'persistent'
            self.register_buffer(name, tensor)
