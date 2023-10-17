import multiprocessing
from typing import Any, List, Optional

import numpy as np
import pytorch_lightning as pl
import xarray as xr
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader

from src.datamodules.torch_datasets import MyTensorDataset
from src.utilities.utils import get_logger, raise_error_if_invalid_value


log = get_logger(__name__)


class BaseDataModule(pl.LightningDataModule):
    """
    ----------------------------------------------------------------------------------------------------------
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    _data_train: MyTensorDataset
    _data_val: MyTensorDataset
    _data_test: MyTensorDataset
    _data_predict: MyTensorDataset

    def __init__(
        self,
        data_dir: str,
        model_config: DictConfig = None,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        num_workers: int = -1,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        drop_last: bool = False,
        verbose: bool = True,
        seed_data: int = 43,
    ):
        """
        Args:
            data_dir (str):  A path to the data folder that contains the input and output files.
            batch_size (int): Batch size for the training dataloader
            eval_batch_size (int): Batch size for the test and validation dataloader's
            num_workers (int): Dataloader arg for higher efficiency (usually set to # of CPU cores).
                                Default: Set to -1 to use all available cores.
            pin_memory (bool): Dataloader arg for higher efficiency. Default: True
            drop_last (bool): Only for training data loading: Drop the last incomplete batch
                                when the dataset size is not divisible by the batch size. Default: False
        """
        super().__init__()
        # The following makes all args available as, e.g., self.hparams.batch_size
        self.save_hyperparameters(ignore=["model_config", "verbose"])
        self.model_config = model_config
        self.test_batch_size = eval_batch_size  # just for testing
        self._data_train = self._data_val = self._data_test = self._data_predict = None
        self._check_args()

    def _check_args(self):
        """Check if the arguments are valid."""
        pass

    def _concat_variables_into_channel_dim(self, data: xr.Dataset, variables: List[str], filename=None) -> np.ndarray:
        """Concatenate xarray variables into numpy channel dimension (last)."""
        data_all = []
        for var in variables:
            # Get the variable from the dataset (as numpy array, by selecting .values)
            var_data = data[var].values
            # add feature dimension (channel)
            var_data = np.expand_dims(var_data, axis=-1)
            # add to list of all variables
            data_all.append(var_data)

        # Concatenate all the variables into a single array along the last (channel/feature) dimension
        dataset = np.concatenate(data_all, axis=-1)
        assert dataset.shape[-1] == len(variables), "Number of variables does not match number of channels."
        return dataset

    def print_data_sizes(self, stage: str = None):
        """Print the sizes of the data."""
        if stage in ["fit", None]:
            log.info(f" Dataset sizes train: {len(self._data_train)}, val: {len(self._data_val)}")
        elif stage in ["test", None]:
            log.info(f" Dataset test size: {len(self._data_test)}")
        elif stage == "predict":
            log.info(f" Dataset predict size: {len(self._data_predict)}")

    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""
        raise_error_if_invalid_value(stage, ["fit", "validate", "test", "predict", None], "stage")

        if stage == "fit" or stage is None:
            self._data_train = ...  # get_tensor_dataset_from_numpy(X_train, Y_train, dataset_id='train')
        if stage in ["fit", "validate", None]:
            self._data_val = ...  # get_tensor_dataset_from_numpy(X_val, Y_val, dataset_id='val')
        if stage in ["test", None]:
            self._data_test = ...  # get_tensor_dataset_from_numpy(X_test, Y_test, dataset_id='test')
        if stage in ["predict"]:
            self._data_predict = ...
        raise NotImplementedError("This class is not implemented yet.")

    @property
    def num_workers(self) -> int:
        if self.hparams.num_workers == -1:
            return multiprocessing.cpu_count()
        return int(self.hparams.num_workers)

    def _shared_dataloader_kwargs(self) -> dict:
        return dict(
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self._data_train,
            batch_size=self.hparams.batch_size,
            drop_last=self.hparams.drop_last,  # drop last incomplete batch (only for training)
            shuffle=True,
            **self._shared_dataloader_kwargs(),
        )

    def _shared_eval_dataloader_kwargs(self) -> dict:
        return dict(**self._shared_dataloader_kwargs(), shuffle=False)

    def val_dataloader(self):
        return (
            DataLoader(
                dataset=self._data_val,
                batch_size=self.hparams.eval_batch_size,
                **self._shared_eval_dataloader_kwargs(),
            )
            if self._data_val is not None
            else None
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._data_test, batch_size=self.test_batch_size, **self._shared_eval_dataloader_kwargs()
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self._data_predict,
            batch_size=self.hparams.eval_batch_size,
            **self._shared_eval_dataloader_kwargs(),
        )

    def boundary_conditions(
        self,
        preds: Tensor,
        targets: Tensor,
        metadata,
        time: float = None,
    ) -> Tensor:
        """Return predictions that satisfy the boundary conditions for a given item (batch element)."""
        return preds

    def get_boundary_condition_kwargs(self, batch: Any, batch_idx: int, split: str) -> dict:
        return dict(t0=0.0, dt=1.0)
