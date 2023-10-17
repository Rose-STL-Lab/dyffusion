from __future__ import annotations

import os
from functools import partial
from os.path import join
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import dask
import numpy as np
import xarray as xr
from einops import rearrange

from src.datamodules.abstract_datamodule import BaseDataModule
from src.datamodules.torch_datasets import MyTensorDataset
from src.utilities.utils import get_logger, raise_error_if_invalid_type, raise_error_if_invalid_value


log = get_logger(__name__)


def drop_lat_lon_info(ds, time_slice=None):
    """Drop any geographical metadata for lat/lon so that xarrays are
    concatenated along example/grid-box instead of lat/lon dim."""
    if time_slice is not None:
        ds = ds.sel(time=time_slice)
    return ds.assign_coords(lat=np.arange(ds.sizes["lat"]), lon=np.arange(ds.sizes["lon"]))


def get_name_for_boxes(boxes: List[int]):
    if boxes == [84, 85, 86, 87, 88, 89, 108, 109, 110, 111, 112]:
        return "Pacific"
    else:
        return ",".join([str(b) for b in boxes])


class OISSTv2DataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        boxes: Union[List, str] = "all",
        stack_boxes_to_batch_dim: bool = True,
        validation_boxes: Union[List, str] = "all",
        predict_boxes: Union[List, str] = "all",
        predict_slice: Optional[slice] = slice("2020-12-01", "2020-12-31"),
        train_start_date: str | int = None,
        box_size: int = 60,
        window: int = 1,
        horizon: int = 1,
        prediction_horizon: int = None,  # None means use horizon
        multi_horizon: bool = False,
        pixelwise_normalization: bool = True,
        save_and_load_as_numpy: bool = False,
        **kwargs,
    ):
        raise_error_if_invalid_type(data_dir, possible_types=[str], name="data_dir")
        raise_error_if_invalid_value(pixelwise_normalization, [True], name="pixelwise_normalization")
        raise_error_if_invalid_value(box_size, [60], name="box_size")
        if "oisst" not in data_dir:
            for name in ["oisstv2-daily", "oisstv2"]:
                if os.path.isdir(join(data_dir, name)):
                    data_dir = join(data_dir, name)
                    break
        if os.path.isdir(join(data_dir, f"subregion-{box_size}x{box_size}boxes-pixelwise_stats")):
            data_dir = join(data_dir, f"subregion-{box_size}x{box_size}boxes-pixelwise_stats")
        super().__init__(data_dir=data_dir, **kwargs)
        self.save_hyperparameters()
        # Set the temporal slices for the train, val, and test sets
        if isinstance(train_start_date, int):
            assert 1980 <= train_start_date <= 2018, f"Invalid train_start_date: {train_start_date}"
            train_start_date = f"{train_start_date}-01-01"
        self.train_slice = slice(train_start_date, "2018-12-31")
        self.val_slice = slice("2019-01-01", "2019-12-31")
        self.test_slice = slice("2020-01-01", "2020-12-31")  # slice("2020-01-01", "2021-12-31")
        self.stage_to_slice = {
            "fit": slice(self.train_slice.start, self.val_slice.stop),
            "validate": self.val_slice,
            "test": self.test_slice,
            "predict": predict_slice,
            None: None,
        }
        log.info(f"Using OISSTv2 data directory: {self.hparams.data_dir}")
        if save_and_load_as_numpy:
            self.numpy_dir = join(data_dir, "numpy")
            os.makedirs(self.numpy_dir, exist_ok=True)  # create the directory if it doesn't exist

    @property
    def dataset_identifier(self) -> str:
        boxes_name = get_name_for_boxes(self.hparams.boxes)
        iden = f"OISSTv2_{boxes_name}_horizon{self.hparams.horizon}"
        if self.hparams.stack_boxes_to_batch_dim:
            iden += "_batch_stacked"
        if self.hparams.multi_horizon:
            iden += "_multi_horizon"
        return iden

    def _get_numpy_filename(self, stage: str):
        split = "train" if stage in ["fit", None] else stage
        if stage == "predict":
            return None
        fname = join(self.numpy_dir, f"{self.dataset_identifier}_{split}")
        if os.path.isfile(fname + ".npy"):
            return fname + ".npy"
        elif os.path.isfile(fname + ".npz"):
            return fname + ".npz"
        return None

    def load_xarray_ds(self, stage: str) -> bool:
        b1 = not self.hparams.save_and_load_as_numpy
        return b1 or self._get_numpy_filename(stage) is None or stage == "predict"

    def save_numpy_arrays(self, numpy_arrays: Dict[str, np.ndarray], split: str):
        fname = join(self.numpy_dir, f"{self.dataset_identifier}_{split}")
        log.info(f"Saving numpy arrays for {split} split to {fname}")
        np.savez_compressed(fname, **numpy_arrays)

    def get_ds_xarray_or_numpy(self, split: str, time_slice) -> Union[xr.DataArray, Dict[str, np.ndarray]]:
        if self.load_xarray_ds(split):
            preprocess = partial(drop_lat_lon_info, time_slice=self.stage_to_slice[split])
            if split == "predict":
                glob_pattern = self.get_glob_pattern(self.hparams.predict_boxes)
                log.info(f"Using data from {self.hparams.predict_boxes} boxes for prediction")
            elif "val" in split:
                glob_pattern = self.get_glob_pattern(self.hparams.validation_boxes)
            else:
                glob_pattern = self.get_glob_pattern(self.hparams.boxes)

            log.info(f" Using data from {self.n_boxes} boxes for ``{split}`` split.")
            with dask.config.set(**{"array.slicing.split_large_chunks": False}):
                try:
                    data = xr.open_mfdataset(
                        glob_pattern, combine="nested", concat_dim="grid_box", preprocess=preprocess
                    ).sst
                except OSError as e:
                    # Raise more informative error message
                    raise ValueError(
                        f"Could not open OISSTv2 data files from {glob_pattern}. "
                        f"Check that the data directory is correct: {self.hparams.data_dir}"
                    ) from e
            return data.sel(time=time_slice)
        else:
            log.info(f"Loading data from numpy file {self._get_numpy_filename(split)}")
            fname = self._get_numpy_filename(split)
            assert fname is not None, f"Could not find numpy file for split {split}"
            npz_file = np.load(fname, allow_pickle=False)
            # print(f'Keys in npz file: {list(npz_file.keys())}, files: {npz_file.files}')
            return {k: npz_file[k] for k in npz_file.files}

    def get_horizon(self, split: str):
        if split in ["predict", "test"]:
            return self.hparams.prediction_horizon or self.hparams.horizon
        else:
            return self.hparams.horizon

    def _check_args(self):
        boxes = self.hparams.boxes
        h = self.hparams.horizon
        w = self.hparams.window
        assert isinstance(h, list) or h > 0, f"horizon must be > 0 or a list, but is {h}"
        assert w > 0, f"window must be > 0, but is {w}"
        assert self.hparams.box_size > 0, f"box_size must be > 0, but is {self.hparams.box_size}"
        assert isinstance(boxes, Sequence) or boxes in [
            "all"
        ], f"boxes must be a list or 'all', but is {self.hparams.boxes}"

    def get_glob_pattern(self, boxes: Union[List, str] = "all"):
        ddir = Path(self.hparams.data_dir)
        if isinstance(boxes, Sequence) and boxes != "all":
            self.n_boxes = len(boxes)
            return [ddir / f"sst.day.mean.box{b}.nc" for b in boxes]
        elif boxes == "all":
            # compute the number of boxes
            self.n_boxes = len(list(ddir.glob("sst.day.mean.box*.nc")))
            return str(ddir / "sst.day.mean.box*.nc")  # os.listdir(self.hparams.data_dir)
        else:
            raise ValueError(f"Unknown value for boxes: {boxes}")

    def update_predict_data(
        self, boxes: Union[List, str] = "all", predict_slice: Optional[slice] = slice("2020-12-01", "2020-12-31")
    ):
        self.hparams.predict_boxes = boxes
        self.stage_to_slice["predict"] = predict_slice

    def create_and_set_dataset(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """Create a torch dataset from the given xarray DataArray and return it."""
        if self.hparams.multi_horizon:
            return self.create_and_set_dataset_multi_horizon(*args, **kwargs)
        else:
            return self.create_and_set_dataset_single_horizon(*args, **kwargs)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""
        # Read all boxes into a single xarray dataset (slice out all innecessary time steps)
        # Set the correct tensor datasets for the train, val, and test sets
        ds_train = self.get_ds_xarray_or_numpy("fit", self.train_slice) if stage in ["fit", None] else None
        ds_val = (
            self.get_ds_xarray_or_numpy("validate", self.val_slice) if stage in ["fit", "validate", None] else None
        )
        ds_test = self.get_ds_xarray_or_numpy("test", self.test_slice) if stage in ["test", None] else None
        ds_predict = (
            self.get_ds_xarray_or_numpy("predict", self.stage_to_slice["predict"]) if stage == "predict" else None
        )
        ds_splits = {"train": ds_train, "val": ds_val, "test": ds_test, "predict": ds_predict}
        for split, split_ds in ds_splits.items():
            if split_ds is None:
                continue

            if isinstance(split_ds, xr.DataArray):
                # Create the numpy arrays from the xarray dataset
                numpy_tensors = self.create_and_set_dataset(split, split_ds)

                # Save the numpy tensors to disk (if requested)
                if self.hparams.save_and_load_as_numpy:
                    self.save_numpy_arrays(numpy_tensors, split)
            else:
                # Alternatively, load the numpy arrays from disk (if requested)
                numpy_tensors = split_ds

            # Create the pytorch tensor dataset
            tensor_ds = MyTensorDataset(numpy_tensors, dataset_id=split)
            # Save the tensor dataset to self._data_{split}
            setattr(self, f"_data_{split}", tensor_ds)
            assert getattr(self, f"_data_{split}") is not None, f"Could not create {split} dataset"

        # Print sizes of the datasets (how many examples)
        self.print_data_sizes(stage)

    def create_and_set_dataset_single_horizon(self, split: str, dataset: xr.DataArray) -> Dict[str, np.ndarray]:
        """Create a torch dataset from the given xarray DataArray and return it."""
        dynamics = self.create_and_set_dataset_multi_horizon(split, dataset)["dynamics"]
        window, horizon = self.hparams.window, self.get_horizon(split)
        assert dynamics.shape[1] == window + horizon, f"Expected dynamics to have shape (b, {window + horizon}, ...)"
        inputs = dynamics[:, :window, ...]
        targets = dynamics[:, -1, ...]
        return {"inputs": inputs, "targets": targets}
        # Split ds into input and target (which is horizon time steps ahead of input, X)
        # X = dataset.isel(time=slice(None, -self.hparams.horizon))
        # Y = dataset.isel(time=slice(self.hparams.horizon, None))
        # # X and Y are 4D tensors with dimensions (grid-box, time, lat, lon)
        #
        # if self.hparams.stack_boxes_to_batch_dim:
        #     # Merge the time and grid_box dimensions into a single example dimension, and reshape
        #     X = X.stack(example=('time', 'grid_box')).transpose('example', 'lat', 'lon').values
        #     Y = Y.stack(example=('time', 'grid_box')).transpose('example', 'lat', 'lon').values
        #     # X and Y are now 3D tensors with dimensions (example, lat, lon)
        #     channel_dim = 1
        # else:
        #     X = X.transpose('time', 'grid_box', 'lat', 'lon').values
        #     Y = Y.transpose('time', 'grid_box', 'lat', 'lon').values
        #     # X and Y are now 4D tensors with dimensions (time, grid-box, lat, lon)
        #     channel_dim = 2
        #
        # # Add dummy channel dimension to first axis (1 channel, since we have only one variable, SST)
        # X, Y = np.expand_dims(X, axis=channel_dim), np.expand_dims(Y, axis=channel_dim)
        # # X and Y are now 4D tensors with dimensions (example, channel, lat, lon), where channel=1
        # return [X, Y]

    def create_and_set_dataset_multi_horizon(self, split: str, dataset: xr.DataArray):
        """Create a torch dataset from the given xarray dataset and return it."""
        # dataset is 4D tensor with dimensions (grid-box, time, lat, lon)
        # Create a tensor, X, of shape (batch-dim, horizon, lat, lon),
        # where each X[i] is a temporal sequence of horizon time steps
        window, horizon = self.hparams.window, self.get_horizon(split)
        dataset = dataset.transpose("time", "grid_box", "lat", "lon").values  # (time, grid-box, lat, lon)

        time_len = dataset.shape[0] - horizon - window + 1  # number of examples per grid-box
        # To save memory, we create the dataset through sliding window views

        X = np.lib.stride_tricks.sliding_window_view(dataset, time_len, axis=0)
        X = rearrange(X, "dynamics gb lat lon time -> (time gb) dynamics 1 lat lon")
        # print(f"X.shape = {X.shape}")   # e.g. (148599, 6, 1, 60, 60)  for horizon=5
        # see tests/test_windowed_data_loading_correctness.py for equivalent code to the above!
        # X is now 4D tensor with dimensions (example, dynamics, lat, lon)
        return {"dynamics": X}
