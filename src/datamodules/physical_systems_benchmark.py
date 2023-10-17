from __future__ import annotations

import math
import os
from os.path import join
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from einops import rearrange
from torch import Tensor

from src.datamodules.abstract_datamodule import BaseDataModule
from src.datamodules.datasets.physical_systems_benchmark import TrajectoryDataset
from src.datamodules.torch_datasets import MyTensorDataset
from src.utilities.utils import (
    get_logger,
    raise_error_if_invalid_type,
    raise_error_if_invalid_value,
    raise_if_invalid_shape,
)


log = get_logger(__name__)


class PhysicalSystemsBenchmarkDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        physical_system: str = "navier-stokes",
        window: int = 1,
        horizon: int = 1,
        prediction_horizon: int = None,  # None means use horizon
        multi_horizon: bool = False,
        num_test_obstacles: int = 1,
        test_out_of_distribution: bool = False,
        num_trajectories: int = None,  # None means all trajectories for training
        **kwargs,
    ):
        raise_error_if_invalid_type(data_dir, possible_types=[str], name="data_dir")
        if "nn-benchmark" not in data_dir:
            for sub_dir in ["physical-nn-benchmark", "nn-benchmark"]:
                if os.path.isdir(join(data_dir, sub_dir)):
                    data_dir = join(data_dir, sub_dir)
                    break
        super().__init__(data_dir=data_dir, **kwargs)
        self.save_hyperparameters()
        self.test_batch_size = 1  # to make sure that the test dataloader returns a single trajectory
        assert window == 1, "window > 1 is not supported yet for this data module."
        raise_error_if_invalid_value(
            physical_system, possible_values=["navier-stokes", "spring-mesh"], name="physical_system"
        )
        ood_infix = "outdist-" if test_out_of_distribution else ""
        if physical_system == "navier-stokes":
            _first_subdir = "navier-stokes-multi"
            assert num_test_obstacles in [1, 4], f"Invalid number of test obstacles {num_test_obstacles}"
            # ns-runs_eval-cors1-navier-stokes-n5-t65-n0_tagcors1_00001 (65, 9282, 2) (65,)
            # ns-runs_eval-cors16-navier-stokes-n5-t4-n0_tagcors16_00001 (4, 9282, 2) (4,)
            # ns-runs_eval-cors4-navier-stokes-n5-t16-n0_tagcors4_00001 (16, 9282, 2) (16,)
            # ns-runs_eval-outdist-cors1-navier-stokes-n5-t65-n0_tagcors1_00001 (65, 9282, 2) (65,)
            # ns-runs_eval-outdist-cors16-navier-stokes-n5-t4-n0_tagcors16_00001 (4, 9282, 2) (4,)
            # ns-runs_eval-outdist-cors4-navier-stokes-n5-t16-n0_tagcors4_00001 (16, 9282, 2) (16,)
            test_t = {1: 65, 4: 16, 16: 4}[num_test_obstacles]
            test_set_name = f"ns-runs_eval-{ood_infix}cors{num_test_obstacles}-navier-stokes-n5-t{test_t}-n0_tagcors{num_test_obstacles}_00001"
            self.subdirs = {
                "train": "ns-runs_train-navier-stokes-n100-t65-n0_00001",
                "val": "ns-runs_val-navier-stokes-n2-t65-n0_00001",
                "test": test_set_name,
            }
            self.subdirs["predict"] = self.subdirs["val"]
        elif physical_system == "spring-mesh":
            _first_subdir = "spring-mesh"
            self.subdirs = {
                "train": "springmesh-10-perturball-runs_train-spring-mesh-n100-t805-n0_00001",
                "val": "springmesh-10-perturball-runs_val-spring-mesh-n3-t805-n0_00001",
                "test": f"springmesh-10-perturball-runs_eval-{ood_infix}spring-mesh-n15-t805-n0_tagcors1_00001",
            }
            self.subdirs["predict"] = self.subdirs["val"]
        else:
            raise NotImplementedError(f"Physical system {physical_system} is not implemented yet.")

        # Check if data directory exists
        if not os.path.isdir(join(self.hparams.data_dir, _first_subdir)):
            if os.path.isdir(join(self.hparams.data_dir, "physical-nn-benchmark", _first_subdir)):
                _first_subdir = join("physical-nn-benchmark", _first_subdir)
        ddir = Path(self.hparams.data_dir)
        assert (
            ddir.is_dir()
        ), f"Could not find data directory {ddir}. Is the data directory correct?. Please specify the data directory using the ``datamodule.data_dir`` option."
        assert (
            ddir / _first_subdir
        ).is_dir(), f"Could not find data directory {ddir / _first_subdir}. Is the data directory correct?. Please specify the data directory using the ``datamodule.data_dir`` option."
        self._first_subdir = join(_first_subdir, "run", "data_gen")
        assert os.path.isdir(
            join(self.hparams.data_dir, self._first_subdir)
        ), f"Could not find data directory {self._first_subdir} in {self.hparams.data_dir}. Did you download the data?"
        log.info(f"Using data directory: {self.hparams.data_dir}")

    @property
    def test_set_name(self):
        """Infix for number of test obstacles and OOD or not."""
        s = ""
        if self.hparams.num_test_obstacles != 1:
            s += f"{self.hparams.num_test_obstacles}obs"
        if self.hparams.test_out_of_distribution:
            s += "-ood"
        return s.lstrip("-")

    def get_horizon(self, split: str):
        if split in ["predict", "test"]:
            return self.hparams.prediction_horizon or self.hparams.horizon
        else:
            return self.hparams.horizon

    def _check_args(self):
        h = self.hparams.horizon
        w = self.hparams.window
        assert isinstance(h, list) or h > 0, f"horizon must be > 0 or a list, but is {h}"
        assert w > 0, f"window must be > 0, but is {w}"

    def _get_split_dir(self, split: str) -> str | None:
        if self.subdirs[split] in ["", None]:
            return None
        else:
            return join(self.hparams.data_dir, self._first_subdir, self.subdirs[split])

    def _get_numpy_filename(self, split: str) -> Optional[str]:
        return join(self.hparams.data_dir, self._first_subdir, self.subdirs[split], "trajectories.npz")

    def get_trajectories_dataset(self, split: str) -> TrajectoryDataset:
        split_dir = self._get_split_dir(split)
        assert split_dir is not None, f"Could not find split directory for split {split}"
        if split == "predict":
            print(f"Using max_samples=1 for prediction set {split_dir}")
            kwargs = {"max_samples": 1}
        else:
            kwargs = {}
        return TrajectoryDataset(split_dir, **kwargs)

    def update_predict_data(self, trajectory_subdir: str):
        self.subdirs["predict"] = trajectory_subdir
        assert os.path.isdir(
            self._get_numpy_filename("predict")
        ), f"Could not find data for prediction in {self._get_numpy_filename('predict')}"

    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""
        assert stage in ["fit", "validate", "test", "predict", None], f"Invalid stage {stage}"
        print(f"Setting up PhysicalSystemsBenchmarkDataModule for stage {stage}...")
        # Set the correct tensor datasets for the train, val, and test sets
        ds_train = self.get_trajectories_dataset("train") if stage in ["fit", None] else None
        ds_val = self.get_trajectories_dataset("val") if stage in ["fit", "validate", None] else None
        ds_test = self.get_trajectories_dataset("test") if stage in ["test", None] else None
        ds_predict = self.get_trajectories_dataset("predict") if stage == "predict" else None
        ds_splits = {"train": ds_train, "val": ds_val, "test": ds_test, "predict": ds_predict}

        for split, split_ds in ds_splits.items():
            dkwargs = {"split": split, "dataset": split_ds, "keep_trajectory_dim": False}  # split == "test"}
            if split_ds is None:
                continue
            elif self.hparams.multi_horizon:
                numpy_tensors = self.create_dataset_multi_horizon(**dkwargs)
            else:
                numpy_tensors = self.create_dataset_single_horizon(**dkwargs)

            # Create the pytorch tensor dataset
            # For the test set, we keep the trajectory dimension, so that we can evaluate the predictions
            # on the full trajectories, thus the test dataset will have a length of num_trajectories
            tensor_ds = MyTensorDataset(numpy_tensors, dataset_id=split)
            # Save the tensor dataset to self._data_{split}
            setattr(self, f"_data_{split}", tensor_ds)
            assert getattr(self, f"_data_{split}") is not None, f"Could not create {split} dataset"

        # Print sizes of the datasets (how many examples)
        self.print_data_sizes(stage)

    def create_dataset_single_horizon(
        self, split: str, dataset: TrajectoryDataset, keep_trajectory_dim: bool = False
    ) -> Dict[str, np.ndarray]:
        """Create a torch dataset from the given TrajectoryDataset and return it."""
        data = self.create_dataset_multi_horizon(split, dataset, keep_trajectory_dim)
        dynamics = data.pop("dynamics")
        window, horizon = self.hparams.window, self.get_horizon(split)
        assert dynamics.shape[1] == window + horizon, f"Expected dynamics to have shape (b, {window + horizon}, ...)"
        inputs = dynamics[:, :window, ...]
        targets = dynamics[:, -1, ...]
        return {"inputs": inputs, "targets": targets, **data}

    def create_dataset_multi_horizon(
        self, split: str, dataset: TrajectoryDataset, keep_trajectory_dim: bool = False
    ) -> Dict[str, np.ndarray]:
        """Create a numpy dataset from the given xarray dataset and return it."""
        # dataset is 4D tensor with dimensions (grid-box, time, lat, lon)
        # Create a tensor, X, of shape (batch-dim, horizon, lat, lon),
        # where each X[i] is a temporal sequence of horizon time steps
        window, horizon = self.hparams.window, self.get_horizon(split)
        trajectories = dict()
        # go through all trajectories and concatenate them in the 2nd dimension (new axis)
        n_trajectories = len(dataset)
        if self.hparams.num_trajectories is not None and split == "train":
            n_trajectories = min(n_trajectories, self.hparams.num_trajectories)

        for i in range(n_trajectories):
            traj_i = dataset[i]
            traj_len = traj_i.trajectory_meta["num_time_steps"]
            time_len = traj_len - horizon - window + 1  # number of examples for this trajectory

            dynamics_i = traj_i.features
            raise_if_invalid_shape(dynamics_i, traj_len, axis=0, name="dynamics_i")
            # Repeat extra_fixed_mask for each example in the trajectory (it is the same for all examples)
            extra_fixed_mask = np.repeat(np.expand_dims(traj_i.condition, axis=0), time_len, axis=0)

            # To save memory, we create the dataset through sliding window views
            dynamics_i = np.lib.stride_tricks.sliding_window_view(dynamics_i, time_len, axis=0)
            dynamics_i = rearrange(dynamics_i, "horizon c h w example -> example horizon c h w")
            raise_if_invalid_shape(dynamics_i, time_len, axis=0, name="dynamics_i")
            raise_if_invalid_shape(extra_fixed_mask, time_len, axis=0, name="extra_fixed_mask")
            if keep_trajectory_dim:
                dynamics_i = np.expand_dims(dynamics_i, axis=0)
                extra_fixed_mask = np.expand_dims(extra_fixed_mask, axis=0)
            # add to the dataset
            condition_name = "condition"  # this should be the same as in the model (the keyword argument)
            traj_i.trajectory_meta["t"] = traj_i.t
            traj_i.trajectory_meta["fixed_mask"] = traj_i.fixed_mask
            if self.hparams.physical_system == "navier-stokes":
                traj_i.trajectory_meta["vertices"] = traj_i.vertices
            elif self.hparams.physical_system == "spring-mesh":
                traj_i.trajectory_meta["features"] = traj_i.features

            traj_metadata = [traj_i.trajectory_meta] * time_len
            if i == 0:
                trajectories["dynamics"] = dynamics_i
                trajectories[condition_name] = extra_fixed_mask
                trajectories["metadata"] = traj_metadata
            else:
                trajectories["dynamics"] = np.concatenate([trajectories["dynamics"], dynamics_i], axis=0)
                trajectories[condition_name] = np.concatenate([trajectories[condition_name], extra_fixed_mask], axis=0)
                trajectories["metadata"] = trajectories["metadata"] + traj_metadata
        # print(f'Shapes={trajectories["dynamics"].shape}, {trajectories["extra_condition"].shape}')
        # E.g. with 90 total examples, horizon=5, window=1: Shapes=(90, 6, 3, 221, 42), (90, 2, 221, 42)
        return trajectories

    def boundary_conditions(
        self,
        preds: Tensor,
        targets: Tensor,
        metadata,
        time: float = None,
    ) -> Tensor:
        batch_size = targets.shape[0]
        if self.hparams.physical_system == "navier-stokes":
            for b_i in range(targets.shape[0]):
                t_i = time if isinstance(time, float) else time[b_i].item()
                in_velocity = float(metadata["in_velocity"][b_i].item())
                # in_velocity = in_velocity[0].item()
                fixed_mask_solutions_pressures = metadata["fixed_mask"][b_i, ...]
                assert (
                    fixed_mask_solutions_pressures.shape == preds.shape[-3:]
                ), f"fixed_mask_solutions_pressures={fixed_mask_solutions_pressures.shape}, predictions={preds.shape}"
                vertex_y = metadata["vertices"][b_i, 1, 0, :]

                left_boundary_indexing = torch.zeros((3, 221, 42), dtype=torch.bool)
                left_boundary_indexing[0, 0, :] = True  # only for first p
                left_boundary = (
                    in_velocity * 4 * vertex_y * (0.41 - vertex_y) / (0.41 * 0.41) * (1 - math.exp(-5 * t_i))
                )
                left_boundary = left_boundary.unsqueeze(0)

                # the predictions should be of shape (*, 3, 221, 42)
                preds[b_i, ..., fixed_mask_solutions_pressures] = 0
                preds[b_i, ..., left_boundary_indexing] = left_boundary
        elif self.hparams.physical_system == "spring-mesh":
            for b_i in range(targets.shape[0]):
                fixed_mask_pq = metadata["fixed_mask"][b_i]  # 4 channels, 2 for p, 2 for q
                assert fixed_mask_pq.shape[0] == 4, f"fixed_mask_pq={fixed_mask_pq.shape}, should be (4, 10, 10)"
                base_q = metadata["features"][b_i, 0, 2:]  # select batch elem and first time step
                # concatenate 0 tensor (for p) to base_q
                boundary_cond = torch.cat([torch.zeros_like(base_q), base_q], dim=0)
                if preds.ndim == 5 and preds.shape[1] == batch_size:
                    preds[:, b_i, ...] = torch.where(fixed_mask_pq, boundary_cond, preds[:, b_i, ...])
                else:
                    preds[b_i, ...] = torch.where(fixed_mask_pq, boundary_cond, preds[b_i, ...])
            # preds[..., fixed_mask_pq] = boundary_cond
            #     Original:
            #     fm_q = trajectory.fixed_mask_q[0].cpu().numpy().reshape((-1, ))
            #     base_q = trajectory.q[0, 0].cpu().numpy().reshape((-1, ))[fm_q]
            #     fm_p = trajectory.fixed_mask_p[0].cpu().numpy().reshape((-1, ))
            #
            #     def spring_mesh_boundary_condition(q, p, t):
            #         q[:, fm_q] = base_q
            #         p[:, fm_p] = 0
            #         return q, p
        else:
            raise NotImplementedError(f"Boundary conditions for {self.hparams.physical_system} not implemented")
        return preds

    def get_boundary_condition_kwargs(self, batch: Any, batch_idx: int, split: str) -> dict:
        metadata = batch["metadata"]
        t0 = metadata["t"][:, 0]  # .item()
        dt = metadata["time_step_size"]  # [:]  #.detach().cpu()  #.item())
        return dict(t0=t0, dt=dt)
