from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class MyTensorDataset(Dataset[Dict[str, Tensor]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    tensors: Dict[str, Tensor]

    def __init__(self, tensors: Dict[str, Tensor] | Dict[str, np.ndarray], dataset_id: str = ""):
        tensors = {
            key: torch.from_numpy(tensor.copy()).float() if isinstance(tensor, np.ndarray) else tensor
            for key, tensor in tensors.items()
        }
        any_tensor = next(iter(tensors.values()))
        self.dataset_size = any_tensor.size(0)
        for k, value in tensors.items():
            if torch.is_tensor(value):
                assert value.size(0) == self.dataset_size, "Size mismatch between tensors"
            elif isinstance(value, Sequence):
                assert (
                    len(value) == self.dataset_size
                ), f"Size mismatch between list ``{k}`` of length {len(value)} and tensors {self.dataset_size}"
            else:
                raise TypeError(f"Invalid type for tensor {k}: {type(value)}")

        self.tensors = tensors
        self.dataset_id = dataset_id

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return self.dataset_size


def get_tensor_dataset_from_numpy(*ndarrays, dataset_id="", dataset_class=MyTensorDataset, **kwargs):
    tensors = [torch.from_numpy(ndarray.copy()).float() for ndarray in ndarrays]
    return dataset_class(*tensors, dataset_id=dataset_id, **kwargs)


class AutoregressiveDynamicsTensorDataset(Dataset[Tuple[Tensor, ...]]):
    data: Tensor

    def __init__(self, data, horizon: int = 1, **kwargs):
        assert horizon > 0, f"horizon must be > 0, but is {horizon}"
        self.data = data
        self.horizon = horizon

    def __getitem__(self, index):
        # input: index time step
        # output: index + horizon time-steps ahead
        return self.data[index], self.data[index + self.horizon]

    def __len__(self):
        return len(self.data) - self.horizon
