"""
Author: Salva Rühling Cachay
"""
from __future__ import annotations

import functools
import logging
import os
import random
import re
import subprocess
from inspect import isfunction
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from tensordict import TensorDict
from torch import Tensor


def no_op(*args, **kwargs):
    pass


def identity(X, *args, **kwargs):
    return X


def get_identity_callable(*args, **kwargs) -> Callable:
    return identity


def exists(x):
    return x is not None


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


distribution_params_to_edit = ["loc", "scale"]


def torch_to_numpy(x: Union[Tensor, Dict[str, Tensor]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, TensorDict):
        return {k: v.detach().cpu().numpy() for k, v in x.items()}
        # return x.detach().cpu()   # numpy() not implemented for TensorDict
    elif isinstance(x, dict):
        return {k: torch_to_numpy(v) for k, v in x.items()}
    elif isinstance(x, torch.distributions.Distribution):
        # only move the parameters to cpu
        for k in distribution_params_to_edit:
            if hasattr(x, k):
                setattr(x, k, getattr(x, k).detach().cpu())
        return x
    else:
        return x


def numpy_to_torch(x: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[Tensor, Dict[str, Tensor]]:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, dict):
        return {k: numpy_to_torch(v) for k, v in x.items()}
    # if it's a namedtuple, convert each element
    elif isinstance(x, tuple) and hasattr(x, "_fields"):
        return type(x)(*[numpy_to_torch(v) for v in x])
    elif torch.is_tensor(x):
        return x
    # if is simple int, float, etc., return as is
    elif isinstance(x, (int, float, str)):
        return x
    else:
        raise ValueError(f"Cannot convert {type(x)} to torch.")


def rrearrange(data: Union[Tensor, torch.distributions.Distribution, TensorDict], pattern: str, **axes_lengths):
    """Extend einops.rearrange to work with distributions."""
    if torch.is_tensor(data) or isinstance(data, np.ndarray):
        return rearrange(data, pattern, **axes_lengths)
    elif isinstance(data, torch.distributions.Distribution):
        dist_params = {
            k: rearrange(getattr(data, k), pattern, **axes_lengths)
            for k in distribution_params_to_edit
            if hasattr(data, k)
        }
        return type(data)(**dist_params)
    elif isinstance(data, TensorDict):
        new_data = {k: rrearrange(v, pattern, **axes_lengths) for k, v in data.items()}
        return TensorDict(new_data, batch_size=new_data[list(new_data.keys())[0]].shape)
    elif isinstance(data, dict):
        return {k: rrearrange(v, pattern, **axes_lengths) for k, v in data.items()}
    else:
        raise ValueError(f"Cannot rearrange {type(data)}")


def torch_select(input: Tensor, dim: int, index: int):
    """Extends torch.select to work with distributions."""
    if isinstance(input, torch.distributions.Distribution):
        dist_params = {
            k: torch.select(getattr(input, k), dim, index) for k in distribution_params_to_edit if hasattr(input, k)
        }
        return type(input)(**dist_params)
    else:
        return torch.select(input, dim, index)


def extract_into_tensor(a, t, x_shape):
    """Extracts the values of tensor, a, at the given indices, t.
    Then, add dummy dimensions to broadcast to x_shape."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise():
        return torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()


def get_activation_function(name: str, functional: bool = False, num: int = 1):
    """Returns the activation function with the given name."""
    name = name.lower().strip()

    def get_functional(s: str) -> Optional[Callable]:
        return {
            "softmax": F.softmax,
            "relu": F.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "identity": nn.Identity(),
            None: None,
            "swish": F.silu,
            "silu": F.silu,
            "elu": F.elu,
            "gelu": F.gelu,
            "prelu": nn.PReLU(),
        }[s]

    def get_nn(s: str) -> Optional[Callable]:
        return {
            "softmax": nn.Softmax(dim=1),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "identity": nn.Identity(),
            "silu": nn.SiLU(),
            "elu": nn.ELU(),
            "prelu": nn.PReLU(),
            "swish": nn.SiLU(),
            "gelu": nn.GELU(),
        }[s]

    if num == 1:
        return get_functional(name) if functional else get_nn(name)
    else:
        return [get_nn(name) for _ in range(num)]


def get_normalization_layer(name, dims, num_groups=None, *args, **kwargs):
    """Returns the normalization layer with the given name.

    Args:
        name: name of the normalization layer. Must be one of ['batch_norm', 'layer_norm' 'group', 'instance', 'none']
    """
    if not isinstance(name, str) or name.lower() == "none":
        return None
    elif "batch_norm" == name:
        return nn.BatchNorm2d(num_features=dims, *args, **kwargs)
    elif "layer_norm" == name:
        return nn.LayerNorm(dims, *args, **kwargs)
    elif "instance" in name:
        return nn.InstanceNorm1d(num_features=dims, *args, **kwargs)
    elif "group" in name:
        if num_groups is None:
            # find an appropriate divisor (not robust against weird dims!)
            pos_groups = [int(dims / N) for N in range(2, 17) if dims % N == 0]
            if len(pos_groups) == 0:
                raise NotImplementedError(f"Group norm could not infer the number of groups for dim={dims}")
            num_groups = max(pos_groups)
        return nn.GroupNorm(num_groups=num_groups, num_channels=dims)
    else:
        raise ValueError("Unknown normalization name", name)


def get_loss(name, reduction="mean"):
    """Returns the loss function with the given name."""
    name = name.lower().strip().replace("-", "_")
    if name in ["l1", "mae", "mean_absolute_error"]:
        loss = nn.L1Loss(reduction=reduction)
    elif name in ["l2", "mse", "mean_squared_error"]:
        loss = nn.MSELoss(reduction=reduction)
    elif name in ["smoothl1", "smooth"]:
        loss = nn.SmoothL1Loss(reduction=reduction)
    else:
        raise ValueError(f"Unknown loss function {name}")
    return loss


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def to_dict(obj: Optional[Union[dict, SimpleNamespace]]):
    if obj is None:
        return dict()
    elif isinstance(obj, dict):
        return obj
    else:
        return vars(obj)


def to_DictConfig(obj: Optional[Union[List, Dict]]):
    """Tries to convert the given object to a DictConfig."""
    if isinstance(obj, DictConfig):
        return obj

    if isinstance(obj, list):
        try:
            dict_config = OmegaConf.from_dotlist(obj)
        except ValueError:
            dict_config = OmegaConf.create(obj)

    elif isinstance(obj, dict):
        dict_config = OmegaConf.create(obj)

    else:
        dict_config = OmegaConf.create()  # empty

    return dict_config


def replace_substrings(string: str, replacements: Dict[str, str], ignore_case: bool = False):
    """
    Given a string and a replacement map, it returns the replaced string.
    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :param bool ignore_case: whether the match should be case insensitive
    :rtype: str
    """
    if not replacements:
        # Edge case that'd produce a funny regex and cause a KeyError
        return string

    # If case-insensitive, we need to normalize the old string so that later a replacement
    # can be found. For instance with {"HEY": "lol"} we should match and find a replacement for "hey",
    # "HEY", "hEy", etc.
    if ignore_case:

        def normalize_old(s):
            return s.lower()

        re_mode = re.IGNORECASE

    else:

        def normalize_old(s):
            return s

        re_mode = 0

    replacements = {normalize_old(key): val for key, val in replacements.items()}

    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
    rep_sorted = sorted(replacements, key=len, reverse=True)
    rep_escaped = map(re.escape, rep_sorted)

    # Create a big OR regex that matches any of the substrings to replace
    pattern = re.compile("|".join(rep_escaped), re_mode)

    # For each match, look up the new string in the replacements, being the key the normalized old string
    return pattern.sub(lambda match: replacements[normalize_old(match.group(0))], string)


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


#####
# The following two functions extend setattr and getattr to support chained objects, e.g. rsetattr(cfg, optim.lr, 1e-4)
# From https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rhasattr(obj, attr, *args):
    def _hasattr(obj, attr):
        return hasattr(obj, attr, *args)

    return functools.reduce(_hasattr, [obj] + attr.split("."))


def to_tensordict(x: Dict[str, torch.Tensor], force_same_device: bool = False, device=None) -> TensorDict:
    """Converts a dictionary of tensors to a TensorDict."""
    if torch.is_tensor(x):
        return x
    any_batch_example = x[list(x.keys())[0]]
    device = any_batch_example.device if force_same_device else device
    return TensorDict(x, batch_size=any_batch_example.shape, device=device)


# Errors
def raise_error_if_invalid_value(value: Any, possible_values: Sequence[Any], name: str = None):
    """Raises an error if the given value (optionally named by `name`) is not one of the possible values."""
    if value not in possible_values:
        name = name or (value.__name__ if hasattr(value, "__name__") else "value")
        raise ValueError(f"{name} must be one of {possible_values}, but was {value}")
    return value


def raise_error_if_has_attr_with_invalid_value(obj: Any, attr: str, possible_values: Sequence[Any]):
    if hasattr(obj, attr):
        raise_error_if_invalid_value(getattr(obj, attr), possible_values, name=f"{obj.__class__.__name__}.{attr}")


def raise_error_if_invalid_type(value: Any, possible_types: Sequence[Any], name: str = None):
    """Raises an error if the given value (optionally named by `name`) is not one of the possible types."""
    if all([not isinstance(value, t) for t in possible_types]):
        name = name or (value.__name__ if hasattr(value, "__name__") else "value")
        raise ValueError(f"{name} must be an instance of either of {possible_types}, but was {type(value)}")
    return value


def raise_if_invalid_shape(
    value: Union[np.ndarray, Tensor], expected_shape: Sequence[int] | int, axis: int = None, name: str = None
):
    if isinstance(expected_shape, int):
        if value.shape[axis] != expected_shape:
            name = name or (value.__name__ if hasattr(value, "__name__") else "value")
            raise ValueError(f"{name} must have shape {expected_shape} along axis {axis}, but shape={value.shape}")
    else:
        if value.shape != expected_shape:
            name = name or (value.__name__ if hasattr(value, "__name__") else "value")
            raise ValueError(f"{name} must have shape {expected_shape}, but was {value.shape}")


# allow checkpointing via USR1
def melk(trainer, ckptdir: str):
    def actual_melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            # print("Is file: last.ckpt ?", os.path.isfile(os.path.join(ckptdir, "last.ckpt")))
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    return actual_melk


def divein(trainer):
    def actual_divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb

            pudb.set_trace()

    return actual_divein


# Random seed (if not using pytorch-lightning)
def set_seed(seed, device="cuda"):
    """
    Sets the random seed for the given device.
    If using pytorch-lightning, preferably to use pl.seed_everything(seed) instead.
    """
    # setting seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != "cpu":
        torch.cuda.manual_seed(seed)


def auto_gpu_selection(
    usage_max: float = 0.2, mem_max: float = 0.6, num_gpus: int = 1, raise_error_if_insufficient_gpus: bool = True
):
    """Auto set CUDA_VISIBLE_DEVICES for gpu  (based on utilization)

    Args:
        usage_max: max percentage of GPU memory
        mem_max: max percentage of GPU utility
        num_gpus: number of GPUs to use
        raise_error_if_insufficient_gpus: raise error if no (not enough) GPU is available
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    try:
        log_output = str(subprocess.check_output("nvidia-smi", shell=True)).split(r"\n")[6:-1]
    except subprocess.CalledProcessError as e:
        print(
            f"Error with code {e.returncode}. There's likely an issue with nvidia-smi."
            f" Returning without setting CUDA_VISIBLE_DEVICES"
        )
        return

    # Maximum of GPUS, 8 is enough for most
    gpu_to_utilization, gpu_to_mem = dict(), dict()
    gpus_available = torch.cuda.device_count()
    for gpu in range(gpus_available):
        idx = gpu * 4 + 3
        if idx > log_output.__len__() - 1:
            break
        inf = log_output[idx].split("|")
        if inf.__len__() < 3:
            break

        try:
            usage = int(inf[3].split("%")[0].strip())
        except ValueError:
            print("Error with code. Returning without setting CUDA_VISIBLE_DEVICES")
            return
        mem_now = int(str(inf[2].split("/")[0]).strip()[:-3])
        mem_all = int(str(inf[2].split("/")[1]).strip()[:-3])

        if usage < 100 * usage_max and mem_now < mem_max * mem_all:
            gpu_to_utilization[gpu] = usage
            gpu_to_mem[gpu] = mem_now
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"GPU {gpu} is vacant: Memory:[{mem_now}/{mem_all}MiB] , GPU-Util:[{usage}%]")
        else:
            print(f"GPU {gpu} is busy: Memory:[{mem_now}/{mem_all}MiB] , GPU-Util:[{usage}%] (> {usage_max * 100}%)")

    if len(gpu_to_utilization) >= num_gpus:
        least_utilized_gpus = sorted(gpu_to_utilization, key=gpu_to_utilization.get)[:num_gpus]
        sorted(gpu_to_mem, key=gpu_to_mem.get)[:num_gpus]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in least_utilized_gpus])
        if num_gpus > 1:
            print(f"Use GPUs {least_utilized_gpus} based on least utilization")
    else:
        if raise_error_if_insufficient_gpus:
            raise ValueError("No vacant GPU")
        print("\nNo vacant GPU, use CPU instead\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_pl_trainer_kwargs_for_evaluation(trainer_config: DictConfig = None) -> (Dict[str, Any], torch.device):
    """Get kwargs for pytorch-lightning Trainer for evaluation and select <=1 GPU if available"""
    # GPU or not:
    if torch.cuda.is_available() and (trainer_config is None or trainer_config.accelerator != "cpu"):
        accelerator, devices, reload_to_device = "gpu", 1, torch.device("cuda:0")
        auto_gpu_selection(usage_max=0.6, mem_max=0.75, num_gpus=devices)
    else:
        accelerator, devices, reload_to_device = "cpu", "auto", torch.device("cpu")
    return dict(accelerator=accelerator, devices=devices, strategy="auto"), reload_to_device


def infer_main_batch_key_from_dataset(dataset: torch.utils.data.Dataset) -> str:
    ds = dataset
    main_data_key = None
    if hasattr(ds, "main_data_key"):
        main_data_key = ds.main_data_key
    else:
        data_example = ds[0]
        if isinstance(data_example, dict):
            if "dynamics" in data_example:
                main_data_key = "dynamics"
            elif "data" in data_example:
                main_data_key = "data"
            else:
                raise ValueError(f"Could not determine main_data_key from data_example: {data_example.keys()}")
    return main_data_key


# Checkpointing
def get_epoch_ckpt_or_last(ckpt_files: List[str], epoch: int = None):
    if epoch is None:
        if "last.ckpt" in ckpt_files:
            model_ckpt_filename = "last.ckpt"
        else:
            ckpt_epochs = [int(name.replace("epoch", "")[:3]) for name in ckpt_files]
            # Use checkpoint with the latest epoch if epoch is not specified
            max_epoch = max(ckpt_epochs)
            model_ckpt_filename = [name for name in ckpt_files if str(max_epoch) in name][0]
        logging.warning(f"Multiple ckpt files exist: {ckpt_files}. Using latest epoch: {model_ckpt_filename}")
    else:
        # Use checkpoint with specified epoch
        model_ckpt_filename = [name for name in ckpt_files if str(epoch) in name]
        if len(model_ckpt_filename) == 0:
            raise ValueError(f"There is no ckpt file for epoch={epoch}. Try one of the ones in {ckpt_files}!")
        model_ckpt_filename = model_ckpt_filename[0]
    return model_ckpt_filename


def get_local_ckpt_path(config: DictConfig, **kwargs):
    ckpt_direc = config.callbacks.model_checkpoint.dirpath
    if not os.path.isdir(ckpt_direc):
        logging.warning(f"Ckpt directory {ckpt_direc} does not exist. Are you sure the ckpt is on this file-system?.")
        return None
    ckpt_filenames = [f for f in os.listdir(ckpt_direc) if os.path.isfile(os.path.join(ckpt_direc, f))]
    filename = get_epoch_ckpt_or_last(ckpt_filenames, **kwargs)
    return os.path.join(ckpt_direc, filename)


def rename_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> (Dict[str, torch.Tensor], bool):
    #  Missing key(s) in state_dict: "model.downs.0.2.fn.fn.to_qkv.1.weight", "model.downs.1.2.fn.fn.to_qkv.1.weight",
    #  Unexpected key(s) in state_dict: "model.downs.0.2.fn.fn.to_qkv.weight", "model.downs.1.2.fn.fn.to_qkv.weight",
    # rename weights
    renamed = False
    for k in list(state_dict.keys()):
        if "fn.to_qkv.weight" in k and "mid_attn" not in k:
            state_dict[k.replace("fn.to_qkv.weight", "fn.to_qkv.1.weight")] = state_dict.pop(k)
            renamed = True

    return state_dict, renamed


def rename_state_dict_keys_and_save(torch_model_state, ckpt_path: str) -> Dict[str, torch.Tensor]:
    """Renames the state dict keys and saves the renamed state dict back to the checkpoint."""
    state_dict, has_been_renamed = rename_state_dict_keys(torch_model_state["state_dict"])
    if has_been_renamed:
        # Save the renamed model state
        torch_model_state["state_dict"] = state_dict
        torch.save(torch_model_state, ckpt_path)
    return state_dict


def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()  # set to eval mode
    return model


def enable_inference_dropout(model: nn.Module):
    """Set all dropout layers to training mode"""
    # find all dropout layers
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d)]
    for layer in dropout_layers:
        layer.train()
    # assert all([layer.training for layer in [m for m in model.modules() if isinstance(m, nn.Dropout)]])


def disable_inference_dropout(model: nn.Module):
    """Set all dropout layers to eval mode"""
    # find all dropout layers
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d)]
    for layer in dropout_layers:
        layer.eval()


def print_gpu_memory_usage(prefix: str = "", tqdm_bar=None, add_description: bool = True, keep_old: bool = False):
    if torch.cuda.is_available():
        used, allocated = torch.cuda.mem_get_info()
        prefix = f"{prefix} GPU mem free/allocated" if add_description else prefix
        info_str = f"{prefix} {used / 1e9:.2f}/{allocated / 1e9:.2f}GB"
        if tqdm_bar is not None:
            if keep_old:
                tqdm_bar.set_postfix_str(f"{tqdm_bar.postfix} | {info_str}")
            else:
                tqdm_bar.set_postfix_str(info_str)
        else:
            logging.info(info_str)


def update_dict_with_other(d1: Dict[str, Any], other: Dict[str, Any]):  # _and_return_difference
    """Updates d1 with other, other can be a dict of dicts with partial updates.

    Returns:
        d1: the updated dict
        diff: the difference between the original d1 and the updated d1 as a string

    Example:
        d1 = {'a': {'b': 1, 'c': 2}, 'x': 99}
        other = {'a': {'b': 3}, 'y': 100}
        d1, diff = update_dict_with_other(d1, other)
        print(d1)
        # {'a': {'b': 3, 'c': 2}, 'x': 99, 'y': 100}
        print(diff)
        # ['a.b: 1 -> 3', 'y: None -> 100']
    """
    diff = []
    for k, v in other.items():
        if isinstance(v, dict) and d1.get(k) is not None:
            d1[k], diff_sub = update_dict_with_other(d1.get(k, {}), v)
            diff += [f"{k}.{x}" for x in diff_sub]
        else:
            if d1.get(k) != v:
                diff.append(f"{k}: {d1.get(k, None)} -> {v}")
            d1[k] = v
    return d1, diff


if __name__ == "__main__":
    d1 = {"a": {"b": 1, "c": 2}, "x": 99}
    other = {"a": {"b": 3, "c": 2}, "y": 100}
    d1, diff = update_dict_with_other(d1, other)
    print(d1)
    print(diff)

    # auto_gpu_selection(0.8, 0.8, raise_error_if_insufficient_gpus=True)
