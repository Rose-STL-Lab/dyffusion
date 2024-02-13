from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import requests
import wandb
from omegaconf import DictConfig, OmegaConf

from src.utilities.utils import get_logger


# Override this in your project
# -----------------------------------------------------------------------
PROJECT = "DYffusion"
_ENTITY = None  # Set your default entity here, e.g. your wandb username
# -----------------------------------------------------------------------

log = get_logger(__name__)


def get_entity(entity: str = None) -> str:
    if entity is None:
        return _ENTITY or wandb.api.default_entity
    return entity


def get_api(wandb_api: wandb.Api = None) -> wandb.Api:
    if wandb_api is None:
        try:
            wandb_api = wandb.Api(timeout=100)
        except wandb.errors.UsageError:
            wandb.login()
            wandb_api = wandb.Api(timeout=100)
    return wandb_api


def get_run_api(
    run_id: str = None, entity: str = None, project: str = None, run_path: str = None, wandb_api: wandb.Api = None
) -> wandb.apis.public.Run:
    entity, project = get_entity(entity), project or PROJECT
    assert run_path is None or run_id is None, "Either run_path or run_id must be None"
    assert run_id is None or isinstance(run_id, str), f"run_id must be a string, but is {type(run_id)}: {run_id}"
    if entity is None:
        # Get default entity from wandb
        entity = wandb.api.default_entity

    run_path = run_path or f"{entity}/{project}/{run_id}"
    return get_api(wandb_api).run(run_path)


def get_project_runs(
    entity: str = None, project: str = None, wandb_api: wandb.Api = None, **kwargs
) -> List[wandb.apis.public.Run]:
    entity, project = get_entity(entity), project or PROJECT
    return get_api(wandb_api).runs(f"{entity}/{project}", **kwargs)


def get_project_groups(
    entity: str = None, project: str = None, wandb_api: wandb.Api = None
) -> List[wandb.apis.public.Run]:
    runs = get_project_runs(entity, project, wandb_api)
    return list(set([run.group for run in runs]))


def get_runs_for_group(
    group: str,
    entity: str = None,
    project: str = None,
    wandb_api: wandb.Api = None,
    filter_functions: Sequence[Callable] = None,
    only_ids: bool = False,
    verbose: bool = True,
) -> Union[List[wandb.apis.public.Run], List[str]]:
    """Get all runs for a given group"""
    group_runs = get_project_runs(entity, project, wandb_api, filters={"group": group})
    if filter_functions is not None:
        n_groups_before = len(group_runs)
        filter_functions = [filter_functions] if callable(filter_functions) else list(filter_functions)
        group_runs = [run for run in group_runs if all([f(run) for f in filter_functions])]
        if len(group_runs) == 0 and len(filter_functions) > 0 and verbose:
            print(f"Filter functions filtered out all {n_groups_before} runs for group {group}")
        elif n_groups_before == 0:
            print(f"----> No runs for group {group}!! Did you mistype the group name?")

    if only_ids:
        group_runs = [run.id for run in group_runs]
    return group_runs


def get_runs_for_group_with_any_metric(
    wandb_group: str,
    options: List[str] | str,
    option_to_key: Callable[[str], str] | None = None,
    wandb_api=None,
    metric: str = "crps",
    **wandb_kwargs,
) -> (Optional[List[wandb.apis.public.Run]], str):
    """Get all runs for a given group that have any of the given metrics."""
    options = [options] if isinstance(options, str) else options
    option_to_key = option_to_key or (lambda x: x)
    wandb_kwargs2 = wandb_kwargs.copy()
    group_runs, any_metric_key = None, None
    tried_options = []
    for s_i, sum_metric in enumerate(options):
        any_metric_key = f"{option_to_key(sum_metric)}/{metric}".replace("//", "/")
        tried_options.append(any_metric_key)
        filter_func = has_summary_metric(any_metric_key)
        if "filter_functions" not in wandb_kwargs:
            wandb_kwargs2["filter_functions"] = filter_func
        elif "filter_functions" in wandb_kwargs and len(options) > 1:
            wandb_kwargs2["filter_functions"] = wandb_kwargs["filter_functions"] + [filter_func]
        else:
            wandb_kwargs2["filter_functions"] = wandb_kwargs["filter_functions"]
        group_runs = get_runs_for_group(wandb_group, wandb_api=wandb_api, verbose=False, **wandb_kwargs2)
        if len(group_runs) > 0:
            break
    if len(group_runs) == 0:
        logging.warning(
            f"No runs found for group {wandb_group}. "
            f"Possible splits: {options}.\nFull keys that were tried: {tried_options}"
        )
        return None, None
    return group_runs, any_metric_key.replace(f"/{metric}", "")


def get_wandb_ckpt_name(run_path: str, epoch: Union[str, int] = "best") -> str:
    """
    Get the wandb ckpt name for a given run_path and epoch.
    Args:
        run_path: ENTITY/PROJECT/RUN_ID
        epoch: If an int, the ckpt name will be the one for that epoch.
            If 'last' ('best') the latest ('best') epoch ckpt will be returned.

    Returns:
        The wandb ckpt file-name, that can be used as follows to restore the checkpoint locally:
           >>> run_path = "<ENTITY/PROJECT/RUN_ID>"
           >>> ckpt_name = get_wandb_ckpt_name(run_path, epoch)
           >>> wandb.restore(ckpt_name, run_path=run_path, replace=True, root=os.getcwd())
    """
    assert epoch in ["best", "last"] or isinstance(
        epoch, int
    ), f"epoch must be 'best', 'last' or an int, but is {epoch}"
    run_api = get_run_api(run_path=run_path)
    ckpt_files = [f.name for f in run_api.files() if f.name.endswith(".ckpt")]
    if epoch == "best":
        if "best.ckpt" in ckpt_files:
            ckpt_filename = "best.ckpt"
        else:
            raise ValueError(f"Could not find best.ckpt in {ckpt_files}")
    elif "last.ckpt" in ckpt_files and epoch == "last":
        ckpt_filename = "last.ckpt"
    else:
        if len(ckpt_files) == 0:
            raise ValueError(f"Wandb run {run_path} has no checkpoint files (.ckpt) saved in the cloud!")
        elif len(ckpt_files) >= 2:
            ckpt_epochs = [int(name.replace("epoch", "")[:3]) for name in ckpt_files]
            if epoch == "last":
                # Use checkpoint of latest epoch if epoch is not specified
                max_epoch = max(ckpt_epochs)
                ckpt_filename = [name for name in ckpt_files if str(max_epoch) in name][0]
                log.info(f"Multiple ckpt files exist: {ckpt_files}. Using latest epoch: {ckpt_filename}")
            else:
                # Use checkpoint with specified epoch
                ckpt_filename = [name for name in ckpt_files if str(epoch) in name]
                if len(ckpt_filename) == 0:
                    raise ValueError(f"There is no ckpt file for epoch={epoch}. Try one of the ones in {ckpt_epochs}!")
                ckpt_filename = ckpt_filename[0]
        else:
            ckpt_filename = ckpt_files[0]
            log.warning(f"Only one ckpt file exists: {ckpt_filename}. Using it...")
    return ckpt_filename


def restore_model_from_wandb_cloud(
    run_path: str, local_checkpoint_path: str = None, ckpt_filename: str = None, **kwargs
) -> str:
    """
    Restore the model from the wandb cloud to local file-system.
    Args:
        run_path: PROJECT/ENTITY/RUN_ID
        local_checkpoint_path: If not None, the model will be restored from this path.
        ckpt_filename: If not None, the model will be restored from this filename (in the cloud).

    Returns:
        The ckpt filename that can be used to reload the model locally.
    """
    if local_checkpoint_path is True:
        run_id = run_path.split("/")[-1]
        # Search for the best model locally (in the current dir, all files ending with .ckpt)
        ckpt_files = [f for f in os.listdir(os.getcwd()) if f.endswith(".ckpt") and run_id in f]
        local_checkpoints = [f for f in ckpt_files if "best" in f]
        if len(local_checkpoints) == 0:
            if len(ckpt_files) == 0:
                pass  # raise ValueError(f"Could not find any model in local dir: {os.getcwd()}")
            else:
                raise ValueError(
                    f"Could not find any best model in local dir: {os.getcwd()}. Please specify "
                    f"local_checkpoint_path explicitly."
                )
        else:
            assert len(local_checkpoints) == 1, f"Found multiple best models: {local_checkpoints}"
            local_checkpoint_path = local_checkpoints[0]

    if isinstance(local_checkpoint_path, (str,)):
        best_model_fname = local_checkpoint_path
        log.info(f"Restoring model from local absolute path: {os.path.abspath(best_model_fname)}")
    else:
        if ckpt_filename is None:
            ckpt_filename = get_wandb_ckpt_name(run_path, **kwargs)
            ckpt_filename = ckpt_filename.split("/")[-1]  # in case the file contains local dir structure
        # IMPORTANT ARGS replace=True: see https://github.com/wandb/client/issues/3247
        best_model_fname = wandb.restore(ckpt_filename, run_path=run_path, replace=True, root=os.getcwd()).name
    # rename best_model_fname to add a unique prefix to avoid conflicts with other runs
    # (e.g. if the same model is reloaded twice)
    # replace only filename part of the path, not the dir structure
    wandb_id = run_path.split("/")[-1]
    ckpt_fname = (
        os.path.basename(best_model_fname)
        if wandb_id in best_model_fname
        else f"{wandb_id}-{os.path.basename(best_model_fname)}"
    )
    ckpt_path = os.path.join(os.path.dirname(best_model_fname), ckpt_fname)
    if os.path.exists(ckpt_path) and ckpt_path != best_model_fname:
        os.remove(ckpt_path)  # remove if one exists from before
    os.rename(best_model_fname, ckpt_path)
    return ckpt_path


def load_hydra_config_from_wandb(
    run_path: str | wandb.apis.public.Run,
    override_config: Optional[DictConfig] = None,
    override_key_value: List[str] = None,
    update_config_in_cloud: bool = False,
) -> DictConfig:
    """
    Args:
        run_path (str): the wandb ENTITY/PROJECT/ID (e.g. ID=2r0l33yc) corresponding to the config to-be-reloaded
        override_config (DictConfig): each of its keys will override the corresponding entry loaded from wandb
        override_key_value: each element is expected to have a "=" in it, like datamodule.num_workers=8
        update_config_in_cloud: if True, the config in the cloud will be updated with the new overrides
    """
    if override_config is not None and override_key_value is not None:
        log.warning("Both override_config and override_key_value are not None! ")
    if isinstance(run_path, wandb.apis.public.Run):
        run = run_path
        run_path = "/".join(run.path)
    else:
        assert isinstance(
            run_path, str
        ), f"run_path must be a string or wandb.apis.public.Run, but is {type(run_path)}"
        run = get_run_api(run_path=run_path)

    override_key_value = override_key_value or []
    if not isinstance(override_key_value, list):
        raise ValueError(f"override_key_value must be a list of strings, but has type {type(override_key_value)}")
    # copy overrides to new list
    overrides = list(override_key_value.copy())
    rank = os.environ.get("RANK", None) or os.environ.get("LOCAL_RANK", 0)

    # Find latest hydra_config-v{VERSION}.yaml file in wandb cloud
    hydra_config_files = [f.name for f in run.files() if "hydra_config" in f.name]
    if len(hydra_config_files) == 0:
        raise ValueError(f"Could not find any hydra_config file in wandb run {run_path}")
    elif len(hydra_config_files) == 1:
        assert hydra_config_files[0] == "hydra_config.yaml", f"Only one hydra_config file found: {hydra_config_files}"
    else:
        hydra_config_files = [f for f in hydra_config_files if "hydra_config-v" in f]
        assert len(hydra_config_files) > 0, f"Could not find any hydra_config-v file in wandb run {run_path}"
        # Sort by version number (largest is last, earliest are hydra_config.yaml and hydra_config-v1.yaml),
        hydra_config_files = sorted(hydra_config_files, key=lambda x: int(x.split("-v")[-1].split(".")[0]))

    hydra_config_file = hydra_config_files[-1]
    if hydra_config_file != "hydra_config.yaml":
        log.info(f" Reloading from hydra config file: {hydra_config_file}")

    # Download from wandb cloud
    wandb_restore_kwargs = dict(run_path=run_path, replace=True, root=os.getcwd())
    if os.path.exists(hydra_config_file) and rank not in ["0", 0]:
        pass
    else:
        wandb.restore(hydra_config_file, **wandb_restore_kwargs)

    # remove overrides of the form k=v, where k has no dot in it. We don't support this.
    overrides = [o for o in overrides if "=" in o and "." in o.split("=")[0]]
    if len(overrides) != len(override_key_value):
        diff = set(overrides) - set(override_key_value)
        log.warning(f"The following overrides were removed because they are not in the form k=v: {diff}")

    overrides += [
        f"logger.wandb.id={run.id}",
        f"logger.wandb.entity={run.entity}",
        f"logger.wandb.project={run.project}",
        f"logger.wandb.tags={run.tags}",
        f"logger.wandb.group={run.group}",
    ]
    config = OmegaConf.load(hydra_config_file)
    overrides = OmegaConf.from_dotlist(overrides)
    config = OmegaConf.unsafe_merge(config, overrides)

    if override_config is not None:
        for k, v in override_config.items():
            if k in ["model", "trainer"] and isinstance(v, str):
                override_config.pop(k)  # remove key from override_config
                log.warning(f"Key {k} is a string, but it should be a DictConfig. Ignoring it.")
        # override config with override_config (which needs to be the second argument of OmegaConf.merge)
        config = OmegaConf.unsafe_merge(config, override_config)  # unsafe_merge since override_config is not needed

    os.remove(hydra_config_file) if os.path.exists(hydra_config_file) else None
    os.remove(f"../../{hydra_config_file}") if os.path.exists(f"../../{hydra_config_file}") else None

    if run.id != config.logger.wandb.id and run.id in config.logger.wandb.name:
        config.logger.wandb.id = run.id
    assert config.logger.wandb.id == run.id, f"{config.logger.wandb.id} != {run.id}. \nFull Hydra config: {config}"
    if update_config_in_cloud:
        with open("hydra_config.yaml", "w") as fp:
            OmegaConf.save(config, f=fp.name, resolve=True)
        run.upload_file("hydra_config.yaml", root=".")
        os.remove("hydra_config.yaml")
    return config


def reload_checkpoint_from_wandb(
    run_id: str,
    entity: str = None,
    project: str = None,
    ckpt_filename: Optional[str] = None,
    epoch: Union[str, int] = "best",
    override_key_value: Union[Sequence[str], dict] = None,
    local_checkpoint_path: str = None,
    **reload_kwargs,
) -> dict:
    """
    Reload model checkpoint based on only the Wandb run ID

    Args:
        run_id (str): the wandb run ID (e.g. 2r0l33yc) corresponding to the model to-be-reloaded
        entity (str): the wandb entity corresponding to the model to-be-reloaded
        project (str): the project entity corresponding to the model to-be-reloaded
        ckpt_filename (str): the filename of the checkpoint to be reloaded (e.g. 'last.ckpt')
        epoch (str or int): If 'best', the reloaded model will be the best one stored, if 'last' the latest one stored),
                             if an int, the reloaded model will be the one save at that epoch (if it was saved, otherwise an error is thrown)
        override_key_value: If a dict, every k, v pair is used to override the reloaded (hydra) config,
                            e.g. pass {datamodule.num_workers: 8} to change the corresponding flag in config.
                            If a sequence, each element is expected to have a "=" in it, like datamodule.num_workers=8
        local_checkpoint_path (str): If not None, the path to the local checkpoint to be reloaded.
    """
    from src.interface import reload_model_from_config_and_ckpt

    entity, project = get_entity(entity), project or PROJECT
    run_id = str(run_id).strip()
    run_path = f"{entity}/{project}/{run_id}"
    config = load_hydra_config_from_wandb(run_path, override_key_value=override_key_value)

    ckpt_path = restore_model_from_wandb_cloud(
        run_path, local_checkpoint_path, epoch=epoch, ckpt_filename=ckpt_filename
    )
    assert os.path.isfile(ckpt_path), f"Could not find {ckpt_path} in {os.getcwd()}"
    assert config.logger.wandb.id == run_id, f"{config.logger.wandb.id} != {run_id}"

    try:
        reloaded_model_data = reload_model_from_config_and_ckpt(config, ckpt_path, **reload_kwargs)
    except RuntimeError as e:
        raise RuntimeError(
            f"You have probably changed the model code, making it incompatible with older model "
            f"versions. Tried to reload the model ckpt for run.id={run_id} from {ckpt_path}.\n"
            f"config.model={config.model}\n{e}"
        )
    if reloaded_model_data.get("wandb") is not None:
        if reloaded_model_data["wandb"].get("id") != run_id:
            raise ValueError(f"run_id={run_id} != state_dict['wandb']['id']={reloaded_model_data['wandb']['id']}")
    # config.trainer.resume_from_checkpoint = ckpt_path
    # os.remove(ckpt_path) if os.path.exists(ckpt_path) else None  # delete the downloaded ckpt
    return {**reloaded_model_data, "config": config, "ckpt_path": ckpt_path}


def does_any_ckpt_file_exist(wandb_run: wandb.apis.public.Run, only_best_and_last: bool = True) -> bool:
    """
    Check if any checkpoint file exists in the wandb run.
    Args:
        wandb_run: the wandb run to check
        only_best_and_last: if True, only checks for 'best.ckpt' and 'last.ckpt' files, otherwise checks for all ckpt files
                Setting to true may speed up the check, since it will stop as soon as it finds one of the two files.
    """
    names = ["last.ckpt", "best.ckpt"] if only_best_and_last else None
    return len([1 for f in wandb_run.files(names=names) if f.name.endswith(".ckpt")]) > 0


def get_existing_wandb_group_runs(
    config: DictConfig, ckpt_must_exist: bool = True, **kwargs
) -> List[wandb.apis.public.Run]:
    if config.get("logger", None) is None or config.logger.get("wandb", None) is None:
        return []
    wandb_cfg = config.logger.wandb
    runs_in_group = get_runs_for_group(wandb_cfg.group, entity=wandb_cfg.entity, project=wandb_cfg.project)
    try:
        _ = len(runs_in_group)
    except ValueError:  # happens if project does not exist
        return []
    other_runs = [
        run
        for run in runs_in_group
        if (not ckpt_must_exist or does_any_ckpt_file_exist(run, **kwargs))  # and run.state not in ['failed']
    ]
    return other_runs
    # other_seeds = [run.config.get('seed') for run in other_runs]
    # if config.seed in other_seeds:
    #    state = runs_in_group[other_seeds.index(config.seed)].state
    #    log.info(f"Found a run (state={state}) with the same seed (={this_seed}) in group {group}.")
    #    return True
    # return False


def reupload_run_history(run):
    """
    This function can be called when for weird reasons your logged metrics do not appear in run.summary.
    All metrics for each epoch (assumes that a key epoch=i for each epoch i was logged jointly with the metrics),
    will be reuploaded to the wandb run summary.
    """
    summary = {}
    for row in run.scan_history():
        if "epoch" not in row.keys() or any(["gradients/" in k for k in row.keys()]):
            continue
        summary.update(row)
    run.summary.update(summary)


#####################################################################
#
# Pre-filtering of wandb runs
#
def has_finished(run):
    return run.state == "finished"


def has_final_metric(run) -> bool:
    return "test/mse" in run.summary.keys() and "test/mse" in run.summary.keys()


def has_run_id(run_ids: str | List[str]) -> Callable:
    if isinstance(run_ids, str):
        run_ids = [run_ids]
    return lambda run: any([run.id == rid for rid in run_ids])


def contains_in_run_name(name: str) -> Callable:
    return lambda run: name in run.name


def has_summary_metric(metric_name: str, check_non_nan: bool = False) -> Callable:
    metric_name = metric_name.replace("//", "/")

    def has_metric(run):
        return metric_name in run.summary.keys()  # or metric_name in run.summary_metrics.keys()

    def has_metric_non_nan(run):
        return metric_name in run.summary.keys() and not np.isnan(run.summary[metric_name])

    return has_metric_non_nan if check_non_nan else has_metric


def has_summary_metric_any(metric_names: List[str], check_non_nan: bool = False) -> Callable:
    metric_names = [m.replace("//", "/") for m in metric_names]

    def has_metric(run):
        return any([m in run.summary.keys() for m in metric_names])

    def has_metric_non_nan(run):
        return any([m in run.summary.keys() and not np.isnan(run.summary[m]) for m in metric_names])

    return has_metric_non_nan if check_non_nan else has_metric


def has_summary_metric_lower_than(metric_name: str, lower_than: float) -> Callable:
    metric_name = metric_name.replace("//", "/")
    return lambda run: metric_name in run.summary.keys() and run.summary[metric_name] < lower_than


def has_summary_metric_greater_than(metric_name: str, greater_than: float) -> Callable:
    metric_name = metric_name.replace("//", "/")
    return lambda run: metric_name in run.summary.keys() and run.summary[metric_name] > greater_than


def has_minimum_runtime(min_minutes: float = 10.0) -> Callable:
    return lambda run: run.summary.get("_runtime", 0) > min_minutes * 60


def has_minimum_epoch(min_epoch: int = 10) -> Callable:
    def has_min_epoch(run):
        hist = run.history(keys=["epoch"])
        return len(hist) > 0 and max(hist["epoch"]) > min_epoch

    return has_min_epoch


def has_minimum_epoch_simple(min_epoch: int = 10) -> Callable:
    return lambda run: run.summary.get("epoch", 0) > min_epoch


def has_keys(keys: Union[str, List[str]]) -> Callable:
    keys = [keys] if isinstance(keys, str) else keys
    return lambda run: all([(k in run.summary.keys() or k in run.config.keys()) for k in keys])


def hasnt_keys(keys: Union[str, List[str]]) -> Callable:
    keys = [keys] if isinstance(keys, str) else keys
    return lambda run: all([(k not in run.summary.keys() and k not in run.config.keys()) for k in keys])


def has_max_metric_value(metric: str = "test/MERRA2/mse_epoch", max_metric_value: float = 1.0) -> Callable:
    return lambda run: run.summary[metric] <= max_metric_value


def has_tags(tags: Union[str, List[str]]) -> Callable:
    if isinstance(tags, str):
        tags = [tags]
    return lambda run: all([tag in run.tags for tag in tags])


def hasnt_tags(tags: Union[str, List[str]]) -> Callable:
    if isinstance(tags, str):
        tags = [tags]
    return lambda run: all([tag not in run.tags for tag in tags])


def hyperparams_list_api(**hyperparams) -> Dict[str, Any]:
    return {f"config.{hyperparam.replace('.', '/')}": value for hyperparam, value in hyperparams.items()}


def has_hyperparam_values(**hyperparams) -> Callable:
    return lambda run: all(
        hyperparam in run.config and run.config[hyperparam] == value for hyperparam, value in hyperparams.items()
    )


def larger_than(**kwargs) -> Callable:
    return lambda run: all(
        hasattr(run.config, hyperparam) and value > run.config[hyperparam] for hyperparam, value in kwargs.items()
    )


def lower_than(**kwargs) -> Callable:
    return lambda run: all(
        hasattr(run.config, hyperparam) and value < run.config[hyperparam] for hyperparam, value in kwargs.items()
    )


str_to_run_pre_filter = {"has_finished": has_finished, "has_final_metric": has_final_metric}


#####################################################################
#
# Post-filtering of wandb runs (usually when you need to compare runs)
#


def non_unique_cols_dropper(df: pd.DataFrame) -> pd.DataFrame:
    nunique = df.nunique()
    cols_to_drop = nunique[nunique == 1].index
    df = df.drop(cols_to_drop, axis=1)
    return df


def groupby(
    df: pd.DataFrame,
    group_by: Union[str, List[str]] = "seed",
    metrics: List[str] = "val/mse_epoch",
    keep_columns: List[str] = "model/name",
) -> pd.DataFrame:
    """
    Args:
        df: pandas DataFrame to be grouped
        group_by: str or list of str defining the columns to group by
        metrics: list of metrics to compute the group mean and std over
        keep_columns: list of columns to keep in the resulting grouped DataFrame
    Returns:
        A dataframe grouped by `group_by` with columns
        `metric`/mean and `metric`/std for each metric passed in `metrics` and all columns in `keep_columns` remain intact.
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    if isinstance(keep_columns, str):
        keep_columns = [keep_columns]
    if isinstance(group_by, str):
        group_by = [group_by]

    grouped_df = df.groupby(group_by, as_index=False, dropna=False)
    agg_metrics = {m: ["mean", "std"] for m in metrics}
    agg_remain_intact = {c: "first" for c in keep_columns}
    # cols = [group_by] + keep_columns + metrics + ['id']
    stats = grouped_df.agg({**agg_metrics, **agg_remain_intact})
    stats.columns = [(f"{c[0]}/{c[1]}" if c[1] in ["mean", "std"] else c[0]) for c in stats.columns]
    for m in metrics:
        stats[f"{m}/std"].fillna(value=0, inplace=True)

    return stats


str_to_run_post_filter = {
    "unique_columns": non_unique_cols_dropper,
}


def get_wandb_filters_dict_list_from_list(filters_list) -> dict:
    if filters_list is None:
        filters_list = []
    elif not isinstance(filters_list, list):
        filters_list: List[Union[Callable, str]] = [filters_list]
    filters_wandb = []  # dict()
    for f in filters_list:
        if isinstance(f, str):
            f = str_to_run_pre_filter[f.lower()]
        filters_wandb.append(f)
        # filters_wandb = {**filters_wandb, **f}
    return filters_wandb


def get_topk_groups_per_hparam(
    hyperparam_filter: Dict[str, Any],
    monitor: str = "val/20ens_mems/avg/crps.min",
    mode: str = None,
    group_aggregation_func: str = "avg",
    top_k: int = 3,
    min_num_of_runs_per_group: int = 1,
    filter_functions: List[Callable] = None,
    entity: str = None,
    project: str = None,
    **kwargs,
) -> Dict[str, List[wandb.apis.public.Run]]:
    assert min_num_of_runs_per_group > 0, f"min_num_of_runs_per_group must be > 0, got {min_num_of_runs_per_group}"
    if mode is None:
        mode = "max" if "max" in monitor else "min"
    filter_functions = filter_functions or []
    filter_functions += [has_summary_metric(monitor)]
    g_to_run = filter_wandb_runs(
        hyperparam_filter=hyperparam_filter,
        filter_functions=filter_functions,
        aggregate_into_groups=True,
        entity=entity,
        project=project,
        **kwargs,
    )

    def get_val_from_summary(summary):
        if isinstance(summary, dict) or hasattr(summary, "keys"):
            return summary[mode]
        return summary

    # Compute statistic for monitor for each group
    aggregate_func = {"avg": np.mean, "min": np.min, "max": np.max, "median": np.median}[group_aggregation_func]
    g_to_summary = {
        g: aggregate_func([get_val_from_summary(r.summary[monitor]) for r in runs])
        for g, runs in g_to_run.items()
        if len(runs) >= min_num_of_runs_per_group
    }
    # Sort groups by statistic
    g_to_summary = {k: v for k, v in sorted(g_to_summary.items(), key=lambda item: item[1], reverse=mode == "max")}
    # Get top k groups
    topk_groups = list(g_to_summary.keys())[:top_k]
    # Get runs for top k groups
    topk_groups = {g: {"runs": g_to_run[g], "summary": g_to_summary[g]} for g in topk_groups}
    if top_k == 1 and len(topk_groups) == 1:
        group_name = list(topk_groups.keys())[0]
        topk_groups = topk_groups[group_name]
        topk_groups["group"] = group_name
    return topk_groups


def get_run_ids_for_hyperparams(hyperparams: dict, **kwargs) -> List[str]:
    runs = filter_wandb_runs(hyperparams, **kwargs)
    run_ids = [run.id for run in runs]
    return run_ids


def filter_wandb_runs(
    hyperparam_filter: Dict[str, Any] = None,
    extra_filters: Dict[str, Any] = None,
    filter_functions: Sequence[Callable] = None,
    order="-created_at",
    aggregate_into_groups: bool = False,
    entity: str = None,
    project: str = None,
    wandb_api=None,
    verbose: bool = True,
) -> List[wandb.apis.public.Run] or Dict[str, List[wandb.apis.public.Run]]:
    """
    Args:
        hyperparam_filter: a dict str -> value, e.g. {'model/name': 'mlp', 'datamodule/exp_type': 'pristine'}
        filter_functions: A set of callable functions that take a wandb run and return a boolean (True/False) so that
                            any run with one or more return values being False is discarded/filtered out

    Note:
        For more complex/logical filters, see https://www.mongodb.com/docs/manual/reference/operator/query/
    """
    entity = get_entity(entity)
    project = project or PROJECT
    filter_functions = filter_functions or []
    if not isinstance(filter_functions, list):
        filter_functions = [filter_functions]
    filter_functions = [(f if callable(f) else str_to_run_pre_filter[f.lower()]) for f in filter_functions]

    hyperparam_filter = hyperparam_filter or dict()
    api = get_api(wandb_api)

    filter_wandb_api = hyperparams_list_api(**hyperparam_filter)
    if isinstance(extra_filters, dict):
        filter_wandb_api = {**filter_wandb_api, **extra_filters}

    filter_wandb_api = {"$and": [filter_wandb_api]}  # MongoDB query lang
    runs = api.runs(f"{entity}/{project}", filters=filter_wandb_api, per_page=100, order=order)
    n_runs1 = len(runs)

    runs = [run for run in runs if all(f(run) for f in filter_functions)]
    if verbose:
        log.info(f"#Filtered runs = {len(runs)}, (wandb API filtered {n_runs1})")
        if len(runs) == 0:
            log.warning(
                f" No runs found for given filters: {filter_wandb_api} in {entity}/{project}"
                f"\n #Runs before post-filtering: {n_runs1}"
            )
        else:
            log.info(f" Found {len(runs)} runs!")
    if aggregate_into_groups:
        groups = defaultdict(list)
        for run in runs:
            groups[run.group].append(run)
        return groups
    return runs


def get_runs_df(
    get_metrics: bool = True,
    hyperparam_filter: dict = None,
    run_pre_filters: Union[str, List[Union[Callable, str]]] = "has_finished",
    run_post_filters: Union[str, List[str]] = None,
    verbose: int = 1,
    make_hashable_df: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """

    get_metrics:
    run_pre_filters:
    run_post_filters:
    verbose: 0, 1, or 2, where 0 = no output at all, 1 is a bit verbose
    """
    if run_post_filters is None:
        run_post_filters = []
    elif not isinstance(run_post_filters, list):
        run_post_filters: List[Union[Callable, str]] = [run_post_filters]
    run_post_filters = [(f if callable(f) else str_to_run_post_filter[f.lower()]) for f in run_post_filters]

    # Project is specified by <entity/project-name>
    runs = filter_wandb_runs(hyperparam_filter, run_pre_filters, **kwargs)
    summary_list = []
    config_list = []
    group_list = []
    name_list = []
    tag_list = []
    id_list = []
    for i, run in enumerate(runs):
        if i % 50 == 0:
            print(f"Going after run {i}")
        # if i > 100: break
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files
        if "model/_target_" not in run.config.keys():
            if verbose >= 1:
                print(f"Run {run.name} filtered out, as model/_target_ not in run.config.")
            continue

        id_list.append(str(run.id))
        tag_list.append(str(run.tags))
        if get_metrics:
            summary_list.append(run.summary._json_dict)
            # run.config is the hyperparameters
            config_list.append({k: v for k, v in run.config.items() if k not in run.summary.keys()})
        else:
            config_list.append(run.config)

        # run.name is the name of the run.
        name_list.append(run.name)
        group_list.append(run.group)

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({"name": name_list, "id": id_list, "tags": tag_list})
    group_df = pd.DataFrame({"group": group_list})
    all_df = pd.concat([name_df, config_df, summary_df, group_df], axis=1)

    cols = [c for c in all_df.columns if not c.startswith("gradients/") and c != "graph_0"]
    all_df = all_df[cols]
    if all_df.empty:
        raise ValueError("Empty DF!")
    for post_filter in run_post_filters:
        all_df = post_filter(all_df)
    all_df = clean_hparams(all_df)
    if make_hashable_df:
        all_df = all_df.applymap(lambda x: tuple(x) if isinstance(x, list) else x)

    return all_df


def fill_nan_if_present(df: pd.DataFrame, column_key: str, fill_value: Any, inplace=True) -> pd.DataFrame:
    if column_key in df.columns:
        df[column_key] = df[column_key].fillna(fill_value)  # , inplace=inplace)
        # df = df[column_key].apply(lambda x: fill_value if x != x else x)
    return df


def clean_hparams(df: pd.DataFrame):
    # Replace string representation of nan with real nan
    df.replace("NaN", np.nan, inplace=True)
    # df = df.where(pd.notnull(df), None).fillna(value=np.nan)

    # Combine/unify columns of optim/scheduler which might be present in stored params more than once
    combine_cols = [col for col in df.columns if col.startswith("model/optim") or col.startswith("model/scheduler")]
    for col in combine_cols:
        new_col = col.replace("model/", "").replace("optimizer", "optim")
        if not hasattr(df, new_col):
            continue
        getattr(df, new_col).fillna(getattr(df, col), inplace=True)
        del df[col]

    return df


def get_datetime_of_run(run: wandb.apis.public.Run, to_local_timezone: bool = True) -> datetime:
    """Get datetime of a run"""
    dt_str = run.createdAt  # a str like '2023-03-09T08:20:25'
    dt_utc = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    if to_local_timezone:
        return dt_utc.astimezone(tz=None)
    else:
        return dt_utc
    return datetime.fromtimestamp(run.summary["_timestamp"])


def get_unique_groups_for_run_ids(run_ids: Sequence[str], wandb_api: wandb.Api = None, **kwargs) -> List[str]:
    """Get unique groups for a list of run ids"""
    api = get_api(wandb_api)
    groups = []
    for run_id in run_ids:
        run = get_run_api(run_id, wandb_api=api, **kwargs)
        groups.append(run.group)
    return list(set(groups))


def get_unique_groups_for_hyperparam_filter(
    hyperparam_filter: dict, filter_functions: str | List[Union[Callable, str]] = None, **kwargs  # 'has_finished'
) -> List[str]:
    """Get unique groups for a hyperparam filter

    Args:
        hyperparam_filter: dict of hyperparam filters.
        filter_functions: list of filter functions to apply to runs before getting groups.

    Examples:
         Use hyperparam_filter={'datamodule/horizon': 1, 'model/dim': 128} to get all runs with horizon=1 and dim=128
         or {'datamodule/horizon': 1, 'diffusion/timesteps': {'$gte': 10}} for horizon=1 and timesteps >= 10
    """
    runs = filter_wandb_runs(hyperparam_filter, filter_functions=filter_functions, **kwargs)
    groups = [run.group for run in runs]
    return list(set(groups))


def add_summary_metrics(
    run_id: str,
    metric_keys: Union[str, List[str]],
    metric_values: Union[float, List[float]],
    wandb_api: wandb.apis.public.Api = None,
    override: bool = False,
    **kwargs,
):
    """
    Add a metric to the summary of a run.
    """
    wandb_api = get_api(wandb_api)
    run = get_run_api(run_id, wandb_api=wandb_api, **kwargs)
    metric_keys = [metric_keys] if isinstance(metric_keys, str) else metric_keys
    metric_values = [metric_values] if isinstance(metric_values, float) else metric_values
    assert len(metric_keys) == len(
        metric_values
    ), f"metric_keys and metric_values must have same length, but got {len(metric_keys)} and {len(metric_values)}"

    for key, value in zip(metric_keys, metric_values):
        if key in run.summary.keys() and not override:
            print(f"Metric {key} already present in run {run_id}, skipping.")
            return
        run.summary[key] = value
    run.summary.update()


def metrics_of_runs_to_arrays(
    runs: Sequence[wandb.apis.public.Run],
    metrics: Sequence[str],
    columns: Sequence[Any],
    column_to_wandb_key: Callable[[Any], str] | Callable[[Any], List[str]],
    dropna_rows: bool = True,
) -> Dict[str, np.ndarray]:
    """Convert metrics of runs to arrays

    Args:
        runs (list): list of wandb runs (will be the rows of the arrays)
        metrics (list): list of metrics (one array will be created for each metric)
        columns (list): list of columns (will be the columns of the arrays)
        column_to_wandb_key (Callable): function to convert a given column to a wandb key (without metric suffix)
         If it returns a list of keys, the first one will be used to get the metric (if present).
    """

    def column_to_wandb_key_with_metric(wandb_key_stem, metric: str):
        if metric not in wandb_key_stem:
            wandb_key_stem = f"{wandb_key_stem}/{metric}"
        return wandb_key_stem.replace("//", "/")

    def get_summary_metric(run: wandb.apis.public.Run, metric: str, column: Any):
        wandb_keys = column_to_wandb_key(column)
        wandb_keys = [wandb_keys] if isinstance(wandb_keys, str) else wandb_keys
        for wandb_key_stem in wandb_keys:
            wandb_key = column_to_wandb_key_with_metric(wandb_key_stem, metric)
            if wandb_key in run.summary.keys():
                return run.summary[wandb_key]
        return np.nan

    nrows, ncols = len(runs), len(columns)
    arrays = {m: np.zeros((nrows, ncols)) for m in metrics}
    for r_i, run in enumerate(runs):
        if (
            run.project != "DYffusion"
            and np.isnan(get_summary_metric(run, metrics[0], columns[0]))
            and "None" not in column_to_wandb_key(None)
        ):
            full_metric_names = [column_to_wandb_key_with_metric(column_to_wandb_key(None), m) for m in metrics]
            run_metrics = get_summary_metrics_from_history(run, full_metric_names, robust=False)
            for m, fm in zip(metrics, full_metric_names):
                assert len(run_metrics[fm]) >= ncols, f"Expected {ncols} columns, got {len(run_metrics[fm])}"
                if len(run_metrics[fm]) > ncols:
                    run_metrics[fm] = run_metrics[fm][ncols]
                else:
                    arrays[m][r_i, :] = run_metrics[fm]
        else:
            for m in metrics:
                arrays[m][r_i, :] = [get_summary_metric(run, m, c) for c in columns]
    if dropna_rows:
        for m in metrics:
            arrays[m] = arrays[m][~np.isnan(arrays[m]).any(axis=1)]
    return arrays


def get_summary_metrics_from_history(run, metrics: Sequence[str], robust: bool = False):
    """Get summary metrics from history"""
    history = run.history(keys=metrics, pandas=True) if not robust else run.scan_history(keys=metrics)
    # history has one column per metric, one row per step, we want to return one numpy array per metric
    if robust:
        return {m: history[m].to_numpy() for m in metrics}
    else:
        return {m: history[m].to_numpy() for m in metrics}


def wandb_run_summary_update(wandb_run: wandb.apis.public.Run):
    try:
        wandb_run.summary.update()
    except wandb.errors.CommError:
        logging.warning("Could not update wandb summary")
    # except requests.exceptions.HTTPError or requests.exceptions.ConnectionError:
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
        # try again
        wandb_run.summary.update()
    except TypeError:
        pass  # wandb_run.update()
